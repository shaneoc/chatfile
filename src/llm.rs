//! OpenAI-compatible chat completion client with SSE streaming.
//!
//! Targets `POST {endpoint}` (default: llama.cpp at
//! `http://localhost:9000/v1/chat/completions`) and yields deltas as they
//! arrive. Thinking models (e.g. Qwen3 served by llama.cpp) stream their
//! reasoning in a separate `reasoning_content` field before the answer
//! starts; both are surfaced as [`StreamDelta`] variants.
//!
//! This module also owns context-window assembly for outgoing requests —
//! [`build_api_messages`] applies the §5 compaction windowing of
//! `docs/format.md`, and [`set_system_prompt`] overlays the hierarchical
//! `AGENTS.md` prompt — plus conversation compaction itself
//! ([`compact_request_turn`], [`run_compact`]).

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use crate::parser;
use crate::tools::{self, LoopEvent, ToolConfig};

/// A chat message in OpenAI wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The wire role: `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// `None` for assistant messages that issued tool calls without text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls issued by an assistant message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For `role: "tool"` messages: the id of the call being answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One tool call in OpenAI wire format (`type` is always `"function"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The unique id the server assigned to this call.
    pub id: String,
    /// Always `"function"` on the OpenAI wire format.
    #[serde(rename = "type", default = "default_call_type")]
    pub call_type: String,
    /// The function being called (name plus JSON arguments).
    pub function: ToolFunction,
}

fn default_call_type() -> String {
    "function".into()
}

/// The function part of a tool call; `arguments` is the JSON-encoded
/// argument object (a string on the wire).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFunction {
    /// The name of the tool to call.
    pub name: String,
    /// The JSON-encoded argument object (a string on the wire).
    pub arguments: String,
}

impl Message {
    /// A message with an explicit role and optional content.
    pub fn new(role: impl Into<String>, content: Option<String>) -> Self {
        Self {
            role: role.into(),
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// A `role: "system"` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", Some(content.into()))
    }

    /// A `role: "user"` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", Some(content.into()))
    }

    /// A `role: "assistant"` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", Some(content.into()))
    }

    /// A tool result answering the call `tool_call_id`.
    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(output.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// The fixed output of a synthetic tool message standing in for a call
/// whose execution was cancelled or interrupted before it could be
/// persisted.
pub const CANCELLED_TOOL_OUTPUT: &str = "Tool execution cancelled or interrupted";

/// Request-prep step that turns a projected history into a valid API
/// payload: for every assistant message, each `tool_call_id` without a
/// matching following `role: "tool"` message gets a synthetic tool message
/// carrying [`CANCELLED_TOOL_OUTPUT`], inserted right after the batch of
/// real tool responses. This is the layer that repairs histories whose
/// file was persisted mid-tool-execution; [`crate::parser::Conversation::to_messages`]
/// itself stays a pure projection.
pub fn complete_tool_calls(messages: &mut Vec<Message>) {
    let mut i = 0usize;
    while i < messages.len() {
        if messages[i].role != "assistant" || messages[i].tool_calls.is_empty() {
            i += 1;
            continue;
        }
        // The batch of tool responses directly following the assistant
        // message.
        let mut j = i + 1;
        while j < messages.len() && messages[j].role == "tool" {
            j += 1;
        }
        for call in messages[i].tool_calls.clone() {
            let answered =
                (i + 1..j).any(|k| messages[k].tool_call_id.as_deref() == Some(call.id.as_str()));
            if !answered {
                messages.insert(
                    j,
                    Message::tool_result(call.id.clone(), CANCELLED_TOOL_OUTPUT),
                );
                j += 1;
            }
        }
        i = j;
    }
}

/// Rebuild the API message list for an outgoing request from a parsed
/// conversation, applying the context-compaction windowing rule (see §5 of
/// `docs/format.md`):
///
/// 1. The original `**System**` section (index 0, when present) is sent
///    first, unchanged.
/// 2. Let the *checkpoint* be the last `**User**` section carrying
///    `compact: true` among the sections **before** a trailing unanswered
///    `**User**` turn at the end of the file (a freshly appended compaction
///    request therefore never truncates the conversation being summarized).
///    When a checkpoint exists, the sections between the system message and
///    the checkpoint are omitted; the window starts at the checkpoint
///    itself (whose `**Assistant**` summary follows it). Without a
///    checkpoint the window starts at the first non-system section.
/// 3. Every section from the window start through the end of the file is
///    projected to wire format (`**Tool**` turns answered, assistant tool
///    calls attached), so a trailing unanswered `**User**` turn — including
///    a new `compact: true` request — is appended last.
///
/// The leading system message carries only the file's `**System**` section
/// content; callers overlay the hierarchical `AGENTS.md` prompt with
/// [`set_system_prompt`] before sending, and repair dangling tool calls
/// with [`complete_tool_calls`].
pub fn build_api_messages(conv: &crate::parser::Conversation) -> Vec<Message> {
    let turns = &conv.turns;
    let n = turns.len();
    let mut messages: Vec<Message> = Vec::new();
    if n > 0 && turns[0].role == crate::parser::Role::System {
        messages.push(Message::system(turns[0].content.clone()));
    }
    // A trailing **User** turn has no **Assistant** reply after it, so it is
    // the prompt being answered (possibly a new `compact: true` request) and
    // is excluded from the checkpoint search.
    let trailing_user = n > 0 && turns[n - 1].role == crate::parser::Role::User;
    let search_end = n - usize::from(trailing_user);
    let checkpoint = (0..search_end)
        .rev()
        .find(|&i| turns[i].role == crate::parser::Role::User && turns[i].meta.compact);
    let start = match checkpoint {
        Some(i) => i,
        None => usize::from(n > 0 && turns[0].role == crate::parser::Role::System),
    };
    messages.extend(conv.project_turns(start..n));
    messages
}

/// Insert or replace the leading `role: "system"` message with
/// `system_prompt` (the hierarchical `AGENTS.md` prompt assembled by
/// [`crate::context::build_system_prompt`]). A no-op when `system_prompt`
/// is empty.
pub fn set_system_prompt(messages: &mut Vec<Message>, system_prompt: &str) {
    if system_prompt.is_empty() {
        return;
    }
    let msg = Message::system(system_prompt.to_string());
    match messages.first_mut() {
        Some(m) if m.role == "system" => *m = msg,
        _ => messages.insert(0, msg),
    }
}

// ---------------------------------------------------------------------------
// Context compaction
// ---------------------------------------------------------------------------

/// The body of the `**User** | compact: true` turn that a compaction
/// appends: it asks the model for a handoff summary (without tool calls)
/// that replaces the pre-checkpoint history in all future requests (§5 of
/// `docs/format.md`).
pub const COMPACT_INSTRUCTION: &str = "\
The conversation above is being compacted to keep its context small. \
Write a handoff summary of it; your reply will be kept while the older \
turns are dropped from future requests. Do not call any tools. Cover, in \
order: **Goal** — what the user asked for; **Key decisions** — technical \
choices made and why; **Files** — files created or modified, with a one-\
line note each; **Current state** — what is done and just verified; \
**Next steps** — what remains, in order.";

/// The `**User** | <timestamp> | compact: true` turn that requests a
/// compaction of the conversation so far.
pub fn compact_request_turn() -> parser::Turn {
    parser::Turn {
        role: parser::Role::User,
        content: COMPACT_INSTRUCTION.to_string(),
        meta: parser::TurnMeta {
            timestamp: Some(parser::now_timestamp()),
            tags: Vec::new(),
            compact: true,
        },
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

/// Drive one context compaction: build the windowed request (the §5
/// compaction rule, the hierarchical system prompt overlaid, and the tool
/// schemas preserved so the server's KV-cache prefix stays warm), stream a
/// single assistant summary, and report its deltas through `on_event`
/// (returning `false` stops early). No tool loop runs: the summary
/// instruction forbids tool calls, so any stray call is ignored. Returns
/// the assembled response; the caller persists it as an `**Assistant**`
/// turn via [`crate::tools::assistant_turn`].
///
/// `conv` must already end with the appended `compact: true` request turn
/// (see [`compact_request_turn`]).
pub async fn run_compact(
    client: &Client,
    conv: &parser::Conversation,
    system_prompt: &str,
    config: &ToolConfig,
    mut on_event: impl FnMut(&LoopEvent) -> bool,
) -> Result<AssistantTurn> {
    let mut messages = build_api_messages(conv);
    set_system_prompt(&mut messages, system_prompt);
    complete_tool_calls(&mut messages);
    let schemas = tools::tool_schemas(config);
    let mut stream = client
        .stream_chat(&messages, (!schemas.is_empty()).then_some(&schemas))
        .await?;
    let mut turn = AssistantTurn::default();
    while let Some(delta) = stream.next().await {
        let delta = delta.context("stream error")?;
        let stop = match &delta {
            StreamDelta::Reasoning(s) => !on_event(&LoopEvent::Reasoning(s.clone())),
            StreamDelta::Content(s) => !on_event(&LoopEvent::Content(s.clone())),
            _ => false,
        };
        turn.push(&delta);
        if stop {
            return Ok(turn);
        }
    }
    Ok(turn)
}

/// Default llama.cpp endpoint.
pub const DEFAULT_ENDPOINT: &str = "http://localhost:9000/v1/chat/completions";

/// Client for an OpenAI-compatible `/v1/chat/completions` endpoint.
#[derive(Debug, Clone)]
pub struct Client {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

impl Client {
    /// Create a client for `endpoint` (a full
    /// `/v1/chat/completions` URL) with the given `model`. When `api_key`
    /// is `Some`, it is sent as a bearer token.
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key,
            http: reqwest::Client::new(),
        }
    }

    /// The endpoint URL this client posts to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The model name sent in every request.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a streaming chat completion request.
    ///
    /// `tools` (JSON schemas in the OpenAI `tools` format) is included in
    /// the request when `Some` and non-empty. Returns a stream of deltas;
    /// the stream ends on `data: [DONE]` or when the connection closes.
    /// HTTP-level failures (non-2xx) are returned as `Err` before any
    /// chunks are produced.
    pub async fn stream_chat(
        &self,
        messages: &[Message],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<SseStream> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
        });
        if let Some(tools) = tools.filter(|t| !t.is_empty()) {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }

        let mut req = self.http.post(&self.endpoint).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("request to {} failed", self.endpoint))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("LLM endpoint returned {status}: {}", text.trim());
        }

        Ok(SseStream::new(resp.bytes_stream()))
    }
}

/// One streamed delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    /// Reasoning text from `delta.reasoning_content` (thinking models).
    Reasoning(String),
    /// Answer text from `delta.content`.
    Content(String),
    /// A tool-call fragment from `delta.tool_calls` (see [`ToolCallDelta`]).
    ToolCall(ToolCallDelta),
    /// The choice's `finish_reason` (e.g. `"stop"`, `"tool_calls"`), sent
    /// once, after all other deltas of the response.
    Finished(Option<String>),
}

/// One tool-call fragment from `delta.tool_calls`.
///
/// Servers stream a tool call in pieces: the first fragment for an index
/// carries the `id` and `function.name`, later fragments carry slices of
/// the JSON `arguments`. The `index` identifies which call a fragment
/// belongs to (multiple calls may interleave).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Position of this call within the assistant message's `tool_calls`.
    pub index: usize,
    /// The call id (usually only in the first fragment).
    pub id: Option<String>,
    /// The function name (usually only in the first fragment).
    pub name: Option<String>,
    /// A slice of the JSON arguments to append.
    pub arguments: String,
}

/// Result of interpreting one SSE line: every field a data payload can
/// carry. A comment/blank/malformed line yields the default (empty) event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The terminal `data: [DONE]` marker.
    pub done: bool,
    /// Reasoning delta from `delta.reasoning_content`, if any.
    pub reasoning: Option<String>,
    /// Content delta from `delta.content`, if any.
    pub content: Option<String>,
    /// Tool-call fragments from `delta.tool_calls`, in order.
    pub tool_calls: Vec<ToolCallDelta>,
    /// The choice's `finish_reason`, when present.
    pub finish_reason: Option<String>,
}

/// Interpret a single SSE line into an [`SseEvent`].
///
/// Handles `data:` with or without a leading space, `:`-comments, and the
/// terminal `[DONE]` marker. Malformed JSON payloads yield an empty event
/// so that keep-alives from quirky servers don't kill the stream.
pub fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim_end();
    if line.is_empty() || line.starts_with(':') {
        return SseEvent::default();
    }
    let data = match line.strip_prefix("data:") {
        Some(d) => d.trim_start(),
        None => return SseEvent::default(),
    };
    if data == "[DONE]" {
        return SseEvent {
            done: true,
            ..Default::default()
        };
    }
    if data.is_empty() {
        return SseEvent::default();
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
        return SseEvent::default();
    };
    let choice = v.get("choices").and_then(|c| c.get(0));
    let mut ev = SseEvent::default();
    if let Some(delta) = choice.and_then(|c| c.get("delta")) {
        ev.reasoning = delta
            .get("reasoning_content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        ev.content = delta
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                ev.tool_calls.push(ToolCallDelta {
                    index: tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize,
                    id: tc.get("id").and_then(|i| i.as_str()).map(str::to_string),
                    name: tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string),
                    arguments: tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }
    }
    ev.finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .map(str::to_string);
    ev
}

/// Accumulates the streamed deltas of one assistant response into a
/// complete message.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantTurn {
    /// Answer text (`delta.content` fragments).
    pub content: String,
    /// Reasoning text (`delta.reasoning_content` fragments).
    pub reasoning: String,
    /// Tool calls accumulated by stream index.
    pub tool_calls: Vec<CompletedToolCall>,
    /// The choice's `finish_reason`, when the server sent one.
    pub finish_reason: Option<String>,
}

/// A fully assembled tool call from a streamed response.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletedToolCall {
    /// The call id (empty if the server never sent one).
    pub id: String,
    /// The function name (empty if the server never sent one).
    pub name: String,
    /// The concatenated JSON arguments.
    pub arguments: String,
}

impl AssistantTurn {
    /// Fold one streamed delta into the accumulator.
    pub fn push(&mut self, delta: &StreamDelta) {
        match delta {
            StreamDelta::Reasoning(s) => self.reasoning.push_str(s),
            StreamDelta::Content(s) => self.content.push_str(s),
            StreamDelta::ToolCall(d) => self.merge_tool_call(d),
            StreamDelta::Finished(r) => self.finish_reason = r.clone(),
        }
    }

    /// True when the response issued tool calls (or the server said so via
    /// `finish_reason`), i.e. the loop must execute tools and continue.
    pub fn wants_tools(&self) -> bool {
        !self.tool_calls.is_empty() || self.finish_reason.as_deref() == Some("tool_calls")
    }

    fn merge_tool_call(&mut self, d: &ToolCallDelta) {
        while self.tool_calls.len() <= d.index {
            self.tool_calls.push(CompletedToolCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        }
        let slot = &mut self.tool_calls[d.index];
        if let Some(id) = &d.id {
            if slot.id.is_empty() {
                slot.id = id.clone();
            }
        }
        if let Some(name) = &d.name {
            if slot.name.is_empty() {
                slot.name = name.clone();
            }
        }
        slot.arguments.push_str(&d.arguments);
    }
}

/// A stream of deltas decoded from an SSE byte stream.
///
/// One data payload can carry several deltas (reasoning + content + tool
/// call fragments), so parsed deltas are buffered and yielded one per poll.
/// The stream always ends with a single [`StreamDelta::Finished`] item.
pub struct SseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buf: String,
    pending: std::collections::VecDeque<StreamDelta>,
    finished: bool,
}

impl std::fmt::Debug for SseStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SseStream")
            .field("buf", &self.buf)
            .finish_non_exhaustive()
    }
}

impl SseStream {
    fn new(inner: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(inner),
            buf: String::new(),
            pending: std::collections::VecDeque::new(),
            finished: false,
        }
    }

    /// Try to consume one complete line from the buffer.
    /// Returns `Some(line)` (without trailing newline) or `None` if the
    /// buffer doesn't yet hold a full line.
    fn take_line(&mut self) -> Option<String> {
        let pos = self.buf.find('\n')?;
        let line: String = self.buf.drain(..=pos).collect();
        Some(line.trim_end_matches(['\r', '\n']).to_string())
    }

    /// Queue the deltas of one parsed event (reasoning, content, tool-call
    /// fragments, finish reason) in order.
    fn queue_event(&mut self, ev: SseEvent) {
        if let Some(r) = ev.reasoning {
            self.pending.push_back(StreamDelta::Reasoning(r));
        }
        if let Some(c) = ev.content {
            self.pending.push_back(StreamDelta::Content(c));
        }
        for tc in ev.tool_calls {
            self.pending.push_back(StreamDelta::ToolCall(tc));
        }
        if let Some(reason) = ev.finish_reason {
            self.finished = true;
            self.pending.push_back(StreamDelta::Finished(Some(reason)));
        }
    }
}

impl Stream for SseStream {
    type Item = Result<StreamDelta>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(delta) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(delta)));
            }
            let line = match self.take_line() {
                Some(line) => line,
                None => match self.inner.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(bytes))) => {
                        self.buf.push_str(&String::from_utf8_lossy(&bytes));
                        continue;
                    }
                    Poll::Ready(Some(Err(e))) => {
                        return Poll::Ready(Some(Err(anyhow::anyhow!("stream error: {e}"))));
                    }
                    Poll::Ready(None) => {
                        // Flush a trailing line without a final newline.
                        let trailing = self.take_line().or_else(|| {
                            (!self.buf.trim().is_empty())
                                .then(|| std::mem::take(&mut self.buf).trim().to_string())
                        });
                        if let Some(line) = trailing {
                            let ev = parse_sse_line(&line);
                            if ev.done {
                                return self.finish_stream();
                            }
                            self.queue_event(ev);
                            continue;
                        }
                        return self.finish_stream();
                    }
                    Poll::Pending => return Poll::Pending,
                },
            };

            let ev = parse_sse_line(&line);
            if ev.done {
                return self.finish_stream();
            }
            self.queue_event(ev);
        }
    }
}

impl SseStream {
    /// End the stream: emit the final [`StreamDelta::Finished`] (with
    /// `None` when the server never sent a `finish_reason`) and then close.
    fn finish_stream(mut self: Pin<&mut Self>) -> Poll<Option<Result<StreamDelta>>> {
        if !self.finished {
            self.finished = true;
            self.pending.push_back(StreamDelta::Finished(None));
        }
        match self.pending.pop_front() {
            Some(delta) => Poll::Ready(Some(Ok(delta))),
            None => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests;
