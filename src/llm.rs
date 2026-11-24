//! OpenAI-compatible chat completion client with SSE streaming.
//!
//! Targets `POST {endpoint}` (default: llama.cpp at
//! `http://localhost:9000/v1/chat/completions`) and yields deltas as they
//! arrive. Thinking models (e.g. Qwen3 served by llama.cpp) stream their
//! reasoning in a separate `reasoning_content` field before the answer
//! starts; both are surfaced as [`StreamDelta`] variants.

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

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
                    index: tc
                        .get("index")
                        .and_then(|i| i.as_u64())
                        .unwrap_or(0) as usize,
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
                        return Poll::Ready(Some(Err(anyhow::anyhow!("stream error: {e}"))))
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
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn parse_data_line_with_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        let ev = parse_sse_line(line);
        assert_eq!(ev.content.as_deref(), Some("Hello"));
        assert_eq!(ev.reasoning, None);
        assert!(ev.tool_calls.is_empty());
        assert!(!ev.done);
    }

    #[test]
    fn parse_data_line_without_space() {
        let line = r#"data:{"choices":[{"delta":{"content":"x"}}]}"#;
        assert_eq!(parse_sse_line(line).content.as_deref(), Some("x"));
    }

    #[test]
    fn parse_reasoning_content_delta() {
        let line = r#"data: {"choices":[{"delta":{"reasoning_content":"We"}}]}"#;
        assert_eq!(parse_sse_line(line).reasoning.as_deref(), Some("We"));
    }

    #[test]
    fn parse_role_only_delta_is_empty_content() {
        let line = r#"data: {"choices":[{"delta":{"role":"assistant","content":null}}]}"#;
        let ev = parse_sse_line(line);
        assert_eq!(ev.content, None);
        assert_eq!(ev.reasoning, None);
    }

    #[test]
    fn parse_done_marker() {
        assert!(parse_sse_line("data: [DONE]").done);
    }

    #[test]
    fn parse_comments_and_blank_lines_ignored() {
        assert_eq!(parse_sse_line(""), SseEvent::default());
        assert_eq!(parse_sse_line(": keep-alive"), SseEvent::default());
        assert_eq!(parse_sse_line("data:"), SseEvent::default());
    }

    #[test]
    fn parse_non_data_fields_ignored() {
        assert_eq!(parse_sse_line("event: message"), SseEvent::default());
        assert_eq!(parse_sse_line("id: 42"), SseEvent::default());
    }

    #[test]
    fn parse_malformed_json_ignored() {
        assert_eq!(parse_sse_line("data: {not json"), SseEvent::default());
    }

    #[test]
    fn parse_missing_delta_ignored() {
        let line = r#"data: {"choices":[{"delta":{}}]}"#;
        assert_eq!(parse_sse_line(line), SseEvent::default());
    }

    #[test]
    fn parse_tool_call_first_fragment() {
        let line = r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read_file","arguments":""}}]}}],"finish_reason":null}"#;
        let ev = parse_sse_line(line);
        assert_eq!(ev.tool_calls.len(), 1);
        assert_eq!(ev.tool_calls[0].index, 0);
        assert_eq!(ev.tool_calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(ev.tool_calls[0].name.as_deref(), Some("read_file"));
        assert_eq!(ev.tool_calls[0].arguments, "");
    }

    #[test]
    fn parse_tool_call_argument_fragments_and_finish() {
        let line = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}],"finish_reason":null}"#;
        let ev = parse_sse_line(line);
        assert_eq!(ev.tool_calls[0].arguments, "{\"path\":");
        assert_eq!(ev.finish_reason, None);

        let line = r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let ev = parse_sse_line(line);
        assert!(ev.tool_calls.is_empty());
        assert_eq!(ev.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn assistant_turn_assembles_tool_calls_by_index() {
        // Two calls, interleaved argument fragments out of order.
        let lines = [
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"list_files","arguments":"{"}},{"index":1,"id":"call_b","function":{"name":"read_file","arguments":"{"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"path\": "}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"subdir\": \".\""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"src/main.rs\"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ];
        let mut turn = AssistantTurn::default();
        for line in lines {
            let ev = parse_sse_line(line);
            for tc in ev.tool_calls {
                turn.push(&StreamDelta::ToolCall(tc));
            }
            if let Some(r) = ev.finish_reason {
                turn.push(&StreamDelta::Finished(Some(r)));
            }
        }
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].id, "call_a");
        assert_eq!(turn.tool_calls[0].name, "list_files");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"subdir": "."}"#);
        assert_eq!(turn.tool_calls[1].id, "call_b");
        assert_eq!(turn.tool_calls[1].name, "read_file");
        assert_eq!(turn.tool_calls[1].arguments, r#"{"path": "src/main.rs"}"#);
        assert_eq!(turn.finish_reason.as_deref(), Some("tool_calls"));
        assert!(turn.wants_tools());
    }

    #[test]
    fn assistant_turn_plain_text_does_not_want_tools() {
        let mut turn = AssistantTurn::default();
        turn.push(&StreamDelta::Content("hello".into()));
        turn.push(&StreamDelta::Finished(Some("stop".into())));
        assert!(!turn.wants_tools());
        assert_eq!(turn.content, "hello");
    }

    /// Spawn a one-shot TCP server that replies with `raw_response`.
    async fn serve_once(raw_response: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Read until the request headers are complete; the body is ignored.
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
            }
            let _ = sock.write_all(&raw_response).await;
            let _ = sock.flush().await;
        });
        format!("http://{addr}/v1/chat/completions")
    }

    #[tokio::test]
    async fn streams_sse_chunks_from_live_endpoint() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\",\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
data: [DONE]\n\n";
        let url = serve_once(raw.to_vec()).await;

        let client = Client::new(url, "test-model", None);
        let mut stream = client
            .stream_chat(&[Message::user("hi")], None)
            .await
            .unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(
            chunks,
            vec![
                StreamDelta::Content("Hello".into()),
                StreamDelta::Content(",".into()),
                StreamDelta::Content(" world".into()),
                StreamDelta::Finished(None)
            ]
        );
    }

    #[tokio::test]
    async fn streams_reasoning_then_content_in_order() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me think.\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" \"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"42\"}}]}\n\n\
data: [DONE]\n\n";
        let url = serve_once(raw.to_vec()).await;

        let client = Client::new(url, "test-model", None);
        let mut stream = client
            .stream_chat(&[Message::user("hi")], None)
            .await
            .unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(
            chunks,
            vec![
                StreamDelta::Reasoning("Let me think.".into()),
                StreamDelta::Reasoning(" ".into()),
                StreamDelta::Content("42".into()),
                StreamDelta::Finished(None)
            ]
        );
    }

    #[tokio::test]
    async fn http_error_is_reported_with_body() {
        let raw = b"HTTP/1.1 400 Bad Request\r\n\
Content-Type: application/json\r\n\
Connection: close\r\n\
Content-Length: 31\r\n\
\r\n\
{\"error\":{\"message\":\"bad req\"}}";
        let url = serve_once(raw.to_vec()).await;

        let client = Client::new(url, "test-model", None);
        let err = client.stream_chat(&[Message::user("hi")], None).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("400"), "unexpected error: {msg}");
        assert!(msg.contains("bad req"), "unexpected error: {msg}");
    }

    #[tokio::test]
    async fn stream_without_done_marker_terminates_on_close() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"only\"}}]}\n\n";
        let url = serve_once(raw.to_vec()).await;

        let client = Client::new(url, "test-model", None);
        let mut stream = client.stream_chat(&[Message::user("hi")], None).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(
            chunks,
            vec![StreamDelta::Content("only".into()), StreamDelta::Finished(None)]
        );
    }

    /// A one-shot server that captures the request body and replies with a
    /// sequence of canned SSE responses, one per connection.
    async fn serve_multi(responses: Vec<Vec<u8>>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let bodies: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&bodies);
        tokio::spawn(async move {
            let mut remaining = responses;
            while let Ok((mut sock, _)) = listener.accept().await {
                let raw = remaining.remove(0);
                // Read the request headers, then exactly the declared body.
                // (Reading to EOF would deadlock: the client keeps the
                // connection open until it gets the response.)
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    match sock.read(&mut tmp).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    }
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..pos + 4]).to_string();
                        let body_len = head
                            .lines()
                            .find_map(|l| {
                                let l = l.trim().to_ascii_lowercase();
                                l.strip_prefix("content-length:")
                                    .and_then(|v| v.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        let have = buf.len() - (pos + 4);
                        let need = body_len.saturating_sub(have);
                        if need > 0 {
                            let mut rest = vec![0u8; need];
                            let _ = sock.read_exact(&mut rest).await;
                            buf.extend_from_slice(&rest);
                        }
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(pos) = text.find("\r\n\r\n") {
                    captured.lock().unwrap().push(text[pos + 4..].to_string());
                } else {
                    captured.lock().unwrap().push(String::new());
                }
                let _ = sock.write_all(&raw).await;
                let _ = sock.flush().await;
            }
        });
        (format!("http://{addr}/v1/chat/completions"), bodies)
    }

    #[tokio::test]
    async fn streamed_tool_calls_assemble_through_stream() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"list_files\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"subdir\\\": \"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\".\\\"}\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";
        let url = serve_once(raw.to_vec()).await;

        let client = Client::new(url, "test-model", None);
        let mut stream = client.stream_chat(&[Message::user("hi")], None).await.unwrap();
        let mut turn = AssistantTurn::default();
        while let Some(delta) = stream.next().await {
            turn.push(&delta.unwrap());
        }
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "call_1");
        assert_eq!(turn.tool_calls[0].name, "list_files");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"subdir": "."}"#);
        assert_eq!(turn.finish_reason.as_deref(), Some("tool_calls"));
        assert!(turn.wants_tools());
    }

    #[tokio::test]
    async fn tools_schemas_are_sent_in_request_body() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
        let (url, bodies) = serve_multi(vec![raw.to_vec()]).await;

        let client = Client::new(url, "test-model", None);
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {"type": "object", "properties": {}},
            }
        })];
        let mut stream = client
            .stream_chat(&[Message::user("hi")], Some(&tools))
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let body = &bodies.lock().unwrap()[0];
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "list_files");
    }

    #[tokio::test]
    async fn no_tools_key_when_disabled() {
        let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
        let (url, bodies) = serve_multi(vec![raw.to_vec()]).await;

        let client = Client::new(url, "test-model", None);
        let mut stream = client.stream_chat(&[Message::user("hi")], None).await.unwrap();
        while stream.next().await.is_some() {}

        let body = &bodies.lock().unwrap()[0];
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert!(v.get("tools").is_none());
    }
}
