use super::*;
use crate::config::ReasoningHistory;
use crate::parser;
use futures::StreamExt;
use std::sync::{Arc, Mutex};
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

// ---------------------------------------------------------------------------
// FinishReason: the finish_reason mapping (§2.5 of docs/format.md)
// ---------------------------------------------------------------------------

#[test]
fn finish_reason_parses_known_values() {
    assert_eq!(FinishReason::parse(Some("stop")), FinishReason::Stop);
    assert_eq!(
        FinishReason::parse(Some("tool_calls")),
        FinishReason::ToolCalls
    );
    assert_eq!(FinishReason::parse(Some("length")), FinishReason::Length);
    assert_eq!(
        FinishReason::parse(Some("content_filter")),
        FinishReason::ContentFilter
    );
}

#[test]
fn finish_reason_missing_or_unknown() {
    assert_eq!(FinishReason::parse(None), FinishReason::Missing);
    assert_eq!(
        FinishReason::parse(Some("stop_with_tools")),
        FinishReason::Unknown("stop_with_tools".into())
    );
}

#[test]
fn finish_reason_flags_and_warning() {
    // Only "length" suppresses tool execution; only "content_filter" sets
    // the filter flag.
    assert!(FinishReason::Length.is_length());
    assert!(!FinishReason::Stop.is_length());
    assert!(!FinishReason::ToolCalls.is_length());
    assert!(!FinishReason::ContentFilter.is_length());
    assert!(FinishReason::ContentFilter.is_content_filter());
    assert!(!FinishReason::Length.is_content_filter());
    // Known values carry no warning; missing and unknown do.
    assert_eq!(FinishReason::Stop.warning(), None);
    assert_eq!(FinishReason::ToolCalls.warning(), None);
    assert_eq!(FinishReason::Length.warning(), None);
    assert_eq!(FinishReason::ContentFilter.warning(), None);
    let missing = FinishReason::Missing.warning().expect("a warning");
    assert!(missing.contains("finish_reason"), "{missing}");
    let unknown = FinishReason::Unknown("weird".into())
        .warning()
        .expect("a warning");
    assert!(unknown.contains("weird"), "{unknown}");
}

#[test]
fn assistant_turn_finish_maps_streamed_reason() {
    let mut turn = AssistantTurn::default();
    turn.push(&StreamDelta::Finished(Some("length".into())));
    assert_eq!(turn.finish(), FinishReason::Length);
    let mut turn = AssistantTurn::default();
    turn.push(&StreamDelta::Finished(None));
    assert_eq!(turn.finish(), FinishReason::Missing);
}

#[test]
fn client_max_output_tokens_builder_and_accessor() {
    // Unset by default: the key is omitted from requests.
    let c = Client::new("http://x", "m", None);
    assert_eq!(c.max_output_tokens(), None);
    let c = c.with_max_output_tokens(Some(99));
    assert_eq!(c.max_output_tokens(), Some(99));
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
    let err = client
        .stream_chat(&[Message::user("hi")], None)
        .await
        .unwrap_err();
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
            StreamDelta::Content("only".into()),
            StreamDelta::Finished(None)
        ]
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
    let mut stream = client
        .stream_chat(&[Message::user("hi")], None)
        .await
        .unwrap();
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
    let mut stream = client
        .stream_chat(&[Message::user("hi")], None)
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let body = &bodies.lock().unwrap()[0];
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    assert!(v.get("tools").is_none());
}

// ---------------------------------------------------------------------------
// build_api_messages: the compaction windowing (§5 of docs/format.md)
// ---------------------------------------------------------------------------

fn conv(md: &str) -> parser::Conversation {
    parser::parse(md).unwrap()
}

/// The `(role, content)` pairs of a built message list.
fn pairs(messages: &[Message]) -> Vec<(&str, &str)> {
    messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_deref().unwrap_or("")))
        .collect()
}

#[test]
fn build_api_messages_without_checkpoint_sends_everything() {
    let c = conv(
        "\
**System**\n\nbe terse\n\n---\n\
**User**\n\nq1\n\n---\n\
**Assistant** | model: m\n\na1\n\n---\n\
**User**\n\nq2\n",
    );
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![
            ("system", "be terse"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
        ]
    );
}

#[test]
fn build_api_messages_checkpoint_windows_history() {
    let c = conv(
        "\
**System**\n\nbe terse\n\n---\n\
**User**\n\nq1\n\n---\n\
**Assistant** | model: m\n\na1\n\n---\n\
**User** | compact: true\n\nsummarize\n\n---\n\
**Assistant** | model: m\n\nsummary\n\n---\n\
**User**\n\nq3\n\n---\n\
**Assistant** | model: m\n\na3\n",
    );
    // q1/a1 (before the checkpoint) are omitted; the window starts at the
    // checkpoint user turn itself.
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![
            ("system", "be terse"),
            ("user", "summarize"),
            ("assistant", "summary"),
            ("user", "q3"),
            ("assistant", "a3"),
        ]
    );
}

#[test]
fn build_api_messages_last_checkpoint_wins() {
    let c = conv(
        "\
**System**\n\nbe terse\n\n---\n\
**User**\n\nq1\n\n---\n\
**Assistant** | model: m\n\na1\n\n---\n\
**User** | compact: true\n\nsummarize 1\n\n---\n\
**Assistant** | model: m\n\nsummary 1\n\n---\n\
**User**\n\nq3\n\n---\n\
**Assistant** | model: m\n\na3\n\n---\n\
**User** | compact: true\n\nsummarize 2\n\n---\n\
**Assistant** | model: m\n\nsummary 2\n\n---\n\
**User**\n\nq5\n",
    );
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![
            ("system", "be terse"),
            ("user", "summarize 2"),
            ("assistant", "summary 2"),
            ("user", "q5"),
        ]
    );
}

#[test]
fn build_api_messages_trailing_compact_request_is_not_a_checkpoint() {
    let c = conv(
        "\
**System**\n\nbe terse\n\n---\n\
**User**\n\nq1\n\n---\n\
**Assistant** | model: m\n\na1\n\n---\n\
**User** | compact: true\n\nsummarize 1\n\n---\n\
**Assistant** | model: m\n\nsummary 1\n\n---\n\
**User** | compact: true\n\nsummarize 2\n",
    );
    // The trailing unanswered `compact: true` request is the prompt being
    // answered, not a checkpoint: it is sent last, the prior active turns
    // (the first checkpoint and its summary) are still sent, and the turns
    // before the previous checkpoint (q1/a1) are omitted.
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![
            ("system", "be terse"),
            ("user", "summarize 1"),
            ("assistant", "summary 1"),
            ("user", "summarize 2"),
        ]
    );
}

#[test]
fn build_api_messages_without_system_section() {
    let c = conv("**User**\n\nq1\n\n---\n**Assistant** | model: m\n\na1\n\n---\n**User**\n\nq2\n");
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![("user", "q1"), ("assistant", "a1"), ("user", "q2")]
    );
}

#[test]
fn build_api_messages_checkpoint_without_system_section() {
    let c = conv(
        "\
**User**\n\nq1\n\n---\n\
**Assistant** | model: m\n\na1\n\n---\n\
**User** | compact: true\n\nsummarize\n\n---\n\
**Assistant** | model: m\n\nsummary\n\n---\n\
**User**\n\nq3\n",
    );
    assert_eq!(
        pairs(&build_api_messages(&c, ReasoningHistory::None)),
        vec![
            ("user", "summarize"),
            ("assistant", "summary"),
            ("user", "q3")
        ]
    );
}

#[test]
fn set_system_prompt_replaces_leading_system_message() {
    let mut messages = build_api_messages(
        &conv("**System**\n\nold\n\n---\n**User**\n\nq\n"),
        ReasoningHistory::None,
    );
    set_system_prompt(&mut messages, "new prompt");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content.as_deref(), Some("new prompt"));
}

#[test]
fn set_system_prompt_inserts_when_no_system_message() {
    let mut messages = build_api_messages(&conv("**User**\n\nq\n"), ReasoningHistory::None);
    set_system_prompt(&mut messages, "new prompt");
    assert_eq!(
        pairs(&messages),
        vec![("system", "new prompt"), ("user", "q")]
    );
}

#[test]
fn set_system_prompt_empty_is_noop() {
    let mut messages = build_api_messages(&conv("**User**\n\nq\n"), ReasoningHistory::None);
    set_system_prompt(&mut messages, "");
    assert_eq!(pairs(&messages), vec![("user", "q")]);
}

// ---------------------------------------------------------------------------
// build_api_messages: reasoning history (reasoning_content)
// ---------------------------------------------------------------------------

/// An assistant turn that issued one tool call and carries persisted
/// reasoning.
fn reasoning_call_turn(reasoning: &str, id: &str) -> parser::Turn {
    parser::Turn {
        role: parser::Role::Assistant,
        content: String::new(),
        meta: parser::TurnMeta::default(),
        reasoning: Some(reasoning.to_string()),
        tool_calls: vec![parser::AssistantToolCall {
            id: id.to_string(),
            name: "list_files".to_string(),
            arguments: "{}".to_string(),
        }],
    }
}

/// `U1, A1 (reasoning + tool call), U2, A2 (reasoning + tool call)`: the
/// last user turn sits between the two assistant turns.
fn reasoning_conv() -> parser::Conversation {
    parser::Conversation {
        turns: vec![
            parser::Turn {
                role: parser::Role::User,
                content: "one".into(),
                meta: parser::TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            reasoning_call_turn("old thought", "call_1"),
            parser::Turn {
                role: parser::Role::User,
                content: "two".into(),
                meta: parser::TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            reasoning_call_turn("new thought", "call_2"),
        ],
        ..Default::default()
    }
}

#[test]
fn build_api_messages_active_turn_resends_reasoning_only_after_last_user() {
    let c = reasoning_conv();
    let msgs = build_api_messages(&c, ReasoningHistory::ActiveTurn);
    assert_eq!(msgs.len(), 4);
    // A1 precedes the last user turn: its reasoning is dropped.
    assert_eq!(msgs[1].reasoning_content, None);
    // A2 is part of the active turn: its reasoning is re-sent.
    assert_eq!(msgs[3].reasoning_content.as_deref(), Some("new thought"));
}

#[test]
fn build_api_messages_all_resends_every_reasoning() {
    let c = reasoning_conv();
    let msgs = build_api_messages(&c, ReasoningHistory::All);
    assert_eq!(msgs[1].reasoning_content.as_deref(), Some("old thought"));
    assert_eq!(msgs[3].reasoning_content.as_deref(), Some("new thought"));
}

#[test]
fn build_api_messages_none_never_resends_reasoning() {
    let c = reasoning_conv();
    let msgs = build_api_messages(&c, ReasoningHistory::None);
    assert!(msgs.iter().all(|m| m.reasoning_content.is_none()));
}

#[test]
fn build_api_messages_non_intermediate_assistant_carries_no_reasoning() {
    // An assistant turn without tool calls is never "intermediate", so its
    // reasoning is dropped even under `all`.
    let c = parser::Conversation {
        turns: vec![
            parser::Turn {
                role: parser::Role::User,
                content: "q".into(),
                meta: parser::TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            parser::Turn {
                role: parser::Role::Assistant,
                content: "answer".into(),
                meta: parser::TurnMeta::default(),
                reasoning: Some("a thought".to_string()),
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let msgs = build_api_messages(&c, ReasoningHistory::All);
    assert!(msgs.iter().all(|m| m.reasoning_content.is_none()));
}

#[test]
fn message_reasoning_content_serializes_only_when_set() {
    let m = Message::assistant("hi");
    assert!(
        !serde_json::to_string(&m)
            .unwrap()
            .contains("reasoning_content")
    );
    let mut m = Message::assistant("hi");
    m.reasoning_content = Some("a thought".into());
    let s = serde_json::to_string(&m).unwrap();
    assert!(s.contains("\"reasoning_content\":\"a thought\""), "{s}");
}
