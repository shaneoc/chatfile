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
