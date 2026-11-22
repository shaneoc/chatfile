//! OpenAI-compatible chat completion client with SSE streaming.
//!
//! Targets `POST {endpoint}` (default: llama.cpp at
//! `http://localhost:9000/v1/chat/completions`) and yields content deltas
//! as they arrive.

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

/// A chat message in OpenAI wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
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

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a streaming chat completion request.
    ///
    /// Returns a stream of content deltas; the stream ends on `data: [DONE]`
    /// or when the connection closes. HTTP-level failures (non-2xx) are
    /// returned as `Err` before any chunks are produced.
    pub async fn stream_chat(&self, messages: &[Message]) -> Result<SseStream> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
        });

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

/// Result of interpreting one SSE line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseLine {
    /// A content delta (possibly empty).
    Data(String),
    /// Terminal `data: [DONE]` marker.
    Done,
    /// Comment, blank, or non-data line: skip.
    Ignore,
}

/// Interpret a single SSE line into an [`SseLine`].
///
/// Handles `data:` with or without a leading space, `:`-comments, and the
/// terminal `[DONE]` marker. Malformed JSON payloads are ignored so that
/// keep-alives from quirky servers don't kill the stream.
pub fn parse_sse_line(line: &str) -> SseLine {
    let line = line.trim_end();
    if line.is_empty() || line.starts_with(':') {
        return SseLine::Ignore;
    }
    let data = match line.strip_prefix("data:") {
        Some(d) => d.trim_start(),
        None => return SseLine::Ignore,
    };
    if data == "[DONE]" {
        return SseLine::Done;
    }
    if data.is_empty() {
        return SseLine::Ignore;
    }
    match serde_json::from_str::<serde_json::Value>(data) {
        Ok(v) => {
            let content = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            SseLine::Data(content)
        }
        Err(_) => SseLine::Ignore,
    }
}

/// A stream of content deltas decoded from an SSE byte stream.
pub struct SseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buf: String,
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
}

impl Stream for SseStream {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
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
                        if let Some(line) = self.take_line().or_else(|| {
                            (!self.buf.trim().is_empty()).then(|| std::mem::take(&mut self.buf).trim().to_string())
                        }) {
                            match parse_sse_line(&line) {
                                SseLine::Data(c) if !c.is_empty() => {
                                    return Poll::Ready(Some(Ok(c)))
                                }
                                _ => {}
                            }
                        }
                        return Poll::Ready(None);
                    }
                    Poll::Pending => return Poll::Pending,
                },
            };

            match parse_sse_line(&line) {
                SseLine::Data(c) if !c.is_empty() => return Poll::Ready(Some(Ok(c))),
                SseLine::Done => return Poll::Ready(None),
                SseLine::Ignore | SseLine::Data(_) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn parse_data_line_with_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(parse_sse_line(line), SseLine::Data("Hello".into()));
    }

    #[test]
    fn parse_data_line_without_space() {
        let line = r#"data:{"choices":[{"delta":{"content":"x"}}]}"#;
        assert_eq!(parse_sse_line(line), SseLine::Data("x".into()));
    }

    #[test]
    fn parse_done_marker() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseLine::Done);
    }

    #[test]
    fn parse_comments_and_blank_lines_ignored() {
        assert_eq!(parse_sse_line(""), SseLine::Ignore);
        assert_eq!(parse_sse_line(": keep-alive"), SseLine::Ignore);
        assert_eq!(parse_sse_line("data:"), SseLine::Ignore);
    }

    #[test]
    fn parse_non_data_fields_ignored() {
        assert_eq!(parse_sse_line("event: message"), SseLine::Ignore);
        assert_eq!(parse_sse_line("id: 42"), SseLine::Ignore);
    }

    #[test]
    fn parse_malformed_json_ignored() {
        assert_eq!(parse_sse_line("data: {not json"), SseLine::Ignore);
    }

    #[test]
    fn parse_missing_delta_ignored() {
        let line = r#"data: {"choices":[{"delta":{}}]}"#;
        assert_eq!(parse_sse_line(line), SseLine::Data(String::new()));
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
            .stream_chat(&[Message::user("hi")])
            .await
            .unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks, vec!["Hello".to_string(), ",".into(), " world".into()]);
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
        let err = client.stream_chat(&[Message::user("hi")]).await.unwrap_err();
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
        let mut stream = client.stream_chat(&[Message::user("hi")]).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks, vec!["only".to_string()]);
    }
}
