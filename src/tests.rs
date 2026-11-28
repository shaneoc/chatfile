use super::*;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A one-shot mock `/v1/chat/completions` server: answers every request with
/// `raw_response` and records each request body (headers stripped).
async fn mock_completions(raw_response: Vec<u8>) -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    mock_completions_multi(vec![raw_response]).await
}

/// A mock `/v1/chat/completions` server that answers the Nth request with
/// `responses[N]` (repeating the last one) and records each request body
/// (headers stripped).
async fn mock_completions_multi(
    responses: Vec<Vec<u8>>,
) -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&bodies);
    tokio::spawn(async move {
        let mut n = 0usize;
        while let Ok((mut sock, _)) = listener.accept().await {
            let raw = responses[n.min(responses.len() - 1)].clone();
            n += 1;
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
async fn compact_appends_request_and_summary_turns() {
    let dir = std::env::temp_dir().join(format!("chatfile-compact-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Bound the AGENTS.md walk to this directory and give it context.
    std::fs::write(dir.join(".chatfile"), "").unwrap();
    std::fs::write(dir.join("AGENTS.md"), "You are terse.").unwrap();

    let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"**Goal** do the thing. **Next steps** finish it.\"}}]}\n\n\
data: [DONE]\n\n";
    let (endpoint, bodies) = mock_completions(raw.to_vec()).await;

    let file = dir.join("conv.chat.md");
    std::fs::write(
        &file,
        format!(
            "---\nmodel: test-model\nendpoint: {endpoint}\ntools: false\n---\n\
             **System**\n\nbe terse\n\n---\n\
             **User**\n\nq1\n\n---\n\
             **Assistant** | model: test-model\n\na1\n\n---\n\
             **User**\n\nq2\n"
        ),
    )
    .unwrap();

    compact(&file, None, &config::AppConfig::default())
        .await
        .unwrap();

    // The request carried the hierarchical system prompt (AGENTS.md + the
    // System section), every section of the file (no prior checkpoint), and
    // ended with the `compact: true` instruction; tools are disabled in the
    // frontmatter, so no `tools` key.
    let body = &bodies.lock().unwrap()[0];
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    let msgs = v["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are terse.\n\nbe terse");
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, vec!["system", "user", "assistant", "user", "user"]);
    assert_eq!(msgs[4]["content"], llm::COMPACT_INSTRUCTION);
    assert!(v.get("tools").is_none());

    // The file keeps its full history: the request turn (persisted before
    // the model call) followed by the summary as an assistant turn.
    let after = std::fs::read_to_string(&file).unwrap();
    let conv = parser::parse(&after).unwrap();
    assert_eq!(conv.turns.len(), 6);
    assert_eq!(conv.turns[4].role, parser::Role::User);
    assert!(conv.turns[4].meta.compact);
    assert_eq!(conv.turns[4].content, llm::COMPACT_INSTRUCTION);
    assert_eq!(conv.turns[5].role, parser::Role::Assistant);
    assert!(conv.turns[5].content.contains("Goal"));
    let model_tag = conv.turns[5]
        .meta
        .tags
        .iter()
        .find(|(k, _)| k == "model")
        .map(|(_, v)| v.as_str());
    assert_eq!(model_tag, Some("test-model"));
    // The file still round-trips exactly (parse/serialize stay bijective).
    assert_eq!(parser::to_markdown(&conv), after);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn run_sends_max_tokens_from_flag_or_frontmatter_and_omits_it_when_unset() {
    let dir = std::env::temp_dir().join(format!("chatfile-runmax-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Bound the AGENTS.md walk to this directory.
    std::fs::write(dir.join(".chatfile"), "").unwrap();

    let raw = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"the reply\"}}]}\n\n\
data: [DONE]\n\n";
    let (endpoint, bodies) = mock_completions(raw.to_vec()).await;

    let write_conv = |name: &str, extra: &str| {
        let file = dir.join(name);
        std::fs::write(
            &file,
            format!(
                "---\nmodel: test-model\nendpoint: {endpoint}\ntools: false\n{extra}---\n\
                 **User**\n\nq\n"
            ),
        )
        .unwrap();
        file
    };

    // The CLI flag lands in the request body.
    let file = write_conv("conv1.chat.md", "");
    run(&file, Some(777), &config::AppConfig::default())
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
    assert_eq!(v["max_tokens"], 777);

    // A frontmatter value applies when the CLI flag is absent.
    let file = write_conv("conv2.chat.md", "max_output_tokens: 512\n");
    run(&file, None, &config::AppConfig::default())
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[1]).unwrap();
    assert_eq!(v["max_tokens"], 512);

    // Nothing set anywhere: the key is omitted from the request.
    let file = write_conv("conv3.chat.md", "");
    run(&file, None, &config::AppConfig::default())
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[2]).unwrap();
    assert!(v.get("max_tokens").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn run_auto_compacts_before_turn_when_persisted_usage_crosses_threshold() {
    let dir = std::env::temp_dir().join(format!("chatfile-autocompact-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Bound the AGENTS.md walk to this directory.
    std::fs::write(dir.join(".chatfile"), "").unwrap();

    // The compaction summary (with usage) and the real reply (with usage).
    let summary = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"**Goal** do the thing. **Next steps** finish it.\"}}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":20,\"total_tokens\":50}}\n\n\
data: [DONE]\n\n";
    let reply = b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Connection: close\r\n\
\r\n\
data: {\"choices\":[{\"delta\":{\"content\":\"the reply\"}}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":10,\"total_tokens\":50}}\n\n\
data: [DONE]\n\n";
    let (endpoint, bodies) = mock_completions_multi(vec![summary.to_vec(), reply.to_vec()]).await;

    let file = dir.join("conv.chat.md");
    std::fs::write(
        &file,
        format!(
            "---\nmodel: test-model\nendpoint: {endpoint}\ntools: false\n---\n\
             **System**\n\nbe terse\n\n---\n\
             **User**\n\nq1\n\n---\n\
             **Assistant** | model: test-model | tokens: 250000\n\na1\n\n---\n\
             **User**\n\nq2\n"
        ),
    )
    .unwrap();

    // context_limit 262144 with the default 80% → threshold 209715 < 250000.
    let mut profiles = std::collections::HashMap::new();
    profiles.insert(
        "p".into(),
        config::ProfileConfig {
            context_limit: Some(262144),
            ..Default::default()
        },
    );
    let config = config::AppConfig {
        default_profile: Some("p".into()),
        profiles,
        ..Default::default()
    };

    run(&file, None, &config).await.unwrap();

    // Two requests: the compaction, then the turn rebuilt from the window.
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let v0: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    let roles0: Vec<&str> = v0["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    // The summary request covers the full history *including* the pending
    // prompt, so the handoff mentions it.
    assert_eq!(roles0, vec!["system", "user", "assistant", "user", "user"]);
    assert_eq!(v0["messages"][4]["content"], llm::COMPACT_INSTRUCTION);

    let v1: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let msgs1 = v1["messages"].as_array().unwrap();
    let roles1: Vec<&str> = msgs1.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles1, vec!["system", "user", "assistant", "user"]);
    // Window = [compact request, summary, the moved prompt].
    assert_eq!(msgs1[1]["content"], llm::COMPACT_INSTRUCTION);
    assert!(msgs1[2]["content"].as_str().unwrap().contains("Goal"));
    assert_eq!(msgs1[3]["content"], "q2");

    // File: history, checkpoint pair, the moved prompt, and the reply.
    let conv = parser::parse(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(conv.turns.len(), 7);
    assert!(conv.turns[3].meta.compact);
    assert_eq!(conv.turns[4].role, parser::Role::Assistant);
    assert_eq!(conv.turns[4].meta.tokens, Some(50));
    assert_eq!(conv.turns[5].content, "q2");
    assert_eq!(conv.turns[6].role, parser::Role::Assistant);
    assert_eq!(conv.turns[6].content, "the reply");
    assert_eq!(conv.turns[6].meta.tokens, Some(50));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_creates_template_and_refuses_overwrite() {
    let dir = std::env::temp_dir().join(format!("chatfile-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let file = dir.join("a").join("b").join("conv.md");

    new(&file, &config::AppConfig::default()).unwrap();
    // A plain `.md` name is normalized to `.chat.md`.
    let created = dir.join("a").join("b").join("conv.chat.md");
    assert!(created.exists());
    let content = std::fs::read_to_string(&created).unwrap();
    assert!(content.starts_with("---\nmodel: local-model\n"));
    assert!(content.contains(llm::DEFAULT_ENDPOINT));
    assert!(content.contains("**User**"));

    // Must parse cleanly, with a single user turn.
    let conv = parser::parse(&content).unwrap();
    assert!(conv.had_frontmatter);
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].role, parser::Role::User);
    assert_eq!(conv.turns[0].content, "Your prompt here");

    // Refuse to overwrite an existing file.
    assert!(new(&file, &config::AppConfig::default()).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_appends_default_extension() {
    let dir = std::env::temp_dir().join(format!("chatfile-newext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let file = dir.join("notes");

    new(&file, &config::AppConfig::default()).unwrap();
    assert!(dir.join("notes.chat.md").exists());

    // An explicit `.chat.md` is left untouched.
    new(&dir.join("exact.chat.md"), &config::AppConfig::default()).unwrap();
    assert!(dir.join("exact.chat.md").exists());
    assert!(!dir.join("exact.chat.chat.md").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_rejects_wrong_extension() {
    let dir = std::env::temp_dir().join(format!("chatfile-runext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("conv.md");
    std::fs::write(&file, "---\n**User**\n\nhi\n").unwrap();

    let err = require_conversation_ext(&file).unwrap_err().to_string();
    assert!(err.contains(".chat.md"), "unexpected error: {err}");
    assert!(require_conversation_ext(&dir.join("ok.chat.md")).is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}
