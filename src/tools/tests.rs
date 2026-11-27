use super::*;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

fn config(dir: &Path) -> ToolConfig {
    ToolConfig {
        enabled: true,
        network: false,
        allow_shell: false,
        work_dir: dir.to_path_buf(),
        allowed_tools: None,
        sandbox: SandboxConfig::default(),
    }
}

// -----------------------------------------------------------------------
// resolve_safe_path
// -----------------------------------------------------------------------

#[test]
fn resolve_safe_path_accepts_relative_paths() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.txt"), "x").unwrap();
    let got = resolve_safe_path(dir.path(), "a.txt").unwrap();
    assert_eq!(got, dir.path().canonicalize().unwrap().join("a.txt"));
}

#[test]
fn resolve_safe_path_nonexistent_nested_path_resolves() {
    let dir = TempDir::new().unwrap();
    let got = resolve_safe_path(dir.path(), "a/b/c.txt").unwrap();
    assert_eq!(got, dir.path().canonicalize().unwrap().join("a/b/c.txt"));
}

#[test]
fn resolve_safe_path_rejects_empty_absolute_and_dotdot() {
    let dir = TempDir::new().unwrap();
    assert!(resolve_safe_path(dir.path(), "").is_err());
    assert!(resolve_safe_path(dir.path(), "   ").is_err());
    assert!(resolve_safe_path(dir.path(), "/etc/passwd").is_err());
    assert!(resolve_safe_path(dir.path(), "../sibling.txt").is_err());
    // `..` in the middle of a path is rejected too.
    assert!(resolve_safe_path(dir.path(), "a/../../b").is_err());
}

#[test]
fn resolve_safe_path_rejects_symlink_escape() {
    let dir = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "top secret").unwrap();
    // A symlinked directory inside the work dir pointing outside.
    std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
    assert!(resolve_safe_path(dir.path(), "link/secret.txt").is_err());
    // A symlinked file pointing outside.
    std::os::unix::fs::symlink(
        outside.path().join("secret.txt"),
        dir.path().join("file_link"),
    )
    .unwrap();
    assert!(resolve_safe_path(dir.path(), "file_link").is_err());
}

// -----------------------------------------------------------------------
// File tools
// -----------------------------------------------------------------------

#[tokio::test]
async fn write_and_read_file_round_trip() {
    let dir = TempDir::new().unwrap();
    let cfg = config(dir.path());
    let out = execute(
        "write_file",
        r#"{"path": "deep/dir/notes.md", "content": "one\ntwo\nthree"}"#,
        &cfg,
    )
    .await;
    assert!(out.starts_with("Wrote"), "{out}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("deep/dir/notes.md")).unwrap(),
        "one\ntwo\nthree"
    );
    let out = execute("read_file", r#"{"path": "deep/dir/notes.md"}"#, &cfg).await;
    assert_eq!(
        out,
        "1 | one\n2 | two\n3 | three\n(3 lines total, showing 1-3)\n"
    );
}

#[tokio::test]
async fn read_file_window_and_past_end() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("n.txt"), "a\nb\nc\nd\ne").unwrap();
    let cfg = config(dir.path());
    let out = execute(
        "read_file",
        r#"{"path": "n.txt", "start_line": 2, "limit": 2}"#,
        &cfg,
    )
    .await;
    assert_eq!(out, "2 | b\n3 | c\n(5 lines total, showing 2-3)\n");
    let out = execute("read_file", r#"{"path": "n.txt", "start_line": 99}"#, &cfg).await;
    assert!(out.contains("past the end"), "{out}");
}

#[tokio::test]
async fn read_file_rejects_binary_and_outside() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("bin.dat"), [0u8, 1, 2]).unwrap();
    let cfg = config(dir.path());
    let out = execute("read_file", r#"{"path": "bin.dat"}"#, &cfg).await;
    assert!(out.contains("binary"), "{out}");
    let out = execute("read_file", r#"{"path": "/etc/passwd"}"#, &cfg).await;
    assert!(out.starts_with("Error:"), "{out}");
}

#[tokio::test]
async fn edit_file_requires_unique_match() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("e.txt"), "foo\nbar\nfoo\n").unwrap();
    let cfg = config(dir.path());

    // Two matches: refused.
    let out = execute(
        "edit_file",
        r#"{"path": "e.txt", "old_text": "foo", "new_text": "qux"}"#,
        &cfg,
    )
    .await;
    assert!(out.contains("matches 2 times"), "{out}");

    // Zero matches: refused with a hint.
    let out = execute(
        "edit_file",
        r#"{"path": "e.txt", "old_text": "nope", "new_text": "x"}"#,
        &cfg,
    )
    .await;
    assert!(out.contains("not found"), "{out}");

    // Unique match: applied exactly once.
    let out = execute(
        "edit_file",
        r#"{"path": "e.txt", "old_text": "bar", "new_text": "baz"}"#,
        &cfg,
    )
    .await;
    assert!(out.starts_with("Edited"), "{out}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("e.txt")).unwrap(),
        "foo\nbaz\nfoo\n"
    );
}

#[tokio::test]
async fn list_files_skips_dotfiles_and_target() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join(".git")).unwrap();
    std::fs::create_dir_all(dir.path().join("target")).unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join(".hidden"), "x").unwrap();
    std::fs::write(dir.path().join("keep.txt"), "12345").unwrap();
    std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
    let cfg = config(dir.path());
    let out = execute("list_files", r#"{"subdir": "."}"#, &cfg).await;
    assert!(out.contains("keep.txt  5"), "{out}");
    assert!(out.contains("src/main.rs  12"), "{out}");
    assert!(!out.contains(".git"), "{out}");
    assert!(!out.contains("target"), "{out}");
    assert!(!out.contains(".hidden"), "{out}");
}

#[tokio::test]
async fn search_files_is_case_insensitive_and_capped() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("a.txt"),
        "Hello world\nnothing\nHELLO again",
    )
    .unwrap();
    let cfg = config(dir.path());
    let out = execute("search_files", r#"{"query": "hello"}"#, &cfg).await;
    assert_eq!(out, "a.txt:1: Hello world\na.txt:3: HELLO again\n");

    // Cap at 50 matches.
    let mut many = String::new();
    for i in 0..60 {
        many.push_str(&format!("needle line {i}\n"));
    }
    std::fs::write(dir.path().join("b.txt"), many).unwrap();
    let out = execute("search_files", r#"{"query": "needle"}"#, &cfg).await;
    assert!(out.contains("truncated at 50 matches"), "{out}");
    let lines: Vec<&str> = out.lines().filter(|l| l.starts_with('b')).collect();
    assert_eq!(lines.len(), 50);
}

#[tokio::test]
async fn unknown_tool_and_bad_json_report_errors() {
    let dir = TempDir::new().unwrap();
    let cfg = config(dir.path());
    let out = execute("nope", "{}", &cfg).await;
    assert!(out.contains("unknown tool"), "{out}");
    let out = execute("read_file", "not json", &cfg).await;
    assert!(out.contains("invalid JSON"), "{out}");
}

// -----------------------------------------------------------------------
// Schemas and frontmatter gating
// -----------------------------------------------------------------------

#[test]
fn tool_schemas_follow_config() {
    let dir = TempDir::new().unwrap();
    let mut cfg = config(dir.path());

    let names: Vec<String> = tool_schemas(&cfg)
        .iter()
        .map(|v| v["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        names,
        vec![
            "list_files",
            "read_file",
            "write_file",
            "edit_file",
            "search_files"
        ]
    );

    cfg.network = true;
    let names: Vec<String> = tool_schemas(&cfg)
        .iter()
        .map(|v| v["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names.last(), Some(&"fetch_url".to_string()));

    cfg.allow_shell = true;
    let names: Vec<String> = tool_schemas(&cfg)
        .iter()
        .map(|v| v["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names.last(), Some(&"bash".to_string()));

    cfg.enabled = false;
    assert!(tool_schemas(&cfg).is_empty());
}

#[tokio::test]
async fn fetch_url_gated_by_network_flag() {
    let dir = TempDir::new().unwrap();
    let mut cfg = config(dir.path());
    let out = execute("fetch_url", r#"{"url": "https://x.test"}"#, &cfg).await;
    assert!(out.contains("network access is disabled"), "{out}");

    cfg.network = true;
    let out = execute("fetch_url", r#"{"url": "ftp://x.test"}"#, &cfg).await;
    assert!(out.contains("only http(s)"), "{out}");
}

#[tokio::test]
async fn bash_gated_by_allow_shell_flag() {
    let dir = TempDir::new().unwrap();
    let cfg = config(dir.path());
    let out = execute("bash", r#"{"command": "echo hi"}"#, &cfg).await;
    assert!(out.contains("allow_shell"), "{out}");
}

/// Run a `bash` tool call in the real sandbox when bwrap is installed;
/// skip otherwise.
async fn sandboxed(command_json: &str, dir: &Path, network: bool) -> Option<String> {
    let ok = std::process::Command::new("bwrap")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: bwrap not installed (e.g. `apt install bubblewrap`)");
        return None;
    }
    let mut cfg = config(dir);
    cfg.allow_shell = true;
    cfg.network = network;
    Some(execute("bash", command_json, &cfg).await)
}

#[tokio::test]
async fn bash_runs_in_sandbox_when_enabled() {
    let dir = TempDir::new().unwrap();
    let Some(out) = sandboxed(r#"{"command": "echo hello && pwd"}"#, dir.path(), false).await
    else {
        return;
    };
    assert!(out.starts_with("exit code: 0"), "{out}");
    assert!(out.contains("hello"), "{out}");
    // The sandboxed cwd is the working directory.
    let cwd = dir
        .path()
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert!(out.contains(&cwd), "{out}");
}

#[tokio::test]
async fn bash_reports_nonzero_exit_codes() {
    let dir = TempDir::new().unwrap();
    let Some(out) = sandboxed(r#"{"command": "echo oops >&2; exit 3"}"#, dir.path(), false).await
    else {
        return;
    };
    assert!(out.starts_with("exit code: 3"), "{out}");
    assert!(out.contains("oops"), "{out}");
}

// -----------------------------------------------------------------------
// The multi-turn loop (mock server)
// -----------------------------------------------------------------------

/// A one-shot-per-connection mock LLM server: each accepted connection
/// gets the next canned response; request bodies are captured.
async fn mock_server(responses: Vec<Vec<u8>>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&bodies);
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut remaining = responses;
        while let Ok((mut sock, _)) = listener.accept().await {
            if remaining.is_empty() {
                break;
            }
            let raw = remaining.remove(0);
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
            }
            let _ = sock.write_all(&raw).await;
            let _ = sock.flush().await;
        }
    });
    (format!("http://{addr}/v1/chat/completions"), bodies)
}

fn sse(body: &str) -> Vec<u8> {
    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}")
        .into_bytes()
}

const TOOL_CALL_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\": \\\"hello.txt\\\", \\\"content\\\": \\\"hi there\\\"}\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";

const FINAL_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Done writing.\"}}]}\n\n\
data: [DONE]\n\n";

const STOP_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"All good.\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

const CONTENT_FILTERED_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Some answer.\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\n\
data: [DONE]\n\n";

const UNKNOWN_FINISH_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Odd end.\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop_with_tools\"}]}\n\n\
data: [DONE]\n\n";

/// A response cut off at the output-token limit mid-stream: partial content
/// and a tool call whose id and arguments are both incomplete.
const LENGTH_TRUNCATED_RESPONSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Partial answ\",\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\": \\\"x.txt\\\"\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
data: [DONE]\n\n";

#[tokio::test]
async fn run_turn_executes_tool_call_then_finalizes() {
    let dir = TempDir::new().unwrap();
    let (url, bodies) = mock_server(vec![sse(TOOL_CALL_RESPONSE), sse(FINAL_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    let mut events = Vec::new();
    run_turn(
        &client,
        vec![Message::user("write hello.txt")],
        &cfg,
        |ev| {
            events.push(format!("{ev:?}"));
            true
        },
    )
    .await
    .unwrap();

    // The tool ran: the file exists with the right content.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
        "hi there"
    );
    // Events: a ToolResult between two ResponseEnds, then final content.
    let tool_idx = events
        .iter()
        .position(|e| e.contains("ToolResult"))
        .expect("a ToolResult event");
    assert!(events[tool_idx].contains("write_file"));
    assert!(events[tool_idx].contains("call_1"));
    let ends = events.iter().filter(|e| e.contains("ResponseEnd")).count();
    assert_eq!(ends, 2);
    assert!(
        events
            .iter()
            .any(|e| e.contains("Content(\"Done writing.\")"))
    );

    // The second request carried the assistant tool call and the tool
    // result message.
    let b = bodies.lock().unwrap();
    assert_eq!(b.len(), 2);
    let v: Json = serde_json::from_str(&b[1]).unwrap();
    let roles: Vec<&str> = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "tool"]);
    assert_eq!(v["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(v["messages"][2]["tool_call_id"], "call_1");
    // The tool message carries the write_file output, not the content.
    assert_eq!(v["messages"][2]["content"], "Wrote 8 bytes to hello.txt\n");
}

#[tokio::test]
async fn run_turn_plain_response_needs_no_tools() {
    let dir = TempDir::new().unwrap();
    let (url, bodies) = mock_server(vec![sse(FINAL_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    let mut content = String::new();
    run_turn(&client, vec![Message::user("hi")], &cfg, |ev| {
        if let LoopEvent::Content(s) = ev {
            content.push_str(s);
        }
        true
    })
    .await
    .unwrap();
    assert_eq!(content, "Done writing.");
    // One request, no tools in the body (tools disabled? no — enabled,
    // but the response simply had no tool calls).
    let b = bodies.lock().unwrap();
    assert_eq!(b.len(), 1);
}

#[tokio::test]
async fn run_turn_stops_at_max_iterations() {
    let dir = TempDir::new().unwrap();
    let responses = vec![sse(TOOL_CALL_RESPONSE); MAX_TOOL_ITERATIONS];
    let (url, _) = mock_server(responses).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    let err = run_turn(&client, vec![Message::user("loop")], &cfg, |_| true)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains(&MAX_TOOL_ITERATIONS.to_string()),
        "{err}"
    );
}

#[tokio::test]
async fn run_turn_stops_when_callback_returns_false() {
    let dir = TempDir::new().unwrap();
    let (url, _) = mock_server(vec![sse(FINAL_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    // The callback refuses after the first event: the loop must stop
    // cleanly without error.
    let mut seen = 0usize;
    run_turn(&client, vec![Message::user("hi")], &cfg, |_| {
        seen += 1;
        false
    })
    .await
    .unwrap();
    assert!(seen >= 1);
}

#[tokio::test]
async fn run_turn_reports_finish_reasons_on_response_end() {
    let dir = TempDir::new().unwrap();
    let (url, _) = mock_server(vec![
        sse(STOP_RESPONSE),
        sse(CONTENT_FILTERED_RESPONSE),
        sse(UNKNOWN_FINISH_RESPONSE),
    ])
    .await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    // One plain response per run_turn: collect the finish each ResponseEnd
    // reports.
    let mut finishes = Vec::new();
    for prompt in ["one", "two", "three"] {
        run_turn(&client, vec![Message::user(prompt)], &cfg, |ev| {
            if let LoopEvent::ResponseEnd { finish, .. } = ev {
                finishes.push(finish.clone());
            }
            true
        })
        .await
        .unwrap();
    }
    assert_eq!(finishes[0], FinishReason::Stop);
    assert_eq!(finishes[1], FinishReason::ContentFilter);
    assert_eq!(finishes[2], FinishReason::Unknown("stop_with_tools".into()));
    // The unknown value is a non-fatal warning, not an error.
    assert!(finishes[2].warning().is_some());
    assert!(FinishReason::Stop.warning().is_none());
    assert!(FinishReason::ContentFilter.warning().is_none());
}

#[tokio::test]
async fn run_turn_length_finish_does_not_execute_partial_calls() {
    let dir = TempDir::new().unwrap();
    let (url, bodies) = mock_server(vec![sse(LENGTH_TRUNCATED_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    let mut events = Vec::new();
    run_turn(&client, vec![Message::user("go")], &cfg, |ev| {
        events.push(format!("{ev:?}"));
        true
    })
    .await
    .unwrap();

    // The partial call is reported on ResponseEnd with the Length finish...
    let end = events
        .iter()
        .find(|e| e.contains("ResponseEnd"))
        .expect("a ResponseEnd event");
    assert!(end.contains("Length"), "{end}");
    assert!(end.contains("call_9"), "{end}");
    // ...but it must not be executed: no ToolResult event, the file was
    // never written, and no follow-up request was made.
    assert!(!events.iter().any(|e| e.contains("ToolResult")));
    assert!(!dir.path().join("x.txt").exists());
    assert_eq!(bodies.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn run_turn_sends_max_tokens_when_set() {
    let dir = TempDir::new().unwrap();
    let (url, bodies) = mock_server(vec![sse(STOP_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None).with_max_output_tokens(Some(123));
    let cfg = config(dir.path());

    run_turn(&client, vec![Message::user("hi")], &cfg, |_| true)
        .await
        .unwrap();
    let v: Json = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
    assert_eq!(v["max_tokens"], 123);
}

#[tokio::test]
async fn run_turn_omits_max_tokens_when_unset() {
    let dir = TempDir::new().unwrap();
    let (url, bodies) = mock_server(vec![sse(STOP_RESPONSE)]).await;
    let client = Client::new(url, "test-model", None);
    let cfg = config(dir.path());

    run_turn(&client, vec![Message::user("hi")], &cfg, |_| true)
        .await
        .unwrap();
    let v: Json = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
    assert!(v.get("max_tokens").is_none());
}

// -----------------------------------------------------------------------
// Persistence helpers
// -----------------------------------------------------------------------

#[test]
fn tool_turn_helper_builds_persistable_turn() {
    let turn = tool_turn("bash", "call_42", "exit code: 0\na b");
    assert_eq!(turn.role, parser::Role::Tool);
    assert_eq!(
        turn.meta.tags,
        vec![
            ("tool".to_string(), "bash".to_string()),
            ("id".to_string(), "call_42".to_string()),
        ]
    );
    let data = parser::parse_tool_turn(&turn).expect("well-formed");
    assert_eq!(data.name, "bash");
    assert_eq!(data.id, "call_42");
    assert_eq!(data.output, "exit code: 0\na b");
}

#[test]
fn assistant_turn_helper_sets_reasoning_field() {
    let turn = assistant_turn("m1", "thinking hard", "the answer", &[], false, false);
    assert_eq!(turn.role, parser::Role::Assistant);
    assert_eq!(turn.content, "the answer");
    assert_eq!(turn.reasoning.as_deref(), Some("thinking hard"));
    assert!(
        turn.meta
            .tags
            .iter()
            .any(|(k, v)| k == "thinking" && v == "true")
    );
    // Empty reasoning is omitted (no tag, no field).
    let turn = assistant_turn("m1", "  ", "just", &[], false, false);
    assert_eq!(turn.content, "just");
    assert_eq!(turn.reasoning, None);
    assert!(!turn.meta.tags.iter().any(|(k, _)| k == "thinking"));
}

#[test]
fn assistant_turn_helper_sets_finish_flags() {
    // Both flags set: they land in the meta (not the free-form tags) and
    // render right after the timestamp.
    let turn = assistant_turn("m1", "", "cut short", &[], true, true);
    assert!(turn.meta.length_limit_reached);
    assert!(turn.meta.content_filtered);
    assert!(
        !turn
            .meta
            .tags
            .iter()
            .any(|(k, _)| { k == "length_limit_reached" || k == "content_filtered" })
    );
    // The flags render right after the timestamp, before the other tags.
    let rendered = turn.meta.render().expect("metadata");
    let parts: Vec<&str> = rendered.split(" | ").collect();
    assert_eq!(parts[1], "length_limit_reached: true");
    assert_eq!(parts[2], "content_filtered: true");
    assert_eq!(parts[3], "model: m1");
    // Unset flags are omitted from the header entirely.
    let turn = assistant_turn("m1", "", "fine", &[], false, false);
    assert!(!turn.meta.length_limit_reached);
    assert!(!turn.meta.content_filtered);
    let rendered = turn.meta.render().expect("metadata");
    assert!(!rendered.contains("length_limit_reached"));
    assert!(!rendered.contains("content_filtered"));
}

#[test]
fn assistant_turn_helper_carries_tool_calls() {
    let calls = vec![
        CompletedToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path": "a.md"}"#.into(),
        },
        CompletedToolCall {
            id: "call_2".into(),
            name: "bash".into(),
            arguments: r#"{"command": "ls"}"#.into(),
        },
    ];
    let turn = assistant_turn("m1", "", "checking", &calls, false, false);
    assert_eq!(turn.tool_calls.len(), 2);
    assert_eq!(
        turn.meta.tags,
        vec![
            ("model".to_string(), "m1".to_string()),
            ("tool_calls".to_string(), "2".to_string()),
        ]
    );
    // No calls: no tag, no field.
    let turn = assistant_turn("m1", "", "plain", &[], false, false);
    assert!(!turn.meta.tags.iter().any(|(k, _)| k == "tool_calls"));
    assert!(turn.tool_calls.is_empty());
}

#[test]
fn append_turn_appends_and_round_trips() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("c.chat.md");
    std::fs::write(
        &path,
        "---\nmodel: m\n---\n**User** | 2026-11-22 14:30 +00:00\n\nhello\n",
    )
    .unwrap();

    let a = assistant_turn("m", "", "hi there", &[], false, false);
    append_turn(&path, &a).unwrap();
    let t = tool_turn("list_files", "c1", "x.md  1");
    append_turn(&path, &t).unwrap();

    let conv = parser::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(conv.turns.len(), 3);
    assert_eq!(conv.turns[1].role, parser::Role::Assistant);
    assert_eq!(conv.turns[1].content, "hi there");
    let data = parser::parse_tool_turn(&conv.turns[2]).expect("well-formed");
    assert_eq!(data.name, "list_files");
    assert_eq!(data.output, "x.md  1");
}
