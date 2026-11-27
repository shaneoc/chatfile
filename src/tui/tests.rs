use super::*;
use ratatui::backend::TestBackend;

fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
    KeyEvent::new(code, mods)
}

/// A minimal app with a short conversation, no file I/O.
fn test_app() -> App {
    let conv = parser::parse(
        "---\nmodel: test-model\n---\n**User** | 2026-11-22 14:30 +00:00\n\nHello there!\n\n---\n**Assistant** | 2026-11-22 14:31 +00:00 | model: test-model\n\nHi, how can I help?\n",
    )
    .unwrap();
    App {
        path: PathBuf::from("test.chat.md"),
        conv,
        model: "test-model".into(),
        endpoint: llm::DEFAULT_ENDPOINT.into(),
        api_key: None,
        tool_config: tools::ToolConfig {
            enabled: false,
            network: false,
            allow_shell: false,
            work_dir: PathBuf::from("."),
            allowed_tools: None,
            sandbox: config::SandboxConfig::default(),
            max_tool_output_bytes: 16384,
            reasoning_history: config::ReasoningHistory::default(),
        },
        compact_prompt: None,
        input: String::new(),
        cursor: 0,
        scroll: 0,
        stick: true,
        streaming: false,
        reason_buf: String::new(),
        stream_buf: String::new(),
        got_reply: false,
        pending_send: None,
        pending_compact: false,
        compacting: false,
        cancel_requested: false,
        quit: false,
        status: String::new(),
        max_output_tokens: None,
        last_finish: None,
    }
}

/// Render `app` into an offscreen buffer and return its lines.
fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| app.draw(f)).unwrap();
    let buf = term.backend().buffer().content().to_vec();
    (0..h)
        .map(|y| {
            let y = y as usize;
            let w = w as usize;
            buf[y * w..(y + 1) * w]
                .iter()
                .map(|c| c.symbol().to_string())
                .collect()
        })
        .collect()
}

#[test]
fn render_80x24_shows_history_and_status() {
    let app = test_app();
    let screen = render(&app, 80, 24).join("\n");
    assert!(screen.contains("User"), "missing User role: {screen}");
    assert!(
        screen.contains("Assistant"),
        "missing Assistant role: {screen}"
    );
    assert!(screen.contains("Hello there!"));
    assert!(screen.contains("Hi, how can I help?"));
    assert!(screen.contains("test.chat.md"));
    assert!(screen.contains("model: test-model"));
    assert!(screen.contains("Enter send"));
}

#[test]
fn render_narrow_20x10_no_panic() {
    let mut app = test_app();
    app.input = "a very long input line that must wrap in a narrow pane".into();
    app.cursor = app.input.len();
    let screen = render(&app, 20, 10).join("\n");
    assert!(screen.contains("test.chat.md"));
}

#[test]
fn render_streaming_shows_partial_reply() {
    let mut app = test_app();
    app.streaming = true;
    app.stream_buf = "partial answer".into();
    let screen = render(&app, 80, 24).join("\n");
    assert!(screen.contains("partial answer"));
    assert!(screen.contains("streaming…"));
}

#[test]
fn render_streaming_shows_reasoning_and_thinking_status() {
    let mut app = test_app();
    app.streaming = true;
    app.reason_buf = "let me work this out".into();
    let screen = render(&app, 80, 24).join("\n");
    assert!(screen.contains("let me work this out"));
    assert!(screen.contains("thinking…"));
}

#[test]
fn render_persisted_turn_splits_reasoning_from_answer() {
    let mut app = test_app();
    app.conv.turns.push(parser::Turn {
        role: parser::Role::Assistant,
        content: "54".into(),
        meta: parser::TurnMeta {
            timestamp: Some("2026-11-23 00:00 +00:00".into()),
            tags: vec![("thinking".into(), "true".into())],
            compact: false,
            length_limit_reached: false,
            content_filtered: false,
        },
        reasoning: Some("because 6*9".into()),
        tool_calls: Vec::new(),
    });
    let screen = render(&app, 80, 24).join("\n");
    assert!(screen.contains("because 6*9"));
    assert!(screen.contains("54"));
}

#[test]
fn typing_inserts_chars_at_cursor() {
    let mut app = test_app();
    for c in "hi".chars() {
        app.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
    }
    assert_eq!(app.input, "hi");
    assert_eq!(app.cursor, 2);
}

#[test]
fn cursor_movement_and_backspace() {
    let mut app = test_app();
    for c in "abc".chars() {
        app.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
    }
    app.handle_key(key(KeyCode::Left, KeyModifiers::NONE));
    app.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
    assert_eq!(app.input, "ac");
    assert_eq!(app.cursor, 1);
    app.handle_key(key(KeyCode::End, KeyModifiers::NONE));
    assert_eq!(app.cursor, 2);
    app.handle_key(key(KeyCode::Home, KeyModifiers::NONE));
    assert_eq!(app.cursor, 0);
}

#[test]
fn alt_enter_and_ctrl_j_insert_newline() {
    let mut app = test_app();
    app.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
    app.handle_key(key(KeyCode::Enter, KeyModifiers::ALT));
    assert_eq!(app.input, "a\n");
    assert!(app.pending_send.is_none());

    app.handle_key(key(KeyCode::Char('b'), KeyModifiers::NONE));
    app.handle_key(key(KeyCode::Char('j'), KeyModifiers::CONTROL));
    assert_eq!(app.input, "a\nb\n");
}

#[test]
fn plain_enter_sends_and_clears_input() {
    let mut app = test_app();
    for c in "hello".chars() {
        app.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
    }
    app.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(app.pending_send.as_deref(), Some("hello"));
    assert!(app.input.is_empty());
    assert_eq!(app.cursor, 0);
    assert!(app.stick);
}

#[test]
fn empty_enter_does_not_send() {
    let mut app = test_app();
    app.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.pending_send.is_none());
}

#[test]
fn input_locked_while_streaming() {
    let mut app = test_app();
    app.streaming = true;
    app.handle_key(key(KeyCode::Char('x'), KeyModifiers::NONE));
    app.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.input.is_empty());
    assert!(app.pending_send.is_none());
}

#[test]
fn scrolling_steps_and_repins() {
    let mut app = test_app();
    app.handle_key(key(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(app.scroll, SCROLL_STEP);
    assert!(!app.stick);

    app.handle_key(key(KeyCode::Char('u'), KeyModifiers::CONTROL));
    assert_eq!(app.scroll, 2 * SCROLL_STEP);

    app.handle_key(key(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(app.scroll, SCROLL_STEP);

    // At the bottom, PageDown re-pins to the bottom.
    app.handle_key(key(KeyCode::Char('d'), KeyModifiers::CONTROL));
    assert!(app.stick);
    assert_eq!(app.scroll, 0);
}

#[test]
fn ctrl_c_cancels_stream_or_quits() {
    let mut app = test_app();
    // Idle: quits.
    app.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(app.quit);

    // Streaming: requests cancel instead.
    let mut app = test_app();
    app.streaming = true;
    app.quit = false;
    app.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(!app.quit);
    assert!(app.cancel_requested);
}

#[test]
fn esc_quits() {
    let mut app = test_app();
    app.handle_key(key(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.quit);
}
