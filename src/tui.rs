//! Interactive TUI for chatting with a conversation file.
//!
//! Layout: a scrollable conversation history on top (distinct styling per
//! role, with timestamps), a multi-line input box below it (`Enter` sends,
//! `Alt+Enter`/`Ctrl+J` insert a newline), and a thin status bar showing the
//! filename, active model, and keybinding hints.
//!
//! Streaming runs on a background tokio task feeding a `tokio::sync::mpsc`
//! channel, so the UI stays responsive while a reply arrives: `Ctrl+C`
//! cancels an active stream, and `PageUp`/`PageDown` (or `Ctrl+U`/`Ctrl+D`)
//! scroll the history at any time.

use std::path::PathBuf;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;

use crate::{context, llm, parser, tools};

/// Environment variable overrides (frontmatter wins over these).
const ENV_ENDPOINT: &str = "MDCHAT_ENDPOINT";
const ENV_MODEL: &str = "MDCHAT_MODEL";
const ENV_API_KEY: &str = "MDCHAT_API_KEY";
const DEFAULT_MODEL: &str = "local-model";

/// Scroll step for PageUp/PageDown/Ctrl+U/Ctrl+D, in lines.
const SCROLL_STEP: usize = 10;

/// Events produced by the background streaming task (one user turn, which
/// may span several LLM round-trips when tools are called).
enum StreamEvent {
    /// A streamed delta: reasoning (thinking) text or answer content.
    Delta(llm::StreamDelta),
    /// The current assistant response has ended (before tool execution, or
    /// as the final response of the turn).
    ResponseEnd,
    /// A tool call has been executed; the fields describe the persisted
    /// result.
    ToolResult {
        id: String,
        name: String,
        arguments: String,
        output: String,
    },
    /// The whole turn finished successfully.
    Done,
    /// The turn failed with an error message.
    Failed(String),
}

/// Decoupled TUI application: UI state plus pure event handling and
/// rendering. No I/O or async inside [`App::handle_key`] or [`App::draw`];
/// side effects are queued in `pending_send` / `cancel_requested` / `quit`
/// for the event loop to execute.
pub struct App {
    /// Path of the conversation file being edited.
    pub path: PathBuf,
    /// Parsed conversation (kept in sync with the file).
    pub conv: parser::Conversation,
    /// Model name shown in the status bar and written to turn metadata.
    pub model: String,
    /// Endpoint used for streaming.
    pub endpoint: String,
    /// API key, when configured via the environment.
    pub api_key: Option<String>,
    /// Multi-line input buffer.
    pub input: String,
    /// Byte offset of the cursor within `input`.
    pub cursor: usize,
    /// Index of the first visible history line (clamped at render time).
    pub scroll: usize,
    /// When true, new content keeps the view pinned to the bottom.
    pub stick: bool,
    /// True while an assistant reply is streaming.
    pub streaming: bool,
    /// In-progress reasoning (thinking) text for the active reply.
    pub reason_buf: String,
    /// In-progress assistant reply text.
    pub stream_buf: String,
    /// Whether the active turn has persisted anything yet (an assistant
    /// response or a tool result); decides the "no reply produced" status.
    pub got_reply: bool,
    /// User message queued by [`App::handle_key`] for the event loop to send.
    pub pending_send: Option<String>,
    /// Set by Ctrl+C while streaming; the event loop aborts the task.
    pub cancel_requested: bool,
    /// Set when the user quits (Esc, or Ctrl+C while idle).
    pub quit: bool,
    /// Short message shown in the status bar.
    pub status: String,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

impl App {
    /// Load a conversation file and prepare an app for it.
    pub fn load(path: PathBuf) -> Result<Self> {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let conv = parser::parse(&raw)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let endpoint = conv
            .frontmatter
            .endpoint
            .clone()
            .or_else(|| env(ENV_ENDPOINT))
            .unwrap_or_else(|| llm::DEFAULT_ENDPOINT.to_string());
        let model = conv
            .frontmatter
            .model
            .clone()
            .or_else(|| env(ENV_MODEL))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        Ok(Self {
            path,
            conv,
            endpoint,
            model,
            api_key: env(ENV_API_KEY),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            stick: true,
            streaming: false,
            reason_buf: String::new(),
            stream_buf: String::new(),
            got_reply: false,
            pending_send: None,
            cancel_requested: false,
            quit: false,
            status: String::new(),
        })
    }

    /// Insert `s` at the cursor and advance the cursor past it.
    fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Delete the character before the cursor (UTF-8 safe).
    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let end = self.cursor;
        let mut start = self.cursor - 1;
        while !self.input.is_char_boundary(start) {
            start -= 1;
        }
        self.input.replace_range(start..end, "");
        self.cursor = start;
    }

    /// Move the cursor one character left.
    fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut start = self.cursor - 1;
        while start > 0 && !self.input.is_char_boundary(start) {
            start -= 1;
        }
        self.cursor = start;
    }

    /// Move the cursor one character right.
    fn cursor_right(&mut self) {
        if self.cursor == self.input.len() {
            return;
        }
        let mut end = self.cursor + 1;
        while end < self.input.len() && !self.input.is_char_boundary(end) {
            end += 1;
        }
        self.cursor = end;
    }
}

impl App {
    /// Process one key event. Pure state transition: no I/O, no async.
    /// Side effects are queued in `pending_send`, `cancel_requested`, or
    /// `quit` for the event loop to execute.
    pub fn handle_key(&mut self, key: KeyEvent) {
        match (key.code, key.modifiers) {
            // Quit: Esc always; Ctrl+C cancels an active stream first.
            (KeyCode::Esc, _) => self.quit = true,
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                if self.streaming {
                    self.cancel_requested = true;
                    self.status = "cancelling…".into();
                } else {
                    self.quit = true;
                }
            }
            // Scrolling works at any time, including while streaming.
            (KeyCode::PageUp, _) | (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.scroll += SCROLL_STEP;
                self.stick = false;
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                if self.scroll <= SCROLL_STEP {
                    // Reached the bottom: re-pin to it.
                    self.stick = true;
                    self.scroll = 0;
                } else {
                    self.scroll -= SCROLL_STEP;
                }
            }
            // Input editing is locked while a reply is streaming.
            _ if self.streaming => {}
            // Newline: Alt+Enter or Ctrl+J.
            (KeyCode::Enter, m) if m.contains(KeyModifiers::ALT) => self.insert_str("\n"),
            (KeyCode::Char('j'), KeyModifiers::CONTROL) => self.insert_str("\n"),
            // Plain Enter sends the input.
            (KeyCode::Enter, m) if m.is_empty() => {
                let text = self.input.trim().to_string();
                if !text.is_empty() {
                    self.pending_send = Some(text);
                    self.input.clear();
                    self.cursor = 0;
                    self.scroll = 0;
                    self.stick = true;
                    self.status = "sending…".into();
                }
            }
            (KeyCode::Backspace, _) => self.backspace(),
            (KeyCode::Left, _) => self.cursor_left(),
            (KeyCode::Right, _) => self.cursor_right(),
            (KeyCode::Home, _) => self.cursor = 0,
            (KeyCode::End, _) => self.cursor = self.input.len(),
            (KeyCode::Tab, _) => self.insert_str("    "),
            (KeyCode::Char(c), m) if !m.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                let mut s = String::new();
                s.push(c);
                self.insert_str(&s);
            }
            _ => {}
        }
    }

    /// The conversation as styled lines for the history pane, including the
    /// in-progress assistant reply while streaming.
    pub fn history_lines(&self) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        if !self.conv.preamble.trim().is_empty() {
            for l in self.conv.preamble.lines() {
                lines.push(Line::from(l.to_string()));
            }
            lines.push(Line::default());
        }
        for t in &self.conv.turns {
            if t.role == parser::Role::Tool {
                push_tool_turn(&mut lines, t);
            } else {
                let (reason, answer) = parser::split_reasoning(&t.content);
                push_turn(&mut lines, t.role, &t.meta, reason.as_deref(), &answer);
            }
            lines.push(Line::default());
        }
        if self.streaming {
            let meta = parser::TurnMeta {
                timestamp: Some(parser::now_timestamp()),
                tags: vec![("model".into(), self.model.clone())],
            };
            let reason = (!self.reason_buf.is_empty()).then_some(self.reason_buf.as_str());
            push_turn(
                &mut lines,
                parser::Role::Assistant,
                &meta,
                reason,
                &self.stream_buf,
            );
        }
        lines
    }

    /// Wrap one logical line into visual segments of display width `width`
    /// (a single over-wide character may exceed it).
    fn wrap_logical(line: &str, width: usize) -> Vec<String> {
        if width == 0 {
            return vec![line.to_string()];
        }
        let mut segs = Vec::new();
        let mut cur = String::new();
        let mut w = 0usize;
        for c in line.chars() {
            let cw = char_width(c);
            if w + cw > width && !cur.is_empty() {
                segs.push(std::mem::take(&mut cur));
                w = 0;
            }
            cur.push(c);
            w += cw;
        }
        segs.push(cur);
        segs
    }

    /// The cursor position in display cells `(row, col)` of the wrapped
    /// input, matching how the input pane is rendered.
    fn cursor_visual(&self, width: usize) -> (usize, usize) {
        let before = &self.input[..self.cursor];
        let line_idx = before.matches('\n').count();
        let line_start = before.rfind('\n').map(|p| p + 1).unwrap_or(0);
        let target_col: usize = self.input[line_start..self.cursor]
            .chars()
            .map(char_width)
            .sum();
        let mut row = 0usize;
        for (i, logical) in self.input.split('\n').enumerate() {
            if i > line_idx {
                break;
            }
            let segs = Self::wrap_logical(logical, width);
            if i == line_idx {
                let mut acc = 0usize;
                for (si, seg) in segs.iter().enumerate() {
                    let sw: usize = seg.chars().map(char_width).sum();
                    if target_col <= acc + sw || si == segs.len() - 1 {
                        return (row + si, target_col.saturating_sub(acc));
                    }
                    acc += sw;
                }
            }
            row += segs.len().max(1);
        }
        (0, 0)
    }

    /// Render the app into `frame`. Pure: reads state only.
    pub fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        if area.height < 2 || area.width == 0 {
            return;
        }
        let history = self.history_lines();
        let total = history.len();

        let input_logical = self.input.lines().count().max(1);
        let input_height = (input_logical as u16).min(area.height - 1).max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .split(area);

        // History pane (scrollable).
        let h_area = chunks[0];
        if h_area.height > 0 {
            let visible = h_area.height as usize;
            let first = if total <= visible {
                0
            } else if self.stick {
                total - visible
            } else {
                self.scroll.min(total - visible)
            };
            let end = (first + visible).min(total);
            let view: Vec<Line<'static>> = history[first..end].to_vec();
            frame.render_widget(Paragraph::new(view).wrap(Wrap { trim: false }), h_area);
        }

        // Input pane (wrapped, with the terminal cursor placed in it).
        let i_area = chunks[1];
        if i_area.height > 0 {
            let width = i_area.width as usize;
            let mut visual: Vec<Line<'static>> = Vec::new();
            for l in self.input.split('\n') {
                for seg in Self::wrap_logical(l, width) {
                    visual.push(Line::from(seg));
                }
            }
            // The input grows upward: keep the last `height` visual lines.
            let skip = visual.len().saturating_sub(i_area.height as usize);
            let view: Vec<Line<'static>> = visual[skip..].to_vec();
            frame.render_widget(Paragraph::new(view), i_area);

            let (row, col) = self.cursor_visual(width);
            let row = row.saturating_sub(skip);
            if (row as u16) < i_area.height {
                frame.set_cursor_position((i_area.x + col as u16, i_area.y + row as u16));
            }
        }

        // Status bar.
        let s_area = chunks[2];
        if s_area.height > 0 {
            let state = if self.streaming {
                if self.stream_buf.is_empty() && !self.reason_buf.is_empty() {
                    format!(
                        "thinking… ({} chars)",
                        self.reason_buf.chars().count()
                    )
                } else if !self.stream_buf.is_empty() {
                    format!(
                        "streaming… ({} chars)",
                        self.stream_buf.chars().count()
                    )
                } else if !self.status.is_empty() {
                    // Between responses (e.g. a tool just ran).
                    self.status.clone()
                } else {
                    "waiting…".to_string()
                }
            } else if self.status.is_empty() {
                "ready".to_string()
            } else {
                self.status.clone()
            };
            let name = self
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            let text = format!(
                " {} | model: {} | {} | Enter send · Alt+Enter newline · PgUp/PgDn scroll · Ctrl+C cancel/quit",
                name, self.model, state
            );
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    text,
                    Style::default().add_modifier(Modifier::DIM),
                ))),
                s_area,
            );
        }
    }
}

/// Push one turn (role line, optional reasoning block, content) into
/// `lines` with role styling. Reasoning renders dimmed and italic so it is
/// clearly secondary to the answer.
fn push_turn(
    lines: &mut Vec<Line<'static>>,
    role: parser::Role,
    meta: &parser::TurnMeta,
    reasoning: Option<&str>,
    content: &str,
) {
    let mut spans = vec![Span::styled(
        role.title().to_string(),
        role_style(role).add_modifier(Modifier::BOLD),
    )];
    if let Some(m) = meta.render() {
        spans.push(Span::styled(
            format!(" · {m}"),
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    lines.push(Line::from(spans));
    let reason_style = Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC);
    if let Some(r) = reasoning {
        for l in r.lines() {
            lines.push(Line::from(Span::styled(l.to_string(), reason_style)));
        }
    }
    for l in content.lines() {
        lines.push(Line::from(l.to_string()));
    }
}

/// Push a persisted `**Tool**` turn into `lines`: the role line, the call
/// (`name(args)`) dimmed, then the output indented. Falls back to the raw
/// content when the turn is not a well-formed tool turn.
fn push_tool_turn(lines: &mut Vec<Line<'static>>, t: &parser::Turn) {
    let mut spans = vec![Span::styled(
        t.role.title().to_string(),
        role_style(t.role).add_modifier(Modifier::BOLD),
    )];
    if let Some(m) = t.meta.render() {
        spans.push(Span::styled(
            format!(" · {m}"),
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    lines.push(Line::from(spans));
    let dim = Style::default().add_modifier(Modifier::DIM);
    match parser::parse_tool_turn(t) {
        Some(data) => {
            lines.push(Line::from(Span::styled(
                format!("{}({})", data.name, data.arguments),
                dim,
            )));
            for l in data.output.lines() {
                lines.push(Line::from(Span::styled(format!("  {l}"), dim)));
            }
        }
        None => {
            for l in t.content.lines() {
                lines.push(Line::from(Span::styled(format!("  {l}"), dim)));
            }
        }
    }
}

/// Display width of a character (control characters count as one cell).
fn char_width(c: char) -> usize {
    unicode_width::UnicodeWidthChar::width(c).unwrap_or(0).max(1)
}

/// Per-role base color for the history pane.
fn role_style(role: parser::Role) -> Style {
    let color = match role {
        parser::Role::User => Color::Blue,
        parser::Role::Assistant => Color::Green,
        parser::Role::System => Color::Yellow,
        parser::Role::Tool => Color::Magenta,
    };
    Style::default().fg(color)
}

/// Run the interactive TUI for `path` (the file must exist).
pub async fn run_tui(path: PathBuf) -> Result<()> {
    let mut app = App::load(path)?;
    enable_raw_mode().context("failed to enable raw terminal mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = event_loop(&mut terminal, &mut app).await;
    // Restore the terminal even when the loop failed.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

/// The main select! loop: terminal events, stream events, and queued
/// side effects (send / cancel).
async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let mut stream_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut key_events = event::EventStream::new();

    loop {
        terminal.draw(|f| app.draw(f))?;

        // Queued user submission: persist the turn, then start streaming.
        if let Some(content) = app.pending_send.take() {
            match submit(app, &tx, content).await {
                Ok(task) => stream_task = Some(task),
                Err(e) => {
                    app.status = format!("error: {e}");
                }
            }
        }

        // Queued cancel (Ctrl+C while streaming).
        if app.cancel_requested {
            app.cancel_requested = false;
            if let Some(h) = stream_task.take() {
                h.abort();
            }
            app.streaming = false;
            app.reason_buf.clear();
            app.stream_buf.clear();
            app.got_reply = false;
            app.status = "stream cancelled".into();
        }

        tokio::select! {
            maybe_ev = key_events.next() => {
                match maybe_ev {
                    Some(Ok(Event::Key(key))) => {
                        app.handle_key(key);
                        if app.quit {
                            break;
                        }
                    }
                    Some(Err(e)) => return Err(e.into()),
                    _ => {}
                }
            }
            maybe_event = rx.recv() => {
                match maybe_event {
                    Some(StreamEvent::Delta(llm::StreamDelta::Reasoning(s))) => {
                        app.reason_buf.push_str(&s);
                    }
                    Some(StreamEvent::Delta(llm::StreamDelta::Content(s))) => {
                        app.stream_buf.push_str(&s);
                    }
                    // Tool-call fragments and finish markers are consumed by
                    // the loop itself; they never reach the UI.
                    Some(StreamEvent::Delta(_)) => {}
                    Some(StreamEvent::ResponseEnd) => {
                        finalize_response(app)?;
                    }
                    Some(StreamEvent::ToolResult {
                        id,
                        name,
                        arguments,
                        output,
                    }) => {
                        let turn = tools::tool_turn(&name, &id, &arguments, &output);
                        tools::append_turn(&app.path, &turn)?;
                        app.conv.turns.push(turn);
                        app.got_reply = true;
                        app.status = format!("tool: {name}");
                    }
                    Some(StreamEvent::Done) => {
                        if let Some(h) = stream_task.take() {
                            let _ = h.await;
                        }
                        app.streaming = false;
                        if !app.got_reply {
                            app.status = "no reply produced".into();
                        } else {
                            app.status.clear();
                        }
                        app.got_reply = false;
                    }
                    Some(StreamEvent::Failed(e)) => {
                        if let Some(h) = stream_task.take() {
                            let _ = h.await;
                        }
                        app.streaming = false;
                        app.reason_buf.clear();
                        app.stream_buf.clear();
                        app.got_reply = false;
                        app.status = format!("error: {e}");
                    }
                    None => {} // sender dropped (task aborted or finished)
                }
            }
        }
    }
    Ok(())
}

/// Persist a submitted user turn immediately, then spawn the streaming task.
async fn submit(
    app: &mut App,
    tx: &mpsc::UnboundedSender<StreamEvent>,
    content: String,
) -> Result<tokio::task::JoinHandle<()>> {
    let ts = parser::now_timestamp();
    let user_turn = parser::Turn {
        role: parser::Role::User,
        content: content.clone(),
        meta: parser::TurnMeta {
            timestamp: Some(ts),
            tags: vec![],
        },
    };
    tools::append_turn(&app.path, &user_turn)?;
    app.conv.turns.push(user_turn);

    let system_prompt = context::build_system_prompt(
        &app.path,
        app.conv.frontmatter.system_prompt.as_deref(),
    );
    let mut messages: Vec<llm::Message> = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(llm::Message::system(system_prompt));
    }
    // `to_messages` folds persisted `**Tool**` turns into `role: "tool"`
    // messages and attaches their `tool_calls` to the preceding assistant
    // message, so reloaded histories are valid API payloads.
    messages.extend(app.conv.to_messages());

    let client = llm::Client::new(app.endpoint.clone(), app.model.clone(), app.api_key.clone());
    let tool_config = tools::ToolConfig::for_file(&app.conv.frontmatter, &app.path);
    let tx = tx.clone();
    let task = tokio::spawn(async move {
        // The whole (possibly multi-round) turn runs here; each event is
        // forwarded to the UI over the channel.
        let res = tools::run_turn(&client, messages, &tool_config, |ev| {
            let ev = match ev {
                tools::LoopEvent::Reasoning(s) => StreamEvent::Delta(llm::StreamDelta::Reasoning(s.clone())),
                tools::LoopEvent::Content(s) => StreamEvent::Delta(llm::StreamDelta::Content(s.clone())),
                tools::LoopEvent::ResponseEnd => StreamEvent::ResponseEnd,
                tools::LoopEvent::ToolResult { id, name, arguments, output } => {
                    StreamEvent::ToolResult {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        output: output.clone(),
                    }
                }
            };
            tx.send(ev).is_ok()
        })
        .await;
        match res {
            Ok(()) => {
                let _ = tx.send(StreamEvent::Done);
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Failed(e.to_string()));
            }
        }
    });
    app.streaming = true;
    app.reason_buf.clear();
    app.stream_buf.clear();
    app.got_reply = false;
    Ok(task)
}

/// Persist the assistant response that just ended (reasoning between
/// `think` / `endthink` tags ahead of the answer) and push it into the
/// in-memory conversation. A response with neither text nor reasoning — a
/// pure tool-call response — is skipped: the parser synthesizes the missing
/// assistant message when the file is reloaded.
fn finalize_response(app: &mut App) -> Result<()> {
    let reply = app.stream_buf.trim_end().to_string();
    let reasoning = app.reason_buf.trim_end().to_string();
    app.reason_buf.clear();
    app.stream_buf.clear();
    if reply.is_empty() && reasoning.is_empty() {
        return Ok(());
    }
    let turn = tools::assistant_turn(&app.model, &reasoning, &reply);
    tools::append_turn(&app.path, &turn)?;
    app.conv.turns.push(turn);
    app.got_reply = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    /// A minimal app with a short conversation, no file I/O.
    fn test_app() -> App {
        let conv = parser::parse(
            "---\nmodel: test-model\n---\n\n---\n**User** · *2026-11-22 14:30*\n\nHello there!\n\n---\n**Assistant** · *2026-11-22 14:31 · model: test-model*\n\nHi, how can I help?\n",
        )
        .unwrap();
        App {
            path: PathBuf::from("test.chat.md"),
            conv,
            model: "test-model".into(),
            endpoint: llm::DEFAULT_ENDPOINT.into(),
            api_key: None,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            stick: true,
            streaming: false,
            reason_buf: String::new(),
            stream_buf: String::new(),
            got_reply: false,
            pending_send: None,
            cancel_requested: false,
            quit: false,
            status: String::new(),
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
        assert!(screen.contains("Assistant"), "missing Assistant role: {screen}");
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
            content: "think\nbecause 6*9\nendthink\n\n54".into(),
            meta: parser::TurnMeta {
                timestamp: Some("2026-11-23 00:00".into()),
                tags: vec![],
            },
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
        app.handle_key(key(
            KeyCode::Enter,
            KeyModifiers::ALT,
        ));
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
}
