//! Interactive TUI for chatting with a conversation file.
//!
//! Layout: a scrollable conversation history on top (distinct styling per
//! role, with timestamps), a multi-line input box below it (`Enter` sends,
//! `Alt+Enter`/`Ctrl+J` insert a newline), and a thin status bar showing the
//! filename, active model, and keybinding hints.
//!
//! Streaming runs on a background tokio task feeding a `tokio::sync::mpsc`
//! channel, so the UI stays responsive while a reply arrives: `Ctrl+C`
//! cancels an active stream, and `PageUp`/`PageDown` (or `Ctrl+U`/`Ctrl+D`,
//! or the mouse wheel when `[tui] mouse` is enabled) scroll the history at
//! any time. Typing `/compact` and pressing Enter
//! appends a `**User** | compact: true` request turn and streams the
//! model's handoff summary as the next `**Assistant**` turn (context
//! compaction, §5 of `docs/format.md`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};
use tokio::sync::mpsc;

use crate::{config, context, llm, parser, tools};

/// Scroll step for PageUp/PageDown/Ctrl+U/Ctrl+D (and the mouse wheel),
/// in lines.
const SCROLL_STEP: usize = 10;

/// Events produced by the background streaming task (one user turn, which
/// may span several LLM round-trips when tools are called).
enum StreamEvent {
    /// A streamed delta: reasoning (thinking) text or answer content.
    Delta(llm::StreamDelta),
    /// The current assistant response has ended (before tool execution, or
    /// as the final response of the turn), with the calls it issued and how
    /// the response ended (§2.5 of `docs/format.md`).
    ResponseEnd {
        tool_calls: Vec<crate::llm::CompletedToolCall>,
        finish: crate::llm::FinishReason,
    },
    /// A tool call has been executed; the fields describe the persisted
    /// result (the call's arguments live on the issuing assistant turn).
    ToolResult {
        id: String,
        name: String,
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
    /// Per-response output-token limit (CLI `--max-output-tokens` >
    /// frontmatter `max_output_tokens` > active profile), sent as
    /// `max_tokens` on every request; `None` omits the field.
    pub max_output_tokens: Option<u32>,
    /// Tool configuration for this conversation (frontmatter over the
    /// active profile from config.toml).
    pub tool_config: tools::ToolConfig,
    /// Custom compaction prompt from `[prompts] compact` in config.toml,
    /// when set.
    pub compact_prompt: Option<String>,
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
    /// A `/compact` command queued by [`App::handle_key`] for the event loop
    /// to run (persists the `compact: true` request turn, then streams the
    /// handoff summary).
    pub pending_compact: bool,
    /// True while a `/compact` summary is streaming; the reply is persisted
    /// as an `**Assistant**` turn when the stream ends.
    pub compacting: bool,
    /// Set by Ctrl+C while streaming; the event loop aborts the task.
    pub cancel_requested: bool,
    /// Set when the user quits (Esc, or Ctrl+C while idle).
    pub quit: bool,
    /// The finish reason of the most recent response that ended this turn
    /// (cleared when the turn completes); drives the status-bar truncation
    /// notice and non-fatal warnings (§2.5 of `docs/format.md`).
    pub last_finish: Option<llm::FinishReason>,
    /// Short message shown in the status bar.
    pub status: String,
}

impl App {
    /// Load a conversation file and prepare an app for it. The endpoint,
    /// model, and API key are resolved from the frontmatter, the
    /// environment, and the active profile (see
    /// [`config::AppConfig::resolve_llm`]); the output-token limit from the
    /// CLI flag, the frontmatter, and the active profile (see
    /// [`config::AppConfig::resolve_max_output_tokens`]); the tool
    /// configuration and the compaction prompt come from the config.
    pub fn load(
        path: PathBuf,
        cli_max_output_tokens: Option<u32>,
        app_config: &config::AppConfig,
    ) -> Result<Self> {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let conv =
            parser::parse(&raw).with_context(|| format!("failed to parse {}", path.display()))?;
        let (endpoint, model, api_key) = app_config.resolve_llm(
            conv.frontmatter.endpoint.as_deref(),
            conv.frontmatter.model.as_deref(),
        );
        let max_output_tokens = app_config
            .resolve_max_output_tokens(cli_max_output_tokens, conv.frontmatter.max_output_tokens);
        let tool_config = tools::ToolConfig::for_file(&conv.frontmatter, &path, app_config);
        let compact_prompt = app_config.prompts.compact.clone();
        Ok(Self {
            path,
            conv,
            endpoint,
            model,
            api_key,
            max_output_tokens,
            tool_config,
            compact_prompt,
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
            last_finish: None,
            status: String::new(),
        })
    }

    /// Scroll up by one step, unpinning from the bottom.
    pub fn scroll_up(&mut self) {
        self.scroll += SCROLL_STEP;
        self.stick = false;
    }

    /// Scroll down by one step; at the bottom, re-pin to it.
    pub fn scroll_down(&mut self) {
        if self.scroll <= SCROLL_STEP {
            // Reached the bottom: re-pin to it.
            self.stick = true;
            self.scroll = 0;
        } else {
            self.scroll -= SCROLL_STEP;
        }
    }

    /// Handle a mouse event: the scroll wheel scrolls the history like
    /// PageUp/PageDown (events are only delivered when mouse capture is
    /// enabled).
    pub fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        match mouse.kind {
            crossterm::event::MouseEventKind::ScrollUp => self.scroll_up(),
            crossterm::event::MouseEventKind::ScrollDown => self.scroll_down(),
            _ => {}
        }
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
            (KeyCode::PageUp, _) | (KeyCode::Char('u'), KeyModifiers::CONTROL) => self.scroll_up(),
            (KeyCode::PageDown, _) | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                self.scroll_down()
            }
            // Input editing is locked while a reply is streaming.
            _ if self.streaming => {}
            // Newline: Alt+Enter or Ctrl+J.
            (KeyCode::Enter, m) if m.contains(KeyModifiers::ALT) => self.insert_str("\n"),
            (KeyCode::Char('j'), KeyModifiers::CONTROL) => self.insert_str("\n"),
            // Plain Enter sends the input (or runs `/compact`).
            (KeyCode::Enter, m) if m.is_empty() => {
                let text = self.input.trim().to_string();
                if !text.is_empty() {
                    if text == "/compact" {
                        self.pending_compact = true;
                        self.status = "compacting…".into();
                    } else {
                        self.pending_send = Some(text);
                        self.status = "sending…".into();
                    }
                    self.input.clear();
                    self.cursor = 0;
                    self.scroll = 0;
                    self.stick = true;
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
        for t in &self.conv.turns {
            if t.role == parser::Role::Tool {
                push_tool_turn(&mut lines, t);
            } else {
                push_turn(
                    &mut lines,
                    t.role,
                    &t.meta,
                    t.reasoning.as_deref(),
                    &t.content,
                );
            }
            lines.push(Line::default());
        }
        if self.streaming {
            let meta = parser::TurnMeta {
                timestamp: Some(parser::now_timestamp()),
                tags: vec![("model".into(), self.model.clone())],
                compact: false,
                length_limit_reached: false,
                content_filtered: false,
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
                    format!("thinking… ({} chars)", self.reason_buf.chars().count())
                } else if !self.stream_buf.is_empty() {
                    format!("streaming… ({} chars)", self.stream_buf.chars().count())
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
                " {} | model: {} | {} | Enter send · /compact summarize · Alt+Enter newline · PgUp/PgDn scroll · Ctrl+C cancel/quit",
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
            format!(" | {m}"),
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

/// Push a persisted `**Tool**` turn into `lines`: the role line, the tool
/// name dimmed, then the output indented. Falls back to the raw content
/// when the turn is not a well-formed tool turn.
fn push_tool_turn(lines: &mut Vec<Line<'static>>, t: &parser::Turn) {
    let mut spans = vec![Span::styled(
        t.role.title().to_string(),
        role_style(t.role).add_modifier(Modifier::BOLD),
    )];
    if let Some(m) = t.meta.render() {
        spans.push(Span::styled(
            format!(" | {m}"),
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    lines.push(Line::from(spans));
    let dim = Style::default().add_modifier(Modifier::DIM);
    match parser::parse_tool_turn(t) {
        Some(data) => {
            lines.push(Line::from(Span::styled(
                format!("{} ({})", data.name, data.id),
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
    unicode_width::UnicodeWidthChar::width(c)
        .unwrap_or(0)
        .max(1)
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

/// Run the interactive TUI for `path` (the file must exist). When `mouse`
/// is true, mouse capture is enabled so the scroll wheel scrolls the
/// history. `cli_max_output_tokens` is the `--max-output-tokens` flag value,
/// which overrides the frontmatter and the active profile.
pub async fn run_tui(
    path: PathBuf,
    mouse: bool,
    cli_max_output_tokens: Option<u32>,
    config: &config::AppConfig,
) -> Result<()> {
    let mut app = App::load(path, cli_max_output_tokens, config)?;
    enable_raw_mode().context("failed to enable raw terminal mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    if mouse {
        execute!(stdout, EnableMouseCapture)?;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = event_loop(&mut terminal, &mut app).await;
    // Restore the terminal even when the loop failed.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    if mouse {
        execute!(terminal.backend_mut(), DisableMouseCapture)?;
    }
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

        // Queued `/compact`: persist the request turn, then stream the
        // handoff summary.
        if app.pending_compact {
            app.pending_compact = false;
            match compact(app, &tx).await {
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
            app.compacting = false;
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
                    // Mouse wheel scrolls the history (only delivered when
                    // mouse capture is enabled).
                    Some(Ok(Event::Mouse(mouse))) => app.handle_mouse(mouse),
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
                    Some(StreamEvent::ResponseEnd {
                        tool_calls,
                        finish,
                    }) => {
                        finalize_response(
                            app,
                            &tool_calls,
                            finish.is_length(),
                            finish.is_content_filter(),
                        )?;
                        app.last_finish = Some(finish);
                    }
                    Some(StreamEvent::ToolResult {
                        id,
                        name,
                        output,
                    }) => {
                        let turn = tools::tool_turn(&name, &id, &output);
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
                        let compacting = app.compacting;
                        app.compacting = false;
                        // The status reflects whether anything was persisted,
                        // and — when a response was — how it ended (§2.5):
                        // the truncation notice for "length", or the
                        // non-fatal warning for a missing/unknown
                        // finish_reason.
                        app.status = if !app.got_reply {
                            (if compacting {
                                "no summary produced"
                            } else {
                                "no reply produced"
                            })
                            .into()
                        } else {
                            app.last_finish
                                .take()
                                .map(finish_status)
                                .unwrap_or_default()
                        };
                        app.got_reply = false;
                    }
                    Some(StreamEvent::Failed(e)) => {
                        if let Some(h) = stream_task.take() {
                            let _ = h.await;
                        }
                        app.streaming = false;
                        app.compacting = false;
                        app.reason_buf.clear();
                        app.stream_buf.clear();
                        app.got_reply = false;
                        app.last_finish = None;
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
            compact: false,
            length_limit_reached: false,
            content_filtered: false,
        },
        reasoning: None,
        tool_calls: Vec::new(),
    };
    tools::append_turn(&app.path, &user_turn)?;
    app.conv.turns.push(user_turn);

    let system_section = app
        .conv
        .turns
        .first()
        .filter(|t| t.role == parser::Role::System)
        .map(|t| t.content.as_str());
    let system_prompt = context::build_system_prompt(&app.path, system_section);
    // `build_api_messages` applies the compaction windowing (§5 of
    // docs/format.md) and projects the surviving turns; `set_system_prompt`
    // overlays the hierarchical AGENTS.md prompt on the leading system
    // message; `complete_tool_calls` answers any call left dangling by a
    // cancelled or interrupted execution, so reloaded histories are valid
    // API payloads.
    let tool_config = app.tool_config.clone();
    let mut messages = llm::build_api_messages(&app.conv, tool_config.reasoning_history);
    llm::set_system_prompt(&mut messages, &system_prompt);
    llm::complete_tool_calls(&mut messages);

    let client = llm::Client::new(app.endpoint.clone(), app.model.clone(), app.api_key.clone())
        .with_max_output_tokens(app.max_output_tokens);
    let tx = tx.clone();
    let task = tokio::spawn(async move {
        // The whole (possibly multi-round) turn runs here; each event is
        // forwarded to the UI over the channel.
        let res = tools::run_turn(&client, messages, &tool_config, |ev| {
            let ev = match ev {
                tools::LoopEvent::Reasoning(s) => {
                    StreamEvent::Delta(llm::StreamDelta::Reasoning(s.clone()))
                }
                tools::LoopEvent::Content(s) => {
                    StreamEvent::Delta(llm::StreamDelta::Content(s.clone()))
                }
                tools::LoopEvent::ResponseEnd { tool_calls, finish } => StreamEvent::ResponseEnd {
                    tool_calls: tool_calls.clone(),
                    finish: finish.clone(),
                },
                tools::LoopEvent::ToolResult {
                    id,
                    name,
                    arguments: _,
                    output,
                } => StreamEvent::ToolResult {
                    id: id.clone(),
                    name: name.clone(),
                    output: output.clone(),
                },
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

/// Persist a `**User** | compact: true` compaction request, then spawn the
/// streaming task for the model's handoff summary. The summary is a single
/// response (no tool loop), so it is persisted by the event loop when the
/// stream ends (see [`StreamEvent::Done`]).
async fn compact(
    app: &mut App,
    tx: &mpsc::UnboundedSender<StreamEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let request = llm::compact_request_turn(app.compact_prompt.as_deref());
    tools::append_turn(&app.path, &request)?;
    app.conv.turns.push(request);

    let system_section = app
        .conv
        .turns
        .first()
        .filter(|t| t.role == parser::Role::System)
        .map(|t| t.content.as_str());
    let system_prompt = context::build_system_prompt(&app.path, system_section);
    let client = llm::Client::new(app.endpoint.clone(), app.model.clone(), app.api_key.clone())
        .with_max_output_tokens(app.max_output_tokens);
    let tool_config = app.tool_config.clone();
    let conv = app.conv.clone();
    let tx = tx.clone();
    let task = tokio::spawn(async move {
        let res = llm::run_compact(&client, &conv, &system_prompt, &tool_config, |ev| {
            match ev {
                tools::LoopEvent::Reasoning(s) => tx
                    .send(StreamEvent::Delta(llm::StreamDelta::Reasoning(s.clone())))
                    .is_ok(),
                tools::LoopEvent::Content(s) => tx
                    .send(StreamEvent::Delta(llm::StreamDelta::Content(s.clone())))
                    .is_ok(),
                // run_compact emits only Reasoning/Content deltas.
                _ => true,
            }
        })
        .await;
        match res {
            Ok(turn) => {
                // The summary is one response; report it like any other
                // ResponseEnd so its finish reason is persisted on the turn
                // and surfaced in the status bar (§2.5).
                let _ = tx.send(StreamEvent::ResponseEnd {
                    tool_calls: Vec::new(),
                    finish: turn.finish(),
                });
                let _ = tx.send(StreamEvent::Done);
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Failed(e.to_string()));
            }
        }
    });
    app.streaming = true;
    app.compacting = true;
    app.reason_buf.clear();
    app.stream_buf.clear();
    app.got_reply = false;
    Ok(task)
}

/// Persist the assistant response that just ended (reasoning in a
/// collapsible Thinking block ahead of the answer, tool calls as trailing
/// blocks, and the `length_limit_reached` / `content_filtered` header tags
/// when set — §2.5 of `docs/format.md`) and push it into the in-memory
/// conversation. A response with neither text, reasoning, nor tool calls is
/// skipped.
fn finalize_response(
    app: &mut App,
    tool_calls: &[crate::llm::CompletedToolCall],
    length_limit_reached: bool,
    content_filtered: bool,
) -> Result<()> {
    let reply = app.stream_buf.trim_end().to_string();
    let reasoning = app.reason_buf.trim_end().to_string();
    app.reason_buf.clear();
    app.stream_buf.clear();
    if reply.is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
        return Ok(());
    }
    let turn = tools::assistant_turn(
        &app.model,
        &reasoning,
        &reply,
        tool_calls,
        length_limit_reached,
        content_filtered,
    );
    tools::append_turn(&app.path, &turn)?;
    app.conv.turns.push(turn);
    app.got_reply = true;
    Ok(())
}

/// The status-bar message for a finished response, from its finish reason
/// (§2.5 of `docs/format.md`): the truncation notice for `"length"`, the
/// non-fatal warning for a missing or unrecognized value, or empty for the
/// known values.
fn finish_status(finish: llm::FinishReason) -> String {
    if finish.is_length() {
        "response truncated at the output-token limit; tool calls not executed".into()
    } else if let Some(w) = finish.warning() {
        format!("warning: {w}")
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests;
