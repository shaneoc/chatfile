//! chatfile — manage LLM conversations as plain Markdown files.
//!
//!   - `chatfile <file>`       open the interactive TUI on a conversation
//!                             (`.chat.md` is appended when no extension is
//!                             given; the file is created with default YAML
//!                             frontmatter if it does not exist yet)
//!   - `chatfile run <file>`   stream the next assistant turn into stdout
//!                             and append it to the file
//!   - `chatfile compact <file>` append a `**User** | compact: true`
//!                             compaction request, stream the model's
//!                             handoff summary, and append it as an
//!                             `**Assistant**` turn
//!   - `chatfile new <file>`   create a starter conversation template

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chatfile::{config, context, llm, parser, tools, tui};
use clap::{Parser, Subcommand};

/// The only file extension accepted for conversation files.
const CONVERSATION_EXT: &str = "chat.md";

#[derive(Parser)]
#[command(
    name = "chatfile",
    version,
    about = "Manage LLM conversations as plain Markdown files"
)]
struct Cli {
    /// Conversation file to open in the interactive TUI.
    #[arg(value_name = "FILE")]
    file: Option<PathBuf>,
    /// Enable or disable mouse capture in the TUI (overrides the `[tui]`
    /// `mouse` setting from config.toml).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    mouse: Option<bool>,
    /// Cap on the model's output per response, sent as `max_tokens` in the
    /// API request. Overrides the conversation frontmatter's
    /// `max_output_tokens` and the active config.toml profile; when unset,
    /// the field is omitted and the endpoint's default applies.
    #[arg(long, global = true, value_name = "N")]
    max_output_tokens: Option<u32>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Read the conversation file and run the next assistant turn.
    ///
    /// The reply is streamed to stdout and appended to the file as an
    /// `**Assistant**` turn.
    Run {
        /// Path to the conversation .chat.md file.
        file: PathBuf,
    },
    /// Create a starter conversation template at the given path.
    New {
        /// Path for the new conversation file (parents are created; the
        /// `.chat.md` extension is applied when missing).
        file: PathBuf,
    },
    /// Compact the conversation in place: append a `**User** | compact: true`
    /// compaction request, stream the model's handoff summary to stdout, and
    /// append it as an `**Assistant**` turn.
    Compact {
        /// Path to the conversation .chat.md file.
        file: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::AppConfig::load()?;
    match cli.command {
        Some(Command::Run { file }) => run(&file, cli.max_output_tokens, &config).await,
        Some(Command::New { file }) => new(&file, &config),
        Some(Command::Compact { file }) => compact(&file, cli.max_output_tokens, &config).await,
        None => match cli.file {
            Some(file) => open_tui(file, cli.mouse, cli.max_output_tokens, &config).await,
            None => bail!("missing conversation file; try `chatfile --help`"),
        },
    }
}

/// True when the path ends with the `.chat.md` extension.
fn has_conversation_ext(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|n| n.ends_with(&format!(".{CONVERSATION_EXT}")))
}

/// Normalize a path for `new`: append `.chat.md` when the name has no
/// `.chat.md` extension (a plain `.md` is replaced).
fn with_conversation_ext(path: &Path) -> PathBuf {
    if has_conversation_ext(path) {
        return path.to_path_buf();
    }
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let new_name = if name.ends_with(".md") {
        format!("{}.chat.md", &name[..name.len() - 3])
    } else {
        format!("{name}.chat.md")
    };
    let mut out = path.to_path_buf();
    out.pop();
    out.push(new_name);
    out
}

/// Reject paths that do not carry the `.chat.md` extension.
fn require_conversation_ext(path: &Path) -> Result<()> {
    if !has_conversation_ext(path) {
        bail!(
            "{} is not a conversation file; expected a .{CONVERSATION_EXT} extension",
            path.display()
        );
    }
    Ok(())
}

/// `chatfile <file>` — open the interactive TUI.
async fn open_tui(
    path: PathBuf,
    mouse: Option<bool>,
    cli_max_output_tokens: Option<u32>,
    config: &config::AppConfig,
) -> Result<()> {
    // Append the extension when none was given; otherwise require it.
    let path = if path.extension().is_none() {
        path.with_extension(CONVERSATION_EXT)
    } else {
        path
    };
    require_conversation_ext(&path)?;
    if !path.exists() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create parent directories for {}", path.display())
                })?;
            }
        }
        std::fs::write(&path, default_template(config))
            .with_context(|| format!("failed to create {}", path.display()))?;
        eprintln!(
            "[chatfile] created {} with default frontmatter",
            path.display()
        );
    }
    // The CLI flag overrides the config file when given.
    let mouse = mouse.unwrap_or(config.tui.mouse);
    tui::run_tui(path, mouse, cli_max_output_tokens, config).await
}

/// `chatfile run <file.chat.md>`
async fn run(
    file: &Path,
    cli_max_output_tokens: Option<u32>,
    config: &config::AppConfig,
) -> Result<()> {
    require_conversation_ext(file)?;
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read {}", file.display()))?;
    let conv =
        parser::parse(&raw).with_context(|| format!("failed to parse {}", file.display()))?;

    if !conv.turns.iter().any(|t| t.role == parser::Role::User) {
        bail!(
            "{} has no **User** section; add one before running",
            file.display()
        );
    }
    let system_section = conv
        .turns
        .first()
        .filter(|t| t.role == parser::Role::System)
        .map(|t| t.content.as_str());
    let system_prompt = context::build_system_prompt(file, system_section);
    let (endpoint, model, api_key) = config.resolve_llm(
        conv.frontmatter.endpoint.as_deref(),
        conv.frontmatter.model.as_deref(),
    );
    // Precedence: CLI flag > frontmatter > active profile.
    let max_output_tokens =
        config.resolve_max_output_tokens(cli_max_output_tokens, conv.frontmatter.max_output_tokens);

    // `build_api_messages` applies the compaction windowing (§5 of
    // docs/format.md: a `compact: true` **User** checkpoint truncates the
    // earlier history) and projects the surviving turns (assistant tool
    // calls ride on the assistant message; `**Tool**` turns become
    // `role: "tool"` messages). `set_system_prompt` overlays the
    // hierarchical AGENTS.md prompt on the leading system message, and
    // `complete_tool_calls` answers any call left dangling by a cancelled
    // or interrupted execution, so reloaded histories are valid API
    // payloads.
    let mut messages = llm::build_api_messages(&conv);
    llm::set_system_prompt(&mut messages, &system_prompt);
    llm::complete_tool_calls(&mut messages);

    eprintln!("[chatfile] model={} endpoint={}", model, endpoint);
    let client = llm::Client::new(endpoint, model.clone(), api_key)
        .with_max_output_tokens(max_output_tokens);
    let tool_config = tools::ToolConfig::for_file(&conv.frontmatter, file, config);

    // Run the (possibly multi-turn) assistant reply. stdout stays a clean
    // stream of answer text: reasoning and tool results are only persisted
    // to the file, with progress notes on stderr.
    let mut reply = String::new();
    let mut reasoning = String::new();
    let mut final_reply = String::new();
    let mut reasoning_chars = 0usize;
    let mut appended = 0usize;
    let stdout = std::io::stdout();
    tools::run_turn(&client, messages, &tool_config, |ev| {
        use tools::LoopEvent::*;
        match ev {
            Reasoning(chunk) => {
                reasoning.push_str(chunk);
                true
            }
            Content(chunk) => {
                reply.push_str(chunk);
                let mut out = stdout.lock();
                let _ = out.write_all(chunk.as_bytes());
                let _ = out.flush();
                true
            }
            ResponseEnd {
                tool_calls,
                finish,
            } => {
                // Surface how the response ended (§2.5): a truncation notice
                // for "length", a non-fatal warning for a missing or
                // unrecognized finish_reason. All streamed content is kept.
                if finish.is_length() {
                    eprintln!(
                        "[chatfile] response truncated at the output-token limit; its tool calls were not executed"
                    );
                } else if let Some(w) = finish.warning() {
                    eprintln!("[chatfile] warning: {w}");
                }
                // Persist the response just finished (skipped when it had no
                // text, no reasoning, and no tool calls).
                if !(reply.trim().is_empty()
                    && reasoning.trim().is_empty()
                    && tool_calls.is_empty())
                {
                    let turn = tools::assistant_turn(
                        &model,
                        &reasoning,
                        &reply,
                        tool_calls,
                        finish.is_length(),
                        finish.is_content_filter(),
                    );
                    if tools::append_turn(file, &turn).is_ok() {
                        appended += 1;
                    }
                    eprintln!(
                        "[chatfile] appended assistant turn ({} chars)",
                        reply.trim().chars().count()
                    );
                }
                reasoning_chars += reasoning.trim().chars().count();
                final_reply = std::mem::take(&mut reply);
                reasoning.clear();
                true
            }
            ToolResult {
                id,
                name,
                arguments: _,
                output,
            } => {
                let turn = tools::tool_turn(name, id, output);
                if tools::append_turn(file, &turn).is_ok() {
                    appended += 1;
                }
                eprintln!(
                    "[chatfile] tool {name}: {} chars of output",
                    output.trim().chars().count()
                );
                true
            }
        }
    })
    .await?;
    if !final_reply.is_empty() && !final_reply.ends_with('\n') {
        let mut out = stdout.lock();
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }

    if appended == 0 {
        eprintln!("[chatfile] no reply produced; file unchanged");
        return Ok(());
    }
    if reasoning_chars > 0 {
        eprintln!("[chatfile] captured {reasoning_chars} chars of reasoning");
    }
    eprintln!("[chatfile] appended {appended} turns to {}", file.display());
    Ok(())
}

/// `chatfile compact <file.chat.md>` — append a `**User** | compact: true`
/// compaction request, stream the model's handoff summary, and append it as
/// an `**Assistant**` turn. The request turn is persisted before the model
/// is called, so a failed run leaves a well-formed file whose trailing
/// unanswered request the next `run` (or `compact`) simply answers.
async fn compact(
    file: &Path,
    cli_max_output_tokens: Option<u32>,
    config: &config::AppConfig,
) -> Result<()> {
    require_conversation_ext(file)?;
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read {}", file.display()))?;
    let mut conv =
        parser::parse(&raw).with_context(|| format!("failed to parse {}", file.display()))?;

    if !conv.turns.iter().any(|t| t.role == parser::Role::User) {
        bail!(
            "{} has no **User** section; add one before compacting",
            file.display()
        );
    }
    let request = llm::compact_request_turn(config.prompts.compact.as_deref());
    tools::append_turn(file, &request)?;
    conv.turns.push(request);

    let system_section = conv
        .turns
        .first()
        .filter(|t| t.role == parser::Role::System)
        .map(|t| t.content.as_str());
    let system_prompt = context::build_system_prompt(file, system_section);
    let (endpoint, model, api_key) = config.resolve_llm(
        conv.frontmatter.endpoint.as_deref(),
        conv.frontmatter.model.as_deref(),
    );
    // Precedence: CLI flag > frontmatter > active profile.
    let max_output_tokens =
        config.resolve_max_output_tokens(cli_max_output_tokens, conv.frontmatter.max_output_tokens);

    eprintln!("[chatfile] model={} endpoint={}", model, endpoint);
    let client = llm::Client::new(endpoint, model.clone(), api_key)
        .with_max_output_tokens(max_output_tokens);
    let tool_config = tools::ToolConfig::for_file(&conv.frontmatter, file, config);

    // Stream the summary: stdout stays a clean stream of answer text, with
    // progress notes on stderr.
    let mut reply = String::new();
    let mut reasoning = String::new();
    let stdout = std::io::stdout();
    let summary = llm::run_compact(&client, &conv, &system_prompt, &tool_config, |ev| {
        use tools::LoopEvent::*;
        match ev {
            Reasoning(chunk) => {
                reasoning.push_str(chunk);
                true
            }
            Content(chunk) => {
                reply.push_str(chunk);
                let mut out = stdout.lock();
                let _ = out.write_all(chunk.as_bytes());
                let _ = out.flush();
                true
            }
            // No tool loop runs for a compaction; stray events are ignored.
            _ => true,
        }
    })
    .await?;
    if !reply.trim().is_empty() && !reply.ends_with('\n') {
        let mut out = stdout.lock();
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }

    // Persist the summary only when the model produced something.
    if reply.trim().is_empty() && reasoning.trim().is_empty() {
        eprintln!("[chatfile] no summary produced; request turn kept, file otherwise unchanged");
        return Ok(());
    }
    let finish = summary.finish();
    if finish.is_length() {
        eprintln!(
            "[chatfile] summary truncated at the output-token limit; its tool calls were not executed"
        );
    } else if let Some(w) = finish.warning() {
        eprintln!("[chatfile] warning: {w}");
    }
    tools::append_turn(
        file,
        &tools::assistant_turn(
            &model,
            &reasoning,
            &reply,
            &[],
            finish.is_length(),
            finish.is_content_filter(),
        ),
    )?;
    eprintln!(
        "[chatfile] appended compaction summary ({} chars) to {}",
        reply.trim().chars().count(),
        file.display()
    );
    Ok(())
}

/// The starter template written by `new` (and by the TUI when it creates a
/// missing file), seeded with the resolved default endpoint and model.
fn default_template(config: &config::AppConfig) -> String {
    let (endpoint, model, _) = config.resolve_llm(None, None);
    format!("---\nmodel: {model}\nendpoint: {endpoint}\n---\n**User**\n\nYour prompt here\n")
}

/// `chatfile new <file>`
fn new(file: &Path, config: &config::AppConfig) -> Result<()> {
    let file = with_conversation_ext(file);
    if file.exists() {
        bail!("{} already exists; refusing to overwrite", file.display());
    }
    if let Some(parent) = file.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create parent directories for {}", file.display())
            })?;
        }
    }
    std::fs::write(&file, default_template(config))
        .with_context(|| format!("failed to write {}", file.display()))?;
    eprintln!("[chatfile] created {}", file.display());
    Ok(())
}

#[cfg(test)]
mod tests;
