//! mdchat — manage LLM conversations as plain Markdown files.
//!
//!   - `mdchat <file>`         open the interactive TUI on a conversation
//!                             (`.chat.md` is appended when no extension is
//!                             given; the file is created with default YAML
//!                             frontmatter if it does not exist yet)
//!   - `mdchat run <file>`     stream the next assistant turn into stdout
//!                             and append it to the file
//!   - `mdchat new <file>`     create a starter conversation template

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use mdchat::{context, llm, parser, tui};

/// Environment variable overrides (frontmatter wins over these).
const ENV_ENDPOINT: &str = "MDCHAT_ENDPOINT";
const ENV_MODEL: &str = "MDCHAT_MODEL";
const ENV_API_KEY: &str = "MDCHAT_API_KEY";
const DEFAULT_MODEL: &str = "local-model";

/// The only file extension accepted for conversation files.
const CONVERSATION_EXT: &str = "chat.md";

#[derive(Parser)]
#[command(name = "mdchat", version, about = "Manage LLM conversations as plain Markdown files")]
struct Cli {
    /// Conversation file to open in the interactive TUI.
    #[arg(value_name = "FILE")]
    file: Option<PathBuf>,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Run { file }) => run(&file).await,
        Some(Command::New { file }) => new(&file),
        None => match cli.file {
            Some(file) => open_tui(file).await,
            None => bail!("missing conversation file; try `mdchat --help`"),
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

/// The metadata portion of a boundary line (` · *...*`), or empty when the
/// metadata is absent.
fn meta_boundary_suffix(meta: &parser::TurnMeta) -> String {
    match meta.render() {
        Some(r) => format!(" · *{r}*"),
        None => String::new(),
    }
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

/// `mdchat <file>` — open the interactive TUI.
async fn open_tui(path: PathBuf) -> Result<()> {
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
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create parent directories for {}", path.display()))?;
            }
        }
        std::fs::write(&path, default_template())
            .with_context(|| format!("failed to create {}", path.display()))?;
        eprintln!("[mdchat] created {} with default frontmatter", path.display());
    }
    tui::run_tui(path).await
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

/// `mdchat run <file.chat.md>`
async fn run(file: &Path) -> Result<()> {
    require_conversation_ext(file)?;
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read {}", file.display()))?;
    let conv = parser::parse(&raw)
        .with_context(|| format!("failed to parse {}", file.display()))?;

    let system_prompt =
        context::build_system_prompt(file, conv.frontmatter.system_prompt.as_deref());
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
    let api_key = env(ENV_API_KEY);

    let mut messages: Vec<llm::Message> = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(llm::Message::system(system_prompt));
    }
    for turn in &conv.turns {
        // Roles pass through verbatim; llama.cpp accepts `tool`. Strict
        // OpenAI requires tool_call_id wiring, which lands with the
        // tool-calling milestone.
        messages.push(llm::Message::new(turn.role.as_str(), turn.content.clone()));
    }
    if messages.is_empty() {
        bail!(
            "{} has no turns and no system prompt; add a `**User**` turn first",
            file.display()
        );
    }

    eprintln!("[mdchat] model={} endpoint={}", model, endpoint);
    let client = llm::Client::new(endpoint, model.clone(), api_key);
    let mut stream = client.stream_chat(&messages).await?;

    // Stream the reply to stdout while accumulating it. Reasoning deltas
    // (thinking models) are collected but not printed, so stdout stays a
    // clean stream of just the answer.
    let mut reply = String::new();
    let mut reasoning = String::new();
    let stdout = std::io::stdout();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[mdchat] interrupted");
                break;
            }
            next = stream.next() => match next {
                Some(Ok(llm::StreamDelta::Reasoning(chunk))) => {
                    reasoning.push_str(&chunk);
                }
                Some(Ok(llm::StreamDelta::Content(chunk))) => {
                    reply.push_str(&chunk);
                    let mut out = stdout.lock();
                    let _ = out.write_all(chunk.as_bytes());
                    let _ = out.flush();
                }
                Some(Err(e)) => bail!("stream error: {e}"),
                None => break,
            },
        }
    }
    if !reply.is_empty() && !reply.ends_with('\n') {
        let mut out = stdout.lock();
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }

    // Append the assistant turn to the conversation file.
    if reply.trim().is_empty() {
        eprintln!("[mdchat] no reply produced; file unchanged");
        return Ok(());
    }
    if !reasoning.trim().is_empty() {
        eprintln!(
            "[mdchat] captured {} chars of reasoning",
            reasoning.trim().chars().count()
        );
    }
    let mut appendage = String::new();
    if !raw.ends_with('\n') {
        appendage.push('\n');
    }
    // Persist reasoning (between think/endthink tags) plus the answer,
    // escaping any boundary pattern so it can't be mistaken for a real turn
    // marker, and stamp the turn with time and model metadata.
    let ts = parser::now_timestamp();
    let meta = parser::TurnMeta {
        timestamp: Some(ts),
        tags: vec![("model".into(), model.clone())],
    };
    let content = parser::with_reasoning(
        (!reasoning.trim().is_empty()).then_some(reasoning.trim()),
        reply.trim_end(),
    );
    let content = parser::escape_boundaries(&content);
    appendage.push_str(&format!("\n---\n**Assistant**{}\n\n{}\n", meta_boundary_suffix(&meta), content));

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .with_context(|| format!("failed to open {} for appending", file.display()))?;
    f.write_all(appendage.as_bytes()).context("failed to write assistant turn")?;
    eprintln!("[mdchat] appended assistant turn to {}", file.display());
    Ok(())
}

/// The starter template written by `new` (and by the TUI when it creates a
/// missing file).
fn default_template() -> String {
    format!(
        "---\nmodel: {0}\nendpoint: {1}\nsystem_prompt:\n---\n\n---\n**User**\n\nYour prompt here\n",
        DEFAULT_MODEL,
        llm::DEFAULT_ENDPOINT
    )
}

/// `mdchat new <file>`
fn new(file: &Path) -> Result<()> {
    let file = with_conversation_ext(file);
    if file.exists() {
        bail!("{} already exists; refusing to overwrite", file.display());
    }
    if let Some(parent) = file.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create parent directories for {}", file.display()))?;
        }
    }
    std::fs::write(&file, default_template()).with_context(|| format!("failed to write {}", file.display()))?;
    eprintln!("[mdchat] created {}", file.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_template_and_refuses_overwrite() {
        let dir = std::env::temp_dir().join(format!("mdchat-new-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("a").join("b").join("conv.md");

        new(&file).unwrap();
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
        assert!(new(&file).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_appends_default_extension() {
        let dir = std::env::temp_dir().join(format!("mdchat-newext-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("notes");

        new(&file).unwrap();
        assert!(dir.join("notes.chat.md").exists());

        // An explicit `.chat.md` is left untouched.
        new(&dir.join("exact.chat.md")).unwrap();
        assert!(dir.join("exact.chat.md").exists());
        assert!(!dir.join("exact.chat.chat.md").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_rejects_wrong_extension() {
        let dir = std::env::temp_dir().join(format!("mdchat-runext-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("conv.md");
        std::fs::write(&file, "---\n**User**\n\nhi\n").unwrap();

        let err = require_conversation_ext(&file).unwrap_err().to_string();
        assert!(err.contains(".chat.md"), "unexpected error: {err}");
        assert!(require_conversation_ext(&dir.join("ok.chat.md")).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
