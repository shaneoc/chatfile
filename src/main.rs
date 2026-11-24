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
use mdchat::{context, llm, parser, tools, tui};

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
    // `to_messages` folds persisted `**Tool**` turns into `role: "tool"`
    // messages and attaches their `tool_calls` to the preceding assistant
    // message, so reloaded histories are valid API payloads.
    messages.extend(conv.to_messages());
    if messages.is_empty() {
        bail!(
            "{} has no turns and no system prompt; add a `**User**` turn first",
            file.display()
        );
    }

    eprintln!("[mdchat] model={} endpoint={}", model, endpoint);
    let client = llm::Client::new(endpoint, model.clone(), api_key);
    let tool_config = tools::ToolConfig::for_file(&conv.frontmatter, file);

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
            ResponseEnd => {
                // Persist the response just finished (skipped when it had no
                // text and no reasoning, i.e. a pure tool-call response).
                if !(reply.trim().is_empty() && reasoning.trim().is_empty()) {
                    let turn = tools::assistant_turn(&model, &reasoning, &reply);
                    if tools::append_turn(file, &turn).is_ok() {
                        appended += 1;
                    }
                    eprintln!(
                        "[mdchat] appended assistant turn ({} chars)",
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
                arguments,
                output,
            } => {
                let turn = tools::tool_turn(name, id, arguments, output);
                if tools::append_turn(file, &turn).is_ok() {
                    appended += 1;
                }
                eprintln!(
                    "[mdchat] tool {name}: {} chars of output",
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
        eprintln!("[mdchat] no reply produced; file unchanged");
        return Ok(());
    }
    if reasoning_chars > 0 {
        eprintln!("[mdchat] captured {reasoning_chars} chars of reasoning");
    }
    eprintln!("[mdchat] appended {appended} turns to {}", file.display());
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
