//! mdchat — manage LLM conversations as plain Markdown files.
//!
//! Milestone 1: core CLI and the Markdown conversation loop.
//!   - `mdchat run <file.md>`  stream the next assistant turn into stdout
//!                             and append it to the file
//!   - `mdchat new <file.md>`  create a starter conversation template

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use mdchat::{context, llm, parser};

/// Environment variable overrides (frontmatter wins over these).
const ENV_ENDPOINT: &str = "MDCHAT_ENDPOINT";
const ENV_MODEL: &str = "MDCHAT_MODEL";
const ENV_API_KEY: &str = "MDCHAT_API_KEY";
const DEFAULT_MODEL: &str = "local-model";

#[derive(Parser)]
#[command(name = "mdchat", version, about = "Manage LLM conversations as plain Markdown files")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read the conversation file and run the next assistant turn.
    ///
    /// The reply is streamed to stdout and appended to the file as a
    /// `<!-- role: assistant -->` turn.
    Run {
        /// Path to the conversation .md file.
        file: PathBuf,
    },
    /// Create a starter conversation template at the given path.
    New {
        /// Path for the new conversation .md file (parents are created).
        file: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run { file } => run(&file).await,
        Command::New { file } => new(&file),
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

/// `mdchat run <file.md>`
async fn run(file: &Path) -> Result<()> {
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
            "{} has no turns and no system prompt; add a `<!-- role: user -->` turn first",
            file.display()
        );
    }

    eprintln!("[mdchat] model={} endpoint={}", model, endpoint);
    let client = llm::Client::new(endpoint, model, api_key);
    let mut stream = client.stream_chat(&messages).await?;

    // Stream the reply to stdout while accumulating it.
    let mut reply = String::new();
    let stdout = std::io::stdout();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[mdchat] interrupted");
                break;
            }
            next = stream.next() => match next {
                Some(Ok(chunk)) => {
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
    let mut appendage = String::new();
    if !raw.ends_with('\n') {
        appendage.push('\n');
    }
    appendage.push_str(&format!("\n<!-- role: assistant -->\n\n{}\n", reply.trim_end()));

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .with_context(|| format!("failed to open {} for appending", file.display()))?;
    f.write_all(appendage.as_bytes()).context("failed to write assistant turn")?;
    eprintln!("[mdchat] appended assistant turn to {}", file.display());
    Ok(())
}

/// `mdchat new <file.md>`
fn new(file: &Path) -> Result<()> {
    if file.exists() {
        bail!("{} already exists; refusing to overwrite", file.display());
    }
    if let Some(parent) = file.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create parent directories for {}", file.display()))?;
        }
    }
    let template = format!(
        "---\nmodel: {0}\nendpoint: {1}\nsystem_prompt:\n---\n\n<!-- role: user -->\n\nYour prompt here\n",
        DEFAULT_MODEL,
        llm::DEFAULT_ENDPOINT
    );
    std::fs::write(file, template).with_context(|| format!("failed to write {}", file.display()))?;
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
        let content = std::fs::read_to_string(&file).unwrap();
        assert!(content.starts_with("---\nmodel: local-model\n"));
        assert!(content.contains(llm::DEFAULT_ENDPOINT));
        assert!(content.contains("<!-- role: user -->"));

        // Must parse cleanly, with a single empty user turn.
        let conv = parser::parse(&content).unwrap();
        assert!(conv.had_frontmatter);
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].role, parser::Role::User);

        // Refuse to overwrite an existing file.
        assert!(new(&file).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
