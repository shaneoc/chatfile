//! chatfile: plain-text LLM conversations as Markdown files, with a TUI and
//! a sandboxed tool-calling agent.
//!
//! Every conversation is a single `.chat.md` file: YAML frontmatter (model,
//! endpoint, tool settings) followed by turns delimited by `---` boundaries
//! (`**User**`, `**Assistant**`, `**Tool**`). Because the format is plain
//! Markdown, conversations are diffable, editable by hand, and portable.
//!
//! # Modules
//!
//! - [`config`] — XDG-compliant TOML configuration
//!   (`$XDG_CONFIG_HOME/chatfile/config.toml`, `$XDG_CONFIG_DIRS`): profiles,
//!   sandbox policy, TUI settings, and prompt overrides.
//! - [`parser`] — parse and serialize `.chat.md` files (frontmatter, turns,
//!   persisted tool results, reasoning blocks).
//! - [`context`] — hierarchical `AGENTS.md` context: walk up from the
//!   conversation file to the vault root (a `.git` or `.chatfile` marker)
//!   and concatenate the instructions found along the way.
//! - [`llm`] — OpenAI-compatible `/v1/chat/completions` client with SSE
//!   streaming, including tool-call fragments and `reasoning_content`.
//! - [`tools`] — built-in tools (`list_files`, `read_file`, `write_file`,
//!   `edit_file`, `search_files`, `fetch_url`, sandboxed `bash`) and the
//!   multi-turn tool-calling loop ([`tools::run_turn`]).
//! - [`sandbox`] — rootless bubblewrap sandbox used by the `bash` tool
//!   (policy comes from [`config::SandboxConfig`]).
//! - [`tui`] — the interactive `ratatui` terminal UI.
//!
//! The binary (`src/main.rs`) is a thin CLI over these modules:
//! `chatfile <file>` opens the TUI, `chatfile run <file>` streams the next
//! assistant turn to stdout, and `chatfile new <file>` creates a starter
//! template.
//!
//! # Example: parse a conversation file
//!
//! ```no_run
//! use chatfile::parser;
//!
//! let raw = std::fs::read_to_string("notes.chat.md").unwrap();
//! let conv = parser::parse(&raw).unwrap();
//! for turn in &conv.turns {
//!     println!("{}: {} chars", turn.role, turn.content.len());
//! }
//! ```

#![warn(missing_docs)]

pub mod config;
pub mod context;
pub mod llm;
pub mod parser;
pub mod sandbox;
pub mod tools;
pub mod tui;
