# Project Specification: chatfile
Build a fast, single-binary Rust CLI and web server for managing LLM
conversations stored as plain Markdown files inside nested directories.

## Core Design Principles
1. **Plain-Text First:** Every conversation is a standard Markdown file with
   YAML frontmatter (`model`, `endpoint`, `tools`) and explicit role markers
   in bold lines (`**User**`, `**Assistant**`, `**Tool**`), optionally
   followed by plain metadata segments separated by ` | ` (a bare timestamp
   first, then `key: value` pairs — e.g. `**Assistant** | 2026-11-23 22:42 |
   model: local-model`). The first section after the frontmatter must be
   `**System**` or `**User**` (the frontmatter's closing fence serves as its
   leading horizontal rule); every later section is preceded by a `---`
   horizontal rule. Conversation files are saved with a `.chat.md` extension.
2. **Hierarchical Context:** Before sending a prompt, walk up from the `.md`
   file's directory to the vault root and concatenate any `AGENTS.md` files
   found into the system prompt. The vault root is identified by the existence
   of a .chatfile file. Every request carries exactly one system message,
   always first: the `AGENTS.md` context followed by the file's `**System**`
   section content (when present).
3. **OpenAI-Compatible Backend:** Target
   `http://localhost:9000/v1/chat/completions` (`llama.cpp`) by default,
   supporting streaming responses and multi-turn tool calling, with
   configurable cloud endpoints/API keys via environment variables.

## Source Layout
- `src/lib.rs` — crate root; exposes the public modules so docs.rs renders
  complete documentation.
- `src/main.rs` — thin CLI entrypoint (`chatfile <file>`, `run`, `new`).
- `src/parser.rs` — parse/serialize `.chat.md` files: frontmatter, turn
  boundaries, persisted tool turns, reasoning tags; `to_messages` folds the
  file into OpenAI wire-format messages.
- `src/context.rs` — hierarchical `AGENTS.md` walk to the vault root (`.git`
  or `.chatfile` marker) and system-prompt assembly.
- `src/llm.rs` — OpenAI-compatible streaming client (SSE deltas, reasoning,
  streamed tool-call fragments).
- `src/tools.rs` — built-in tools (`list_files`, `read_file`, `write_file`,
  `edit_file`, `search_files`, `fetch_url`, `bash`) with path-containment
  safety (`resolve_safe_path`), and the multi-turn tool-calling loop
  (`run_turn`) that executes calls against the conversation's working
  directory and persists `**Tool**` turns.
- `src/sandbox.rs` — config-driven rootless bubblewrap sandbox
  (`/etc/chatfile/config.yaml`, `~/.config/chatfile/config.yaml`) used by the
  `bash` tool: fresh mount/network namespaces, the work dir as the only
  writable location, timeouts and output truncation.
- `src/tui.rs` — the `ratatui` interactive UI (streaming, cancel, scrolling).

## Build & Verification Rules
- Use `cargo check` and `cargo test` after every module change.
- Keep dependencies minimal and well-maintained.

## Rust Dependencies
- Only use crates available as `librust-*-dev` packages in Debian (installed in
  `/usr/share/cargo/registry`).
- Before adding any new dependency to `Cargo.toml`, check if it exists in
  Debian via `apt-cache search librust-<name>-dev`. If the package exists as a
  Debian package however it has not been installed to
  `/usr/share/cargo/registry`, then temporarily stop your work and inform the
  user that the package needs to be installed.
