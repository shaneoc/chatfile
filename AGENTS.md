# Project Specification: mdchat
Build a fast, single-binary Rust CLI and web server for managing LLM
conversations stored as plain Markdown files inside nested directories.

## Core Design Principles
1. **Plain-Text First:** Every conversation is a standard Markdown file with
   YAML frontmatter (`model`, `endpoint`, `tools`) and explicit role markers
   identified by markdown headers (`# User`, `# Assistant`, `# Tool`).
   Conversation files are saved with a `.chat.md` extension.
2. **Hierarchical Context:** Before sending a prompt, walk up from the `.md`
   file's directory to the vault root and concatenate any `AGENTS.md` files
   found into the system prompt. The vault root is identified by the existance
   of a .mdchat file.
3. **OpenAI-Compatible Backend:** Target
   `http://localhost:9000/v1/chat/completions` (`llama.cpp`) by default,
   supporting streaming responses and multi-turn tool calling, with
   configurable cloud endpoints/API keys via environment variables.
4. **Thin Interfaces:**
   - `mdchat run <file.md>`: Reads the file, executes the next assistant turn
     (including any tool calls), streams/appends the output directly back to
     `<file.md>`, and exits. (Designed to be called from Neovim or CLI).
   - `mdchat serve --root <dir> --port 4096`: Serves an Axum + HTMX web UI for
     browsing folders, viewing PDFs/images inline, and chatting from desktop or
     Android browsers.

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
