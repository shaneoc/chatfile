# Project Specification: mdchat
Build a fast, single-binary Rust CLI and web server for managing LLM
conversations stored as plain Markdown files inside nested directories.

## Core Design Principles
1. **Plain-Text First:** Every conversation is a standard `.md` file with YAML
   frontmatter (`model`, `endpoint`, `tools`) and explicit role markers (`<!-- role: user -->`, `<!-- role: assistant -->`, `<!-- role: tool -->`).
2. **Hierarchical Context:** Before sending a prompt, walk up from the `.md`
   file's directory to the vault root and concatenate any `AGENTS.md` files
   found into the system prompt.
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
