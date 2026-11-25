# chatfile

Plain-text LLM conversations as Markdown files, with an interactive terminal UI
and a bubblewrap-sandboxed tool-calling agent — for local models (llama.cpp and
friends) and any OpenAI-compatible endpoint.

Every conversation is a single `.chat.md` file: YAML frontmatter plus turns
delimited by `---` boundaries (`**User**`, `**Assistant**`, `**Tool**`).
Because the format is plain Markdown, conversations are diffable in git,
editable by hand, and portable between tools.

> **Early development (`v0.1.0`):** `chatfile` is under active early
> development. While the core TUI, conversation engine, and
> bubblewrap-sandboxed tools work today, the `.chat.md` file format, YAML
> frontmatter options, CLI flags, and Rust library API may still change before
> `1.0.0`.

## Features

- **Plain-file conversations** — one `.chat.md` per conversation; reasoning
  (from thinking models) is persisted inline between `think` / `endthink` tags.
- **Hierarchical context** — before each turn, chatfile walks up from the
  file's directory to the vault root (the nearest ancestor containing a `.git`
  entry or a `.chatfile` marker file) and concatenates every `AGENTS.md` found
  along the way into the system prompt, most general first.
- **Interactive TUI** — a `ratatui` terminal UI with per-role styling, live
  streaming, cancel (Ctrl+C), and scrolling; `run` mode streams the next
  assistant turn to stdout for scripting (e.g. from Neovim).
- **Sandboxed agent tools** — built-in `list_files`, `read_file`, `write_file`,
  `edit_file`, `search_files`, `fetch_url`, and a `bash` tool executed inside a
  rootless [`bubblewrap`](https://github.com/containers/bubblewrap) sandbox:
  the model can only see its working directory, and nothing else — no host
  filesystem, no network (unless you opt in per conversation).
- **Multi-turn tool loop** — tool calls are streamed, executed against the
  conversation's working directory, persisted as `**Tool**` turns, and fed back
  to the model until it produces a final answer.

## Quickstart

```sh
# Open (or create) a conversation in the TUI
chatfile notes.chat.md

# Stream the next assistant turn to stdout and append it to the file
chatfile run notes.chat.md

# Create a starter template
chatfile new notes
```

With no configuration, chatfile targets llama.cpp at
`http://localhost:9000/v1/chat/completions` with model `local-model`.

### Environment variables

Frontmatter in the `.chat.md` file wins over environment variables; the
`CHATFILE_*` variables win over the standard OpenAI-compatible ones.

| Variable            | Meaning                                          |
| ------------------- | ------------------------------------------------ |
| `CHATFILE_ENDPOINT` | Full `/v1/chat/completions` URL                  |
| `OPENAI_API_BASE`   | Same, honored when `CHATFILE_ENDPOINT` is unset  |
| `CHATFILE_MODEL`    | Model name                                       |
| `CHATFILE_API_KEY`  | Bearer token for cloud endpoints                 |
| `OPENAI_API_KEY`    | Same, honored when `CHATFILE_API_KEY` is unset   |

### Conversation frontmatter

```yaml
---
model: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_0
endpoint: http://localhost:9000/v1/chat/completions
scope: folder        # tools operate in the file's directory (default) or the vault root
tools: true          # enable built-in tools (default true; false disables them all)
network: false       # allow fetch_url and shared-net bash (default false)
allow_shell: false   # allow the bash tool (default false)
---
```

### Turn headers

Each section starts with a bold role, optionally followed by plain metadata
segments separated by ` | ` — a bare timestamp first, then `key: value` pairs
(all optional):

```markdown
**Assistant** | 2026-11-23 22:42 | model: unsloth/Qwen3.8-27B-GGUF:Q8_K_XL
```

The first section of the body must be `**System**` or `**User**`; a
`**System**` section is only valid there (its content is folded into the
single leading system message). The parser is strict about the header format
and reports violations with line numbers.

The complete specification — file structure, header grammar, fenced blocks,
tool turns, and the full error catalog — lives in
[FORMAT.md](FORMAT.md).

- `scope: folder` (default) — file tools operate in the conversation file's own
  directory.
- `scope: vault` — file tools operate in the vault root (nearest ancestor with
  a `.git` entry or `.chatfile` marker). The `AGENTS.md` context walk always
  goes to the vault root regardless of scope.

## Sandbox configuration

The `bash` tool runs through bubblewrap with a policy loaded from optional YAML
config files — `/etc/chatfile/config.yaml` (system) and
`~/.config/chatfile/config.yaml` (user, overrides system) — falling back to
built-in distro-agnostic defaults when neither exists. A config file may name
any subset of the fields; unnamed fields keep the value from the layer below:

```yaml
bwrap_path: bwrap            # path to the bwrap binary (bare names use PATH)
ro_bind: [/usr, /etc]        # host paths bound read-only
ro_bind_try: [/nix, /opt]    # ...only if they exist
symlink_or_ro_bind: [/bin]   # recreate symlinks / bind dirs (usr-merged distros)
mask_tmpfs: [/etc/ssh]       # cover with an empty tmpfs
mask_devnull: [/etc/shadow]  # replace with /dev/null
env:                         # the complete environment (--clearenv is used)
  PATH: /usr/local/bin:/usr/bin:/bin
extra_args: []               # raw bwrap flags appended last
```

Inside the sandbox, the conversation's working directory is the only writable
location and becomes the child's cwd; `/proc`, `/dev`, and a private `/tmp` are
provided by bubblewrap. The network namespace is unshared unless `network:
true` is set in the conversation's frontmatter (loopback works either way).
Commands are killed after a timeout and their combined output is truncated
before being fed back to the model.

## Development notes

This project was developed iteratively via [OpenCode](https://opencode.ai): the
code was generated by a local LLM —
[unsloth/Qwen3.8-27B-GGUF:Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
— which wrote, tested, and reviewed its own output as it went, with a single
human steering the milestones; some advice was provided by [Google
Search AI Mode](https://www.google.com/search). External pull requests are not
accepted, but bug reports and feature ideas in GitHub Issues are very welcome.

## License

Dual-licensed under the MIT License or the Apache License, Version 2.0 — see
[LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
