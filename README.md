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
  streaming, cancel (Ctrl+C), and keyboard/mouse-wheel scrolling; `run` mode
  streams the next assistant turn to stdout for scripting (e.g. from Neovim).
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
`CHATFILE_*` variables win over the standard OpenAI-compatible ones, and both
win over a named profile from `config.toml` (see [Configuration](#configuration)).

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

- `scope: folder` (default) — file tools operate in the conversation file's own
  directory.
- `scope: vault` — file tools operate in the vault root (nearest ancestor with
  a `.git` entry or `.chatfile` marker). The `AGENTS.md` context walk always
  goes to the vault root regardless of scope.

### Turn headers

Each section starts with a bold role, optionally followed by plain metadata
segments separated by ` | ` — a bare timestamp first (with an explicit numeric
UTC offset, `YYYY-MM-DD HH:MM ±HH:MM`), then `key: value` pairs (all
optional):

```markdown
**Assistant** | 2026-11-23 22:42 +00:00 | model: unsloth/Qwen3.8-27B-GGUF:Q8_K_XL
```

The first section of the body must be `**System**` or `**User**`; a
`**System**` section is only valid there (its content is folded into the
single leading system message). The parser is strict about the header format
and reports violations with line numbers.

The complete specification — file structure, header grammar, fenced blocks,
tool turns, and the full error catalog — lives in
[docs/format.md](docs/format.md).

## Configuration

chatfile reads an optional TOML config file from the standard XDG locations,
lowest priority first:

1. `<dir>/chatfile/config.toml` for each directory in `$XDG_CONFIG_DIRS`
   (colon-separated; per the XDG spec the *first* entry has the *highest*
   priority), falling back to `/etc/xdg/chatfile/config.toml` when the
   variable is unset or empty.
2. `$XDG_CONFIG_HOME/chatfile/config.toml`, falling back to
   `~/.config/chatfile/config.toml` when `$XDG_CONFIG_HOME` is unset or empty.

A file may name any subset of the keys; unnamed keys keep the value from the
layer below, and missing files are skipped. A file that exists but is not
valid TOML is a hard error. Every field in the schema below is optional:

```toml
# The profile used when frontmatter and CHATFILE_* env vars are unset.
default_profile = "local"

[profiles.local]
endpoint = "http://localhost:9000/v1/chat/completions"
model = "qwen"
api_key_env = "MY_KEY"           # name of the env var holding the key (never the key itself)
scope = "vault"                  # default tool scope: "folder" (default) or "vault"
tools = ["read_file", "bash"]    # optional allow-list of built-in tools

[profiles.cloud]
endpoint = "https://api.example.com/v1/chat/completions"
model = "gpt-x"

[sandbox]
bwrap_path = "bwrap"             # path to the bwrap binary (bare names use PATH)
ro_bind = ["/usr", "/etc"]       # host paths bound read-only
ro_bind_try = ["/nix", "/opt"]   # ...only if they exist
symlink_or_ro_bind = ["/bin"]    # recreate symlinks / bind dirs (usr-merged distros)
mask_tmpfs = ["/etc/ssh"]        # cover with an empty tmpfs
mask_devnull = ["/etc/shadow"]   # replace with /dev/null
env = { PATH = "/usr/local/bin:/usr/bin:/bin" }  # the complete environment (--clearenv is used)
extra_args = []                  # raw bwrap flags appended last

[tui]
mouse = true                     # capture mouse events (scroll wheel scrolls the history)

[prompts]
compact = """Summarize in three bullets.
Keep it short."""
```

### LLM resolution precedence

`.chat.md` frontmatter > `CHATFILE_*` environment variables > the active
profile (`default_profile`) > standard `OPENAI_*` variables > built-in
defaults (llama.cpp at `http://localhost:9000/v1/chat/completions`, model
`local-model`). The API key resolves as `CHATFILE_API_KEY` > the active
profile's `api_key_env` variable > `OPENAI_API_KEY`. A profile's `tools`
allow-list and `scope` apply to conversations that do not override them in
frontmatter; `fetch_url` and `bash` still require the frontmatter's
`network`/`allow_shell` flags.

The `[tui] mouse` setting (default `true`) can be overridden per invocation
with `chatfile --mouse <file>` or `chatfile --mouse=false <file>`.

### Sandbox behavior

The `bash` tool runs through bubblewrap with the policy from the `[sandbox]`
table above, falling back to built-in distro-agnostic defaults when no config
names a key. Inside the sandbox, the conversation's working directory is the
only writable location and becomes the child's cwd; `/proc`, `/dev`, and a
private `/tmp` are provided by bubblewrap. The network namespace is unshared
unless `network: true` is set in the conversation's frontmatter (loopback
works either way). Commands are killed after a timeout and their combined
output is truncated before being fed back to the model.

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
