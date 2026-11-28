//! Built-in tools and the multi-turn tool-calling loop.
//!
//! Tools are pure-Rust file operations plus a sandboxed `bash`. Every path
//! argument is resolved against the conversation's working directory and
//! verified to stay inside it (symlinks included), so a model — or a
//! hand-edited file — cannot read or write outside the vault.
//!
//! The loop in [`run_turn`] drives one user turn end-to-end: stream an
//! assistant response, execute any tool calls it issued, feed the results
//! back as `role: "tool"` messages, and repeat until the model produces a
//! final text reply (or [`MAX_TOOL_ITERATIONS`] is reached).

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde_json::{Value as Json, json};

use crate::config::{AppConfig, ReasoningHistory, SandboxConfig};
use crate::context;
use crate::llm::{
    AssistantTurn, Client, CompletedToolCall, FinishReason, Message, StreamDelta, TokenUsage,
    ToolCall, ToolFunction, build_api_messages, complete_tool_calls, set_system_prompt,
    should_auto_compact,
};
use crate::parser;
use crate::sandbox;

/// Maximum LLM round-trips per user turn before the loop gives up.
pub const MAX_TOOL_ITERATIONS: usize = 15;

/// All built-in tool names. This is the default allow-list when the active
/// profile does not restrict tools with a `tools` list; `fetch_url` and
/// `bash` are additionally gated by the frontmatter's `network`/`allow_shell`.
pub const DEFAULT_TOOLS: &[&str] = &[
    "list_files",
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "fetch_url",
    "bash",
];

/// Default hard cap on tool output bytes fed back to the model (and
/// persisted); the active profile's `max_tool_output_bytes` overrides it.
const MAX_TOOL_OUTPUT: usize = 16384;

/// Cap on `list_files` entries.
const LIST_CAP: usize = 200;

/// Cap on `search_files` matches.
const SEARCH_CAP: usize = 50;

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

/// Resolve a model-supplied path against `work_dir`, refusing anything that
/// escapes it.
///
/// Rules: the path must be non-empty, relative (no leading `/`), and free of
/// `..` components. The result is then verified by canonicalizing the
/// deepest *existing* ancestor and re-appending the missing remainder — this
/// catches symlink escapes where an existing directory inside the work dir
/// points outside of it.
pub fn resolve_safe_path(work_dir: &Path, user_path: &str) -> Result<PathBuf> {
    let user = user_path.trim();
    if user.is_empty() {
        bail!("path is empty");
    }
    let p = Path::new(user);
    if p.is_absolute() {
        bail!(
            "absolute paths are not allowed; use a path relative to the working directory ({})",
            work_dir.display()
        );
    }
    for comp in p.components() {
        if matches!(comp, Component::ParentDir | Component::Prefix(_)) {
            bail!("path escapes the working directory: contains '..'");
        }
    }

    let candidate = work_dir.join(p);
    let (existing, remainder) = split_existing(&candidate);
    let canon_base = existing
        .canonicalize()
        .with_context(|| format!("path does not exist: {user}"))?;
    let resolved = if remainder.as_os_str().is_empty() {
        canon_base
    } else {
        canon_base.join(remainder)
    };

    let canon_work = work_dir
        .canonicalize()
        .with_context(|| format!("working directory {} does not exist", work_dir.display()))?;
    if !resolved.starts_with(&canon_work) {
        bail!("path escapes the working directory: {user}");
    }
    Ok(resolved)
}

/// Split `path` into the deepest existing ancestor and the remaining
/// (non-existent) components. The filesystem root always exists, so this
/// always succeeds.
fn split_existing(path: &Path) -> (PathBuf, PathBuf) {
    let comps: Vec<Component> = path.components().collect();
    for split_at in (0..=comps.len()).rev() {
        let base: PathBuf = comps[..split_at].iter().collect();
        if base.exists() {
            let rem: PathBuf = comps[split_at..].iter().collect();
            return (base, rem);
        }
    }
    unreachable!("the filesystem root always exists")
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-conversation tool configuration, derived from frontmatter with the
/// app config supplying defaults (frontmatter wins).
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// Whether built-in tools are enabled at all (frontmatter `tools`).
    pub enabled: bool,
    /// Allow network access (`fetch_url`, shared-net sandboxed bash).
    pub network: bool,
    /// Allow the `bash` tool.
    pub allow_shell: bool,
    /// The directory file tools operate in (from frontmatter `scope`, with
    /// the active profile's `scope` as the fallback default).
    pub work_dir: PathBuf,
    /// Tool names enabled for this conversation (the active profile's
    /// `tools` list); `None` means all built-in file tools.
    pub allowed_tools: Option<Vec<String>>,
    /// Sandbox policy for the `bash` tool (from `[sandbox]` in
    /// `config.toml`, with built-in defaults).
    pub sandbox: SandboxConfig,
    /// Hard cap on tool output bytes fed back to the model (and persisted),
    /// from the active profile's `max_tool_output_bytes` (falling back to
    /// [`MAX_TOOL_OUTPUT`]).
    pub max_tool_output_bytes: usize,
    /// Which assistant turns' persisted reasoning is re-sent as
    /// `reasoning_content`, from the active profile's `reasoning_history`
    /// (falling back to the default, [`ReasoningHistory::ActiveTurn`]).
    pub reasoning_history: ReasoningHistory,
}

impl ToolConfig {
    /// Build the config for a conversation file: `scope: folder` selects the
    /// file's own directory, `scope: vault` the nearest ancestor carrying a
    /// `.chatfile` marker or `.git` entry (falling back to the file's dir).
    /// The active profile supplies the default scope and tool list; explicit
    /// frontmatter settings win.
    pub fn for_file(fm: &parser::Frontmatter, file: &Path, config: &AppConfig) -> Self {
        let file_dir = file
            .parent()
            .map(Path::to_path_buf)
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("."));
        // The frontmatter's scope is a concrete value (default `folder`), so
        // an explicit `scope: vault` wins; otherwise the profile's default
        // applies (the format itself treats `folder` as the absent value).
        let scope = if fm.scope == parser::Scope::Vault
            || config
                .active_profile()
                .and_then(|p| p.scope.as_deref())
                .is_some_and(|s| s == "vault")
        {
            parser::Scope::Vault
        } else {
            parser::Scope::Folder
        };
        let work_dir = match scope {
            parser::Scope::Folder => file_dir,
            parser::Scope::Vault => context::agent_chain(&file_dir)
                .last()
                .cloned()
                .unwrap_or(file_dir),
        };
        // Canonicalize so relative `work_dir`s (e.g. `.` when the file was
        // passed without a directory component) still prefix-match the
        // canonicalized paths produced by [`resolve_safe_path`].
        let work_dir = work_dir.canonicalize().unwrap_or(work_dir);
        Self {
            enabled: fm.tools,
            network: fm.network,
            allow_shell: fm.allow_shell,
            work_dir,
            allowed_tools: config.active_profile().and_then(|p| p.tools.clone()),
            sandbox: config.sandbox.clone(),
            max_tool_output_bytes: config
                .active_profile()
                .and_then(|p| p.max_tool_output_bytes)
                .unwrap_or(MAX_TOOL_OUTPUT),
            reasoning_history: config
                .active_profile()
                .and_then(|p| p.reasoning_history)
                .unwrap_or_default(),
        }
    }
}

/// Everything needed to run automatic compaction (§5.4 of `docs/format.md`):
/// the conversation file the checkpoint turns are persisted into, the
/// threshold settings from the active profile, and the request ingredients
/// (system prompt, compaction prompt override, model name).
#[derive(Debug, Clone)]
pub struct AutoCompact {
    /// The conversation file to persist the `compact: true` request turn and
    /// the summary turn into.
    pub file: PathBuf,
    /// The model's context window in tokens; `None` disables automatic
    /// compaction entirely.
    pub context_limit: Option<u64>,
    /// Percentage (`1..=100`) of `context_limit` at which to compact;
    /// `None` keeps the built-in default of 80.
    pub auto_compact_percent: Option<u8>,
    /// The hierarchical system prompt overlaid on the compaction request
    /// (see [`crate::context::build_system_prompt`]).
    pub system_prompt: String,
    /// Override for the `compact: true` request body (`[prompts] compact`),
    /// when set.
    pub compact_prompt: Option<String>,
    /// Model name recorded on the persisted summary turn.
    pub model: String,
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

/// The OpenAI `tools` array for the request body. Empty when tools are
/// disabled or no tool is permitted by the frontmatter/profile.
pub fn tool_schemas(config: &ToolConfig) -> Vec<Json> {
    if !config.enabled {
        return Vec::new();
    }
    let allowed: Vec<String> = config
        .allowed_tools
        .clone()
        .unwrap_or_else(|| DEFAULT_TOOLS.iter().map(|s| s.to_string()).collect());
    let mut v = Vec::new();
    for name in DEFAULT_TOOLS.iter().copied() {
        if !allowed.iter().any(|a| a == name) {
            continue;
        }
        match name {
            "list_files" => v.push(schema(
            "list_files",
            "List files recursively under a subdirectory of the working directory, \
             skipping .git, target, and dotfiles. Returns relative paths with byte \
             sizes (max 200 entries). Use this before reading or editing files.",
            json!({
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Subdirectory to list (default \".\")"
                    }
                }
            }),
            &[],
            )),
            "read_file" => v.push(schema(
            "read_file",
            "Read a text file from the working directory. Returns numbered lines \
             in the form `N | text` plus a total-line-count note (defaults: first \
             200 lines).",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the working directory"},
                    "start_line": {"type": "integer", "description": "First line to read, 1-based (default 1)"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read (default 200)"}
                },
                "required": ["path"]
            }),
            &["path"],
            )),
            "write_file" => v.push(schema(
            "write_file",
            "Create or overwrite a file in the working directory with the given \
             content. Parent directories are created as needed.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the working directory"},
                    "content": {"type": "string", "description": "Full file content"}
                },
                "required": ["path", "content"]
            }),
            &["path", "content"],
            )),
            "edit_file" => v.push(schema(
            "edit_file",
            "Replace one exact occurrence of `old_text` with `new_text` in a file. \
             Fails when `old_text` is missing or matches more than once — include \
             surrounding context to make it unique.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the working directory"},
                    "old_text": {"type": "string", "description": "Exact text to find (must be unique in the file)"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_text", "new_text"]
            }),
            &["path", "old_text", "new_text"],
            )),
            "search_files" => v.push(schema(
            "search_files",
            "Case-insensitive substring search over the text files under a \
             subdirectory of the working directory. Returns `path:line: text` \
             matches (max 50). Prefer this over reading whole files to locate code.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for"},
                    "subdir": {"type": "string", "description": "Subdirectory to search (default \".\")"}
                },
                "required": ["query"]
            }),
            &["query"],
            )),
            "fetch_url" if config.network => v.push(schema(
            "fetch_url",
            "Fetch an http(s) URL and return its content as plain text (HTML \
             stripped), truncated to 16 KB.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http(s) URL to fetch"}
                },
                "required": ["url"]
            }),
            &["url"],
            )),
            "bash" if config.allow_shell => v.push(schema(
            "bash",
            "Run a shell command in a sandboxed environment whose only writable \
             location is the working directory. Prefer read_file, write_file, \
             edit_file, and search_files for file work; use bash for everything \
             else (running tests or builds, git, scripts). Returns the exit code \
             plus combined stdout/stderr.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout_secs": {"type": "integer", "description": "Timeout in seconds (default 60, max 300)"}
                },
                "required": ["command"]
            }),
            &["command"],
            )),
            _ => {}
        }
    }
    v
}

fn schema(name: &str, description: &str, parameters: Json, required: &[&str]) -> Json {
    let mut params = parameters;
    if !required.is_empty() {
        params["required"] = Json::Array(
            required
                .iter()
                .map(|s| Json::String((*s).to_string()))
                .collect(),
        );
    }
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        }
    })
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Execute one tool call by name with its JSON arguments.
///
/// The result is always returned as text suitable to feed back to the model:
/// failures are reported as `Error: ...` lines (truncated to a fixed byte
/// cap) rather than panics, so the model can see and react to them.
pub async fn execute(name: &str, arguments: &str, config: &ToolConfig) -> String {
    let args: Json = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(e) => return format!("Error: invalid JSON arguments: {e}"),
    };
    let out = match name {
        "list_files" => list_files(&args, config),
        "read_file" => read_file(&args, config),
        "write_file" => write_file(&args, config),
        "edit_file" => edit_file(&args, config),
        "search_files" => search_files(&args, config),
        "fetch_url" => fetch_url(&args, config).await,
        "bash" => bash(&args, config).await,
        other => format!("Error: unknown tool '{other}'"),
    };
    truncate_output(out, config.max_tool_output_bytes)
}

fn arg_str<'a>(args: &'a Json, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn arg_usize(args: &Json, key: &str) -> Option<usize> {
    args.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// Truncate `s` to `max_bytes` on a char boundary.
fn truncate_output(s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated]", &s[..end])
}

/// Recursively collect regular files (and symlinks) under `dir`, skipping
/// dotfiles, `.git`, and `target` directories. Symlinked directories are not
/// followed, which also prevents cycles.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => collect_files(&path, out),
            _ => out.push(path),
        }
    }
}

fn list_files(args: &Json, config: &ToolConfig) -> String {
    let subdir = arg_str(args, "subdir").unwrap_or(".");
    let root = match resolve_safe_path(&config.work_dir, subdir) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if !root.is_dir() {
        return format!("Error: not a directory: {subdir}");
    }
    let mut files = Vec::new();
    collect_files(&root, &mut files);
    files.sort();
    if files.is_empty() {
        return "(no files)".to_string();
    }
    let truncated = files.len() > LIST_CAP;
    files.truncate(LIST_CAP);
    let mut out = String::new();
    for path in &files {
        let rel = path
            .strip_prefix(&config.work_dir)
            .unwrap_or(path)
            .to_string_lossy();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        out.push_str(&format!("{rel}  {size}\n"));
    }
    if truncated {
        out.push_str("… (truncated at 200 entries)\n");
    }
    out
}

fn read_file(args: &Json, config: &ToolConfig) -> String {
    let Some(path_s) = arg_str(args, "path") else {
        return "Error: missing 'path'".to_string();
    };
    let start = arg_usize(args, "start_line").unwrap_or(1).max(1);
    let limit = arg_usize(args, "limit").unwrap_or(200).max(1);
    let path = match resolve_safe_path(&config.work_dir, path_s) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => return format!("Error: reading {path_s}: {e}"),
    };
    if bytes.contains(&0) {
        return "Error: binary file (not supported)".to_string();
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    if start > total {
        return format!("(file has {total} lines; start_line {start} is past the end)");
    }
    let end = (start - 1 + limit).min(total);
    let mut out = String::new();
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        out.push_str(&format!("{} | {line}\n", start + i));
    }
    out.push_str(&format!("({total} lines total, showing {start}-{end})\n"));
    out
}

fn write_file(args: &Json, config: &ToolConfig) -> String {
    let Some(path_s) = arg_str(args, "path") else {
        return "Error: missing 'path'".to_string();
    };
    let Some(content) = arg_str(args, "content") else {
        return "Error: missing 'content'".to_string();
    };
    let path = match resolve_safe_path(&config.work_dir, path_s) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return format!("Error: creating directories for {path_s}: {e}");
        }
    }
    match std::fs::write(&path, content) {
        Ok(()) => format!("Wrote {} bytes to {path_s}\n", content.len()),
        Err(e) => format!("Error: writing {path_s}: {e}"),
    }
}

fn edit_file(args: &Json, config: &ToolConfig) -> String {
    let Some(path_s) = arg_str(args, "path") else {
        return "Error: missing 'path'".to_string();
    };
    let Some(old) = arg_str(args, "old_text") else {
        return "Error: missing 'old_text'".to_string();
    };
    let Some(new) = arg_str(args, "new_text") else {
        return "Error: missing 'new_text'".to_string();
    };
    if old.is_empty() {
        return "Error: 'old_text' is empty".to_string();
    }
    let path = match resolve_safe_path(&config.work_dir, path_s) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return format!("Error: reading {path_s}: {e}"),
    };
    match text.matches(old).count() {
        0 => "Error: 'old_text' not found in file; re-read the file to see its exact content"
            .to_string(),
        1 => {
            let updated = text.replacen(old, new, 1);
            match std::fs::write(&path, &updated) {
                Ok(()) => format!("Edited {path_s} ({} bytes written)\n", updated.len()),
                Err(e) => format!("Error: writing {path_s}: {e}"),
            }
        }
        n => format!(
            "Error: 'old_text' matches {n} times in the file; include more surrounding context to make it unique"
        ),
    }
}

fn search_files(args: &Json, config: &ToolConfig) -> String {
    let Some(query) = arg_str(args, "query") else {
        return "Error: missing 'query'".to_string();
    };
    if query.is_empty() {
        return "Error: 'query' is empty".to_string();
    }
    let subdir = arg_str(args, "subdir").unwrap_or(".");
    let root = match resolve_safe_path(&config.work_dir, subdir) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if !root.is_dir() {
        return format!("Error: not a directory: {subdir}");
    }
    let q = query.to_lowercase();
    let mut files = Vec::new();
    collect_files(&root, &mut files);
    files.sort();
    let mut matches_out: Vec<String> = Vec::new();
    'outer: for path in &files {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        if bytes.contains(&0) {
            continue; // binary
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if line.to_lowercase().contains(&q) {
                let rel = path
                    .strip_prefix(&config.work_dir)
                    .unwrap_or(path)
                    .to_string_lossy();
                matches_out.push(format!("{rel}:{}: {}", i + 1, line.trim_end()));
                if matches_out.len() >= SEARCH_CAP {
                    break 'outer;
                }
            }
        }
    }
    if matches_out.is_empty() {
        return "No matches.".to_string();
    }
    let mut out = matches_out.join("\n");
    out.push('\n');
    if matches_out.len() == SEARCH_CAP {
        out.push_str("… (truncated at 50 matches)\n");
    }
    out
}

async fn fetch_url(args: &Json, config: &ToolConfig) -> String {
    if !config.network {
        return "Error: network access is disabled for this conversation (set `network: true` in the frontmatter)".to_string();
    }
    let Some(url) = arg_str(args, "url") else {
        return "Error: missing 'url'".to_string();
    };
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return "Error: only http(s) URLs are supported".to_string();
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("Error: building HTTP client: {e}"),
    };
    let resp = match client.get(url).send().await {
        Ok(r) => r,
        Err(e) => return format!("Error: request failed: {e}"),
    };
    if !resp.status().is_success() {
        return format!("Error: HTTP {}", resp.status());
    }
    let body = match resp.text().await {
        Ok(t) => t,
        Err(e) => return format!("Error: reading response body: {e}"),
    };
    strip_html(&body)
}

/// HTML block-level tag prefixes that become line breaks when stripped.
const BLOCK_TAGS: &[&str] = &[
    "<p",
    "</p",
    "<div",
    "</div",
    "<br",
    "<li",
    "</li",
    "<ul",
    "</ul",
    "<ol",
    "</ol",
    "<h1",
    "</h1",
    "<h2",
    "</h2",
    "<h3",
    "</h3",
    "<h4",
    "</h4",
    "<h5",
    "</h5",
    "<h6",
    "</h6",
    "<tr",
    "</tr",
    "<table",
    "</table",
    "<pre",
    "</pre",
    "<blockquote",
    "</blockquote",
    "<hr",
    "<section",
    "</section",
    "<article",
    "</article",
    "<header",
    "</header",
    "<footer",
    "</footer",
    "<main",
    "</main",
    "<nav",
    "</nav",
    "<form",
    "</form",
];

/// Strip HTML down to readable text: drop `<script>`/`<style>` blocks
/// entirely, turn block-level tags into line breaks, remove all other tags,
/// decode common entities, and collapse runs of blank lines.
fn strip_html(html: &str) -> String {
    let chars: Vec<char> = html.chars().collect();
    let mut out = String::with_capacity(html.len() / 2);
    let mut skip_depth = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '<' {
            if let Some(end_rel) = chars[i..].iter().position(|&c| c == '>') {
                let tag: String = chars[i..i + end_rel + 1].iter().collect();
                let lower = tag.to_ascii_lowercase();
                let is_open = lower.starts_with("<script") || lower.starts_with("<style");
                let is_close = lower.starts_with("</script") || lower.starts_with("</style");
                if skip_depth > 0 {
                    if is_close {
                        skip_depth -= 1;
                    }
                } else if is_open {
                    skip_depth += 1;
                } else if BLOCK_TAGS.iter().any(|p| lower.starts_with(p)) {
                    out.push('\n');
                }
                i += end_rel + 1;
                continue;
            }
        }
        if skip_depth == 0 {
            out.push(chars[i]);
        }
        i += 1;
    }
    let text = decode_entities(&out);
    // Trim trailing whitespace per line and collapse 3+ newlines to two.
    let mut collapsed = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.split('\n') {
        let line = line.trim_end();
        if line.is_empty() {
            blank_run += 1;
            if blank_run <= 2 {
                collapsed.push('\n');
            }
        } else {
            blank_run = 0;
            collapsed.push_str(line);
            collapsed.push('\n');
        }
    }
    collapsed.trim().to_string()
}

/// Decode the handful of entities that dominate real-world HTML. `&amp;` is
/// decoded last so double-encoded sequences decode by exactly one level.
fn decode_entities(s: &str) -> String {
    let mut out = s.replace("&lt;", "<");
    out = out.replace("&gt;", ">");
    out = out.replace("&quot;", "\"");
    out = out.replace("&#39;", "'");
    out = out.replace("&apos;", "'");
    out = out.replace("&nbsp;", " ");
    out.replace("&amp;", "&")
}

async fn bash(args: &Json, config: &ToolConfig) -> String {
    if !config.allow_shell {
        return "Error: the bash tool is disabled for this conversation (set `allow_shell: true` in the frontmatter)".to_string();
    }
    let Some(command) = arg_str(args, "command") else {
        return "Error: missing 'command'".to_string();
    };
    if command.trim().is_empty() {
        return "Error: 'command' is empty".to_string();
    }
    let timeout_secs = arg_usize(args, "timeout_secs").unwrap_or(60).clamp(1, 300);
    match sandbox::run_sandboxed(
        &config.sandbox,
        &config.work_dir,
        config.network,
        Duration::from_secs(timeout_secs as u64),
        config.max_tool_output_bytes,
        "/bin/sh",
        &["-c", command],
    )
    .await
    {
        Ok(out) => {
            let code = out
                .exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "killed".to_string());
            let body = if out.output.trim().is_empty() {
                "(no output)".to_string()
            } else {
                out.output
            };
            format!("exit code: {code}\n{body}")
        }
        Err(e) => format!("Error: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

/// Build the persisted form of a tool result turn (`**Tool**` with `tool:`
/// and `id:` meta tags, `<details><summary>Output</summary>` body). The
/// call's name/id/arguments live on the issuing assistant turn, not here.
pub fn tool_turn(name: &str, id: &str, output: &str) -> parser::Turn {
    parser::Turn {
        role: parser::Role::Tool,
        content: parser::tool_turn_body(output),
        meta: parser::TurnMeta {
            timestamp: Some(parser::now_timestamp()),
            tags: vec![
                ("tool".to_string(), name.to_string()),
                ("id".to_string(), id.to_string()),
            ],
            compact: false,
            length_limit_reached: false,
            content_filtered: false,
            tokens: None,
        },
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

/// Build the persisted form of an assistant response: the visible answer,
/// plus — when there is reasoning — a `thinking: true` header tag and the
/// reasoning carried in [`parser::Turn::reasoning`] (serialized as a
/// collapsible `<details><summary>Thinking</summary>` block), — when tool
/// calls were issued — a `tool_calls: <N>` header tag and the calls carried
/// in [`parser::Turn::tool_calls`] (serialized as trailing Tool Call
/// blocks), — when set — the `length_limit_reached: true` /
/// `content_filtered: true` header tags recording how the response ended
/// (§2.5 of `docs/format.md`), and — when the server reported a positive
/// token count — the `tokens: <n>` header tag (rendered after every other
/// tag, §2.2 of `docs/format.md`).
pub fn assistant_turn(
    model: &str,
    reasoning: &str,
    content: &str,
    tool_calls: &[CompletedToolCall],
    length_limit_reached: bool,
    content_filtered: bool,
    tokens: Option<u64>,
) -> parser::Turn {
    let has_reasoning = !reasoning.trim().is_empty();
    let mut tags = vec![("model".to_string(), model.to_string())];
    if has_reasoning {
        tags.push(("thinking".to_string(), "true".to_string()));
    }
    if !tool_calls.is_empty() {
        tags.push(("tool_calls".to_string(), tool_calls.len().to_string()));
    }
    parser::Turn {
        role: parser::Role::Assistant,
        content: content.trim_end().to_string(),
        meta: parser::TurnMeta {
            timestamp: Some(parser::now_timestamp()),
            tags,
            compact: false,
            length_limit_reached,
            content_filtered,
            tokens: tokens.filter(|&n| n > 0),
        },
        reasoning: has_reasoning.then(|| reasoning.trim().to_string()),
        tool_calls: tool_calls
            .iter()
            .map(|c| parser::AssistantToolCall {
                id: c.id.clone(),
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            })
            .collect(),
    }
}

/// Append one formatted turn (boundary line plus content) to a conversation
/// file, matching what [`parser::to_markdown`] would emit for that turn.
pub fn append_turn(path: &Path, turn: &parser::Turn) -> Result<()> {
    let needs_newline = std::fs::read_to_string(path)
        .map(|c| !c.ends_with('\n'))
        .unwrap_or(false);
    let mut s = String::new();
    if needs_newline {
        s.push('\n');
    }
    s.push_str("\n---\n");
    s.push_str(&format!("**{}**", turn.role.title()));
    if let Some(m) = turn.meta.render() {
        s.push_str(&format!(" | {m}"));
    }
    s.push_str("\n\n");
    s.push_str(&parser::render_turn_body(turn));
    s.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {} for appending", path.display()))?;
    use std::io::Write;
    f.write_all(s.as_bytes()).context("failed to write turn")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The multi-turn loop
// ---------------------------------------------------------------------------

/// One event emitted by [`run_turn`], in stream order.
#[derive(Debug, Clone)]
pub enum LoopEvent {
    /// Reasoning (thinking) text delta.
    Reasoning(String),
    /// Answer text delta.
    Content(String),
    /// The current assistant response has ended — either before tool
    /// execution or as the final response of the turn.
    ResponseEnd {
        /// The calls the response issued (ids normalized), so the caller
        /// can persist them on the assistant turn even when execution is
        /// interrupted before every result arrives.
        tool_calls: Vec<CompletedToolCall>,
        /// How the response ended, mapped from the server's
        /// `finish_reason` (§2.5 of `docs/format.md`). When it is
        /// [`FinishReason::Length`], the calls are incomplete and were
        /// **not** executed; the caller persists them on the turn with the
        /// `length_limit_reached` flag.
        finish: FinishReason,
    },
    /// A tool call has been executed; the fields describe the persisted
    /// result.
    ToolResult {
        /// The tool call id.
        id: String,
        /// The tool name.
        name: String,
        /// The JSON arguments exactly as the model sent them.
        arguments: String,
        /// The tool's output (or `Error: ...` text on failure).
        output: String,
    },
    /// A response's token usage arrived (the final chunk's `usage`,
    /// §2.2/§5.4 of `docs/format.md`); its `total_tokens` feeds the context
    /// meter and the automatic-compaction check.
    Usage(TokenUsage),
    /// Automatic compaction has started (§5.4): the `compact: true` request
    /// turn is persisted and the model's handoff summary is streaming.
    CompactionStart,
    /// Automatic compaction finished: both checkpoint turns are persisted
    /// and the message list was rebuilt from the new window. `tokens` is
    /// the summary response's total usage when reported (the new context
    /// size), else `None`.
    CompactionDone {
        /// The summary response's total token usage, when reported.
        tokens: Option<u64>,
    },
}

/// Run one automatic compaction (§5.4 of `docs/format.md`): re-read
/// `ac.file`, append a `**User** | compact: true` request turn, stream the
/// model's handoff summary (no tool loop), and append it as an
/// `**Assistant**` turn carrying its streamed usage as the `tokens:` count —
/// persisting both checkpoint turns. Emits [`LoopEvent::CompactionStart`]
/// before streaming and forwards the summary's deltas (reasoning, content,
/// usage) through `on_event`; a summary with no text and no reasoning is not
/// persisted (the file keeps its well-formed trailing request). Returns the
/// conversation re-parsed from the file, ready for
/// [`build_api_messages`] to window at the new checkpoint.
pub async fn run_auto_compact(
    client: &Client,
    ac: &AutoCompact,
    config: &ToolConfig,
    mut on_event: impl FnMut(&LoopEvent) -> bool,
) -> Result<parser::Conversation> {
    let raw = std::fs::read_to_string(&ac.file)
        .with_context(|| format!("failed to read {}", ac.file.display()))?;
    let mut conv =
        parser::parse(&raw).with_context(|| format!("failed to parse {}", ac.file.display()))?;

    // The request turn is persisted before the model is called, so a failed
    // run leaves a well-formed file whose trailing unanswered request the
    // next run (or compact) simply answers.
    let request = crate::llm::compact_request_turn(ac.compact_prompt.as_deref());
    append_turn(&ac.file, &request)?;
    conv.turns.push(request);

    if !on_event(&LoopEvent::CompactionStart) {
        return Ok(conv);
    }
    let summary = crate::llm::run_compact(client, &conv, &ac.system_prompt, config, |ev| {
        // Forward the summary's streamed deltas to the outer handler; the
        // compaction runs no tool loop, so nothing else is emitted.
        match ev {
            LoopEvent::Reasoning(s) => on_event(&LoopEvent::Reasoning(s.clone())),
            LoopEvent::Content(s) => on_event(&LoopEvent::Content(s.clone())),
            LoopEvent::Usage(u) => on_event(&LoopEvent::Usage(*u)),
            _ => true,
        }
    })
    .await?;

    // Persist the summary only when the model produced something.
    let tokens = summary.usage.map(|u| u.total_tokens);
    if !(summary.content.trim().is_empty() && summary.reasoning.trim().is_empty()) {
        let finish = summary.finish();
        let turn = assistant_turn(
            &ac.model,
            &summary.reasoning,
            &summary.content,
            &[],
            finish.is_length(),
            finish.is_content_filter(),
            tokens,
        );
        append_turn(&ac.file, &turn)?;
        conv.turns.push(turn);
    }
    let _ = on_event(&LoopEvent::CompactionDone { tokens });
    Ok(conv)
}

/// Pre-turn automatic compaction (§5.4 of `docs/format.md`): when the
/// persisted `tokens:` count of the most recent `**Assistant**` turn inside
/// the window has reached the threshold, run [`run_auto_compact`] and return
/// the re-parsed conversation; otherwise return `None`.
///
/// When `conv` ends with an unanswered, non-`compact` `**User**` turn — the
/// prompt the caller is about to answer — it would land *before* the new
/// checkpoint and be dropped from the windowed request, so after compaction
/// it is moved to after the summary (the file is rewritten with the turn at
/// the end). The next request then asks the model to answer the prompt with
/// the handoff summary as context. Compaction runs *before* the move, so an
/// aborted run leaves the prompt in the file (ahead of the unanswered
/// `compact: true` request) instead of losing it.
pub async fn pre_turn_auto_compact(
    client: &Client,
    ac: &AutoCompact,
    config: &ToolConfig,
    conv: &parser::Conversation,
    mut on_event: impl FnMut(&LoopEvent) -> bool,
) -> Result<Option<parser::Conversation>> {
    let Some(used) = conv.last_assistant_tokens_in_window() else {
        return Ok(None);
    };
    if !should_auto_compact(used, ac.context_limit, ac.auto_compact_percent) {
        return Ok(None);
    }
    let pending = conv
        .turns
        .last()
        .filter(|t| t.role == parser::Role::User && !t.meta.compact)
        .cloned();
    // The compaction only appends turns, so the pending prompt — the last
    // turn of `conv` — keeps its original index, immediately before the new
    // checkpoint.
    let pending_pos = pending.as_ref().map(|_| conv.turns.len() - 1);
    let mut conv = run_auto_compact(client, ac, config, &mut on_event).await?;
    if let (Some(turn), Some(pos)) = (pending, pending_pos) {
        if conv.turns[pos] != turn {
            bail!(
                "internal error: pending user turn not found before the new checkpoint in {}",
                ac.file.display()
            );
        }
        let moved = conv.turns.remove(pos);
        conv.turns.push(moved);
        std::fs::write(&ac.file, parser::to_markdown(&conv))
            .with_context(|| format!("failed to write {}", ac.file.display()))?;
    }
    Ok(Some(conv))
}

/// Drive one user turn end-to-end: stream an assistant response, execute any
/// tool calls it issued (appending `role: "tool"` results to the message
/// list), and repeat until a response arrives with no tool calls.
///
/// Every streamed delta and executed tool is reported through `on_event` so
/// callers can render progress (TUI) or print the answer (CLI). Returning
/// `false` from the callback stops the loop early (e.g. the UI went away).
/// The function returns when the final response has been fully streamed; its
/// content is whatever `Content` events were emitted after the last
/// `ResponseEnd`.
///
/// A response that ends with `finish_reason: "length"` (§2.5 of
/// `docs/format.md`) is truncated: its `ResponseEnd` carries
/// [`FinishReason::Length`] and any calls it carried are **not** executed —
/// the caller persists them on the assistant turn with the
/// `length_limit_reached` flag, and [`crate::llm::complete_tool_calls`]
/// answers them synthetically on the next request.
///
/// When `auto_compact` is `Some` and the response that just issued tool
/// calls reported usage past the threshold (`should_auto_compact`), the loop
/// runs an automatic compaction ([`run_auto_compact`], §5.4) before the next
/// request and rebuilds `messages` from the new checkpoint window via
/// [`build_api_messages`], [`set_system_prompt`], and
/// [`complete_tool_calls`]. The caller must persist each `ResponseEnd` /
/// `ToolResult` into `auto_compact.file` as it arrives, since the
/// compaction re-reads the file from disk.
pub async fn run_turn(
    client: &Client,
    mut messages: Vec<Message>,
    config: &ToolConfig,
    auto_compact: Option<&AutoCompact>,
    mut on_event: impl FnMut(&LoopEvent) -> bool,
) -> Result<()> {
    let schemas = tool_schemas(config);
    for _ in 0..MAX_TOOL_ITERATIONS {
        let mut stream = client
            .stream_chat(&messages, (!schemas.is_empty()).then_some(&schemas))
            .await?;
        let mut turn = AssistantTurn::default();
        while let Some(delta) = stream.next().await {
            let delta = delta.context("stream error")?;
            let stop = match &delta {
                StreamDelta::Reasoning(s) => !on_event(&LoopEvent::Reasoning(s.clone())),
                StreamDelta::Content(s) => !on_event(&LoopEvent::Content(s.clone())),
                StreamDelta::Usage(u) => !on_event(&LoopEvent::Usage(*u)),
                _ => false,
            };
            turn.push(&delta);
            if stop {
                return Ok(());
            }
        }
        // Normalize the calls first: servers that omit the id would break
        // the follow-up request (and the persisted pairing), so generate
        // one. The normalized list rides on `ResponseEnd` so the caller
        // can persist it on the assistant turn even when execution is
        // interrupted before every result arrives.
        let calls: Vec<CompletedToolCall> = turn
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, c)| CompletedToolCall {
                id: if c.id.is_empty() {
                    format!("call_gen_{i}")
                } else {
                    c.id.clone()
                },
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            })
            .collect();
        let finish = turn.finish();
        if !on_event(&LoopEvent::ResponseEnd {
            tool_calls: calls.clone(),
            finish: finish.clone(),
        }) {
            return Ok(());
        }
        if calls.is_empty() {
            return Ok(());
        }
        // `finish_reason: "length"` means the response was cut off at the
        // output-token limit, mid-stream: any tool calls it carried (even
        // partially streamed ones) are incomplete and must not be executed.
        // The caller persists them on the assistant turn with the
        // `length_limit_reached` flag; `complete_tool_calls` answers them
        // synthetically on the next request.
        if finish.is_length() {
            return Ok(());
        }

        // Append the assistant message (with its tool calls) to the history,
        // then execute each call and append a `role: "tool"` result. The
        // response's reasoning rides along as `reasoning_content` so the
        // server can condition on its own thinking — unless the configured
        // history mode is `None` (the active turn is always in scope).
        let mut assistant = Message::new(
            "assistant",
            (!turn.content.trim().is_empty()).then(|| turn.content.clone()),
        );
        if config.reasoning_history != ReasoningHistory::None && !turn.reasoning.is_empty() {
            assistant.reasoning_content = Some(turn.reasoning.clone());
        }
        for c in &calls {
            assistant.tool_calls.push(ToolCall {
                id: c.id.clone(),
                call_type: "function".to_string(),
                function: ToolFunction {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                },
            });
        }
        messages.push(assistant);
        for c in &calls {
            let output = execute(&c.name, &c.arguments, config).await;
            if !on_event(&LoopEvent::ToolResult {
                id: c.id.clone(),
                name: c.name.clone(),
                arguments: c.arguments.clone(),
                output: output.clone(),
            }) {
                return Ok(());
            }
            messages.push(Message::tool_result(&c.id, output));
        }

        // Mid-loop automatic compaction (§5.4): when the response that just
        // issued these tool calls reported usage past the threshold, compact
        // before the next request. The caller has persisted the assistant
        // turn (on `ResponseEnd`) and every tool result (on `ToolResult`),
        // so the compaction — which re-reads the file — sees the complete
        // history; the rebuilt message list then replaces the one built
        // before the turn.
        if let Some(ac) = auto_compact {
            if let Some(usage) = turn.usage {
                if should_auto_compact(
                    usage.total_tokens,
                    ac.context_limit,
                    ac.auto_compact_percent,
                ) {
                    let conv = run_auto_compact(client, ac, config, &mut on_event).await?;
                    messages = build_api_messages(&conv, config.reasoning_history);
                    set_system_prompt(&mut messages, &ac.system_prompt);
                    complete_tool_calls(&mut messages);
                }
            }
        }
    }
    bail!("no final reply after {MAX_TOOL_ITERATIONS} tool iterations")
}

#[cfg(test)]
mod tests;
