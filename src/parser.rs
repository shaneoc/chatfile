//! Parsing and serialization of conversation Markdown files.
//!
//! File layout:
//! ```markdown
//! ---
//! model: local-model
//! endpoint: http://localhost:9000/v1/chat/completions
//! ---
//! **System**
//!
//! You are terse.
//!
//! ---
//! **User** | 2026-11-22 14:30
//!
//! Hello!
//!
//! ---
//! **Assistant** | 2026-11-22 14:31 | model: local-model
//!
//! Hi there.
//! ```
//!
//! Sections are introduced by the role in bold, optionally followed by plain
//! (unitalicized) metadata segments separated by ` | `:
//! `**<role>** | YYYY-MM-DD HH:MM | key: value | ...`. The first metadata
//! segment may be a bare timestamp; every later segment must be `key: value`.
//! All metadata is optional. The first section
//! must be `**System**` or `**User**`; the frontmatter's closing fence (or
//! the start of the file, when there is no frontmatter) serves as its
//! leading horizontal rule, so it carries no `---` of its own. Every later
//! section is preceded by a `---` horizontal rule. A `**System**` section is
//! only valid as the first section: its content is combined with the
//! hierarchical `AGENTS.md` context into the single leading system message
//! and never appears in the conversation itself. When serializing, any
//! occurrence of the boundary pattern in unfenced message content is escaped
//! (a backslash is added before the role's first asterisk) so it cannot be
//! mistaken for a real section boundary on the next parse.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Build a parse error pointing at a 1-based line number in the source file.
///
/// Every strict format check reports its failures through this helper so the
/// user gets an actionable, line-precise diagnostic (the file path is added
/// by the caller's context).
pub fn line_error(line: usize, msg: impl fmt::Display) -> anyhow::Error {
    anyhow!("line {line}: {msg}")
}

/// Roles recognized in turn boundaries (`**User**`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A `**System**` turn.
    System,
    /// A `**User**` turn.
    User,
    /// An `**Assistant**` turn.
    Assistant,
    /// A `**Tool**` turn (a persisted tool result).
    Tool,
}

impl Role {
    /// The lowercase wire form (`"user"`, ...), as sent to the LLM API.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// The title used in turn boundaries (`**User**`, ...).
    pub fn title(self) -> &'static str {
        match self {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::Tool => "Tool",
        }
    }

    /// Parse a role name, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata attached to a turn boundary line.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnMeta {
    /// Timestamp in `YYYY-MM-DD HH:MM` form, if present.
    pub timestamp: Option<String>,
    /// Arbitrary `key: value` tags (e.g. `("model", "local-model")`).
    pub tags: Vec<(String, String)>,
}

impl TurnMeta {
    /// The metadata as it appears after the role in a header line — plain
    /// segments joined by ` | ` (timestamp first, then `key: value` pairs) —
    /// or `None` when there is none.
    pub fn render(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ts) = &self.timestamp {
            parts.push(ts.clone());
        }
        for (k, v) in &self.tags {
            // Always `key: value` (a bare key would not re-parse); an empty
            // value leaves a trailing space, which the parser trims.
            parts.push(format!("{k}: {v}"));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" | "))
        }
    }
}

/// Which directory built-in tools operate in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// The conversation file's own directory (default).
    #[default]
    Folder,
    /// The vault root (nearest ancestor with a `.chatfile` marker or `.git`).
    Vault,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Scope::Folder => "folder",
            Scope::Vault => "vault",
        })
    }
}

fn default_true() -> bool {
    true
}
fn is_true(b: &bool) -> bool {
    *b
}
fn is_false(b: &bool) -> bool {
    !b
}
fn is_folder(s: &Scope) -> bool {
    matches!(s, Scope::Folder)
}

/// Deserialize a boolean that tolerates legacy frontmatter values: any
/// non-boolean YAML value (e.g. the old free-form `tools:` list) reads as
/// `true`, so pre-tool-calling files keep parsing.
fn lenient_bool<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match serde_yaml::Value::deserialize(deserializer)? {
        serde_yaml::Value::Bool(b) => b,
        _ => true,
    })
}

/// YAML frontmatter fields. Unknown keys are ignored on parse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frontmatter {
    /// The model name to request (e.g. `local-model`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The full `/v1/chat/completions` endpoint URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Whether built-in tools are enabled (default `true`).
    #[serde(
        default = "default_true",
        deserialize_with = "lenient_bool",
        skip_serializing_if = "is_true"
    )]
    pub tools: bool,
    /// Which directory built-in tools operate in (default `folder`).
    #[serde(default, skip_serializing_if = "is_folder")]
    pub scope: Scope,
    /// Allow network access (`fetch_url`; sandboxed bash with shared net).
    #[serde(default, skip_serializing_if = "is_false")]
    pub network: bool,
    /// Allow the `bash` tool (default `false`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_shell: bool,
}

impl Default for Frontmatter {
    fn default() -> Self {
        Self {
            model: None,
            endpoint: None,
            tools: true,
            scope: Scope::Folder,
            network: false,
            allow_shell: false,
        }
    }
}

impl Frontmatter {
    /// True when no field is set (tools defaults to enabled and is not
    /// counted).
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.endpoint.is_none()
            && self.scope == Scope::Folder
            && !self.network
            && !self.allow_shell
    }
}

/// A single conversation turn: a boundary line plus its content.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// The turn's role.
    pub role: Role,
    /// The turn body (for assistant turns, reasoning tags included).
    pub content: String,
    /// Optional timestamp and `key: value` tags from the boundary line.
    pub meta: TurnMeta,
}

/// Structured data of a persisted `**Tool**` turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolTurnData {
    /// The tool name (from the `tool:` meta tag).
    pub name: String,
    /// The tool call id (from the `id:` meta tag).
    pub id: String,
    /// The JSON arguments exactly as the model sent them (from the summary
    /// line), or an empty string when the summary is missing/malformed.
    pub arguments: String,
    /// The unwrapped tool output (the fence content).
    pub output: String,
}

/// Parse a `Role::Tool` turn's persisted `<details>` body into structured
/// data. Returns `None` when the turn is not a well-formed tool turn (the
/// name and id meta tags are required; the body must carry a summary line
/// and a tilde-fenced output block).
pub fn parse_tool_turn(turn: &Turn) -> Option<ToolTurnData> {
    if turn.role != Role::Tool {
        return None;
    }
    let name = turn.meta.tags.iter().find(|(k, _)| k == "tool")?.1.clone();
    let id = turn.meta.tags.iter().find(|(k, _)| k == "id")?.1.clone();

    // Summary line: `<summary><code>name(args)</code></summary>`
    let mut arguments = String::new();
    for line in turn.content.lines() {
        let line = line.trim();
        let Some(inner) = line
            .strip_prefix("<summary><code>")
            .and_then(|s| s.strip_suffix("</code></summary>"))
        else {
            continue;
        };
        if let Some(open) = inner.find('(') {
            let name_part = &inner[..open];
            if !name_part.is_empty() && !name_part.chars().any(|c| c.is_whitespace()) {
                // The arguments run to the final closing paren.
                if let Some(close) = inner.rfind(')') {
                    if close > open {
                        arguments = inner[open + 1..close].to_string();
                    }
                }
            }
        }
        break; // only the first summary line is considered
    }

    // Output block: a tilde fence (info `text`) opened after the summary,
    // closed by a bare tilde run at least as long.
    let lines: Vec<&str> = turn.content.lines().collect();
    let mut open_at: Option<usize> = None;
    let mut open_len = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let run = t.chars().take_while(|&c| c == '~').count();
        if run >= 3 && (t[run..].trim() == "text" || t[run..].trim().is_empty()) {
            open_at = Some(i);
            open_len = run;
            break;
        }
    }
    let open_at = open_at?;
    let mut close_at: Option<usize> = None;
    for (i, line) in lines.iter().enumerate().skip(open_at + 1) {
        let t = line.trim();
        let run = t.chars().take_while(|&c| c == '~').count();
        if run >= open_len && t.len() == run {
            close_at = Some(i);
            break;
        }
    }
    let close_at = close_at?;
    let output = lines[open_at + 1..close_at].join("\n").trim().to_string();

    Some(ToolTurnData {
        name,
        id,
        arguments,
        output,
    })
}

/// Serialize a tool result into the canonical persisted body: a
/// `<details>` block whose summary shows the call and whose output sits in
/// a tilde fence. The fence is at least four tildes long and strictly
/// longer than any tilde run inside the output, so embedded ` ``` ` (or
/// shorter `~~~`) cannot terminate it early.
pub fn tool_turn_body(name: &str, arguments: &str, output: &str) -> String {
    let fence = "~".repeat(fence_len(output));
    format!(
        "<details>\n<summary><code>{name}({arguments})</code></summary>\n\n{fence}text\n{output}\n{fence}\n\n</details>"
    )
}

/// The tilde fence length for a tool output: `max(4, longest run + 1)`.
fn fence_len(output: &str) -> usize {
    let mut longest = 0usize;
    for line in output.lines() {
        let t = line.trim_start();
        if t.starts_with('~') {
            longest = longest.max(t.chars().take_while(|&c| c == '~').count());
        }
    }
    longest.saturating_add(1).max(4)
}

/// A parsed conversation file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    /// The parsed YAML frontmatter (defaults when absent).
    pub frontmatter: Frontmatter,
    /// Whether the source file began with a YAML frontmatter block.
    pub had_frontmatter: bool,
    /// The sections in file order; the first is always `System` or `User`.
    pub turns: Vec<Turn>,
}

/// Current local time formatted as `YYYY-MM-DD HH:MM`.
pub fn now_timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M").to_string()
}

/// True for a horizontal-rule line: exactly `---` (whitespace tolerated).
fn is_hr_line(line: &str) -> bool {
    line.trim() == "---"
}

/// True when `s` looks like a `YYYY-MM-DD HH:MM` timestamp.
fn looks_like_timestamp(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 16
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b' '
        && b[13] == b':'
        && b.iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7 | 10 | 13) || c.is_ascii_digit())
}

/// True when the line begins with a bold recognized role (`**User**`, ...),
/// i.e. it is attempting to be a section header line. Used to decide whether
/// a `---` line introduces a new section (versus being plain content) and to
/// escape would-be boundaries inside message bodies.
fn is_role_header_line(line: &str) -> bool {
    let t = line.trim_start();
    let Some(rest) = t.strip_prefix("**") else {
        return false;
    };
    let Some(end) = rest.find("**") else {
        return false;
    };
    Role::parse(&rest[..end]).is_some()
}

/// Strictly parse a section header line: `**<role>**` optionally followed by
/// plain (unitalicized) metadata segments separated by exactly ` | ` — an
/// optional bare `YYYY-MM-DD HH:MM` timestamp first, then `key: value`
/// pairs.
///
/// Every deviation is a parse error reported with the 1-based line number
/// via [`line_error`]: a different separator (`·`, `-`, missing spaces),
/// bold/italic markers anywhere in the metadata, a pipe inside a segment, a
/// timestamp that is not the first segment, or a non-timestamp segment
/// without a `key: ` prefix.
fn parse_header_line(line: &str, line_no: usize) -> Result<(Role, TurnMeta)> {
    let t = line.trim();
    let bad_role = || {
        line_error(
            line_no,
            "invalid header; expected a bold role like **User**",
        )
    };
    let rest = t.strip_prefix("**").ok_or_else(bad_role)?;
    let end = rest.find("**").ok_or_else(bad_role)?;
    let role = Role::parse(&rest[..end]).ok_or_else(|| {
        line_error(line_no, format!("unknown role in header: {}", &rest[..end]))
    })?;
    let mut meta = TurnMeta::default();
    let after = &rest[end + 2..];
    if !after.trim().is_empty() {
        // The separator between the role and the first segment must be
        // exactly ` | `.
        if !after.starts_with(" | ") {
            return Err(line_error(
                line_no,
                "invalid header metadata separator; expected ' | '",
            ));
        }
        for seg in after[3..].trim_end().split(" | ") {
            if seg.contains('|') {
                return Err(line_error(
                    line_no,
                    "invalid header metadata separator; expected ' | '",
                ));
            }
            if seg.contains('*') {
                return Err(line_error(
                    line_no,
                    "header metadata must be plain text; bold/italic markers are not allowed",
                ));
            }
            if looks_like_timestamp(seg) {
                if meta.timestamp.is_some() {
                    return Err(line_error(line_no, "duplicate timestamp in header"));
                }
                if !meta.tags.is_empty() {
                    return Err(line_error(
                        line_no,
                        "timestamp must be the first metadata segment after the role",
                    ));
                }
                meta.timestamp = Some(seg.to_string());
            } else if let Some((k, v)) = seg.split_once(':') {
                let k = k.trim();
                if k.is_empty() {
                    return Err(line_error(
                        line_no,
                        "invalid header metadata; 'key: value' requires a non-empty key",
                    ));
                }
                meta.tags.push((k.to_string(), v.trim().to_string()));
            } else {
                return Err(line_error(
                    line_no,
                    "invalid header metadata; expected a timestamp or 'key: value'",
                ));
            }
        }
    }
    Ok((role, meta))
}

/// Tracks whether the scanner is inside a fenced code block (CommonMark subset).
#[derive(Debug, Clone, Copy, Default)]
struct FenceTracker {
    /// The fence character (`'`'` or `'~'`) while inside a fence.
    char: Option<char>,
    /// Length of the opening fence run.
    len: usize,
}

impl FenceTracker {
    /// Feed one line (without its trailing newline).
    ///
    /// Returns `true` when the line is content inside a fenced code block;
    /// `false` for fence delimiter lines and for lines outside any fence.
    fn next_line(&mut self, line: &str) -> bool {
        let indent = line.len() - line.trim_start_matches(' ').len();
        if indent > 3 {
            return self.char.is_some();
        }
        let rest = &line[indent..];
        let (fence_char, run) = match rest.chars().next() {
            Some(c @ ('`' | '~')) => {
                let n = rest.chars().take_while(|&c2| c2 == c).count();
                (c, n)
            }
            _ => return self.char.is_some(),
        };
        match self.char {
            None => {
                // Candidate opening fence. Backtick fences may not carry
                // backticks in their info string.
                if run >= 3 {
                    let info = &rest[run..];
                    let ok = if fence_char == '`' { !info.contains('`') } else { true };
                    if ok {
                        self.char = Some(fence_char);
                        self.len = run;
                        return false;
                    }
                }
                false
            }
            Some(open) => {
                // Candidate closing fence: same character, at least as long,
                // and nothing but whitespace after it.
                if fence_char == open && run >= self.len && rest[run..].trim().is_empty() {
                    self.char = None;
                    return false;
                }
                true
            }
        }
    }
}

/// Split off a leading `---`-fenced YAML block.
///
/// A file has frontmatter if and only if its first line is a fence (`---` or
/// `...`); the block ends at the next fence line, and a block that never
/// closes is a format error. Returns `(yaml, body, body_first_line)` where
/// `body_first_line` is the 1-based line number of the first body line.
fn split_frontmatter(input: &str) -> Result<(Option<&str>, &str, usize)> {
    let mut offset = 0usize;
    let mut yaml_start: Option<usize> = None;
    let mut open_line = 1usize;
    let mut line_no = 0usize;
    for line in input.split_inclusive('\n') {
        line_no += 1;
        let is_fence = line.trim_end() == "---" || line.trim_end() == "...";
        match yaml_start {
            None => {
                if !is_fence {
                    return Ok((None, input, 1));
                }
                open_line = line_no;
                offset += line.len();
                yaml_start = Some(offset);
            }
            Some(start) => {
                if is_fence {
                    let yaml = &input[start..offset].trim_end_matches('\n');
                    offset += line.len();
                    return Ok((Some(yaml), &input[offset..], line_no + 1));
                }
                offset += line.len();
            }
        }
    }
    Err(line_error(open_line, "frontmatter block is not closed with a `---` line"))
}

/// Parse conversation Markdown into a [`Conversation`].
///
/// The body (everything after the frontmatter, or the whole file when there
/// is none) must begin with a `**System**` or `**User**` header line; every
/// later section is introduced by a `---` horizontal rule. A `**System**`
/// section is only valid as the first section. Violations are reported with
/// a 1-based line number via [`line_error`].
pub fn parse(input: &str) -> Result<Conversation> {
    let (yaml, body, first_line) = split_frontmatter(input)?;
    let mut conv = Conversation::default();
    if let Some(yaml) = yaml {
        conv.had_frontmatter = true;
        if yaml.trim().is_empty() {
            conv.frontmatter = Frontmatter::default();
        } else {
            conv.frontmatter = serde_yaml::from_str::<Frontmatter>(yaml).with_context(|| {
                format!("invalid YAML frontmatter (lines 2..={})", first_line - 1)
            })?;
        }
    }

    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        return Err(line_error(
            first_line,
            "file has no sections; expected a **System** or **User** header",
        ));
    }

    // The first section carries no leading horizontal rule of its own.
    let bad_first = || {
        line_error(
            first_line,
            "first section must be a **System** or **User** header",
        )
    };
    if !is_role_header_line(lines[0]) {
        return Err(bad_first());
    }
    let (first_role, first_meta) = parse_header_line(lines[0], first_line)?;
    if !matches!(first_role, Role::System | Role::User) {
        return Err(bad_first());
    }

    let mut turns: Vec<Turn> = Vec::new();
    let mut cur_role = first_role;
    let mut cur_meta = first_meta;
    let mut cur_lines: Vec<&str> = Vec::new();
    let mut fence = FenceTracker::default();
    fence.next_line(lines[0]); // keep the tracker in sync with the header line

    let mut i = 1usize;
    while i < lines.len() {
        // `true` = the line is inside a fenced code block.
        let in_fence = fence.next_line(lines[i]);
        let boundary = !in_fence
            && is_hr_line(lines[i])
            && i + 1 < lines.len()
            && is_role_header_line(lines[i + 1]);
        if boundary {
            // A line that starts a section must be a *valid* header; a
            // malformed one is a format error, not content.
            let (role, meta) = parse_header_line(lines[i + 1], first_line + i + 1)?;
            if role == Role::System {
                return Err(line_error(
                    first_line + i + 1,
                    "**System** is only allowed as the first section",
                ));
            }
            fence.next_line(lines[i + 1]); // keep the tracker in sync
            turns.push(Turn {
                role: cur_role,
                meta: cur_meta,
                content: cur_lines.join("\n").trim().to_string(),
            });
            cur_role = role;
            cur_meta = meta;
            cur_lines.clear();
            i += 2;
        } else {
            cur_lines.push(lines[i]);
            i += 1;
        }
    }
    turns.push(Turn {
        role: cur_role,
        meta: cur_meta,
        content: cur_lines.join("\n").trim().to_string(),
    });

    conv.turns = turns;
    Ok(conv)
}

/// Escape any turn-boundary pattern in `content` that would be parsed as a
/// real boundary later: for every unfenced `---` line directly followed by a
/// role line, a backslash is inserted before the role's first asterisk
/// (`\**User**`). In rendered Markdown the backslash shows as a literal `*`,
/// and on the next parse the line no longer matches the boundary pattern.
/// Lines inside fenced code blocks are left untouched.
pub fn escape_boundaries(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut fence = FenceTracker::default();
    let mut prev_hr = false;
    for line in content.split_inclusive('\n') {
        let (text, nl) = match line.ends_with('\n') {
            true => (&line[..line.len() - 1], "\n"),
            false => (line, ""),
        };
        let in_fence = fence.next_line(text);
        let escaped = if !in_fence && prev_hr && is_role_header_line(text) {
            // Backslash before the role's first asterisk; both asterisks stay.
            let trimmed = text.trim_start();
            let prefix = &text[..text.len() - trimmed.len()];
            format!("{prefix}\\{trimmed}")
        } else {
            text.to_string()
        };
        out.push_str(&escaped);
        out.push_str(nl);
        prev_hr = !in_fence && is_hr_line(text);
    }
    out
}

/// The opening tag of a persisted reasoning block (line-anchored).
pub const THINK_OPEN: &str = "think";
/// The closing tag of a persisted reasoning block (line-anchored).
pub const THINK_CLOSE: &str = "endthink";

/// Split turn content into `(reasoning, answer)` at line-anchored
/// `think` / `endthink` tags.
///
/// Returns `(None, content)` when no complete tag pair is present, so
/// content that merely mentions the words is left untouched. Only the
/// first pair is treated as reasoning; any text before it is prepended to
/// the answer.
pub fn split_reasoning(content: &str) -> (Option<String>, String) {
    let lines: Vec<&str> = content.lines().collect();
    let Some(open) = lines.iter().position(|l| l.trim() == THINK_OPEN) else {
        return (None, content.to_string());
    };
    let Some(rel_close) = lines[open + 1..]
        .iter()
        .position(|l| l.trim() == THINK_CLOSE)
    else {
        return (None, content.to_string());
    };
    let close = open + 1 + rel_close;
    let reasoning = lines[open + 1..close].join("\n").trim().to_string();
    let before = lines[..open].join("\n").trim().to_string();
    let after = lines[close + 1..].join("\n").trim().to_string();
    let answer = match (before.is_empty(), after.is_empty()) {
        (true, true) => String::new(),
        (true, false) => after,
        (false, true) => before,
        (false, false) => format!("{before}\n\n{after}"),
    };
    (Some(reasoning), answer)
}

/// Format reasoning plus answer into the canonical persisted form:
/// `think` / `endthink` tags around the reasoning, a blank line, then the
/// answer. With no reasoning, just the (trimmed) answer.
pub fn with_reasoning(reasoning: Option<&str>, answer: &str) -> String {
    let answer = answer.trim();
    match reasoning.map(str::trim).filter(|r| !r.is_empty()) {
        Some(r) => format!("{THINK_OPEN}\n{r}\n{THINK_CLOSE}\n\n{answer}"),
        None => answer.to_string(),
    }
}

/// Serialize a [`Conversation`] back to canonical Markdown.
///
/// The first section is emitted without a leading horizontal rule: the
/// frontmatter's closing fence (or the start of the file, when there is no
/// frontmatter) serves that visual purpose. Every later section is preceded
/// by its own `---`. Content has any colliding boundary patterns escaped
/// (see [`escape_boundaries`]). The output always ends with a single
/// trailing newline, and `parse(to_markdown(c)) == c` for any `c` produced
/// by `parse`.
pub fn to_markdown(conv: &Conversation) -> String {
    let mut out = String::new();

    if conv.had_frontmatter {
        out.push_str("---\n");
        let mut mapping = serde_yaml::Mapping::new();
        if let Some(v) = &conv.frontmatter.model {
            mapping.insert(key("model"), serde_yaml::Value::String(v.clone()));
        }
        if let Some(v) = &conv.frontmatter.endpoint {
            mapping.insert(key("endpoint"), serde_yaml::Value::String(v.clone()));
        }
        // Booleans/enums are written only when they differ from the
        // defaults, keeping files clean.
        if !conv.frontmatter.tools {
            mapping.insert(key("tools"), serde_yaml::Value::Bool(false));
        }
        if conv.frontmatter.scope != Scope::Folder {
            mapping.insert(
                key("scope"),
                serde_yaml::Value::String(conv.frontmatter.scope.to_string()),
            );
        }
        if conv.frontmatter.network {
            mapping.insert(key("network"), serde_yaml::Value::Bool(true));
        }
        if conv.frontmatter.allow_shell {
            mapping.insert(key("allow_shell"), serde_yaml::Value::Bool(true));
        }
        if mapping.is_empty() {
            out.push_str("---\n"); // empty frontmatter block
        } else {
            let yaml = serde_yaml::to_string(&serde_yaml::Value::Mapping(mapping))
                .expect("frontmatter serialization cannot fail");
            out.push_str(yaml.trim_end());
            out.push_str("\n---\n");
        }
    }

    for (i, t) in conv.turns.iter().enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str("---\n");
        }
        out.push_str(&format!("**{}**", t.role.title()));
        if let Some(meta) = t.meta.render() {
            out.push_str(&format!(" | {meta}"));
        }
        out.push_str("\n\n");
        out.push_str(&escape_boundaries(t.content.trim()));
        out.push('\n');
    }

    out
}

fn key(s: &str) -> serde_yaml::Value {
    serde_yaml::Value::String(s.to_string())
}

impl Conversation {
    /// Parse then re-serialize (canonical form).
    pub fn canonicalize(input: &str) -> Result<String> {
        Ok(to_markdown(&parse(input)?))
    }

    /// The sections as OpenAI wire-format messages.
    ///
    /// `**System**` sections are skipped: their content is folded into the
    /// single leading system message by the caller (see
    /// [`crate::context::build_system_prompt`]) and never appears in the
    /// conversation itself. `**Tool**` turns become `role: "tool"` messages
    /// carrying the `tool_call_id`, and the `tool_calls` (id, name,
    /// arguments) are attached to the preceding assistant message — a
    /// message with `content: None` is synthesized when a tool turn has no
    /// preceding assistant turn (e.g. a hand-edited file), so reloaded
    /// histories are always valid API payloads. Malformed tool turns are
    /// skipped.
    pub fn to_messages(&self) -> Vec<crate::llm::Message> {
        use crate::llm::{Message, ToolCall, ToolFunction};

        let mut messages: Vec<Message> = Vec::new();
        for turn in &self.turns {
            match turn.role {
                Role::System => {}
                Role::Tool => {
                    let Some(data) = parse_tool_turn(turn) else {
                        continue;
                    };
                    let call = ToolCall {
                        id: data.id.clone(),
                        call_type: "function".into(),
                        function: ToolFunction {
                            name: data.name.clone(),
                            arguments: data.arguments.clone(),
                        },
                    };
                    match messages.last_mut() {
                        Some(last) if last.role == "assistant" => last.tool_calls.push(call),
                        _ => {
                            let mut m = Message::new("assistant", None);
                            m.tool_calls.push(call);
                            messages.push(m);
                        }
                    }
                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(data.output.clone()),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(data.id.clone()),
                    });
                }
                other => {
                    // Only assistant messages may omit `content` (they
                    // carried tool calls instead); other roles always send
                    // a string, even when empty.
                    let content = if other == Role::Assistant && turn.content.trim().is_empty() {
                        None
                    } else {
                        Some(turn.content.clone())
                    };
                    messages.push(Message::new(other.as_str(), content));
                }
            }
        }
        messages
    }
}

#[cfg(test)]
mod tests;
