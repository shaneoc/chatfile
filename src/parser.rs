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
//! **User** | 2026-11-22 14:30 +00:00
//!
//! Hello!
//!
//! ---
//! **Assistant** | 2026-11-22 14:31 +00:00 | model: local-model
//!
//! Hi there.
//! ```
//!
//! Sections are introduced by the role in bold, optionally followed by plain
//! (unitalicized) metadata segments separated by ` | `:
//! `**<role>** | YYYY-MM-DD HH:MM ±HH:MM | key: value | ...`. The first
//! metadata segment may be a bare timestamp with an explicit numeric UTC
//! offset; every later segment must be `key: value`.
//! All metadata is optional. The first section
//! must be `**System**` or `**User**`; the frontmatter's closing fence (or
//! the start of the file, when there is no frontmatter) serves as its
//! leading horizontal rule, so it carries no `---` of its own. Every later
//! section is preceded by a blank line and a `---` horizontal rule (the
//! blank line is mandatory on parse). A `**System**` section is only valid
//! as the first section: its content is combined with the
//! hierarchical `AGENTS.md` context into the single leading system message
//! and never appears in the conversation itself. When serializing, colliding
//! lines in unfenced message content — a `---` line directly before a line
//! starting with `**`, backslash-prefixed code-fence lines, and unclosed
//! opening fences — are escaped with the bijective `N -> 2N + 1` odd-backslash
//! rule so the round trip is lossless (see [`escape_boundaries`] and
//! `docs/format.md`).

use anyhow::{Context, Result, anyhow};
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
    /// Timestamp in `YYYY-MM-DD HH:MM ±HH:MM` form, if present (an
    /// ambiguous `-00:00` offset is normalized to `+00:00` on parse).
    pub timestamp: Option<String>,
    /// Arbitrary `key: value` tags (e.g. `("model", "local-model")`).
    pub tags: Vec<(String, String)>,
    /// Whether the header carried `compact: true` (a `**User**`-only tag
    /// marking a context-compaction request; see §2.2 of `docs/format.md`).
    /// Lifted out of [`TurnMeta::tags`] on parse so it has a stable,
    /// role-checked position in the rendered header.
    pub compact: bool,
    /// Whether the header carried `length_limit_reached: true` (an
    /// `**Assistant**`-only tag recording that the response was cut off at
    /// the model's output-token limit; see §2.5 of `docs/format.md`). Lifted
    /// out of [`TurnMeta::tags`] on parse like [`TurnMeta::compact`].
    pub length_limit_reached: bool,
    /// Whether the header carried `content_filtered: true` (an
    /// `**Assistant**`-only tag recording that the provider's content filter
    /// cut the response; see §2.5 of `docs/format.md`). Lifted out of
    /// [`TurnMeta::tags`] on parse like [`TurnMeta::compact`].
    pub content_filtered: bool,
}

impl TurnMeta {
    /// The metadata as it appears after the role in a header line — plain
    /// segments joined by ` | ` (timestamp first, then `compact: true`,
    /// `length_limit_reached: true`, and `content_filtered: true` when set,
    /// then the remaining `key: value` pairs) — or `None` when there is
    /// none.
    pub fn render(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ts) = &self.timestamp {
            parts.push(ts.clone());
        }
        if self.compact {
            parts.push("compact: true".to_string());
        }
        if self.length_limit_reached {
            parts.push("length_limit_reached: true".to_string());
        }
        if self.content_filtered {
            parts.push("content_filtered: true".to_string());
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

/// Deserialize an optional positive integer: `0` (like any non-integer or
/// negative value) is a type error, so the frontmatter rejects useless caps.
/// An explicit YAML null reads as unset.
fn opt_positive_u32<'de, D>(deserializer: D) -> std::result::Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<u32>::deserialize(deserializer)?;
    if v == Some(0) {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Unsigned(0),
            &"a positive integer",
        ));
    }
    Ok(v)
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
    /// Cap on the model's output per response, sent as `max_tokens` in the
    /// API request; `None` leaves the endpoint's default in effect. Must be
    /// a positive integer when present.
    #[serde(
        default,
        deserialize_with = "opt_positive_u32",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_output_tokens: Option<u32>,
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
            max_output_tokens: None,
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
            && self.max_output_tokens.is_none()
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
    /// The turn body (for assistant turns, the visible answer — the
    /// reasoning block is carried separately in [`Turn::reasoning`]).
    pub content: String,
    /// Optional timestamp and `key: value` tags from the boundary line.
    pub meta: TurnMeta,
    /// Assistant turns only: the model's reasoning, persisted in a
    /// `<details><summary>Thinking</summary>` block that requires the
    /// header tag `thinking: true`. `None` for every other turn, and for
    /// assistant turns without the tag (whose body is verbatim content).
    pub reasoning: Option<String>,
    /// Assistant turns only: the tool calls issued with this reply, parsed
    /// from the trailing Tool Call blocks (see [`Turn::tool_call_count`]).
    /// Empty for every other turn and for assistant turns without the
    /// `tool_calls:` header tag.
    pub tool_calls: Vec<AssistantToolCall>,
}

/// One tool call persisted on an `**Assistant**` turn (a trailing
/// `<details>` block after the visible content).
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantToolCall {
    /// The tool call id (as the server assigned it).
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The JSON arguments exactly as the model sent them.
    pub arguments: String,
}

impl Turn {
    /// The `tool_calls: <N>` header tag value, or `0` when the tag is
    /// absent (the parser enforces that the tag and [`Turn::tool_calls`]
    /// agree).
    pub fn tool_call_count(&self) -> usize {
        self.meta
            .tags
            .iter()
            .find(|(k, _)| k == "tool_calls")
            .and_then(|(_, v)| v.parse::<usize>().ok())
            .unwrap_or(0)
    }
}

/// Structured data of a persisted `**Tool**` turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolTurnData {
    /// The tool name (from the `tool:` meta tag).
    pub name: String,
    /// The tool call id (from the `id:` meta tag).
    pub id: String,
    /// The unwrapped tool output (the fence content).
    pub output: String,
}

/// Parse a `Role::Tool` turn's persisted `<details>` body into structured
/// data. Returns `None` when the turn is not a well-formed tool turn (the
/// name and id meta tags are required; the body must carry the canonical
/// output wrapper). [`parse`] rejects such turns with a line-numbered
/// error, so this predicate only fails for hand-constructed [`Turn`]s. The
/// arguments live on the issuing assistant turn, not here.
pub fn parse_tool_turn(turn: &Turn) -> Option<ToolTurnData> {
    if turn.role != Role::Tool {
        return None;
    }
    let name = turn.meta.tags.iter().find(|(k, _)| k == "tool")?.1.clone();
    let id = turn.meta.tags.iter().find(|(k, _)| k == "id")?.1.clone();

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

    Some(ToolTurnData { name, id, output })
}

/// Serialize a tool result into the canonical persisted body: a
/// `<details>` block whose summary is `Output` and whose output sits in a
/// tilde fence. The fence is at least four tildes long and strictly longer
/// than any tilde run inside the output, so embedded tildes cannot
/// terminate it early.
pub fn tool_turn_body(output: &str) -> String {
    let fence = "~".repeat(fence_len(output));
    format!("<details>\n<summary>Output</summary>\n\n{fence}text\n{output}\n{fence}\n\n</details>")
}

/// The tilde fence length for tool arguments or output:
/// `max(4, longest run + 1)`.
fn fence_len(text: &str) -> usize {
    let mut longest = 0usize;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with('~') {
            longest = longest.max(t.chars().take_while(|&c| c == '~').count());
        }
    }
    longest.saturating_add(1).max(4)
}

/// Serialize one tool call into its canonical trailing block: a
/// `<details>` wrapper whose summary names the call and whose arguments sit
/// in a tilde fence (see [`fence_len`]).
pub fn tool_call_block(call: &AssistantToolCall) -> String {
    let fence = "~".repeat(fence_len(&call.arguments));
    format!(
        "<details>\n<summary>Tool Call: <code>{}</code> (<code>{}</code>)</summary>\n\n{fence}json\n{}\n{fence}\n\n</details>",
        call.name, call.id, call.arguments
    )
}

/// Parse the `<summary>Tool Call: <code>{name}</code> (<code>{id}</code>)</summary>`
/// line into `(name, id)`. Both must be non-empty and contain no
/// whitespace.
fn parse_tool_call_summary(line: &str) -> Option<(String, String)> {
    // After stripping the fixed prefix and suffix the line is
    // `{name}</code> (<code>{id}</code>`.
    let inner = line
        .trim()
        .strip_prefix("<summary>Tool Call: <code>")?
        .strip_suffix(")</summary>")?;
    let close_code = inner.find("</code>")?;
    let name = inner[..close_code].to_string();
    if name.is_empty() || name.chars().any(|c| c.is_whitespace()) {
        return None;
    }
    let id = inner[close_code + "</code>".len()..]
        .strip_prefix(" (<code>")?
        .strip_suffix("</code>")?
        .to_string();
    if id.is_empty() || id.chars().any(|c| c.is_whitespace()) {
        return None;
    }
    Some((name, id))
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

/// Current local time formatted as `YYYY-MM-DD HH:MM ±HH:MM`, with the
/// system's numeric UTC offset (e.g. `-05:00`) recorded automatically.
pub fn now_timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%d %H:%M %:z")
        .to_string()
}

/// If `line` consists of `N` leading backslashes followed by `---` and
/// optional trailing whitespace, return `N`. Such a line directly before a
/// line starting with `**` collides with the turn-boundary pattern and is
/// handled with the bijective `N -> 2N + 1` odd-backslash escape.
fn hr_backslashes(line: &str) -> Option<usize> {
    let b = line.as_bytes();
    let mut n = 0;
    while n < b.len() && b[n] == b'\\' {
        n += 1;
    }
    match line[n..].strip_prefix("---") {
        Some(tail) if tail.chars().all(|c| c.is_whitespace()) => Some(n),
        _ => None,
    }
}

/// If `line` starts with `N >= 1` backslashes immediately followed by three
/// or more backticks or tildes, return `(N, fence char)`. Such a line
/// collides with the code-fence pattern and is handled with the bijective
/// `N -> 2N + 1` odd-backslash escape.
fn fence_backslashes(line: &str) -> Option<(usize, char)> {
    let b = line.as_bytes();
    let mut n = 0;
    while n < b.len() && b[n] == b'\\' {
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let rest = &line[n..];
    let c = *rest.as_bytes().first()?;
    if !matches!(c, b'`' | b'~') {
        return None;
    }
    let run = rest.bytes().take_while(|&x| x == c).count();
    (run >= 3).then_some((n, c as char))
}

/// Replace the first `k` leading backslashes of `line` with `(k - 1) / 2` —
/// the inverse of the `N -> 2N + 1` escape for odd `k` (`1 -> 0`, `3 -> 1`,
/// `5 -> 2`).
fn restore_backslashes(line: &str, k: usize) -> String {
    format!("{}{}", "\\".repeat((k - 1) / 2), &line[k..])
}

/// True when `s` begins with a `YYYY-MM-DD HH:MM` prefix — i.e. it is
/// attempting to be a timestamp. The strict form (`YYYY-MM-DD HH:MM ±HH:MM`)
/// is checked by [`validate_timestamp`]; matching this loose shape keeps
/// malformed attempts (a missing offset, a `Z` suffix, a timezone
/// abbreviation) out of the `key: value` fallback so they fail with a
/// descriptive timestamp error instead of parsing as a tag.
fn looks_like_timestamp(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 16
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b' '
        && b[13] == b':'
        && b[..16]
            .iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7 | 10 | 13) || c.is_ascii_digit())
}

/// Strictly validate a timestamp segment: exactly 23 characters of the form
/// `YYYY-MM-DD HH:MM ±HH:MM` — a valid calendar date and time, a single
/// ASCII space, a `'+'` or `'-'` sign, then offset hours in `00..=23` and
/// offset minutes in `00..=59`. Both `-00:00` and `+00:00` are accepted;
/// serialization prefers `+00:00` (see [`parse_header_line`]).
fn validate_timestamp(s: &str) -> Result<()> {
    let b = s.as_bytes();
    let bad = || {
        anyhow!(
            "invalid timestamp '{s}'; expected 'YYYY-MM-DD HH:MM ±HH:MM' \
             (a valid date/time, one space, then a numeric UTC offset like +05:30)"
        )
    };
    if b.len() != 23
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b' '
        || b[13] != b':'
        || b[16] != b' '
        || !(b[17] == b'+' || b[17] == b'-')
        || b[20] != b':'
    {
        return Err(bad());
    }
    let two = |i: usize| -> Result<u32> {
        if !b[i].is_ascii_digit() || !b[i + 1].is_ascii_digit() {
            return Err(bad());
        }
        Ok(u32::from(b[i] - b'0') * 10 + u32::from(b[i + 1] - b'0'))
    };
    let (year, month, day) = (two(0)?, two(5)?, two(8)?);
    let (hour, minute) = (two(11)?, two(14)?);
    let (off_hours, off_minutes) = (two(18)?, two(21)?);
    chrono::NaiveDate::from_ymd_opt(i32::try_from(year).unwrap(), month, day)
        .ok_or_else(|| anyhow!("invalid timestamp '{s}'; not a valid calendar date"))?;
    chrono::NaiveTime::from_hms_opt(hour, minute, 0).ok_or_else(|| {
        anyhow!("invalid timestamp '{s}'; hour must be 00..=23 and minute 00..=59")
    })?;
    if off_hours > 23 || off_minutes > 59 {
        return Err(anyhow!(
            "invalid timestamp '{s}'; offset hours must be 00..=23 and offset minutes 00..=59"
        ));
    }
    Ok(())
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
/// optional bare `YYYY-MM-DD HH:MM ±HH:MM` timestamp first, then `key: value`
/// pairs. A segment that begins with a `YYYY-MM-DD HH:MM` prefix is
/// validated strictly ([`validate_timestamp`]); an ambiguous `-00:00`
/// offset is normalized to `+00:00` so serialization never emits it.
///
/// Every deviation is a parse error reported with the 1-based line number
/// via [`line_error`]: a different separator (`·`, `-`, missing spaces),
/// bold/italic markers anywhere in the metadata, a pipe inside a segment, a
/// timestamp that is not the first segment or that fails strict validation
/// (missing offset, `Z` suffix, timezone abbreviation, out-of-range date,
/// time, or offset), or a non-timestamp segment without a `key: ` prefix.
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
    let role = Role::parse(&rest[..end])
        .ok_or_else(|| line_error(line_no, format!("unknown role in header: {}", &rest[..end])))?;
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
                validate_timestamp(seg).map_err(|e| line_error(line_no, e.to_string()))?;
                // Normalize the ambiguous `-00:00` to `+00:00` so a later
                // serialization never emits it.
                meta.timestamp = Some(
                    seg.strip_suffix("-00:00")
                        .map(|prefix| format!("{prefix}+00:00"))
                        .unwrap_or_else(|| seg.to_string()),
                );
            } else if let Some((k, v)) = seg.split_once(':') {
                let k = k.trim();
                if k.is_empty() {
                    return Err(line_error(
                        line_no,
                        "invalid header metadata; 'key: value' requires a non-empty key",
                    ));
                }
                if meta.tags.iter().any(|(seen, _)| seen == k) {
                    return Err(line_error(
                        line_no,
                        format!("duplicate metadata key '{k}' in header"),
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
    // The `thinking` tag is an assistant-only switch and must be exactly
    // the lowercase literal `true`.
    for (k, v) in &meta.tags {
        if k == "thinking" {
            if role != Role::Assistant {
                return Err(line_error(
                    line_no,
                    "'thinking' metadata is only allowed on **Assistant** headers",
                ));
            }
            if v != "true" {
                return Err(line_error(
                    line_no,
                    "'thinking' metadata must be the literal value 'true'",
                ));
            }
        }
    }
    // The `compact` tag is a **User**-only switch marking a context-
    // compaction request; it must be exactly the lowercase literal `true`.
    // When present it is lifted out of the free-form tags into
    // [`TurnMeta::compact`] (rendered right after the timestamp).
    for (k, v) in &meta.tags {
        if k == "compact" {
            if role != Role::User {
                return Err(line_error(
                    line_no,
                    "'compact' metadata is only allowed on **User** headers",
                ));
            }
            if v != "true" {
                return Err(line_error(
                    line_no,
                    "'compact' metadata must be the literal value 'true'",
                ));
            }
        }
    }
    if meta.tags.iter().any(|(k, _)| k == "compact") {
        meta.compact = true;
        meta.tags.retain(|(k, _)| k != "compact");
    }
    // The `length_limit_reached` and `content_filtered` tags are
    // assistant-only switches recording how the response ended (§2.5); each
    // must be exactly the lowercase literal `true`. When present they are
    // lifted out of the free-form tags into [`TurnMeta`] (rendered right
    // after the timestamp).
    for (k, v) in &meta.tags {
        if k == "length_limit_reached" || k == "content_filtered" {
            if role != Role::Assistant {
                return Err(line_error(
                    line_no,
                    format!("'{k}' metadata is only allowed on **Assistant** headers"),
                ));
            }
            if v != "true" {
                return Err(line_error(
                    line_no,
                    format!("'{k}' metadata must be the literal value 'true'"),
                ));
            }
        }
    }
    if meta.tags.iter().any(|(k, _)| k == "length_limit_reached") {
        meta.length_limit_reached = true;
        meta.tags.retain(|(k, _)| k != "length_limit_reached");
    }
    if meta.tags.iter().any(|(k, _)| k == "content_filtered") {
        meta.content_filtered = true;
        meta.tags.retain(|(k, _)| k != "content_filtered");
    }
    // The `tool_calls` tag is an assistant-only positive integer counting
    // the trailing Tool Call blocks in the body.
    for (k, v) in &meta.tags {
        if k == "tool_calls" {
            if role != Role::Assistant {
                return Err(line_error(
                    line_no,
                    "'tool_calls' metadata is only allowed on **Assistant** headers",
                ));
            }
            if v.parse::<usize>().map_or(true, |n| n == 0) {
                return Err(line_error(
                    line_no,
                    "'tool_calls' metadata must be a positive integer",
                ));
            }
        }
    }
    // The `tool` and `id` tags name a tool call and are only allowed on
    // **Tool** headers.
    for (k, _) in &meta.tags {
        if (k == "tool" || k == "id") && role != Role::Tool {
            return Err(line_error(
                line_no,
                format!("'{k}' metadata is only allowed on **Tool** headers"),
            ));
        }
    }
    // A **Tool** header must name the call: non-empty `tool` and `id` tags.
    if role == Role::Tool {
        for k in ["tool", "id"] {
            let ok = meta.tags.iter().any(|(seen, v)| seen == k && !v.is_empty());
            if !ok {
                return Err(line_error(
                    line_no,
                    format!("**Tool** header requires a non-empty '{k}:' metadata tag"),
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
                    let ok = if fence_char == '`' {
                        !info.contains('`')
                    } else {
                        true
                    };
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
    Err(line_error(
        open_line,
        "frontmatter block is not closed with a `---` line",
    ))
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

    // Sections are collected as raw line runs first; the `N -> 2N + 1`
    // boundary escapes are restored per section afterwards (reasoning
    // extraction must see the backslashes before they are stripped).
    let mut raw_turns: Vec<(Role, TurnMeta, Vec<String>, usize)> = Vec::new();
    let mut cur_role = first_role;
    let mut cur_meta = first_meta;
    let mut cur_lines: Vec<String> = Vec::new();
    let mut cur_start = 1usize; // body-line index of the first content line
    let mut fence = FenceTracker::default();
    fence.next_line(lines[0]); // keep the tracker in sync with the header line

    let mut i = 1usize;
    while i < lines.len() {
        // `true` = the line is inside a fenced code block. Lines inside a
        // fence are content, never boundaries or escape candidates.
        let in_fence = fence.next_line(lines[i]);
        if in_fence {
            cur_lines.push(lines[i].to_string());
            i += 1;
            continue;
        }
        // A `---` line (with backslash escapes) directly before a line that
        // starts with `**`: k == 0 is a boundary, odd k is escaped content
        // (restored), even k is a format error.
        if let Some(k) = hr_backslashes(lines[i]) {
            if i + 1 < lines.len() && lines[i + 1].starts_with("**") {
                if k > 0 && k % 2 == 0 {
                    return Err(line_error(
                        first_line + i,
                        "even number of backslashes before '---'; expected an odd count",
                    ));
                }
                if k > 0 {
                    // Escaped content: keep the raw line; the backslashes
                    // are restored when the section is finalized.
                    cur_lines.push(lines[i].to_string());
                    i += 1;
                    continue;
                }
                // k == 0: a real separator candidate (escaped content,
                // k > 0, is exempt). It must be preceded by a blank or
                // whitespace-only line.
                if !lines[i - 1].trim().is_empty() {
                    return Err(line_error(
                        first_line + i,
                        "turn separator '---' must be preceded by a blank line",
                    ));
                }
                // A real boundary only when the next line is a recognized
                // role header; anything else is content. A line that starts
                // a section must be a *valid* header; a malformed one is a
                // format error, not content.
                if is_role_header_line(lines[i + 1]) {
                    let (role, meta) = parse_header_line(lines[i + 1], first_line + i + 1)?;
                    if role == Role::System {
                        return Err(line_error(
                            first_line + i + 1,
                            "**System** is only allowed as the first section",
                        ));
                    }
                    fence.next_line(lines[i + 1]); // keep the tracker in sync
                    raw_turns.push((
                        cur_role,
                        cur_meta,
                        std::mem::take(&mut cur_lines),
                        cur_start,
                    ));
                    cur_role = role;
                    cur_meta = meta;
                    cur_start = i + 2;
                    i += 2;
                    continue;
                }
            }
            cur_lines.push(lines[i].to_string());
            i += 1;
            continue;
        }
        // A backslash-escaped code-fence line outside a fence: odd k is
        // escaped content (restored when the section is finalized), even k
        // is a format error.
        if let Some((k, _)) = fence_backslashes(lines[i]) {
            if k % 2 == 0 {
                return Err(line_error(
                    first_line + i,
                    "even number of backslashes before a code fence; expected an odd count",
                ));
            }
            cur_lines.push(lines[i].to_string());
            i += 1;
            continue;
        }
        cur_lines.push(lines[i].to_string());
        i += 1;
    }
    raw_turns.push((cur_role, cur_meta, cur_lines, cur_start));

    // Finalize each section: for assistant turns carrying `thinking: true`
    // split off the reasoning block, then — for assistant turns carrying
    // `tool_calls: N` — extract the trailing tool call blocks. Both run on
    // the raw lines, before boundary escapes are restored.
    let mut turns: Vec<Turn> = Vec::new();
    for (role, meta, raw_lines, start) in raw_turns {
        let first = first_line + start;
        let mut reasoning = None;
        let mut body_lines = raw_lines;
        let mut body_first = first;
        if role == Role::Assistant && meta.tags.iter().any(|(k, _)| k == "thinking") {
            let (r, lines, f) = finalize_thinking_turn(&body_lines, body_first)?;
            reasoning = Some(r);
            body_lines = lines;
            body_first = f;
        }
        if role == Role::Tool {
            // Strict validation (§3.3.2): the header tags are checked in
            // parse_header_line, the body shape here. The unwrapped output
            // is re-extracted by parse_tool_turn when building messages.
            finalize_tool_turn(&body_lines, body_first)?;
        }
        let mut content_lines = body_lines;
        let mut tool_calls: Vec<AssistantToolCall> = Vec::new();
        if role == Role::Assistant {
            if let Some(n) = meta
                .tags
                .iter()
                .find(|(k, _)| k == "tool_calls")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
            {
                let (calls, rest) = extract_tool_calls(&content_lines, body_first, n)?;
                tool_calls = calls;
                content_lines = rest;
            }
        }
        turns.push(Turn {
            role,
            content: restore_boundaries(&content_lines, body_first)?,
            meta,
            reasoning,
            tool_calls,
        });
    }

    conv.turns = turns;
    Ok(conv)
}

/// Escape turn-boundary and code-fence collisions in `content` so the next
/// parse reads every line back as content, using the bijective `N -> 2N + 1`
/// odd-backslash rule (parse restores `(k - 1) / 2` for odd `k`):
///
/// - Any unfenced line consisting of `N` leading backslashes followed by
///   `---` (and optional trailing whitespace) directly before a line that
///   starts with `**` is rewritten with `2N + 1` leading backslashes
///   (`0 -> 1`, `1 -> 3`, `2 -> 5`).
/// - Any unfenced line starting with `N >= 1` backslashes immediately
///   followed by three or more backticks or tildes is rewritten with
///   `2N + 1` leading backslashes.
/// - An opening code fence that is never closed before the end of the body
///   gets one backslash prepended to its opener (`0 -> 1`); no synthetic
///   closing fence is added.
///
/// Lines inside a fenced code block are left untouched, and for every input
/// `c` the round trip is lossless: parsing the escaped form restores `c`.
pub fn escape_boundaries(content: &str) -> String {
    let raw: Vec<&str> = content.split_inclusive('\n').collect();
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    let mut fence = FenceTracker::default();
    let mut open_fence_at: Option<usize> = None;
    for (idx, &line) in raw.iter().enumerate() {
        let (text, nl) = match line.ends_with('\n') {
            true => (&line[..line.len() - 1], "\n"),
            false => (line, ""),
        };
        let was_open = fence.char.is_some();
        let in_fence = fence.next_line(text);
        let now_open = fence.char.is_some();
        if in_fence {
            // Content inside a fenced code block: untouched.
            out.push(line.to_string());
            continue;
        }
        if !was_open && now_open {
            // This line opens a fence; remember it in case it is never
            // closed (the opener is escaped at the end).
            open_fence_at = Some(out.len());
            out.push(line.to_string());
            continue;
        }
        if was_open && !now_open {
            // This line closes the fence.
            open_fence_at = None;
            out.push(line.to_string());
            continue;
        }
        // A plain line outside any fence: apply the escape rules.
        let mut t = text.to_string();
        if let Some(k) = hr_backslashes(text) {
            let next_role = idx + 1 < raw.len() && raw[idx + 1].starts_with("**");
            if next_role {
                t = format!("{}{}", "\\".repeat(2 * k + 1), &text[k..]);
            }
        } else if let Some((k, _)) = fence_backslashes(text) {
            t = format!("{}{}", "\\".repeat(2 * k + 1), &text[k..]);
        }
        out.push(format!("{t}{nl}"));
    }
    if let Some(at) = open_fence_at {
        // Unclosed opening fence: prepend one backslash (`0 -> 1`).
        let mut s = out[at].clone();
        s.insert(0, '\\');
        out[at] = s;
    }
    out.concat()
}

// Reasoning-block helpers live below [`escape_boundaries`].

/// Restore the `N -> 2N + 1` odd-backslash boundary escapes in a run of
/// body lines — the inverse of [`escape_boundaries`]. A line of `k >= 1`
/// leading backslashes before `---` (directly above a line that begins
/// with `**`) or immediately before a code fence comes back with
/// `(k - 1) / 2` backslashes; an even count is a format error. Lines
/// inside fenced code blocks are verbatim, exactly as
/// [`escape_boundaries`] left them. The result is trimmed of surrounding
/// whitespace.
fn restore_boundaries(lines: &[String], first_line: usize) -> Result<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut fence = FenceTracker::default();
    for (i, line) in lines.iter().enumerate() {
        if fence.next_line(line) {
            // Content inside a fenced code block: verbatim.
            out.push(line.clone());
            continue;
        }
        if let Some(k) = hr_backslashes(line) {
            let next_role = i + 1 < lines.len() && lines[i + 1].starts_with("**");
            if next_role {
                if k % 2 == 0 {
                    return Err(line_error(
                        first_line + i,
                        "even number of backslashes before '---'; expected an odd count",
                    ));
                }
                if k > 0 {
                    out.push(restore_backslashes(line, k));
                    continue;
                }
            }
        } else if let Some((k, _)) = fence_backslashes(line) {
            if k % 2 == 0 {
                return Err(line_error(
                    first_line + i,
                    "even number of backslashes before a code fence; expected an odd count",
                ));
            }
            out.push(restore_backslashes(line, k));
            continue;
        }
        out.push(line.clone());
    }
    Ok(out.join("\n").trim().to_string())
}

/// Escape stray `<details>` / `</details>` tags in persisted reasoning so
/// the block a later parse extracts is exactly this text.
///
/// A single forward pass tracks fenced code blocks and inline backtick
/// spans; details tags are matched case-insensitively outside both. An
/// already-backslashed tag (with `k >= 1` leading backslashes) is rewritten
/// with `2 * k + 1` backslashes (`N -> 2N + 1`, no stack effect); an
/// unescaped opener is remembered; an unescaped closer pops the stack when
/// a matching opener exists (the nested pair stays untouched) or gets one
/// backslash prepended when it does not. Openers left on the stack at the
/// end (unmatched within the reasoning) also get one backslash prepended.
/// Edits are applied in descending byte-offset order; the mapping is the
/// exact inverse of the restoration performed by
/// [`finalize_thinking_turn`].
pub fn escape_details_tags(text: &str) -> String {
    let mut out = text.to_string();
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut open_stack: Vec<usize> = Vec::new();
    let mut fence = FenceTracker::default();
    let mut base = 0usize;
    for line in text.split_inclusive('\n') {
        let l = match line.strip_suffix('\n') {
            Some(l) => l,
            None => line,
        };
        let was_open = fence.char.is_some();
        let in_fence = fence.next_line(l);
        let now_open = fence.char.is_some();
        if in_fence || was_open != now_open {
            // Fenced content or a fence delimiter line: untouched.
            base += line.len();
            continue;
        }
        for (p, is_closer, k) in details_tags_in_line(l) {
            let abs = base + p;
            if k > 0 {
                // Rewrite the backslash run itself: N -> 2N + 1.
                edits.push((abs - k, k, "\\".repeat(2 * k + 1)));
            } else if is_closer {
                if open_stack.pop().is_none() {
                    edits.push((abs, 0, "\\".to_string()));
                }
            } else {
                open_stack.push(abs);
            }
        }
        base += line.len();
    }
    for &off in &open_stack {
        edits.push((off, 0, "\\".to_string()));
    }
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, len, rep) in edits {
        out.replace_range(start..start + len, &rep);
    }
    out
}

/// The byte ranges of inline backtick code spans in `line`: a run of N
/// backticks opens a span closed by the next run of exactly N backticks;
/// an unclosed run extends to the end of the line.
fn inline_code_spans(line: &str) -> Vec<(usize, usize)> {
    let b = line.as_bytes();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != b'`' {
            i += 1;
            continue;
        }
        let start = i;
        let run = b[i..].iter().take_while(|&&c| c == b'`').count();
        let mut j = i + run;
        let mut closed: Option<usize> = None;
        while j < b.len() {
            if b[j] == b'`' {
                let r2 = b[j..].iter().take_while(|&&c| c == b'`').count();
                if r2 == run {
                    closed = Some(j + r2);
                    break;
                }
                j += r2;
            } else {
                j += 1;
            }
        }
        match closed {
            Some(end) => {
                spans.push((start, end));
                i = end;
            }
            None => {
                spans.push((start, b.len()));
                break;
            }
        }
    }
    spans
}

/// The details tags in `line` outside inline code spans, as
/// `(byte offset of '<', is_closer, leading backslashes)`.
///
/// An opener is `<details` (no whitespace between `<` and `details`)
/// followed by `>` or by attributes before `>`; a closer is `</details`
/// followed only by optional whitespace before `>`. Matching is
/// case-insensitive; `< details>` and `</ details>` are not tags.
fn details_tags_in_line(line: &str) -> Vec<(usize, bool, usize)> {
    let spans = inline_code_spans(line);
    let in_span = |p: usize| spans.iter().any(|&(s, e)| p >= s && p < e);
    let lower = line.to_ascii_lowercase();
    let b = lower.as_bytes();
    let mut out: Vec<(usize, bool, usize)> = Vec::new();
    let mut i = 0usize;
    while i + 8 <= b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        if b[i..].starts_with(b"</details") && !in_span(i) {
            if let Some(gt) = line[i + 9..].find('>') {
                let between = &line[i + 9..i + 9 + gt];
                if between.bytes().all(|c| c.is_ascii_whitespace()) {
                    out.push((i, true, backslash_run(line, i)));
                    i += 9;
                    continue;
                }
            }
        } else if b[i..].starts_with(b"<details") && !in_span(i) {
            if let Some(gt) = line[i + 8..].find('>') {
                let between = &line[i + 8..i + 8 + gt];
                if between.is_empty()
                    || between
                        .bytes()
                        .next()
                        .is_some_and(|c| c.is_ascii_whitespace())
                {
                    out.push((i, false, backslash_run(line, i)));
                    i += 8;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// The number of consecutive backslashes ending immediately before byte
/// position `pos` in `line`.
fn backslash_run(line: &str, pos: usize) -> usize {
    let b = line.as_bytes();
    let mut k = 0usize;
    let mut j = pos;
    while j > 0 && b[j - 1] == b'\\' {
        j -= 1;
        k += 1;
    }
    k
}

/// Finalize an assistant section whose header carries `thinking: true`:
/// split the raw body lines into the reasoning block (restored) and the
/// visible answer (raw lines, still carrying boundary escapes), together
/// with the 1-based file line of the first answer line.
///
/// The trimmed body must begin with a bare `<details>` line immediately
/// followed by a bare `<summary>Thinking</summary>` line. The scan then
/// tracks fenced code blocks and inline backtick spans; an unescaped
/// `<details…>` opens a level, an unescaped `</details…>` closes one, and
/// the wrapper ends at the first closer that returns the depth to zero,
/// which must be a bare `</details>` line. Backslashes immediately before
/// a details tag must come in odd counts (the inverse of the `N -> 2N + 1`
/// escape applied by [`escape_details_tags`]); `(k - 1) / 2` are restored.
fn finalize_thinking_turn(
    raw_lines: &[String],
    first_line: usize,
) -> Result<(String, Vec<String>, usize)> {
    // The first non-blank line must be the bare opening tag.
    let mut s = 0usize;
    while s < raw_lines.len() && raw_lines[s].trim().is_empty() {
        s += 1;
    }
    if s >= raw_lines.len() || raw_lines[s].trim() != "<details>" {
        return Err(line_error(
            first_line + s.min(raw_lines.len().saturating_sub(1)),
            "expected a bare '<details>' line to open the reasoning block",
        ));
    }
    if s + 1 >= raw_lines.len() || raw_lines[s + 1].trim() != "<summary>Thinking</summary>" {
        return Err(line_error(
            first_line + (s + 1).min(raw_lines.len().saturating_sub(1)),
            "expected a bare '<summary>Thinking</summary>' line after the opening tag",
        ));
    }

    let mut fence = FenceTracker::default();
    let mut depth = 1usize;
    let mut reason_lines: Vec<String> = Vec::new();
    let mut i = s + 2;
    let mut close_at: Option<usize> = None;
    while i < raw_lines.len() {
        let line = &raw_lines[i];
        let was_open = fence.char.is_some();
        let in_fence = fence.next_line(line);
        let now_open = fence.char.is_some();
        if in_fence || was_open != now_open {
            // Fenced content or a fence delimiter line: part of the
            // reasoning, never a details tag.
            reason_lines.push(line.clone());
            i += 1;
            continue;
        }
        let mut edits: Vec<(usize, usize, String)> = Vec::new();
        let mut closes = false;
        for (p, is_closer, k) in details_tags_in_line(line) {
            if k >= 2 && k % 2 == 0 {
                return Err(line_error(
                    first_line + i,
                    "even number of backslashes before a <details> tag; expected an odd count",
                ));
            }
            if k > 0 {
                // Escaped tag: restore `(k - 1) / 2` backslashes; the tag
                // is inert and does not affect the depth.
                edits.push((p - k, k, "\\".repeat((k - 1) / 2)));
            } else if is_closer {
                depth -= 1;
                if depth == 0 {
                    if line.trim() != "</details>" {
                        return Err(line_error(
                            first_line + i,
                            "a closing '</details>' must stand alone on its own line",
                        ));
                    }
                    closes = true;
                    break;
                }
            } else {
                depth += 1;
            }
        }
        let mut t = line.clone();
        edits.sort_by(|a, b| b.0.cmp(&a.0));
        for (start, len, rep) in &edits {
            t.replace_range(*start..*start + *len, rep);
        }
        if closes {
            close_at = Some(i);
            break;
        }
        reason_lines.push(t);
        i += 1;
    }
    let Some(close_at) = close_at else {
        return Err(line_error(
            first_line + raw_lines.len().saturating_sub(1),
            "unclosed reasoning block; expected a closing '</details>' line",
        ));
    };

    // The reasoning piece is restored here (the wrapper lines separate the
    // pieces, so no boundary collision can straddle them); the answer piece
    // is returned as raw lines — tool-call extraction runs before boundary
    // restoration. `first_line + close_at + 1` is the 1-based file line of
    // the first answer line.
    let reasoning = restore_boundaries(&reason_lines, first_line + s + 2)?;
    Ok((
        reasoning,
        raw_lines[close_at + 1..].to_vec(),
        first_line + close_at + 1,
    ))
}

/// Extract the trailing `n` tool call blocks from the raw body lines of an
/// assistant turn whose header carries `tool_calls: {n}`. Returns the calls
/// in file order and the remaining lines (the visible content, still
/// carrying boundary escapes).
///
/// Extraction walks backward from the end of the section, so literal
/// examples of tool call blocks inside the visible content are left
/// untouched.
fn extract_tool_calls(
    lines: &[String],
    first_line: usize,
    n: usize,
) -> Result<(Vec<AssistantToolCall>, Vec<String>)> {
    let mut calls: Vec<AssistantToolCall> = Vec::new();
    let mut end = lines.len();
    for _ in 0..n {
        while end > 0 && lines[end - 1].trim().is_empty() {
            end -= 1;
        }
        if end == 0 {
            return Err(line_error(
                first_line + lines.len().saturating_sub(1),
                format!("expected {n} tool call blocks at the end of the section; found fewer"),
            ));
        }
        let (call, new_end) = extract_one_tool_call_block(&lines[..end], first_line)?;
        calls.push(call);
        end = new_end;
    }
    calls.reverse();
    Ok((calls, lines[..end].to_vec()))
}

/// Extract one well-formed tool call block ending at the last line of
/// `lines`. Returns the call and the index of the block's opening
/// `<details>` line (the remainder runs up to it).
fn extract_one_tool_call_block(
    lines: &[String],
    first_line: usize,
) -> Result<(AssistantToolCall, usize)> {
    let e = lines.len() - 1;
    if lines[e].trim() != "</details>" {
        return Err(line_error(
            first_line + e,
            "expected a closing '</details>' line to end the tool call block",
        ));
    }
    // Closing fence: the first non-blank line above the closer.
    let mut c = e;
    while c > 0 && lines[c - 1].trim().is_empty() {
        c -= 1;
    }
    if c == 0 {
        return Err(line_error(
            first_line + e,
            "malformed tool call block; expected a tilde-fenced arguments block",
        ));
    }
    c -= 1;
    let close = lines[c].trim();
    let close_len = close.chars().take_while(|&ch| ch == '~').count();
    if close_len < 3 || close.len() != close_len {
        return Err(line_error(
            first_line + e,
            "malformed tool call block; expected a tilde-fenced arguments block",
        ));
    }
    // Summary line: the first line above the closing fence (walking back)
    // that parses as a Tool Call summary.
    let mut s: Option<usize> = None;
    for i in (0..c).rev() {
        if parse_tool_call_summary(&lines[i]).is_some() {
            s = Some(i);
            break;
        }
    }
    let Some(s) = s else {
        return Err(line_error(
            first_line + e,
            "malformed tool call block; expected a Tool Call summary line",
        ));
    };
    let (name, id) = parse_tool_call_summary(&lines[s]).unwrap();
    // Opening fence: the first tilde fence below the summary. The
    // serializer makes it strictly longer than any embedded tilde run, so
    // an embedded fence line cannot be mistaken for it.
    let mut o: Option<usize> = None;
    for (i, line) in lines[s + 1..c].iter().enumerate() {
        let t = line.trim();
        let run = t.chars().take_while(|&ch| ch == '~').count();
        if run >= 3 && (t[run..].trim() == "json" || t[run..].trim().is_empty()) {
            o = Some(s + 1 + i);
            break;
        }
    }
    let Some(o) = o else {
        return Err(line_error(
            first_line + s,
            "malformed tool call block; expected a tilde-fenced arguments block",
        ));
    };
    let open_len = lines[o].trim().chars().take_while(|&ch| ch == '~').count();
    if open_len > close_len {
        return Err(line_error(
            first_line + c,
            "malformed tool call block; expected a tilde-fenced arguments block",
        ));
    }
    // Opening tag: the first non-blank line above the summary.
    let mut d = s;
    while d > 0 && lines[d - 1].trim().is_empty() {
        d -= 1;
    }
    if d == 0 || lines[d - 1].trim() != "<details>" {
        return Err(line_error(
            first_line + s,
            "malformed tool call block; expected an opening '<details>' line",
        ));
    }
    let arguments = lines[o + 1..c].join("\n").trim().to_string();
    Ok((
        AssistantToolCall {
            id,
            name,
            arguments,
        },
        d - 1,
    ))
}

/// Validate the persisted body of a `**Tool**` turn and return the unwrapped
/// output. The body must be the canonical `<details>` wrapper (§3.3.2): a
/// bare `<details>` line immediately followed by a bare
/// `<summary>Output</summary>` line, then a tilde-fenced output (info string
/// `text` or empty, run of at least three tildes), and finally a bare
/// `</details>` line; blank lines may separate the pieces.
fn finalize_tool_turn(lines: &[String], first_line: usize) -> Result<String> {
    let last_line = first_line + lines.len().saturating_sub(1);
    // 1. First non-blank line: a bare <details>.
    let mut d = 0;
    while d < lines.len() && lines[d].trim().is_empty() {
        d += 1;
    }
    if d == lines.len() || lines[d].trim() != "<details>" {
        return Err(line_error(
            last_line,
            "expected a bare '<details>' line to open the tool output block",
        ));
    }
    // 2. Next line: a bare <summary>Output</summary>.
    if d + 1 >= lines.len() || lines[d + 1].trim() != "<summary>Output</summary>" {
        return Err(line_error(
            first_line + (d + 1).min(lines.len() - 1),
            "expected a bare '<summary>Output</summary>' line after the opening tag",
        ));
    }
    let s = d + 1;
    // 3. Opening fence: the first non-blank line below the summary must be a
    // tilde fence (info string `text` or empty).
    let mut o: Option<usize> = None;
    for (i, line) in lines.iter().enumerate().skip(s + 1) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let run = t.chars().take_while(|&c| c == '~').count();
        if run >= 3 && (t[run..].trim() == "text" || t[run..].trim().is_empty()) {
            o = Some(i);
        } else {
            return Err(line_error(
                first_line + i,
                "expected a tilde-fenced output block in the tool turn body",
            ));
        }
        break;
    }
    let Some(o) = o else {
        return Err(line_error(
            last_line,
            "expected a tilde-fenced output block in the tool turn body",
        ));
    };
    let open_len = lines[o].trim().chars().take_while(|&c| c == '~').count();
    // 4. Closing fence: a bare tilde run at least as long as the opener.
    let mut c: Option<usize> = None;
    for (i, line) in lines.iter().enumerate().skip(o + 1) {
        let t = line.trim();
        let run = t.chars().take_while(|&ch| ch == '~').count();
        if run >= open_len && t.len() == run {
            c = Some(i);
            break;
        }
    }
    let Some(c) = c else {
        return Err(line_error(
            last_line,
            "unclosed tool output block; expected a closing tilde fence",
        ));
    };
    // 5. After the closing fence: only blank lines, then a final bare
    // </details> line.
    let mut e = lines.len();
    while e > 0 && lines[e - 1].trim().is_empty() {
        e -= 1;
    }
    if e == 0 || lines[e - 1].trim() != "</details>" {
        return Err(line_error(
            last_line,
            "expected a bare '</details>' line to close the tool output block",
        ));
    }
    for (i, line) in lines.iter().enumerate().take(e - 1).skip(c + 1) {
        if !line.trim().is_empty() {
            return Err(line_error(
                first_line + i,
                "expected a bare '</details>' line to close the tool output block",
            ));
        }
    }
    Ok(lines[o + 1..c].join("\n").trim().to_string())
}

/// Serialize a [`Conversation`] back to canonical Markdown.
///
/// The first section is emitted without a leading horizontal rule: the
/// frontmatter's closing fence (or the start of the file, when there is no
/// frontmatter) serves that visual purpose. Every later section is preceded
/// by a blank line and its own `---` (`\n\n---\n`). Content has any colliding
/// boundary patterns escaped
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
        if let Some(v) = &conv.frontmatter.max_output_tokens {
            mapping.insert(key("max_output_tokens"), serde_yaml::Value::from(*v));
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
        out.push_str(&render_turn_body(t));
        out.push('\n');
    }

    out
}

/// The persisted body of a turn: the visible content, or — for an
/// assistant turn with non-empty reasoning — the collapsible Thinking
/// block ahead of it, and — for an assistant turn with tool calls — one
/// trailing Tool Call block per call (see [`tool_call_block`]):
///
/// ```markdown
/// <details>
/// <summary>Thinking</summary>
///
/// {reasoning}
///
/// </details>
///
/// {answer}
///
/// <details>
/// <summary>Tool Call: <code>{name}</code> (<code>{id}</code>)</summary>
///
/// ~~~~json
/// {arguments}
/// ~~~~
///
/// </details>
/// ```
///
/// (each optional piece is omitted when empty). Boundary-colliding lines
/// in the reasoning and answer pieces are escaped (see
/// [`escape_boundaries`]); stray details tags inside the reasoning are
/// escaped by [`escape_details_tags`]. The tool call blocks are emitted
/// verbatim; their tilde fences are sized by [`fence_len`].
pub fn render_turn_body(t: &Turn) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(r) = t
        .reasoning
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty())
    {
        let escaped_r = escape_boundaries(&escape_details_tags(r));
        parts.push(format!(
            "<details>\n<summary>Thinking</summary>\n\n{escaped_r}\n\n</details>"
        ));
    }
    let answer = t.content.trim();
    if !answer.is_empty() {
        parts.push(escape_boundaries(answer));
    }
    for call in &t.tool_calls {
        parts.push(tool_call_block(call));
    }
    parts.join("\n\n")
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
    /// conversation itself. Assistant turns carry their parsed tool calls
    /// (the trailing Tool Call blocks of §3.3); `**Tool**` turns become
    /// `role: "tool"` messages carrying the `tool_call_id`. [`parse`]
    /// rejects malformed tool turns, so only hand-constructed ones are
    /// skipped defensively. This is a pure projection: it never synthesizes
    /// messages — see [`crate::llm::complete_tool_calls`] for the
    /// request-prep step that answers dangling tool calls.
    pub fn to_messages(&self) -> Vec<crate::llm::Message> {
        self.project_turns(0..self.turns.len(), crate::config::ReasoningHistory::None)
    }

    /// Project a range of sections into OpenAI wire-format messages with the
    /// same rules as [`Conversation::to_messages`] (System sections skipped,
    /// `**Tool**` turns answered, assistant tool calls attached). Used by
    /// [`crate::llm::build_api_messages`] to window the history.
    ///
    /// `reasoning` selects which intermediate assistant turns (those that
    /// issued tool calls) re-send their persisted reasoning as
    /// `reasoning_content` (see [`crate::config::ReasoningHistory`]).
    pub(crate) fn project_turns(
        &self,
        range: std::ops::Range<usize>,
        reasoning: crate::config::ReasoningHistory,
    ) -> Vec<crate::llm::Message> {
        use crate::llm::{Message, ToolCall, ToolFunction};

        // The active turn starts after the last user message; under
        // `ActiveTurn` only assistant turns from there on re-send reasoning.
        let last_user = self.turns.iter().rposition(|t| t.role == Role::User);
        let mut messages: Vec<Message> = Vec::new();
        for i in range {
            let turn = &self.turns[i];
            match turn.role {
                Role::System => {}
                Role::Tool => {
                    let Some(data) = parse_tool_turn(turn) else {
                        continue;
                    };
                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(data.output.clone()),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(data.id.clone()),
                        reasoning_content: None,
                    });
                }
                other => {
                    // Only assistant messages may omit `content` (they
                    // carried tool calls instead); other roles always send
                    // a string, even when empty.
                    let mut m = Message::new(
                        other.as_str(),
                        if turn.content.trim().is_empty() && other == Role::Assistant {
                            None
                        } else {
                            Some(turn.content.clone())
                        },
                    );
                    for c in &turn.tool_calls {
                        m.tool_calls.push(ToolCall {
                            id: c.id.clone(),
                            call_type: "function".into(),
                            function: ToolFunction {
                                name: c.name.clone(),
                                arguments: c.arguments.clone(),
                            },
                        });
                    }
                    // Intermediate assistant turns (those that issued tool
                    // calls) may re-send their persisted reasoning, subject
                    // to the configured history mode.
                    if other == Role::Assistant && !m.tool_calls.is_empty() {
                        let include = match reasoning {
                            crate::config::ReasoningHistory::All => true,
                            crate::config::ReasoningHistory::None => false,
                            crate::config::ReasoningHistory::ActiveTurn => {
                                last_user.is_none_or(|u| i > u)
                            }
                        };
                        if include && turn.reasoning.as_deref().is_some_and(|r| !r.is_empty()) {
                            m.reasoning_content = turn.reasoning.clone();
                        }
                    }
                    messages.push(m);
                }
            }
        }
        messages
    }
}

#[cfg(test)]
mod tests;
