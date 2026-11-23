//! Parsing and serialization of conversation Markdown files.
//!
//! File layout:
//! ```markdown
//! ---
//! model: local-model
//! endpoint: http://localhost:9000/v1/chat/completions
//! system_prompt: You are terse.
//! ---
//!
//! Optional preamble text before the first turn boundary.
//!
//! ---
//! **User** · *2026-11-22 14:30*
//!
//! Hello!
//!
//! ---
//! **Assistant** · *2026-11-22 14:31 · model: local-model*
//!
//! Hi there.
//! ```
//!
//! Turns are delimited by a horizontal rule (`---`) followed on the next line
//! by the role in bold, optionally with metadata in italics:
//! `**<role>** · *YYYY-MM-DD HH:MM · key: value · ...*`. The metadata block
//! and every tag inside it are optional. When serializing, any occurrence of
//! this exact pattern in unfenced message content is escaped (a backslash is
//! added before the role's first asterisk) so it cannot be mistaken for a
//! real turn boundary on the next parse.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Roles recognized in turn boundaries (`**User**`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
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
    /// The metadata as it appears inside the italic block of a boundary line,
    /// or `None` when there is none.
    pub fn render(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ts) = &self.timestamp {
            parts.push(ts.clone());
        }
        for (k, v) in &self.tags {
            parts.push(if v.is_empty() {
                k.clone()
            } else {
                format!("{k}: {v}")
            });
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" · "))
        }
    }
}

/// YAML frontmatter fields. Unknown keys are ignored on parse.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Frontmatter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Free-form tool configuration, reserved for the tool-calling milestone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_yaml::Value>,
}

impl Frontmatter {
    /// True when no field is set.
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.endpoint.is_none()
            && self.system_prompt.is_none()
            && self.tools.is_none()
    }
}

/// A single conversation turn: a boundary line plus its content.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    pub meta: TurnMeta,
}

/// A parsed conversation file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    pub frontmatter: Frontmatter,
    /// Whether the source file began with a YAML frontmatter block.
    pub had_frontmatter: bool,
    /// Body text before the first turn boundary (kept for lossless round-trips).
    pub preamble: String,
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

/// Parse a boundary role line: `**<role>**` optionally followed by
/// ` · *<timestamp> · <key>: <value> · ...*`.
///
/// The metadata block is optional, and so is every part of it: a line may be
/// just `**User**`, or carry only a timestamp, only tags, or both.
fn parse_role_line(line: &str) -> Option<(Role, TurnMeta)> {
    let t = line.trim();
    let rest = t.strip_prefix("**")?;
    let end = rest.find("**")?;
    let role = Role::parse(&rest[..end])?;
    let mut meta = TurnMeta::default();
    let after = rest[end + 2..].trim_start();
    if !after.is_empty() {
        let after_dot = after.strip_prefix('·')?;
        let inner = after_dot.trim_start().strip_prefix('*')?;
        let inner = inner.trim_end().strip_suffix('*')?;
        for part in inner.split('·').map(str::trim).filter(|s| !s.is_empty()) {
            if meta.timestamp.is_none() && looks_like_timestamp(part) {
                meta.timestamp = Some(part.to_string());
            } else if let Some((k, v)) = part.split_once(':') {
                meta.tags.push((k.trim().to_string(), v.trim().to_string()));
            } else {
                meta.tags.push((part.to_string(), String::new()));
            }
        }
    }
    Some((role, meta))
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
/// Returns `(Some(yaml), body)` when the file starts with frontmatter,
/// otherwise `(None, input)`. The closing fence may be `---` or `...`.
/// A line that parses as a turn boundary aborts frontmatter detection, so a
/// file that begins directly with a conversation (no frontmatter) is not
/// misread as YAML.
fn split_frontmatter(input: &str) -> (Option<&str>, &str) {
    let mut offset = 0usize;
    let mut yaml_start: Option<usize> = None;
    for line in input.split_inclusive('\n') {
        let is_fence = line.trim_end() == "---" || line.trim_end() == "...";
        match yaml_start {
            None => {
                if !is_fence {
                    return (None, input);
                }
                offset += line.len();
                yaml_start = Some(offset);
            }
            Some(start) => {
                if is_fence {
                    let yaml = &input[start..offset].trim_end_matches('\n');
                    offset += line.len();
                    return (Some(yaml), &input[offset..]);
                }
                if parse_role_line(line).is_some() {
                    return (None, input);
                }
                offset += line.len();
            }
        }
    }
    // Opening fence without a closing fence: not frontmatter.
    (None, input)
}

/// Parse conversation Markdown into a [`Conversation`].
pub fn parse(input: &str) -> Result<Conversation> {
    let (yaml, body) = split_frontmatter(input);
    let mut conv = Conversation::default();
    if let Some(yaml) = yaml {
        conv.frontmatter =
            serde_yaml::from_str::<Frontmatter>(yaml).context("invalid YAML frontmatter")?;
        conv.had_frontmatter = true;
    }

    let lines: Vec<&str> = body.lines().collect();
    let mut preamble: Vec<&str> = Vec::new();
    let mut turn: Option<(Role, TurnMeta, Vec<&str>)> = None;
    let mut fence = FenceTracker::default();
    let mut i = 0usize;
    while i < lines.len() {
        // `true` = the line is inside a fenced code block.
        let in_fence = fence.next_line(lines[i]);
        let boundary = !in_fence
            && is_hr_line(lines[i])
            && i + 1 < lines.len()
            && parse_role_line(lines[i + 1]).is_some();
        if boundary {
            let (role, meta) = parse_role_line(lines[i + 1]).unwrap();
            fence.next_line(lines[i + 1]); // keep the tracker in sync
            match turn.take() {
                Some((r, m, ls)) => conv.turns.push(Turn {
                    role: r,
                    meta: m,
                    content: ls.join("\n").trim().to_string(),
                }),
                None => conv.preamble = preamble.join("\n").trim().to_string(),
            }
            turn = Some((role, meta, Vec::new()));
            i += 2;
        } else {
            if let Some((_, _, ls)) = &mut turn {
                ls.push(lines[i]);
            } else {
                preamble.push(lines[i]);
            }
            i += 1;
        }
    }
    if let Some((r, m, ls)) = turn {
        conv.turns.push(Turn {
            role: r,
            meta: m,
            content: ls.join("\n").trim().to_string(),
        });
    } else {
        conv.preamble = preamble.join("\n").trim().to_string();
    }
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
        let escaped = if !in_fence && prev_hr && parse_role_line(text).is_some() {
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

/// Serialize a [`Conversation`] back to canonical Markdown.
///
/// Each turn is emitted as `---` + `**<Role>**` plus its metadata (when any),
/// followed by the content with any colliding boundary patterns escaped (see
/// [`escape_boundaries`]). The output always ends with a single trailing
/// newline, and `parse(to_markdown(c)) == c` for any `c` produced by `parse`.
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
        if let Some(v) = &conv.frontmatter.system_prompt {
            mapping.insert(
                key("system_prompt"),
                serde_yaml::Value::String(v.clone()),
            );
        }
        if let Some(v) = &conv.frontmatter.tools {
            mapping.insert(key("tools"), v.clone());
        }
        let yaml = serde_yaml::to_string(&serde_yaml::Value::Mapping(mapping))
            .expect("frontmatter serialization cannot fail");
        out.push_str(yaml.trim_end());
        out.push_str("\n---\n");
    }

    let has_preamble = !conv.preamble.trim().is_empty();
    if has_preamble {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(conv.preamble.trim());
        out.push('\n');
    }

    for (i, t) in conv.turns.iter().enumerate() {
        if i > 0 || has_preamble || !out.is_empty() {
            out.push('\n');
        }
        out.push_str("---\n");
        out.push_str(&format!("**{}**", t.role.title()));
        if let Some(meta) = t.meta.render() {
            out.push_str(&format!(" · *{meta}*"));
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
}

#[cfg(test)]
mod tests;
