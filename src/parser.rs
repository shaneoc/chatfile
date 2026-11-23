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
//! Optional preamble text before the first role header.
//!
//! # User
//!
//! Hello!
//!
//! # Assistant
//!
//! Hi there.
//! ```
//!
//! Turns are delimited by depth-1 ATX headers (`# User`, `# Assistant`,
//! `# Tool`, `# System`). When a conversation is serialized, every ATX
//! header found in turn content (outside fenced code blocks) is demoted one
//! level so that message content can never be confused with mdchat's own
//! section headers; the parser reverses this on the way back in.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Roles recognized by their depth-1 header (`# User`, ...).
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

    /// The title used in section headers (`# User`, ...).
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

/// A single conversation turn: a role header plus its content.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

/// A parsed conversation file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    pub frontmatter: Frontmatter,
    /// Whether the source file began with a YAML frontmatter block.
    pub had_frontmatter: bool,
    /// Body text before the first role header (kept for lossless round-trips).
    pub preamble: String,
    pub turns: Vec<Turn>,
}

/// Recognize an ATX header line (CommonMark subset).
///
/// Returns `(depth, text)` when the line is a header: at most 3 leading
/// spaces, 1–6 `#` characters, then a space/tab or end of line. An optional
/// closing sequence (`## Title ##`) is stripped from the returned text.
fn atx_header(line: &str) -> Option<(usize, &str)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let depth = rest.chars().take_while(|&c| c == '#').count();
    if !(1..=6).contains(&depth) {
        return None;
    }
    let after = &rest[depth..];
    if !after.is_empty() && !after.starts_with([' ', '\t']) {
        return None;
    }
    let mut text = after.trim();
    // Strip an optional closing sequence: a trailing run of `#` preceded by
    // whitespace (or starting the text, e.g. `## ###`).
    let bytes = text.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] == b'#' {
        i -= 1;
    }
    if i < bytes.len() && (i == 0 || bytes[i - 1] == b' ' || bytes[i - 1] == b'\t') {
        text = text[..i].trim_end();
    }
    Some((depth, text))
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

/// A line that is a depth-1 ATX header whose text names a role (`# User`, ...).
fn parse_marker(line: &str) -> Option<Role> {
    let (depth, text) = atx_header(line)?;
    if depth != 1 {
        return None;
    }
    Role::parse(text)
}

/// Split off a leading `---`-fenced YAML block.
///
/// Returns `(Some(yaml), body)` when the file starts with frontmatter,
/// otherwise `(None, input)`. The closing fence may be `---` or `...`.
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

    let mut preamble: Vec<String> = Vec::new();
    let mut turn: Option<(Role, Vec<String>)> = None;
    let mut fence = FenceTracker::default();

    for line in body.lines() {
        // `true` = inside a fenced code block: the line is content, never a marker.
        let in_fence = fence.next_line(line);
        if !in_fence {
            if let Some(role) = parse_marker(line) {
                match turn.take() {
                    Some((r, lines)) => conv.turns.push(Turn {
                        role: r,
                        content: undemote_headers(lines.join("\n").trim()),
                    }),
                    None => conv.preamble = preamble.join("\n").trim().to_string(),
                }
                turn = Some((role, Vec::new()));
                continue;
            }
        }
        if let Some((_, lines)) = &mut turn {
            lines.push(line.to_string());
        } else {
            preamble.push(line.to_string());
        }
    }
    if let Some((r, lines)) = turn {
        conv.turns.push(Turn {
            role: r,
            content: undemote_headers(lines.join("\n").trim()),
        });
    } else {
        conv.preamble = preamble.join("\n").trim().to_string();
    }
    Ok(conv)
}

/// Apply `transform` to every ATX header line of `content` that is not inside
/// a fenced code block; all other lines pass through verbatim.
fn map_header_lines(content: &str, mut transform: impl FnMut(&str) -> String) -> String {
    let mut out = String::with_capacity(content.len());
    let mut fence = FenceTracker::default();
    for line in content.split_inclusive('\n') {
        let (text, nl) = match line.ends_with('\n') {
            true => (&line[..line.len() - 1], "\n"),
            false => (line, ""),
        };
        if fence.next_line(text) {
            out.push_str(line);
        } else {
            out.push_str(&transform(text));
            out.push_str(nl);
        }
    }
    out
}

/// Demote every ATX header in `content` that is not inside a fenced code
/// block by one level (`#` → `##`, ...). Headers already at the maximum
/// depth of 6 are left unchanged.
pub fn demote_headers(content: &str) -> String {
    map_header_lines(content, |line| match atx_header(line) {
        Some((depth, _)) if depth < 6 => {
            let indent = line.len() - line.trim_start_matches(' ').len();
            format!("{}#{}", &line[..indent], &line[indent..])
        }
        _ => line.to_string(),
    })
}

/// The inverse of [`demote_headers`]: promote headers of depth 2–6 back by one
/// level. Depth-1 headers are left alone (they cannot be the product of a
/// demotion). A depth-6 header is ambiguous — it may have been depth 5 before
/// demotion or an unchanged depth 6 — and is promoted either way, so a file
/// round-trips stably even if one original level-6 header loses a level.
fn undemote_headers(content: &str) -> String {
    map_header_lines(content, |line| match atx_header(line) {
        Some((depth, _)) if (2..=6).contains(&depth) => {
            let indent = line.len() - line.trim_start_matches(' ').len();
            format!("{}{}", &line[..indent], &line[indent + 1..])
        }
        _ => line.to_string(),
    })
}

/// Serialize a [`Conversation`] back to canonical Markdown.
///
/// Turn content is emitted with its ATX headers demoted one level (see
/// [`demote_headers`]), so message headers always sit one level below the
/// depth-1 role headers and can never be mistaken for them. The output always
/// ends with a single trailing newline, and `parse(to_markdown(c)) == c` for
/// any `c` produced by `parse`.
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
        out.push_str(&format!("# {}\n\n", t.role.title()));
        out.push_str(&demote_headers(t.content.trim()));
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
mod tests {
    use super::*;

    #[test]
    fn parse_full_document() {
        let input = "\
---
model: local-model
endpoint: http://localhost:9000/v1/chat/completions
system_prompt: Be terse.
tools:
  - name: shell
---

# My convo

# User

Hello!

# Assistant

Hi there.

# Tool

exit code 0
";
        let conv = parse(input).unwrap();
        assert!(conv.had_frontmatter);
        assert_eq!(conv.frontmatter.model.as_deref(), Some("local-model"));
        assert_eq!(
            conv.frontmatter.endpoint.as_deref(),
            Some("http://localhost:9000/v1/chat/completions")
        );
        assert_eq!(conv.frontmatter.system_prompt.as_deref(), Some("Be terse."));
        assert!(conv.frontmatter.tools.is_some());
        assert_eq!(conv.preamble, "# My convo");
        assert_eq!(conv.turns.len(), 3);
        assert_eq!(conv.turns[0].role, Role::User);
        assert_eq!(conv.turns[0].content, "Hello!");
        assert_eq!(conv.turns[1].role, Role::Assistant);
        assert_eq!(conv.turns[1].content, "Hi there.");
        assert_eq!(conv.turns[2].role, Role::Tool);
        assert_eq!(conv.turns[2].content, "exit code 0");
    }

    #[test]
    fn parse_no_frontmatter_and_preamble_only() {
        let conv = parse("Just some notes.\n").unwrap();
        assert!(!conv.had_frontmatter);
        assert!(conv.frontmatter.is_empty());
        assert_eq!(conv.preamble, "Just some notes.");
        assert!(conv.turns.is_empty());
    }

    #[test]
    fn parse_empty_file() {
        let conv = parse("").unwrap();
        assert!(!conv.had_frontmatter);
        assert_eq!(conv.preamble, "");
        assert!(conv.turns.is_empty());
    }

    #[test]
    fn parse_multiline_content_preserved() {
        let input = "\
# User

line one
line two

  indented line
";
        let conv = parse(input).unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].content, "line one\nline two\n\n  indented line");
    }

    #[test]
    fn parse_marker_whitespace_tolerant() {
        let input = "  #   USER   \n\ntext\n";
        let conv = parse(input).unwrap();
        assert_eq!(conv.turns[0].role, Role::User);
        assert_eq!(conv.turns[0].content, "text");
    }

    #[test]
    fn parse_marker_case_insensitive() {
        let conv = parse("# assistant\n\nhi\n").unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].role, Role::Assistant);
    }

    #[test]
    fn parse_deeper_headers_are_not_markers() {
        // Depth-2 role names are content, not turn boundaries.
        let conv = parse("# User\n\n## Assistant should not start a turn\n").unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].role, Role::User);
        // Demoted on the way in, restored on the way out.
        assert_eq!(conv.turns[0].content, "# Assistant should not start a turn");
    }

    #[test]
    fn parse_marker_inside_code_block_ignored() {
        let input = "\
# User

```markdown
# Assistant

this is just text in a fence
```

still user content
";
        let conv = parse(input).unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].role, Role::User);
        assert_eq!(
            conv.turns[0].content,
            "```markdown\n# Assistant\n\nthis is just text in a fence\n```\n\nstill user content"
        );
    }

    #[test]
    fn parse_undemotes_content_headers() {
        let input = "\
# User

## Original h1 became h2

plain

###### Already at the bottom
";
        let conv = parse(input).unwrap();
        assert_eq!(
            conv.turns[0].content,
            "# Original h1 became h2\n\nplain\n\n##### Already at the bottom"
        );
    }

    #[test]
    fn parse_unterminated_frontmatter_is_preamble() {
        let input = "---\nmodel: x\nno closing fence\n";
        let conv = parse(input).unwrap();
        assert!(!conv.had_frontmatter);
        assert_eq!(conv.preamble, "---\nmodel: x\nno closing fence");
    }

    #[test]
    fn parse_invalid_frontmatter_fails() {
        let input = "---\nmodel: [unclosed\n---\ntext\n";
        assert!(parse(input).is_err());
    }

    #[test]
    fn atx_header_recognition() {
        assert_eq!(atx_header("# User"), Some((1, "User")));
        assert_eq!(atx_header("   # Spaced"), Some((1, "Spaced")));
        assert_eq!(atx_header("###### Six"), Some((6, "Six")));
        assert_eq!(atx_header("# Closing ##"), Some((1, "Closing")));
        assert_eq!(atx_header("#Empty"), None); // no space after #
        assert_eq!(atx_header("####### Seven"), None); // depth 7 is not a header
        assert_eq!(atx_header("    # Four spaces"), None); // too much indent
        assert_eq!(atx_header("##Title#"), None); // no space, trailing # kept as text
        assert_eq!(atx_header("# Title#"), Some((1, "Title#")));
    }

    #[test]
    fn demote_headers_one_level() {
        let input = "# One\n## Two\n### Three\n#### Four\n##### Five\n###### Six\nplain text\n";
        assert_eq!(
            demote_headers(input),
            "## One\n### Two\n#### Three\n##### Four\n###### Five\n###### Six\nplain text\n"
        );
    }

    #[test]
    fn demote_leaves_depth_six_unchanged() {
        assert_eq!(demote_headers("###### Max"), "###### Max");
    }

    #[test]
    fn demote_skips_headers_inside_backtick_fences() {
        let input = "# Real\n\n```\n# In fence\n## Also in fence\n```\n\n# After\n";
        assert_eq!(
            demote_headers(input),
            "## Real\n\n```\n# In fence\n## Also in fence\n```\n\n## After\n"
        );
    }

    #[test]
    fn demote_skips_headers_inside_tilde_fences() {
        let input = "~~~\n# Tilde fence\n~~~\n# Outside\n";
        assert_eq!(demote_headers(input), "~~~\n# Tilde fence\n~~~\n## Outside\n");
    }

    #[test]
    fn demote_unclosed_fence_treated_as_fenced() {
        let input = "```\n# Never closed\nstill in fence\n";
        assert_eq!(demote_headers(input), "```\n# Never closed\nstill in fence\n");
    }

    #[test]
    fn demote_respects_fence_length_and_char() {
        // A shorter or differently-charred run inside a fence is content.
        let input = "````\n```\n# Content line\n````\n# Outside\n";
        assert_eq!(demote_headers(input), "````\n```\n# Content line\n````\n## Outside\n");
    }

    #[test]
    fn demote_backtick_info_string_rule() {
        // Backticks in the info string prevent a backtick fence from opening.
        let input = "`` `not a fence`\n# Header here\n";
        assert_eq!(demote_headers(input), "`` `not a fence`\n## Header here\n");
    }

    #[test]
    fn demote_preserves_indent_and_closing_sequence() {
        let input = "  ## Indented ##\n";
        assert_eq!(demote_headers(input), "  ### Indented ##\n");
    }

    #[test]
    fn undemote_is_inverse_of_demote() {
        // Strict inverse for headers that demotion can represent (depth <= 5).
        let original = "# A\n## B\n### C\n#### D\n##### E\n";
        assert_eq!(undemote_headers(&demote_headers(original)), original);
    }

    #[test]
    fn depth_six_header_is_ambiguous_but_stable() {
        // A depth-6 header cannot be demoted, so parse must pick one reading;
        // it promotes to depth 5. The file form is stable across round trips.
        let original = "###### F\n";
        let demoted = demote_headers(original);
        assert_eq!(demoted, "###### F\n"); // unchanged at max depth
        assert_eq!(undemote_headers(&demoted), "##### F\n");
        // File-level round trip is idempotent even though one level is lost.
        let conv = parse(&format!("# User\n\n{demoted}")).unwrap();
        let once = to_markdown(&conv);
        assert_eq!(to_markdown(&parse(&once).unwrap()), once);
    }

    #[test]
    fn demote_no_newline_at_end() {
        assert_eq!(demote_headers("# X"), "## X");
    }

    #[test]
    fn serialize_round_trip_is_idempotent() {
        let input = "\
---
model: local-model
system_prompt: Be terse.
---

# User

Hello!

# Assistant

Hi there.
";
        let once = to_markdown(&parse(input).unwrap());
        let twice = to_markdown(&parse(&once).unwrap());
        assert_eq!(once, twice);
    }

    #[test]
    fn serialize_round_trip_with_headers_in_content() {
        // Message content carrying its own headers must survive a round trip.
        // The file stores demoted headers; parse restores the originals.
        let input = "\
# User

## My question

some body

# Assistant

### Answer heading

text
";
        let once = to_markdown(&parse(input).unwrap());
        let twice = to_markdown(&parse(&once).unwrap());
        assert_eq!(once, twice);
        // The in-memory content keeps the original (undemoted) headers.
        let conv = parse(&once).unwrap();
        assert_eq!(conv.turns[0].content, "# My question\n\nsome body");
        assert_eq!(conv.turns[1].content, "## Answer heading\n\ntext");
    }

    #[test]
    fn serialize_canonical_form() {
        let conv = Conversation {
            frontmatter: Frontmatter {
                model: Some("m".into()),
                endpoint: None,
                system_prompt: Some("Be terse.".into()),
                tools: None,
            },
            had_frontmatter: true,
            preamble: String::new(),
            turns: vec![Turn {
                role: Role::User,
                content: "Hello!".into(),
            }],
        };
        let md = to_markdown(&conv);
        assert_eq!(md, "---\nmodel: m\nsystem_prompt: Be terse.\n---\n\n# User\n\nHello!\n");
    }

    #[test]
    fn serialize_demotes_content_headers() {
        let conv = Conversation {
            turns: vec![Turn {
                role: Role::Assistant,
                content: "# Title\nbody".into(),
            }],
            ..Default::default()
        };
        assert_eq!(to_markdown(&conv), "# Assistant\n\n## Title\nbody\n");
    }

    #[test]
    fn serialize_empty_conversation() {
        assert_eq!(to_markdown(&Conversation::default()), "");
    }

    #[test]
    fn serialize_multiline_system_prompt_quotes() {
        let conv = Conversation {
            frontmatter: Frontmatter {
                system_prompt: Some("line1\nline2".into()),
                ..Default::default()
            },
            had_frontmatter: true,
            ..Default::default()
        };
        let md = to_markdown(&conv);
        // Must round-trip through YAML.
        let back = parse(&md).unwrap();
        assert_eq!(back.frontmatter.system_prompt.as_deref(), Some("line1\nline2"));
    }

    #[test]
    fn role_parse_case_insensitive() {
        assert_eq!(Role::parse("Assistant"), Some(Role::Assistant));
        assert_eq!(Role::parse(" tool "), Some(Role::Tool));
        assert_eq!(Role::parse("narrator"), None);
    }

    #[test]
    fn role_titles() {
        assert_eq!(Role::System.title(), "System");
        assert_eq!(Role::User.title(), "User");
        assert_eq!(Role::Assistant.title(), "Assistant");
        assert_eq!(Role::Tool.title(), "Tool");
    }
}
