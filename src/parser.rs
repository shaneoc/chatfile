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
//! Optional preamble text before the first role marker.
//!
//! <!-- role: user -->
//!
//! Hello!
//!
//! <!-- role: assistant -->
//!
//! Hi there.
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Roles recognized in `<!-- role: ... -->` markers.
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

/// A single conversation turn: a role marker plus its content.
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
    /// Body text before the first role marker (kept for lossless round-trips).
    pub preamble: String,
    pub turns: Vec<Turn>,
}

/// A line that is exactly `<!-- role: <name> -->` (whitespace-tolerant).
fn parse_marker(line: &str) -> Option<Role> {
    let inner = line.trim().strip_prefix("<!--")?.strip_suffix("-->")?;
    let value = inner.trim().strip_prefix("role:")?.trim();
    Role::parse(value)
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

    for line in body.lines() {
        if let Some(role) = parse_marker(line) {
            match turn.take() {
                Some((r, lines)) => conv.turns.push(Turn {
                    role: r,
                    content: lines.join("\n").trim().to_string(),
                }),
                None => conv.preamble = preamble.join("\n").trim().to_string(),
            }
            turn = Some((role, Vec::new()));
        } else if let Some((_, lines)) = &mut turn {
            lines.push(line.to_string());
        } else {
            preamble.push(line.to_string());
        }
    }
    if let Some((r, lines)) = turn {
        conv.turns.push(Turn {
            role: r,
            content: lines.join("\n").trim().to_string(),
        });
    } else {
        conv.preamble = preamble.join("\n").trim().to_string();
    }
    Ok(conv)
}

/// Serialize a [`Conversation`] back to canonical Markdown.
///
/// The output always ends with a single trailing newline, and
/// `parse(to_markdown(c)) == c` for any `c` produced by `parse`.
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
        out.push_str(&format!("<!-- role: {} -->\n\n", t.role));
        out.push_str(t.content.trim());
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

<!-- role: user -->

Hello!

<!-- role: assistant -->

Hi there.

<!-- role: tool -->

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
<!-- role: user -->

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
        let input = "  <!--   role:   USER   -->  \n\ntext\n";
        let conv = parse(input).unwrap();
        assert_eq!(conv.turns[0].role, Role::User);
        assert_eq!(conv.turns[0].content, "text");
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
    fn serialize_round_trip_is_idempotent() {
        let input = "\
---
model: local-model
system_prompt: Be terse.
---

<!-- role: user -->

Hello!

<!-- role: assistant -->

Hi there.
";
        let once = to_markdown(&parse(input).unwrap());
        let twice = to_markdown(&parse(&once).unwrap());
        assert_eq!(once, twice);
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
        assert_eq!(
            md,
            "---\nmodel: m\nsystem_prompt: Be terse.\n---\n\n<!-- role: user -->\n\nHello!\n"
        );
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
}
