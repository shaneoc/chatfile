//! Tests for [`super`]: parsing, serialization, metadata, and escaping.

use super::*;

#[test]
fn parse_full_document() {
    let input = "\
---
model: local-model
endpoint: http://localhost:9000/v1/chat/completions
tools:
  - name: shell
---
**System**

Be terse.

---
**User** | 2026-11-22 14:30

Hello!

---
**Assistant** | 2026-11-22 14:31 | model: local-model

Hi there.

---
**Tool** | 2026-11-22 14:31 | tool: bash | id: call_1

<details>
<summary>Output</summary>

~~~~text
exit code 0
~~~~

</details>
";
    let conv = parse(input).unwrap();
    assert!(conv.had_frontmatter);
    assert_eq!(conv.frontmatter.model.as_deref(), Some("local-model"));
    assert_eq!(
        conv.frontmatter.endpoint.as_deref(),
        Some("http://localhost:9000/v1/chat/completions")
    );
    // The legacy free-form `tools:` list is tolerated and reads as enabled.
    assert!(conv.frontmatter.tools);
    assert_eq!(conv.turns.len(), 4);
    assert_eq!(conv.turns[0].role, Role::System);
    assert_eq!(conv.turns[0].content, "Be terse.");
    assert_eq!(conv.turns[1].role, Role::User);
    assert_eq!(conv.turns[1].content, "Hello!");
    assert_eq!(
        conv.turns[1].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:30".into()),
            tags: vec![],
        }
    );
    assert_eq!(conv.turns[2].role, Role::Assistant);
    assert_eq!(conv.turns[2].content, "Hi there.");
    assert_eq!(
        conv.turns[2].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags: vec![("model".into(), "local-model".into())],
        }
    );
    assert_eq!(conv.turns[3].role, Role::Tool);
    let data = parse_tool_turn(&conv.turns[3]).expect("well-formed tool turn");
    assert_eq!(data.name, "bash");
    assert_eq!(data.id, "call_1");
    assert_eq!(data.output, "exit code 0");
}

#[test]
fn parse_first_line_must_be_system_or_user() {
    // No frontmatter: plain text is not a valid first section.
    let err = parse("Just some notes.\n").unwrap_err();
    assert!(err.to_string().contains("line 1"), "{err}");
    assert!(err.to_string().contains("first section"), "{err}");

    // With frontmatter: the error points at the first body line.
    let err = parse("---\nmodel: m\n---\nJust some notes.\n").unwrap_err();
    assert!(err.to_string().contains("line 4"), "{err}");

    // An empty file has no sections at all.
    assert!(parse("").unwrap_err().to_string().contains("line 1"));
}

#[test]
fn parse_system_only_allowed_as_first_section() {
    // Lines: 1 **User**, 2 blank, 3 hi, 4 blank, 5 ---, 6 **System**.
    let err = parse("**User**\n\nhi\n\n---\n**System**\n\nlate\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 6"), "{msg}");
    assert!(msg.contains("**System**"), "{msg}");
}

#[test]
fn parse_header_rejects_wrong_separators() {
    // The legacy `·` separator is no longer a header.
    for (input, at) in [
        ("**User** · *2026-11-22 14:30*\n\nhi\n", "line 1"),
        ("**User**\n\nhi\n\n---\n**Assistant** - 2026-11-22 14:31\n\nlate\n", "line 6"),
        // Missing the spaces around `|`.
        ("**User**| 2026-11-22 14:30\n\nhi\n", "line 1"),
    ] {
        let err = parse(input).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(at), "{msg}");
        assert!(
            msg.contains("invalid header metadata separator; expected ' | '"),
            "{msg}"
        );
    }
}

#[test]
fn parse_header_rejects_italicized_metadata() {
    // The legacy italic block around the metadata.
    let err = parse("**User** | *2026-11-22 14:30*\n\nhi\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 1"), "{msg}");
    assert!(msg.contains("plain text"), "{msg}");

    // Italicized key: value segment.
    let err = parse("**User** | 2026-11-22 14:30 | *model: m*\n\nhi\n").unwrap_err();
    assert!(err.to_string().contains("plain text"), "{err}");
}

#[test]
fn parse_header_requires_key_prefix_after_timestamp() {
    // A bare word after the timestamp is not `key: value`.
    let err = parse("**User** | 2026-11-22 14:30 | note\n\nhi\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 1"), "{msg}");
    assert!(
        msg.contains("expected a timestamp or 'key: value'"),
        "{msg}"
    );

    // Same violation in a later section: the line number tracks it.
    let err = parse("**User**\n\nhi\n\n---\n**Assistant** | 2026-11-22 14:31 | model\n\nlate\n")
        .unwrap_err();
    assert!(err.to_string().contains("line 6"), "{err}");

    // An empty key is rejected too.
    let err = parse("**User** | : value\n\nhi\n").unwrap_err();
    assert!(err.to_string().contains("non-empty key"), "{err}");
}

#[test]
fn parse_header_timestamp_must_be_first_segment() {
    let err = parse("**User** | model: m | 2026-11-22 14:30\n\nhi\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 1"), "{msg}");
    assert!(msg.contains("first metadata segment"), "{msg}");

    // A second timestamp is rejected as well.
    let err = parse("**User** | 2026-11-22 14:30 | 2026-11-22 14:31\n\nhi\n").unwrap_err();
    assert!(err.to_string().contains("duplicate timestamp"), "{err}");
}

#[test]
fn parse_header_tags_without_timestamp_ok() {
    // The timestamp is optional; `key: value` segments may lead.
    let conv = parse("**User** | model: m1 | endpoint: local\n\nhi\n").unwrap();
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: None,
            tags: vec![
                ("model".into(), "m1".into()),
                ("endpoint".into(), "local".into()),
            ],
        }
    );
}

#[test]
fn parse_multiline_content_preserved() {
    let input = "\
**User**

line one
line two

  indented line
";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].content, "line one\nline two\n\n  indented line");
}

#[test]
fn parse_boundary_whitespace_tolerant() {
    let input = "  ** USER **  \n\ntext\n";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns[0].role, Role::User);
    assert_eq!(conv.turns[0].content, "text");
}

#[test]
fn parse_role_line_case_insensitive() {
    let conv = parse("**User**\n\nhello\n\n---\n**assistant**\n\nhi\n").unwrap();
    assert_eq!(conv.turns.len(), 2);
    assert_eq!(conv.turns[1].role, Role::Assistant);
}

#[test]
fn parse_metadata_optional_variants() {
    // No metadata at all.
    let conv = parse("**User**\n\nhi\n").unwrap();
    assert_eq!(conv.turns[0].meta, TurnMeta::default());

    // Timestamp only (model tag missing).
    let conv = parse("**User** | 2026-11-22 14:30\n\nhi\n").unwrap();
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:30".into()),
            tags: vec![],
        }
    );

    // Tags only (no timestamp).
    let conv = parse("**User** | model: m1 | endpoint: local\n\nhi\n").unwrap();
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: None,
            tags: vec![("model".into(), "m1".into()), ("endpoint".into(), "local".into())],
        }
    );

    // Timestamp plus several tags; tag values may contain colons.
    let conv = parse(
        "**User**\n\nhello\n\n---\n**Assistant** | 2026-11-22 14:31 | model: unsloth/Qwen:Q8 | note: a: b\n\nhi\n",
    )
    .unwrap();
    assert_eq!(
        conv.turns[1].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags: vec![
                ("model".into(), "unsloth/Qwen:Q8".into()),
                ("note".into(), "a: b".into()),
            ],
        }
    );
}

#[test]
fn parse_hr_without_role_line_is_content() {
    let input = "**User**\n\nbefore\n---\nafter\n";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].content, "before\n---\nafter");
}

#[test]
fn parse_role_line_without_hr_is_content() {
    // A role line without a preceding hr (and not the file's first line) is
    // just content.
    let input = "**User**\n\nbefore\n**Assistant**\n\nnot a section\n";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].content, "before\n**Assistant**\n\nnot a section");
}

#[test]
fn parse_boundary_inside_code_block_ignored() {
    let input = "\
**User**

```markdown
---
**Assistant** | 2026-11-22 14:31

this is just text in a fence
```

still user content
";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].role, Role::User);
    assert_eq!(
        conv.turns[0].content,
        "```markdown\n---\n**Assistant** | 2026-11-22 14:31\n\nthis is just text in a fence\n```\n\nstill user content"
    );
}

#[test]
fn parse_opening_fence_requires_closing_fence() {
    // A file that starts with `---` claims to have frontmatter; without a
    // closing fence that is a format error pointing at the opening line.
    let err = parse("---\n**User** | 2026-11-22 14:30\n\nHello\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 1"), "{msg}");
    assert!(msg.contains("not closed"), "{msg}");

    let err = parse("---\nmodel: x\nno closing fence\n").unwrap_err();
    assert!(err.to_string().contains("line 1"));
}

#[test]
fn parse_empty_frontmatter_block_is_defaults() {
    let input = "---\n---\n**User**\n\nhi\n";
    let conv = parse(input).unwrap();
    assert!(conv.had_frontmatter);
    assert!(conv.frontmatter.is_empty());
    assert_eq!(conv.turns.len(), 1);
    // Round-trips exactly.
    assert_eq!(to_markdown(&conv), input);
}

#[test]
fn parse_invalid_frontmatter_fails() {
    let err = parse("---\nmodel: [unclosed\n---\ntext\n").unwrap_err();
    assert!(err.to_string().contains("invalid YAML frontmatter"));
}

#[test]
fn serialize_round_trip_is_idempotent() {
    let input = "\
---
model: local-model
---
**System**

Be terse.

---
**User** | 2026-11-22 14:30

Hello!

---
**Assistant** | 2026-11-22 14:31 | model: local-model

Hi there.
";
    let once = to_markdown(&parse(input).unwrap());
    assert_eq!(once, input); // already canonical
    let twice = to_markdown(&parse(&once).unwrap());
    assert_eq!(once, twice);
}

#[test]
fn serialize_canonical_form() {
    let conv = Conversation {
        frontmatter: Frontmatter {
            model: Some("m".into()),
            ..Default::default()
        },
        had_frontmatter: true,
        turns: vec![
            Turn {
                role: Role::System,
                content: "Be terse.".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::User,
                content: "Hello!".into(),
                meta: TurnMeta {
                    timestamp: Some("2026-11-22 14:30".into()),
                    tags: vec![],
                },
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
    };
    let md = to_markdown(&conv);
    assert_eq!(
        md,
        "---\nmodel: m\n---\n**System**\n\nBe terse.\n\n---\n**User** | 2026-11-22 14:30\n\nHello!\n"
    );
}

#[test]
fn serialize_omits_missing_metadata() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: "Hi".into(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    assert_eq!(to_markdown(&conv), "**Assistant**\n\nHi\n");
}

#[test]
fn serialize_system_section_round_trips() {
    let conv = Conversation {
        had_frontmatter: true,
        turns: vec![
            Turn {
                role: Role::System,
                content: "line1\nline2".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let md = to_markdown(&conv);
    let back = parse(&md).unwrap();
    assert_eq!(back.turns[0].role, Role::System);
    assert_eq!(back.turns[0].content, "line1\nline2");
    assert_eq!(to_markdown(&back), md);
}

#[test]
fn role_parse_case_insensitive() {
    assert_eq!(Role::parse("System"), Some(Role::System));
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

#[test]
fn escape_boundaries_escapes_unfenced_collision() {
    // `N -> 2N + 1` leading backslashes before the hr line (`0 -> 1`,
    // `1 -> 3`, `2 -> 5`).
    let b = |n: usize| "\\".repeat(n);
    assert_eq!(
        escape_boundaries("look at this:\n---\n**User** | 2026-11-22 14:30\n\nand more"),
        "look at this:\n\\---\n**User** | 2026-11-22 14:30\n\nand more"
    );
    assert_eq!(
        escape_boundaries(&format!("note:\n{}---\n**User**", b(1))),
        format!("note:\n{}---\n**User**", b(3))
    );
    assert_eq!(
        escape_boundaries(&format!("note:\n{}---\n**User**", b(2))),
        format!("note:\n{}---\n**User**", b(5))
    );
}

#[test]
fn escape_boundaries_leaves_non_collisions_alone() {
    // No `**` line after the hr.
    assert_eq!(escape_boundaries("---\nplain text"), "---\nplain text");
    // Role line without a preceding hr.
    assert_eq!(
        escape_boundaries("**User** | 2026-11-22 14:30"),
        "**User** | 2026-11-22 14:30"
    );
    // The hr is the last line: nothing follows it.
    assert_eq!(escape_boundaries("text\n---"), "text\n---");
    // An unknown role still collides (the rule keys off `**`, not the role).
    assert_eq!(escape_boundaries("---\n**Narrator**"), "\\---\n**Narrator**");
}

#[test]
fn escape_boundaries_skips_fenced_content() {
    let content = "```\n---\n**User** | 2026-11-22 14:30\n```\nreal text";
    assert_eq!(escape_boundaries(content), content);
}

#[test]
fn escape_boundaries_ignores_indented_role_line() {
    // The collision rule requires the following line to *begin* with `**`;
    // an indented role line is plain content and needs no escaping.
    let content = "---\n  **User**";
    assert_eq!(escape_boundaries(content), content);
}

#[test]
fn escape_boundaries_escapes_backslashed_fence_lines() {
    // `N -> 2N + 1` for backslash-prefixed fence lines outside a fence.
    let b = |n: usize| "\\".repeat(n);
    assert_eq!(
        escape_boundaries(&format!("{}```rust", b(1))),
        format!("{}```rust", b(3))
    );
    assert_eq!(
        escape_boundaries(&format!("{}~~~x", b(2))),
        format!("{}~~~x", b(5))
    );
}

#[test]
fn escape_boundaries_escapes_unclosed_fence_opener() {
    // An opening fence that never closes gets one backslash prepended
    // (`0 -> 1`); no synthetic closing fence is added.
    let b = |n: usize| "\\".repeat(n);
    assert_eq!(
        escape_boundaries("```rust\nfn main() {"),
        format!("{}```rust\nfn main() {{", b(1))
    );
    // An already-backslashed opener is a fence line with N >= 1 backslashes:
    // `N -> 2N + 1`, not the unclosed-fence rule.
    assert_eq!(
        escape_boundaries(&format!("{}```rust", b(1))),
        format!("{}```rust", b(3))
    );
}

#[test]
fn escaped_collision_round_trips_as_content() {
    // A message containing the raw boundary pattern must survive a round
    // trip byte-exactly: serialized with the `N -> 2N + 1` escape, parsed
    // back to the original content, stable thereafter.
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: "---\n**Assistant** | 2026-11-22 14:31".into(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("\\---\n**Assistant**"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].role, Role::User);
    assert_eq!(back.turns[0].content, "---\n**Assistant** | 2026-11-22 14:31");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn boundary_collision_round_trips_losslessly() {
    // parse(serialize(msg)) == msg for every backslash count N (0..=4)
    // before a `---` line directly above a `**` line.
    let b = |n: usize| "\\".repeat(n);
    for n in 0..=4usize {
        let content = format!("{}---\n**User**", b(n));
        let conv = Conversation {
            turns: vec![Turn {
                role: Role::User,
                content: content.clone(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            }],
            ..Default::default()
        };
        let once = to_markdown(&conv);
        let back = parse(&once).unwrap();
        assert_eq!(back.turns.len(), 1, "n={n}");
        assert_eq!(back.turns[0].content, content, "n={n}");
        assert_eq!(to_markdown(&back), once, "n={n}");
    }
}

#[test]
fn closed_fence_with_boundary_collision_round_trips() {
    // A valid closed fenced code block containing the raw boundary pattern
    // is left untouched by serialization and parses back byte-exactly.
    let content = "```\n---\n**User** | 2026-11-22 14:30\n```\nreal text";
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: content.to_string(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].content, content);
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn unclosed_fence_at_end_of_message_round_trips() {
    // An unclosed opening fence at the end of a message body (plain or
    // already backslashed) survives a round trip with another turn after it.
    for opener in ["```rust", "\\```rust"] {
        let content = format!("{opener}\nfn main() {{");
        let conv = Conversation {
            turns: vec![
                Turn {
                    role: Role::User,
                    content: content.clone(),
                    meta: TurnMeta::default(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                },
                Turn {
                    role: Role::Assistant,
                    content: "done".into(),
                    meta: TurnMeta::default(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                },
            ],
            ..Default::default()
        };
        let once = to_markdown(&conv);
        let back = parse(&once).unwrap();
        assert_eq!(back.turns.len(), 2, "opener={opener}");
        assert_eq!(back.turns[0].content, content, "opener={opener}");
        assert_eq!(back.turns[1].content, "done", "opener={opener}");
        assert_eq!(to_markdown(&back), once, "opener={opener}");
    }
}

#[test]
fn parse_restores_odd_backslash_escapes() {
    // Odd k before `---` above a `**` line: restore (k - 1) / 2 backslashes.
    let b = |n: usize| "\\".repeat(n);
    for (k, restored) in [(1usize, 0usize), (3, 1), (5, 2)] {
        let input = format!("**User**\n\n{}---\n**Assistant**\n\nhi\n", b(k));
        let conv = parse(&input).unwrap();
        assert_eq!(conv.turns.len(), 1, "k={k}");
        assert_eq!(
            conv.turns[0].content,
            format!("{}---\n**Assistant**\n\nhi", b(restored)),
            "k={k}"
        );
    }
    // Same for backslash-prefixed fence lines.
    for (k, restored) in [(1usize, 0usize), (3, 1), (5, 2)] {
        let input = format!("**User**\n\n{}```rust\nfn main() {{\n", b(k));
        let conv = parse(&input).unwrap();
        assert_eq!(conv.turns.len(), 1, "k={k}");
        assert_eq!(
            conv.turns[0].content,
            format!("{}```rust\nfn main() {{", b(restored)),
            "k={k}"
        );
    }
}

#[test]
fn parse_rejects_even_backslash_counts() {
    // Even k before `---` directly above a `**` line.
    let input = format!("**User**\n\n{}---\n**Assistant**\n", "\\".repeat(2));
    let err = parse(&input).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 3"), "{msg}");
    assert!(msg.contains("even number of backslashes before '---'"), "{msg}");

    // Even k before a code fence.
    let input = format!("**User**\n\n{}```rust\nfn main() {{\n", "\\".repeat(2));
    let err = parse(&input).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 3"), "{msg}");
    assert!(msg.contains("even number of backslashes before a code fence"), "{msg}");
}

#[test]
fn parse_requires_blank_line_before_separator() {
    // A `---` directly above a `**` line must be preceded by a blank line.
    // Lines: 1 **User**, 2 blank, 3 hi, 4 ---.
    let err = parse("**User**\n\nhi\n---\n**Assistant**\n\nhello\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 4"), "{msg}");
    assert!(
        msg.contains("turn separator '---' must be preceded by a blank line"),
        "{msg}"
    );

    // A whitespace-only line counts as blank.
    let conv = parse("**User**\n\nhi\n   \n---\n**Assistant**\n\nhello\n").unwrap();
    assert_eq!(conv.turns.len(), 2);
    assert_eq!(conv.turns[0].content, "hi");
}

#[test]
fn empty_section_serializes_with_blank_line_before_separator() {
    // An empty section still gets a blank line above its separator.
    let conv = parse("**User**\n\n\n---\n**Assistant**\n\nhello\n").unwrap();
    assert_eq!(conv.turns.len(), 2);
    assert_eq!(conv.turns[0].content, "");
    let md = to_markdown(&conv);
    assert!(md.contains("\n\n---\n"), "{md}");
    // Round-trips stably.
    let back = parse(&md).unwrap();
    assert_eq!(back.turns[0].content, "");
    assert_eq!(to_markdown(&back), md);
}

#[test]
fn boundary_collision_without_blank_line_round_trips_losslessly() {
    // Content with a `---` directly above a `**` line and no blank line is
    // escaped on serialize (`N -> 2N + 1`) and parses back byte-exactly,
    // without tripping the missing-blank-line rule.
    let content = "Heading\n---\n**Bold**";
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: content.to_string(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("\\---\n**Bold**"), "{once}");
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].content, content);
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn unclosed_fence_round_trips_without_lingering_backslash() {
    // An unclosed opening fence is escaped on serialize (`0 -> 1`) and the
    // parser restores `1 -> 0`, so the content comes back byte-exactly with
    // no backslash left behind.
    let content = "```rust\nfn main() {";
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: content.to_string(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("\\```rust"), "{once}");
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].content, content);
    assert!(!back.turns[0].content.contains('\\'));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn turn_meta_render() {
    assert_eq!(TurnMeta::default().render(), None);
    let meta = TurnMeta {
        timestamp: Some("2026-11-22 14:30".into()),
        tags: vec![("model".into(), "m".into()), ("flag".into(), String::new())],
    };
    // Empty values keep the `key: ` prefix (a bare key would not re-parse).
    assert_eq!(
        meta.render().as_deref(),
        Some("2026-11-22 14:30 | model: m | flag: ")
    );
}

#[test]
fn looks_like_timestamp_validation() {
    assert!(looks_like_timestamp("2026-11-22 14:30"));
    assert!(!looks_like_timestamp("2026/11/22 14:30"));
    assert!(!looks_like_timestamp("2026-11-22 14:3"));
    assert!(!looks_like_timestamp("2026-11-22 14:30:59"));
    assert!(!looks_like_timestamp("model: m"));
}

#[test]
fn now_timestamp_shape() {
    let ts = now_timestamp();
    assert!(looks_like_timestamp(&ts), "bad timestamp shape: {ts}");
}

// ---------------------------------------------------------------------------
// Reasoning: `thinking: true` + collapsible <details> block
// ---------------------------------------------------------------------------

/// A two-turn conversation: a user question and an assistant reply with
/// reasoning (the `thinking: true` header tag is set accordingly).
fn thinking_conv(reasoning: &str, answer: &str) -> Conversation {
    Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "q".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Assistant,
                content: answer.to_string(),
                meta: TurnMeta {
                    timestamp: None,
                    tags: vec![("thinking".into(), "true".into())],
                },
                reasoning: Some(reasoning.to_string()),
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    }
}

#[test]
fn reasoning_round_trip_with_answer() {
    // Byte-exact: the wrapper is `<details>` / `<summary>Thinking</summary>`
    // with blank lines around the reasoning, then a blank line and the
    // answer.
    let conv = thinking_conv("because 6*9", "54");
    let once = to_markdown(&conv);
    assert_eq!(
        once,
        "**User**\n\nq\n\n---\n**Assistant** | thinking: true\n\n<details>\n\
         <summary>Thinking</summary>\n\nbecause 6*9\n\n</details>\n\n54\n"
    );
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 2);
    assert_eq!(back.turns[1].role, Role::Assistant);
    assert_eq!(back.turns[1].reasoning.as_deref(), Some("because 6*9"));
    assert_eq!(back.turns[1].content, "54");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_round_trip_empty_answer() {
    // An empty answer omits the trailing blank line and answer entirely.
    let conv = thinking_conv("only thoughts", "");
    let once = to_markdown(&conv);
    assert_eq!(
        once,
        "**User**\n\nq\n\n---\n**Assistant** | thinking: true\n\n<details>\n\
         <summary>Thinking</summary>\n\nonly thoughts\n\n</details>\n"
    );
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some("only thoughts"));
    assert_eq!(back.turns[1].content, "");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_matched_nested_details_stays_verbatim() {
    // A case-insensitively matched nested pair is left untouched by
    // serialization and parses back byte-exactly (the depth tracking keeps
    // the wrapper from ending early).
    let r = "<DETAILS><summary>Sub</summary>inner</DETAILS>";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("<DETAILS><summary>Sub</summary>inner</DETAILS>"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(back.turns[1].content, "done");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_unmatched_opener_is_escaped() {
    // An unmatched opener (even one with attributes) gets one backslash
    // prepended and round-trips.
    let r = "note <details open> then more";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("\\<details open>"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_unmatched_closer_on_own_line_is_escaped() {
    let r = "line one\n</Details>\nline two";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("\\</Details>"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_unmatched_closer_mid_line_is_escaped() {
    // `</details >` (whitespace before `>`) is a valid closer shape but
    // must stand alone to end the block, so mid-line it is escaped.
    let r = "see <span></span></details > here";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("\\</details >"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_already_escaped_closer_re_escapes() {
    // An already-backslashed tag is rewritten N -> 2N + 1 (1 -> 3) and
    // parses back to the original single backslash.
    let r = "keep \\</details> literal";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("\\\\\\</details>"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_invalid_spacing_is_not_a_tag() {
    // `< details>` and `</ details>` (whitespace after `<`) are not tags.
    let r = "not a tag: < details> and </ details>";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("< details> and </ details>"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_closer_in_inline_code_is_ignored() {
    let r = "use `</details>` in markdown";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("`</details>`"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn reasoning_closer_in_fenced_block_is_ignored() {
    let r = "```\n</details>\n```\nafter";
    let conv = thinking_conv(r, "done");
    let once = to_markdown(&conv);
    assert!(once.contains("```\n</details>\n```"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning.as_deref(), Some(r));
    assert_eq!(back.turns[1].content, "done");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn no_thinking_tag_wrapper_stays_content() {
    // Without `thinking: true` the whole body is verbatim content, even if
    // it looks exactly like a Thinking wrapper.
    let body = "<details>\n<summary>Thinking</summary>\n\nfake\n\n</details>\n\nreal answer";
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "q".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Assistant,
                content: body.to_string(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(!once.contains("thinking: true"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].reasoning, None);
    assert_eq!(back.turns[1].content, body);
    assert_eq!(to_markdown(&back), once);
}

/// A user preamble so the tested assistant section is not the first one.
const USER_PREAMBLE: &str = "**User** | 2026-11-22 14:30\n\nq\n\n---\n";

#[test]
fn thinking_tag_rejects_non_true_value() {
    let input = format!("{USER_PREAMBLE}**Assistant** | thinking: false\n\nhi\n");
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: 'thinking' metadata must be the literal value 'true'"
    );
}

#[test]
fn thinking_tag_rejects_non_assistant_role() {
    let input = "**User** | thinking: true\n\nhi\n";
    let err = parse(input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 1: 'thinking' metadata is only allowed on **Assistant** headers"
    );
}

#[test]
fn thinking_tag_requires_lowercase_details_opener() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | thinking: true\n\n<DETAILS>\n<summary>Thinking</summary>\n\nr\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 8: expected a bare '<details>' line to open the reasoning block"
    );
}

#[test]
fn thinking_tag_requires_exact_summary_line() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | thinking: true\n\n<details>\n<summary>thinking</summary>\n\nr\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 9: expected a bare '<summary>Thinking</summary>' line after the opening tag"
    );
}

#[test]
fn thinking_tag_requires_closing_details_line() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | thinking: true\n\n<details>\n<summary>Thinking</summary>\n\nreasoning...\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 11: unclosed reasoning block; expected a closing '</details>' line"
    );
}

#[test]
fn thinking_tag_rejects_even_backslash_count() {
    // Two backslashes before the closer: an even count is a format error.
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | thinking: true\n\n<details>\n<summary>Thinking</summary>\n\n\\\\</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 11: even number of backslashes before a <details> tag; expected an odd count"
    );
}

// ---------------------------------------------------------------------------
// Milestone 2: tool turns and message assembly
// ---------------------------------------------------------------------------

fn tool_turn_fixture(name: &str, id: &str, output: &str) -> Turn {
    Turn {
        role: Role::Tool,
        content: tool_turn_body(output),
        meta: TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags: vec![
                ("tool".into(), name.into()),
                ("id".into(), id.into()),
            ],
        },
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

#[test]
fn tool_turn_round_trips() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "List the files.".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            tool_turn_fixture("list_files", "call_1", "notes.md  120\nsrc/main.rs  4567"),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 2);
    let data = parse_tool_turn(&back.turns[1]).expect("well-formed tool turn");
    assert_eq!(data.name, "list_files");
    assert_eq!(data.id, "call_1");
    assert_eq!(data.output, "notes.md  120\nsrc/main.rs  4567");
    // Stable under re-serialization.
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn tool_turn_output_with_details_and_backticks_round_trips() {
    // The output contains a literal `</details>` and a triple-backtick
    // fence; the tilde fence must outlast both.
    let output = "look:\n```\n---\n**User** | 2026-11-22 14:30\n```\nand </details> inside";
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "run it".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            tool_turn_fixture("bash", "call_9", output),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 2);
    let data = parse_tool_turn(&back.turns[1]).expect("well-formed tool turn");
    assert_eq!(data.output, output);
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn tool_turn_output_with_long_tilde_run_round_trips() {
    // A five-tilde run in the output forces a six-tilde fence.
    let output = "line one\n~~~~~\nline two";
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "read it".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            tool_turn_fixture("read_file", "c1", output),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("~~~~~~text"), "fence must exceed the embedded run:\n{once}");
    let back = parse(&once).unwrap();
    let data = parse_tool_turn(&back.turns[1]).expect("well-formed tool turn");
    assert_eq!(data.output, output);
}

#[test]
fn parse_tool_turn_rejects_malformed() {
    // Missing the `id:` tag.
    let no_id = Turn {
        role: Role::Tool,
        content: tool_turn_body("out"),
        meta: TurnMeta {
            timestamp: None,
            tags: vec![("tool".into(), "bash".into())],
        },
        reasoning: None,
        tool_calls: Vec::new(),
    };
    assert!(parse_tool_turn(&no_id).is_none());
    // Not a tool role.
    let user = Turn {
        role: Role::User,
        content: tool_turn_body("out"),
        meta: TurnMeta::default(),
        reasoning: None,
        tool_calls: Vec::new(),
    };
    assert!(parse_tool_turn(&user).is_none());
    // Body without a summary line or fence.
    let no_body = Turn {
        role: Role::Tool,
        content: "just some text".into(),
        meta: TurnMeta {
            timestamp: None,
            tags: vec![
                ("tool".into(), "bash".into()),
                ("id".into(), "c1".into()),
            ],
        },
        reasoning: None,
        tool_calls: Vec::new(),
    };
    assert!(parse_tool_turn(&no_body).is_none());
}

#[test]
fn to_messages_projects_assistant_tool_calls_and_tool_turns() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "List the files.".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Assistant,
                content: "Let me check.".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: vec![AssistantToolCall {
                    id: "call_1".into(),
                    name: "list_files".into(),
                    arguments: r#"{"subdir": "."}"#.into(),
                }],
            },
            tool_turn_fixture("list_files", "call_1", "a.md  10"),
        ],
        ..Default::default()
    };
    let messages = conv.to_messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content.as_deref(), Some("List the files."));
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content.as_deref(), Some("Let me check."));
    assert_eq!(messages[1].tool_calls.len(), 1);
    assert_eq!(messages[1].tool_calls[0].id, "call_1");
    assert_eq!(messages[1].tool_calls[0].function.name, "list_files");
    assert_eq!(
        messages[1].tool_calls[0].function.arguments,
        r#"{"subdir": "."}"#
    );
    assert_eq!(messages[2].role, "tool");
    assert_eq!(messages[2].content.as_deref(), Some("a.md  10"));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn to_messages_skips_system_sections() {
    // The system text never appears in the wire messages; it is folded into
    // the single leading system message by the caller.
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::System,
                content: "Be terse.".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Assistant,
                content: "hello".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let messages = conv.to_messages();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[1].role, "assistant");
    assert!(!messages.iter().any(|m| m.role == "system"));
}

#[test]
fn to_messages_projects_leading_tool_turn_without_synthesis() {
    // A file that starts with a tool turn (hand-edited) projects to a bare
    // `role: "tool"` message: no assistant message is synthesized. The
    // projection stays pure; payload repair belongs to the request-prep
    // layer (see `llm::complete_tool_calls`).
    let conv = Conversation {
        turns: vec![tool_turn_fixture("read_file", "c7", "data")],
        ..Default::default()
    };
    let messages = conv.to_messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "tool");
    assert_eq!(messages[0].content.as_deref(), Some("data"));
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("c7"));
}

#[test]
fn to_messages_skips_malformed_tool_turns() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Tool,
                content: "not a tool turn".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let messages = conv.to_messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
}

#[test]
fn frontmatter_tool_fields_round_trip() {
    let conv = Conversation {
        frontmatter: Frontmatter {
            tools: false,
            scope: Scope::Vault,
            network: true,
            allow_shell: true,
            ..Default::default()
        },
        had_frontmatter: true,
        turns: vec![Turn {
            role: Role::User,
            content: "hi".into(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
    };
    let md = to_markdown(&conv);
    assert!(md.contains("tools: false"), "{md}");
    assert!(md.contains("scope: vault"), "{md}");
    assert!(md.contains("network: true"), "{md}");
    assert!(md.contains("allow_shell: true"), "{md}");
    let back = parse(&md).unwrap();
    assert_eq!(back.frontmatter, conv.frontmatter);
}

#[test]
fn frontmatter_defaults_are_omitted() {
    let conv = Conversation {
        had_frontmatter: true,
        turns: vec![Turn {
            role: Role::User,
            content: "hi".into(),
            meta: TurnMeta::default(),
            reasoning: None,
            tool_calls: Vec::new(),
        }],
        ..Default::default()
    };
    let md = to_markdown(&conv);
    // An all-default frontmatter serializes as an empty block.
    assert!(md.starts_with("---\n---\n"), "{md}");
    assert!(!md.contains("tools:"), "{md}");
    assert!(!md.contains("scope:"), "{md}");
    assert!(!md.contains("network:"), "{md}");
    assert!(!md.contains("allow_shell:"), "{md}");
    let back = parse(&md).unwrap();
    assert_eq!(back.frontmatter, Frontmatter::default());
}

#[test]
fn frontmatter_legacy_tools_list_reads_as_enabled() {
    let md = "---\ntools:\n  - name: shell\n---\n**User**\n\nhi\n";
    let conv = parse(md).unwrap();
    assert!(conv.frontmatter.tools);
}

// ---------------------------------------------------------------------------
// Milestone 3: explicit tool call persistence (`tool_calls: N`)
// ---------------------------------------------------------------------------

/// One canonical trailing Tool Call block, for hand-written fixtures.
fn tc_block(name: &str, id: &str, args: &str) -> String {
    format!(
        "<details>\n<summary>Tool Call: <code>{name}</code> (<code>{id}</code>)</summary>\n\n~~~~json\n{args}\n~~~~\n\n</details>"
    )
}

/// An assistant turn carrying the `model` tag and — when `calls` is
/// non-empty — the `tool_calls: <N>` tag.
fn assistant_turn_fixture(content: &str, calls: &[AssistantToolCall]) -> Turn {
    let mut tags = vec![("model".to_string(), "m1".to_string())];
    if !calls.is_empty() {
        tags.push(("tool_calls".to_string(), calls.len().to_string()));
    }
    Turn {
        role: Role::Assistant,
        content: content.to_string(),
        meta: TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags,
        },
        reasoning: None,
        tool_calls: calls.to_vec(),
    }
}

fn user_turn_fixture(content: &str) -> Turn {
    Turn {
        role: Role::User,
        content: content.into(),
        meta: TurnMeta::default(),
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

#[test]
fn tool_call_single_round_trips() {
    let call = AssistantToolCall {
        id: "call_1".into(),
        name: "list_files".into(),
        arguments: r#"{"subdir": "."}"#.into(),
    };
    let conv = Conversation {
        turns: vec![
            user_turn_fixture("List the files."),
            assistant_turn_fixture("Let me check.", &[call.clone()]),
            tool_turn_fixture("list_files", "call_1", "a.md  10"),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(
        once.contains("**Assistant** | 2026-11-22 14:31 | model: m1 | tool_calls: 1"),
        "{once}"
    );
    let back = parse(&once).unwrap();
    assert_eq!(back, conv);
}

#[test]
fn tool_call_parallel_round_trips() {
    let calls = vec![
        AssistantToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path": "a.md"}"#.into(),
        },
        AssistantToolCall {
            id: "call_2".into(),
            name: "bash".into(),
            arguments: r#"{"command": "ls"}"#.into(),
        },
    ];
    let conv = Conversation {
        turns: vec![
            user_turn_fixture("Check both."),
            assistant_turn_fixture("Checking both.", &calls),
            tool_turn_fixture("read_file", "call_1", "content-a"),
            tool_turn_fixture("bash", "call_2", "a b"),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back, conv);
    // Order is preserved.
    assert_eq!(back.turns[1].tool_calls[0].id, "call_1");
    assert_eq!(back.turns[1].tool_calls[1].id, "call_2");
}

#[test]
fn tool_call_with_thinking_and_content_round_trips() {
    let calls = vec![
        AssistantToolCall {
            id: "call_1".into(),
            name: "list_files".into(),
            arguments: r#"{"subdir": "."}"#.into(),
        },
        AssistantToolCall {
            id: "call_2".into(),
            name: "read_file".into(),
            arguments: r#"{"path": "a.md"}"#.into(),
        },
    ];
    let conv = Conversation {
        turns: vec![
            user_turn_fixture("List and read."),
            Turn {
                role: Role::Assistant,
                content: "Here you go.".into(),
                meta: TurnMeta {
                    timestamp: Some("2026-11-22 14:31".into()),
                    tags: vec![
                        ("model".to_string(), "m1".to_string()),
                        ("thinking".to_string(), "true".to_string()),
                        ("tool_calls".to_string(), "2".to_string()),
                    ],
                },
                reasoning: Some("Let me think.".into()),
                tool_calls: calls.clone(),
            },
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(
        once.contains(
            "**Assistant** | 2026-11-22 14:31 | model: m1 | thinking: true | tool_calls: 2"
        ),
        "{once}"
    );
    let back = parse(&once).unwrap();
    assert_eq!(back, conv);
    assert_eq!(back.turns[1].reasoning.as_deref(), Some("Let me think."));
}

#[test]
fn tool_call_with_thinking_without_content_round_trips() {
    // A pure tool-call response from a thinking model: reasoning and calls,
    // no visible answer.
    let call = AssistantToolCall {
        id: "call_1".into(),
        name: "bash".into(),
        arguments: r#"{"command": "ls"}"#.into(),
    };
    let conv = Conversation {
        turns: vec![
            user_turn_fixture("List."),
            Turn {
                role: Role::Assistant,
                content: String::new(),
                meta: TurnMeta {
                    timestamp: Some("2026-11-22 14:31".into()),
                    tags: vec![
                        ("thinking".to_string(), "true".to_string()),
                        ("tool_calls".to_string(), "1".to_string()),
                    ],
                },
                reasoning: Some("Quick check.".into()),
                tool_calls: vec![call.clone()],
            },
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back, conv);
    assert_eq!(back.turns[1].content, "");
    assert_eq!(back.turns[1].tool_calls, vec![call]);
}

#[test]
fn tool_call_extraction_ignores_fake_example_in_content() {
    // The visible content quotes a literal Tool Call block; extraction must
    // take the real trailing block only and leave the example verbatim.
    let fake = tc_block("fake_tool", "fake_1", r#"{"x": 1}"#);
    let content = format!(
        "Here is what a tool call looks like:\n\n{fake}\n\nAnd here is the real one below."
    );
    let real = AssistantToolCall {
        id: "real_1".into(),
        name: "read_file".into(),
        arguments: r#"{"path": "b.md"}"#.into(),
    };
    let conv = Conversation {
        turns: vec![user_turn_fixture("Show me."), assistant_turn_fixture(&content, &[real.clone()])],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back.turns[1].tool_calls, vec![real]);
    assert_eq!(back.turns[1].content, content);
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn tool_call_block_fence_grows_with_embedded_tilde_runs() {
    // A five-tilde run at the start of an arguments line forces a
    // six-tilde fence.
    let call = AssistantToolCall {
        id: "c1".into(),
        name: "write_file".into(),
        arguments: "a\n~~~~~\nb".into(),
    };
    let block = tool_call_block(&call);
    assert!(block.contains("~~~~~~json"), "{block}");
    let input = format!("{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n{block}\n");
    let conv = parse(&input).unwrap();
    assert_eq!(conv.turns[1].tool_calls, vec![call]);
    assert_eq!(to_markdown(&conv), input);
}

#[test]
fn to_messages_is_pure_and_request_layer_recovers_dangling_calls() {
    use crate::llm::{complete_tool_calls, CANCELLED_TOOL_OUTPUT};

    let calls = vec![
        AssistantToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path": "a.md"}"#.into(),
        },
        AssistantToolCall {
            id: "call_2".into(),
            name: "bash".into(),
            arguments: r#"{"command": "ls"}"#.into(),
        },
    ];
    // Execution was cancelled after the first result: the file carries the
    // assistant turn (two calls) and a single **Tool** turn.
    let conv = Conversation {
        turns: vec![
            user_turn_fixture("Do both."),
            assistant_turn_fixture("Working on it.", &calls),
            tool_turn_fixture("read_file", "call_1", "content-a"),
        ],
        ..Default::default()
    };
    let mut messages = conv.to_messages();
    // Pure projection: the dangling call stays unanswered.
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].tool_calls.len(), 2);
    assert_eq!(messages[2].role, "tool");
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
    // The request-prep layer answers the dangling call.
    complete_tool_calls(&mut messages);
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[3].role, "tool");
    assert_eq!(messages[3].content.as_deref(), Some(CANCELLED_TOOL_OUTPUT));
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_2"));
}

#[test]
fn tool_calls_zero_rejected() {
    let input = format!("{USER_PREAMBLE}**Assistant** | tool_calls: 0\n\nhi\n");
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: 'tool_calls' metadata must be a positive integer"
    );
}

#[test]
fn tool_calls_non_integer_rejected() {
    for v in ["true", "-1", "1.5", ""] {
        let input = format!("{USER_PREAMBLE}**Assistant** | tool_calls: {v}\n\nhi\n");
        let err = parse(&input).unwrap_err();
        assert_eq!(
            err.to_string(),
            "line 6: 'tool_calls' metadata must be a positive integer",
            "value: {v:?}"
        );
    }
}

#[test]
fn tool_calls_on_non_assistant_header_rejected() {
    let input = "**User** | 2026-11-22 14:30\n\nq\n\n---\n**User** | tool_calls: 1\n\nhi\n";
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: 'tool_calls' metadata is only allowed on **Assistant** headers"
    );
}

#[test]
fn tool_calls_empty_body_rejected() {
    let input = format!("{USER_PREAMBLE}**Assistant** | tool_calls: 2\n\n");
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 7: expected 2 tool call blocks at the end of the section; found fewer"
    );
}

#[test]
fn tool_calls_fewer_blocks_than_tag_rejected() {
    let input = format!("{USER_PREAMBLE}**Assistant** | tool_calls: 2\n\n{}", tc_block("a", "c1", "{}"));
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 15: expected 2 tool call blocks at the end of the section; found fewer"
    );
}

#[test]
fn tool_call_block_missing_closer_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n<details>\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\n~~~~json\n{{}}\n~~~~\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 13: expected a closing '</details>' line to end the tool call block"
    );
}

#[test]
fn tool_call_block_missing_fence_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n<details>\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\nnot a fence\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 13: malformed tool call block; expected a tilde-fenced arguments block"
    );
}

#[test]
fn tool_call_block_missing_summary_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n<details>\n\n~~~~json\n{{}}\n~~~~\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 14: malformed tool call block; expected a Tool Call summary line"
    );
}

#[test]
fn tool_call_block_missing_opening_tag_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\n~~~~json\n{{}}\n~~~~\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 8: malformed tool call block; expected an opening '<details>' line"
    );
}

#[test]
fn tool_call_block_invalid_summary_rejected() {
    // An empty name fails the summary grammar, so the block is malformed.
    let input = format!(
        "{USER_PREAMBLE}**Assistant** | tool_calls: 1\n\n<details>\n<summary>Tool Call: <code></code> (<code>c1</code>)</summary>\n\n~~~~json\n{{}}\n~~~~\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 15: malformed tool call block; expected a Tool Call summary line"
    );
}

// ---------------------------------------------------------------------------
// Strict header metadata keys & strict **Tool** turn validation
// ---------------------------------------------------------------------------

#[test]
fn duplicate_metadata_key_rejected() {
    let input = "**User** | model: a | model: b\n\nhi\n";
    let err = parse(input).unwrap_err();
    assert_eq!(err.to_string(), "line 1: duplicate metadata key 'model' in header");
}

#[test]
fn duplicate_metadata_key_non_adjacent_rejected() {
    let input = "**User** | model: a | note: x | model: b\n\nhi\n";
    let err = parse(input).unwrap_err();
    assert_eq!(err.to_string(), "line 1: duplicate metadata key 'model' in header");
}

#[test]
fn duplicate_metadata_key_on_tool_header_rejected() {
    let input = "**User**\n\nq\n\n---\n**Tool** | tool: a | id: c1 | tool: b\n\n";
    let err = parse(input).unwrap_err();
    assert_eq!(err.to_string(), "line 6: duplicate metadata key 'tool' in header");
}

#[test]
fn duplicate_thinking_key_rejected() {
    // The header error fires before any body validation.
    let input = "**Assistant** | thinking: true | thinking: true\n\nhi\n";
    let err = parse(input).unwrap_err();
    assert_eq!(err.to_string(), "line 1: duplicate metadata key 'thinking' in header");
}

#[test]
fn tool_tag_on_non_tool_header_rejected() {
    for header in ["**User** | tool: bash", "**Assistant** | tool: bash", "**System** | tool: bash"] {
        let input = format!("{header}\n\nhi\n");
        let err = parse(&input).unwrap_err();
        assert_eq!(
            err.to_string(),
            "line 1: 'tool' metadata is only allowed on **Tool** headers",
            "header: {header}"
        );
    }
}

#[test]
fn id_tag_on_non_tool_header_rejected() {
    for header in ["**User** | id: c1", "**Assistant** | id: c1", "**System** | id: c1"] {
        let input = format!("{header}\n\nhi\n");
        let err = parse(&input).unwrap_err();
        assert_eq!(
            err.to_string(),
            "line 1: 'id' metadata is only allowed on **Tool** headers",
            "header: {header}"
        );
    }
}

#[test]
fn tool_turn_missing_tool_tag_rejected() {
    let input = format!("{USER_PREAMBLE}**Tool** | id: c1\n\n{}", tool_turn_body("out"));
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: **Tool** header requires a non-empty 'tool:' metadata tag"
    );
}

#[test]
fn tool_turn_missing_id_tag_rejected() {
    let input = format!("{USER_PREAMBLE}**Tool** | tool: bash\n\n{}", tool_turn_body("out"));
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: **Tool** header requires a non-empty 'id:' metadata tag"
    );
}

#[test]
fn tool_turn_empty_tool_tag_rejected() {
    let input = format!("{USER_PREAMBLE}**Tool** | tool: | id: c1\n\n{}", tool_turn_body("out"));
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: **Tool** header requires a non-empty 'tool:' metadata tag"
    );
}

#[test]
fn legacy_free_form_tool_turn_rejected() {
    // The pre-strict free-form shape (no tool:/id: tags, bare output) is now
    // a hard error.
    let input = "**User**\n\nhi\n\n---\n**Tool** | model: m\n\nexit code 0\n";
    let err = parse(input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 6: **Tool** header requires a non-empty 'tool:' metadata tag"
    );
}

#[test]
fn tool_turn_empty_body_rejected() {
    let input = format!("{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n");
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 7: expected a bare '<details>' line to open the tool output block"
    );
}

#[test]
fn tool_turn_missing_summary_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n\n~~~~text\nout\n~~~~\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 9: expected a bare '<summary>Output</summary>' line after the opening tag"
    );
}

#[test]
fn tool_turn_missing_fence_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n<summary>Output</summary>\n\nnot a fence\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 11: expected a tilde-fenced output block in the tool turn body"
    );
}

#[test]
fn tool_turn_no_fence_at_all_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n<summary>Output</summary>\n\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 11: expected a tilde-fenced output block in the tool turn body"
    );
}

#[test]
fn tool_turn_unclosed_fence_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n<summary>Output</summary>\n\n~~~~text\nout\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 12: unclosed tool output block; expected a closing tilde fence"
    );
}

#[test]
fn tool_turn_missing_closer_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n<summary>Output</summary>\n\n~~~~text\nout\n~~~~\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 13: expected a bare '</details>' line to close the tool output block"
    );
}

#[test]
fn tool_turn_trailing_content_after_closer_rejected() {
    let input = format!(
        "{USER_PREAMBLE}**Tool** | tool: bash | id: c1\n\n<details>\n<summary>Output</summary>\n\n~~~~text\nout\n~~~~\nextra\n</details>\n"
    );
    let err = parse(&input).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 14: expected a bare '</details>' line to close the tool output block"
    );
}

#[test]
fn tool_turn_body_with_extra_blanks_round_trips() {
    // Blank lines between the wrapper's pieces are optional; a hand-edited
    // wrapper with extra blanks parses and round-trips verbatim.
    let content = "<details>\n<summary>Output</summary>\n\n\n~~~~text\nout\n~~~~\n\n\n</details>";
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "run it".into(),
                meta: TurnMeta::default(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
            Turn {
                role: Role::Tool,
                content: content.into(),
                meta: TurnMeta {
                    timestamp: Some("2026-11-22 14:31".into()),
                    tags: vec![
                        ("tool".to_string(), "bash".to_string()),
                        ("id".to_string(), "c1".to_string()),
                    ],
                },
                reasoning: None,
                tool_calls: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back, conv);
    let data = parse_tool_turn(&back.turns[1]).expect("well-formed tool turn");
    assert_eq!(data.output, "out");
}
