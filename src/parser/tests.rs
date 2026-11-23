//! Tests for [`super`]: parsing, serialization, metadata, and escaping.

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

---
**User** · *2026-11-22 14:30*

Hello!

---
**Assistant** · *2026-11-22 14:31 · model: local-model*

Hi there.

---
**Tool** · *2026-11-22 14:31 · model: local-model*

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
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:30".into()),
            tags: vec![],
        }
    );
    assert_eq!(conv.turns[1].role, Role::Assistant);
    assert_eq!(conv.turns[1].content, "Hi there.");
    assert_eq!(
        conv.turns[1].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags: vec![("model".into(), "local-model".into())],
        }
    );
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
---
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
    let input = "---\n  ** USER **  \n\ntext\n";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns[0].role, Role::User);
    assert_eq!(conv.turns[0].content, "text");
}

#[test]
fn parse_role_line_case_insensitive() {
    let conv = parse("---\n**assistant**\n\nhi\n").unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].role, Role::Assistant);
}

#[test]
fn parse_metadata_optional_variants() {
    // No metadata at all.
    let conv = parse("---\n**User**\n\nhi\n").unwrap();
    assert_eq!(conv.turns[0].meta, TurnMeta::default());

    // Timestamp only (model tag missing).
    let conv = parse("---\n**User** · *2026-11-22 14:30*\n\nhi\n").unwrap();
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: Some("2026-11-22 14:30".into()),
            tags: vec![],
        }
    );

    // Tags only (no timestamp).
    let conv = parse("---\n**User** · *model: m1 · endpoint: local*\n\nhi\n").unwrap();
    assert_eq!(
        conv.turns[0].meta,
        TurnMeta {
            timestamp: None,
            tags: vec![("model".into(), "m1".into()), ("endpoint".into(), "local".into())],
        }
    );

    // Timestamp plus several tags; tag values may contain colons.
    let conv = parse(
        "---\n**Assistant** · *2026-11-22 14:31 · model: unsloth/Qwen:Q8 · note: a: b*\n\nhi\n",
    )
    .unwrap();
    assert_eq!(
        conv.turns[0].meta,
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
    let input = "---\n**User**\n\nbefore\n---\nafter\n";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].content, "before\n---\nafter");
}

#[test]
fn parse_role_line_without_hr_is_content() {
    let input = "**User** · *2026-11-22 14:30*\n\nnot a turn\n";
    let conv = parse(input).unwrap();
    assert!(conv.turns.is_empty());
    assert_eq!(conv.preamble, "**User** · *2026-11-22 14:30*\n\nnot a turn");
}

#[test]
fn parse_boundary_inside_code_block_ignored() {
    let input = "\
---
**User**

```markdown
---
**Assistant** · *2026-11-22 14:31*

this is just text in a fence
```

still user content
";
    let conv = parse(input).unwrap();
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].role, Role::User);
    assert_eq!(
        conv.turns[0].content,
        "```markdown\n---\n**Assistant** · *2026-11-22 14:31*\n\nthis is just text in a fence\n```\n\nstill user content"
    );
}

#[test]
fn parse_file_starting_with_boundary_is_not_frontmatter() {
    let input = "---\n**User** · *2026-11-22 14:30*\n\nHello\n";
    let conv = parse(input).unwrap();
    assert!(!conv.had_frontmatter);
    assert!(conv.turns.len() == 1);
    assert_eq!(conv.turns[0].role, Role::User);
    assert_eq!(conv.turns[0].content, "Hello");
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

---
**User** · *2026-11-22 14:30*

Hello!

---
**Assistant** · *2026-11-22 14:31 · model: local-model*

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
            meta: TurnMeta {
                timestamp: Some("2026-11-22 14:30".into()),
                tags: vec![],
            },
        }],
    };
    let md = to_markdown(&conv);
    assert_eq!(
        md,
        "---\nmodel: m\nsystem_prompt: Be terse.\n---\n\n---\n**User** · *2026-11-22 14:30*\n\nHello!\n"
    );
}

#[test]
fn serialize_omits_missing_metadata() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: "Hi".into(),
            meta: TurnMeta::default(),
        }],
        ..Default::default()
    };
    assert_eq!(to_markdown(&conv), "---\n**Assistant**\n\nHi\n");
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

#[test]
fn escape_boundaries_escapes_unfenced_collision() {
    let content = "look at this:\n---\n**User** · *2026-11-22 14:30*\n\nand more";
    assert_eq!(
        escape_boundaries(content),
        "look at this:\n---\n\\**User** · *2026-11-22 14:30*\n\nand more"
    );
}

#[test]
fn escape_boundaries_leaves_non_collisions_alone() {
    // No role line after the hr.
    assert_eq!(escape_boundaries("---\nplain text"), "---\nplain text");
    // Role line without a preceding hr.
    assert_eq!(
        escape_boundaries("**User** · *2026-11-22 14:30*"),
        "**User** · *2026-11-22 14:30*"
    );
    // Unknown role.
    assert_eq!(escape_boundaries("---\n**Narrator**"), "---\n**Narrator**");
    // Already escaped (stable on re-serialization).
    let once = escape_boundaries("---\n**User**");
    assert_eq!(escape_boundaries(&once), once);
}

#[test]
fn escape_boundaries_skips_fenced_content() {
    let content = "```\n---\n**User** · *2026-11-22 14:30*\n```\nreal text";
    assert_eq!(escape_boundaries(content), content);
}

#[test]
fn escape_boundaries_preserves_indent_of_role_line() {
    let content = "---\n  **User**";
    assert_eq!(escape_boundaries(content), "---\n  \\**User**");
}

#[test]
fn escaped_collision_round_trips_as_content() {
    // A message containing the raw boundary pattern must survive a round
    // trip: serialized escaped, parsed back as content, stable thereafter.
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: "---\n**Assistant** · *2026-11-22 14:31*".into(),
            meta: TurnMeta::default(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("\\**Assistant**"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].role, Role::User);
    assert_eq!(back.turns[0].content, "---\n\\**Assistant** · *2026-11-22 14:31*");
    assert_eq!(to_markdown(&back), once);
}

#[test]
fn turn_meta_render() {
    assert_eq!(TurnMeta::default().render(), None);
    let meta = TurnMeta {
        timestamp: Some("2026-11-22 14:30".into()),
        tags: vec![("model".into(), "m".into()), ("flag".into(), String::new())],
    };
    assert_eq!(meta.render().as_deref(), Some("2026-11-22 14:30 · model: m · flag"));
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
