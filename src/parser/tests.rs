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
**Tool** | 2026-11-22 14:31 | model: local-model

exit code 0
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
    assert_eq!(conv.turns[3].content, "exit code 0");
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
    // Lines: 1 **User**, 2 blank, 3 hi, 4 ---, 5 **System**.
    let err = parse("**User**\n\nhi\n---\n**System**\n\nlate\n").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("line 5"), "{msg}");
    assert!(msg.contains("**System**"), "{msg}");
}

#[test]
fn parse_header_rejects_wrong_separators() {
    // The legacy `·` separator is no longer a header.
    for (input, at) in [
        ("**User** · *2026-11-22 14:30*\n\nhi\n", "line 1"),
        ("**User**\n\nhi\n---\n**Assistant** - 2026-11-22 14:31\n\nlate\n", "line 5"),
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
    let err = parse("**User**\n\nhi\n---\n**Assistant** | 2026-11-22 14:31 | model\n\nlate\n")
        .unwrap_err();
    assert!(err.to_string().contains("line 5"), "{err}");

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
    let conv = parse("**User**\n\nhello\n---\n**assistant**\n\nhi\n").unwrap();
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
        "**User**\n\nhello\n---\n**Assistant** | 2026-11-22 14:31 | model: unsloth/Qwen:Q8 | note: a: b\n\nhi\n",
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
            },
            Turn {
                role: Role::User,
                content: "Hello!".into(),
                meta: TurnMeta {
                    timestamp: Some("2026-11-22 14:30".into()),
                    tags: vec![],
                },
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
            },
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
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
    let content = "look at this:\n---\n**User** | 2026-11-22 14:30\n\nand more";
    assert_eq!(
        escape_boundaries(content),
        "look at this:\n---\n\\**User** | 2026-11-22 14:30\n\nand more"
    );
    // A would-be header with *invalid* metadata is escaped too: on the next
    // parse it would be a format error, not content.
    let content = "note:\n---\n**User** · *2026-11-22 14:30*\n\nmore";
    assert_eq!(
        escape_boundaries(content),
        "note:\n---\n\\**User** · *2026-11-22 14:30*\n\nmore"
    );
}

#[test]
fn escape_boundaries_leaves_non_collisions_alone() {
    // No role line after the hr.
    assert_eq!(escape_boundaries("---\nplain text"), "---\nplain text");
    // Role line without a preceding hr.
    assert_eq!(
        escape_boundaries("**User** | 2026-11-22 14:30"),
        "**User** | 2026-11-22 14:30"
    );
    // Unknown role.
    assert_eq!(escape_boundaries("---\n**Narrator**"), "---\n**Narrator**");
    // Already escaped (stable on re-serialization).
    let once = escape_boundaries("---\n**User**");
    assert_eq!(escape_boundaries(&once), once);
}

#[test]
fn escape_boundaries_skips_fenced_content() {
    let content = "```\n---\n**User** | 2026-11-22 14:30\n```\nreal text";
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
            content: "---\n**Assistant** | 2026-11-22 14:31".into(),
            meta: TurnMeta::default(),
        }],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    assert!(once.contains("\\**Assistant**"));
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 1);
    assert_eq!(back.turns[0].role, Role::User);
    assert_eq!(back.turns[0].content, "---\n\\**Assistant** | 2026-11-22 14:31");
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

#[test]
fn split_reasoning_no_tags() {
    let (reason, answer) = split_reasoning("just an answer");
    assert_eq!(reason, None);
    assert_eq!(answer, "just an answer");
}

#[test]
fn split_reasoning_full_block() {
    let content = "think\nstep one\nstep two\nendthink\n\nthe answer";
    let (reason, answer) = split_reasoning(content);
    assert_eq!(reason.as_deref(), Some("step one\nstep two"));
    assert_eq!(answer, "the answer");
}

#[test]
fn split_reasoning_unclosed_tag_is_content() {
    let content = "think\nstill thinking...\nmore text";
    let (reason, answer) = split_reasoning(content);
    assert_eq!(reason, None);
    assert_eq!(answer, content);
}

#[test]
fn split_reasoning_word_in_prose_is_ignored() {
    // "think" mid-line is not a tag.
    let content = "you should think\nabout this\nendthink\n\nhmm";
    let (reason, answer) = split_reasoning(content);
    assert_eq!(reason, None);
    assert_eq!(answer, content);
}

#[test]
fn split_reasoning_text_before_block_goes_to_answer() {
    let content = "preamble line\nthink\nreason\nendthink\n\nfinal";
    let (reason, answer) = split_reasoning(content);
    assert_eq!(reason.as_deref(), Some("reason"));
    assert_eq!(answer, "preamble line\n\nfinal");
}

#[test]
fn with_reasoning_round_trips_through_split() {
    let rendered = with_reasoning(Some("because 6*9"), "54");
    assert_eq!(rendered, "think\nbecause 6*9\nendthink\n\n54");
    let (reason, answer) = split_reasoning(&rendered);
    assert_eq!(reason.as_deref(), Some("because 6*9"));
    assert_eq!(answer, "54");
}

#[test]
fn with_reasoning_empty_reasoning_is_plain_answer() {
    assert_eq!(with_reasoning(Some("   "), "hi"), "hi");
    assert_eq!(with_reasoning(None, "  hi  "), "hi");
}

// ---------------------------------------------------------------------------
// Milestone 2: tool turns and message assembly
// ---------------------------------------------------------------------------

fn tool_turn_fixture(name: &str, id: &str, arguments: &str, output: &str) -> Turn {
    Turn {
        role: Role::Tool,
        content: tool_turn_body(name, arguments, output),
        meta: TurnMeta {
            timestamp: Some("2026-11-22 14:31".into()),
            tags: vec![
                ("tool".into(), name.into()),
                ("id".into(), id.into()),
            ],
        },
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
            },
            tool_turn_fixture(
                "list_files",
                "call_1",
                r#"{"subdir": "."}"#,
                "notes.md  120\nsrc/main.rs  4567",
            ),
        ],
        ..Default::default()
    };
    let once = to_markdown(&conv);
    let back = parse(&once).unwrap();
    assert_eq!(back.turns.len(), 2);
    let data = parse_tool_turn(&back.turns[1]).expect("well-formed tool turn");
    assert_eq!(data.name, "list_files");
    assert_eq!(data.id, "call_1");
    assert_eq!(data.arguments, r#"{"subdir": "."}"#);
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
            },
            tool_turn_fixture("bash", "call_9", r#"{"command": "ls"}"#, output),
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
            },
            tool_turn_fixture("read_file", "c1", r#"{"path": "a.md"}"#, output),
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
        content: tool_turn_body("bash", "{}", "out"),
        meta: TurnMeta {
            timestamp: None,
            tags: vec![("tool".into(), "bash".into())],
        },
    };
    assert!(parse_tool_turn(&no_id).is_none());
    // Not a tool role.
    let user = Turn {
        role: Role::User,
        content: tool_turn_body("bash", "{}", "out"),
        meta: TurnMeta::default(),
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
    };
    assert!(parse_tool_turn(&no_body).is_none());
}

#[test]
fn to_messages_folds_tool_turns_into_assistant_and_tool() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "List the files.".into(),
                meta: TurnMeta::default(),
            },
            Turn {
                role: Role::Assistant,
                content: "Let me check.".into(),
                meta: TurnMeta::default(),
            },
            tool_turn_fixture("list_files", "call_1", r#"{"subdir": "."}"#, "a.md  10"),
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
            },
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
            },
            Turn {
                role: Role::Assistant,
                content: "hello".into(),
                meta: TurnMeta::default(),
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
fn to_messages_synthesizes_assistant_when_tool_turn_leads() {
    // A file that starts with a tool turn (hand-edited) must still yield a
    // valid payload: an assistant message with content None carries the call.
    let conv = Conversation {
        turns: vec![tool_turn_fixture("read_file", "c7", r#"{"path": "x"}"#, "data")],
        ..Default::default()
    };
    let messages = conv.to_messages();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "assistant");
    assert_eq!(messages[0].content, None);
    assert_eq!(messages[0].tool_calls.len(), 1);
    assert_eq!(messages[0].tool_calls[0].id, "c7");
    assert_eq!(messages[1].role, "tool");
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("c7"));
}

#[test]
fn to_messages_skips_malformed_tool_turns() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::User,
                content: "hi".into(),
                meta: TurnMeta::default(),
            },
            Turn {
                role: Role::Tool,
                content: "not a tool turn".into(),
                meta: TurnMeta::default(),
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
