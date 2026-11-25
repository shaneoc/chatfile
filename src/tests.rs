use super::*;

#[test]
fn new_creates_template_and_refuses_overwrite() {
    let dir = std::env::temp_dir().join(format!("chatfile-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let file = dir.join("a").join("b").join("conv.md");

    new(&file).unwrap();
    // A plain `.md` name is normalized to `.chat.md`.
    let created = dir.join("a").join("b").join("conv.chat.md");
    assert!(created.exists());
    let content = std::fs::read_to_string(&created).unwrap();
    assert!(content.starts_with("---\nmodel: local-model\n"));
    assert!(content.contains(llm::DEFAULT_ENDPOINT));
    assert!(content.contains("**User**"));

    // Must parse cleanly, with a single user turn.
    let conv = parser::parse(&content).unwrap();
    assert!(conv.had_frontmatter);
    assert_eq!(conv.turns.len(), 1);
    assert_eq!(conv.turns[0].role, parser::Role::User);
    assert_eq!(conv.turns[0].content, "Your prompt here");

    // Refuse to overwrite an existing file.
    assert!(new(&file).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_appends_default_extension() {
    let dir = std::env::temp_dir().join(format!("chatfile-newext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let file = dir.join("notes");

    new(&file).unwrap();
    assert!(dir.join("notes.chat.md").exists());

    // An explicit `.chat.md` is left untouched.
    new(&dir.join("exact.chat.md")).unwrap();
    assert!(dir.join("exact.chat.md").exists());
    assert!(!dir.join("exact.chat.chat.md").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_rejects_wrong_extension() {
    let dir = std::env::temp_dir().join(format!("chatfile-runext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("conv.md");
    std::fs::write(&file, "---\n**User**\n\nhi\n").unwrap();

    let err = require_conversation_ext(&file).unwrap_err().to_string();
    assert!(err.contains(".chat.md"), "unexpected error: {err}");
    assert!(require_conversation_ext(&dir.join("ok.chat.md")).is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}
