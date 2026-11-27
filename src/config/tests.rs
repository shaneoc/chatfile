use super::*;
use std::ffi::OsStr;
use tempfile::TempDir;

fn opt(s: &str) -> Option<&OsStr> {
    Some(s.as_ref())
}

// ---------------------------------------------------------------------------
// XDG path resolution
// ---------------------------------------------------------------------------

#[test]
fn config_paths_custom_xdg_config_home_and_dirs() {
    let paths = config_paths_from(
        opt("/home/u/.cfg"),
        opt("/etc/xdg:/usr/share/xdg"),
        opt("/home/u"),
    );
    // System dirs in reverse (spec's first entry = highest priority), then
    // the user config.
    assert_eq!(
        paths,
        vec![
            PathBuf::from("/usr/share/xdg/chatfile/config.toml"),
            PathBuf::from("/etc/xdg/chatfile/config.toml"),
            PathBuf::from("/home/u/.cfg/chatfile/config.toml"),
        ]
    );
}

#[test]
fn config_paths_xdg_config_dirs_priority_order() {
    let paths = config_paths_from(opt("/u"), opt("a:b:c"), opt("/h"));
    // Lowest first: c, b, a; the user layer is highest.
    assert_eq!(
        paths,
        vec![
            PathBuf::from("c/chatfile/config.toml"),
            PathBuf::from("b/chatfile/config.toml"),
            PathBuf::from("a/chatfile/config.toml"),
            PathBuf::from("/u/chatfile/config.toml"),
        ]
    );
}

#[test]
fn config_paths_fallbacks() {
    // Unset XDG_CONFIG_HOME falls back to $HOME/.config; unset
    // XDG_CONFIG_DIRS falls back to /etc/xdg.
    let paths = config_paths_from(None, None, opt("/home/u"));
    assert_eq!(
        paths,
        vec![
            PathBuf::from("/etc/xdg/chatfile/config.toml"),
            PathBuf::from("/home/u/.config/chatfile/config.toml"),
        ]
    );

    // Empty variables behave like unset.
    let paths = config_paths_from(opt(""), opt(""), opt("/home/u"));
    assert_eq!(
        paths,
        vec![
            PathBuf::from("/etc/xdg/chatfile/config.toml"),
            PathBuf::from("/home/u/.config/chatfile/config.toml"),
        ]
    );

    // No HOME and no XDG_CONFIG_HOME: no user layer at all.
    let paths = config_paths_from(None, None, None);
    assert_eq!(paths, vec![PathBuf::from("/etc/xdg/chatfile/config.toml")]);
}

// ---------------------------------------------------------------------------
// TOML parsing
// ---------------------------------------------------------------------------

#[test]
fn load_empty_and_missing_files_yield_defaults() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("nope.toml");
    assert_eq!(
        AppConfig::load_from(&[missing.to_path_buf()]).unwrap(),
        AppConfig::default()
    );

    let empty = dir.path().join("empty.toml");
    std::fs::write(&empty, "").unwrap();
    assert_eq!(
        AppConfig::load_from(&[empty.to_path_buf()]).unwrap(),
        AppConfig::default()
    );
}

#[test]
fn load_full_toml_sets_every_table() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(
        &file,
        r#"
default_profile = "local"

[profiles.local]
endpoint = "http://localhost:9000/v1/chat/completions"
model = "qwen"
api_key_env = "MY_KEY"
max_output_tokens = 512
max_tool_output_bytes = 4096
scope = "vault"
tools = ["read_file", "bash"]

[profiles.cloud]
endpoint = "https://api.example.com/v1/chat/completions"
model = "gpt-x"

[sandbox]
bwrap_path = "/usr/local/bin/bwrap"
ro_bind = ["/opt/only"]
env = { PATH = "/bin", HOME = "/tmp" }

[tui]
mouse = false

[prompts]
compact = """
Summarize in three bullets.
Keep it short.
"""
"#,
    )
    .unwrap();

    let config = AppConfig::load_from(&[file.to_path_buf()]).unwrap();
    assert_eq!(config.default_profile.as_deref(), Some("local"));
    assert_eq!(config.profiles.len(), 2);

    let local = &config.profiles["local"];
    assert_eq!(
        local.endpoint.as_deref(),
        Some("http://localhost:9000/v1/chat/completions")
    );
    assert_eq!(local.model.as_deref(), Some("qwen"));
    assert_eq!(local.api_key_env.as_deref(), Some("MY_KEY"));
    assert_eq!(local.max_output_tokens, Some(512));
    assert_eq!(local.max_tool_output_bytes, Some(4096));
    assert_eq!(local.scope.as_deref(), Some("vault"));
    assert_eq!(
        local.tools.as_deref(),
        Some(&["read_file".to_string(), "bash".to_string()][..])
    );

    let cloud = &config.profiles["cloud"];
    assert_eq!(
        cloud.endpoint.as_deref(),
        Some("https://api.example.com/v1/chat/completions")
    );
    assert_eq!(cloud.model.as_deref(), Some("gpt-x"));
    assert_eq!(cloud.api_key_env, None);
    assert_eq!(cloud.max_output_tokens, None);
    assert_eq!(cloud.max_tool_output_bytes, None);
    assert_eq!(cloud.scope, None);
    assert_eq!(cloud.tools, None);

    assert_eq!(config.sandbox.bwrap_path, "/usr/local/bin/bwrap");
    assert_eq!(config.sandbox.ro_bind, vec!["/opt/only".to_string()]);
    // Unnamed sandbox keys keep their defaults.
    let defaults = SandboxConfig::default();
    assert_eq!(
        config.sandbox.symlink_or_ro_bind,
        defaults.symlink_or_ro_bind
    );
    assert_eq!(config.sandbox.mask_devnull, defaults.mask_devnull);
    assert_eq!(
        config.sandbox.env.get("PATH").map(String::as_str),
        Some("/bin")
    );

    assert!(!config.tui.mouse);

    assert_eq!(
        config.prompts.compact.as_deref(),
        Some("Summarize in three bullets.\nKeep it short.\n")
    );

    // The default profile resolves.
    assert_eq!(
        config.active_profile().unwrap().model.as_deref(),
        Some("qwen")
    );
}

#[test]
fn load_partial_toml_overrides_only_named_keys() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "[tui]\nmouse = false\n").unwrap();

    let config = AppConfig::load_from(&[file.to_path_buf()]).unwrap();
    assert!(!config.tui.mouse);
    // Everything else is untouched.
    let defaults = AppConfig::default();
    assert_eq!(config.sandbox, defaults.sandbox);
    assert!(config.profiles.is_empty());
    assert_eq!(config.default_profile, None);
    assert_eq!(config.prompts, PromptsConfig::default());
}

#[test]
fn load_merges_layers_lowest_first() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("first.toml");
    std::fs::write(
        &first,
        r#"
[profiles.p]
endpoint = "http://first"
model = "m1"

[sandbox]
bwrap_path = "first-bwrap"
ro_bind = ["/first"]

[tui]
mouse = false
"#,
    )
    .unwrap();
    let second = dir.path().join("second.toml");
    std::fs::write(
        &second,
        r#"
[profiles.p]
model = "m2"

[sandbox]
bwrap_path = "second-bwrap"
"#,
    )
    .unwrap();

    let config = AppConfig::load_from(&[first.to_path_buf(), second.to_path_buf()]).unwrap();
    // Key by key: the later file wins where it names a key.
    assert_eq!(config.sandbox.bwrap_path, "second-bwrap");
    assert_eq!(config.sandbox.ro_bind, vec!["/first".to_string()]);
    let p = &config.profiles["p"];
    assert_eq!(p.endpoint.as_deref(), Some("http://first"));
    assert_eq!(p.model.as_deref(), Some("m2"));
    // A key only the earlier file names survives.
    assert!(!config.tui.mouse);

    // Reversed order flips the winner: `first` (with `mouse = false`) is now
    // the highest-priority layer.
    let config = AppConfig::load_from(&[second.to_path_buf(), first.to_path_buf()]).unwrap();
    assert_eq!(config.sandbox.bwrap_path, "first-bwrap");
    assert_eq!(config.sandbox.ro_bind, vec!["/first".to_string()]);
    let p = &config.profiles["p"];
    assert_eq!(p.endpoint.as_deref(), Some("http://first"));
    assert_eq!(p.model.as_deref(), Some("m1"));
    assert!(!config.tui.mouse);
}

#[test]
fn load_malformed_toml_is_an_error() {
    let dir = TempDir::new().unwrap();
    let bad = dir.path().join("bad.toml");
    std::fs::write(&bad, "bwrap_path = [unclosed\n").unwrap();

    let err = AppConfig::load_from(&[bad.to_path_buf()]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("invalid TOML"), "unexpected error: {msg}");
    assert!(msg.contains("bad.toml"), "unexpected error: {msg}");
}

#[test]
fn load_unknown_fields_are_rejected() {
    let dir = TempDir::new().unwrap();

    let bad_top = dir.path().join("top.toml");
    std::fs::write(&bad_top, "bogus_key = 1\n").unwrap();
    assert!(AppConfig::load_from(&[bad_top.to_path_buf()]).is_err());

    let bad_table = dir.path().join("table.toml");
    std::fs::write(&bad_table, "[sandbox]\nrobinds = [\"/x\"]\n").unwrap();
    assert!(AppConfig::load_from(&[bad_table.to_path_buf()]).is_err());

    let bad_profile = dir.path().join("profile.toml");
    std::fs::write(&bad_profile, "[profiles.p]\nendpoints = \"x\"\n").unwrap();
    assert!(AppConfig::load_from(&[bad_profile.to_path_buf()]).is_err());
}

#[test]
fn load_default_profile_must_name_a_profile() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "default_profile = \"nope\"\n").unwrap();

    let err = AppConfig::load_from(&[file.to_path_buf()]).unwrap_err();
    assert!(
        err.to_string().contains("default_profile"),
        "unexpected error: {err}"
    );
}

#[test]
fn load_invalid_scope_is_rejected() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "[profiles.p]\nscope = \"everywhere\"\n").unwrap();

    let err = AppConfig::load_from(&[file.to_path_buf()]).unwrap_err();
    assert!(err.to_string().contains("scope"), "unexpected error: {err}");
}

#[test]
fn load_zero_max_output_tokens_is_rejected() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "[profiles.p]\nmax_output_tokens = 0\n").unwrap();

    let err = AppConfig::load_from(&[file.to_path_buf()]).unwrap_err();
    assert!(
        err.to_string().contains("max_output_tokens"),
        "unexpected error: {err}"
    );
}

#[test]
fn load_max_tool_output_bytes_parses_and_defaults() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "[profiles.p]\nmax_tool_output_bytes = 4096\n").unwrap();

    let config = AppConfig::load_from(&[file.to_path_buf()]).unwrap();
    assert_eq!(config.profiles["p"].max_tool_output_bytes, Some(4096));
    // Unnamed keys keep the unset value.
    assert_eq!(config.profiles["p"].max_output_tokens, None);
}

#[test]
fn load_zero_max_tool_output_bytes_is_rejected() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("config.toml");
    std::fs::write(&file, "[profiles.p]\nmax_tool_output_bytes = 0\n").unwrap();

    let err = AppConfig::load_from(&[file.to_path_buf()]).unwrap_err();
    assert!(
        err.to_string().contains("max_tool_output_bytes"),
        "unexpected error: {err}"
    );
}

#[test]
fn active_profile_none_when_unset() {
    let config = AppConfig::default();
    assert_eq!(config.active_profile(), None);
}

// ---------------------------------------------------------------------------
// LLM resolution precedence
// ---------------------------------------------------------------------------

fn profiled_config() -> AppConfig {
    let mut profiles = HashMap::new();
    profiles.insert(
        "p".into(),
        ProfileConfig {
            endpoint: Some("http://profile".into()),
            model: Some("profile-model".into()),
            api_key_env: Some("PROFILE_KEY".into()),
            max_output_tokens: Some(4096),
            max_tool_output_bytes: None,
            scope: None,
            tools: None,
        },
    );
    AppConfig {
        default_profile: Some("p".into()),
        profiles,
        ..Default::default()
    }
}

/// A fixed environment for the precedence tests.
fn env_map<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |name| {
        vars.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn resolve_llm_frontmatter_wins_over_everything() {
    let config = profiled_config();
    let env = env_map(&[
        ("CHATFILE_ENDPOINT", "http://env"),
        ("CHATFILE_MODEL", "env-model"),
        ("OPENAI_API_BASE", "http://openai"),
        ("PROFILE_KEY", "k-profile"),
        ("OPENAI_API_KEY", "k-openai"),
    ]);
    let (endpoint, model, api_key) =
        resolve_llm(&config, Some("http://fm"), Some("fm-model"), &env);
    assert_eq!(endpoint, "http://fm");
    assert_eq!(model, "fm-model");
    // The key still comes from the profile (CHATFILE_API_KEY unset).
    assert_eq!(api_key.as_deref(), Some("k-profile"));
}

#[test]
fn resolve_llm_env_wins_over_profile() {
    let config = profiled_config();
    let env = env_map(&[
        ("CHATFILE_ENDPOINT", "http://env"),
        ("PROFILE_KEY", "k-profile"),
    ]);
    let (endpoint, model, api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(endpoint, "http://env");
    // CHATFILE_MODEL unset: the profile's model applies.
    assert_eq!(model, "profile-model");
    assert_eq!(api_key.as_deref(), Some("k-profile"));
}

#[test]
fn resolve_llm_profile_wins_over_openai_env() {
    let config = profiled_config();
    let env = env_map(&[("OPENAI_API_BASE", "http://openai")]);
    let (endpoint, model, _api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(endpoint, "http://profile");
    assert_eq!(model, "profile-model");
    // No PROFILE_KEY and no CHATFILE_API_KEY: fall through to OPENAI_API_KEY.
    let env = env_map(&[("OPENAI_API_KEY", "k-openai")]);
    let (_, _, api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(api_key.as_deref(), Some("k-openai"));
}

#[test]
fn resolve_llm_openai_env_wins_over_defaults() {
    let config = AppConfig::default();
    let env = env_map(&[
        ("OPENAI_API_BASE", "http://openai"),
        ("OPENAI_API_KEY", "k-openai"),
    ]);
    let (endpoint, model, api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(endpoint, "http://openai");
    assert_eq!(model, DEFAULT_MODEL);
    assert_eq!(api_key.as_deref(), Some("k-openai"));
}

#[test]
fn resolve_llm_chatfile_api_key_wins_over_profile_key() {
    let config = profiled_config();
    let env = env_map(&[
        ("CHATFILE_API_KEY", "k-chatfile"),
        ("PROFILE_KEY", "k-profile"),
        ("OPENAI_API_KEY", "k-openai"),
    ]);
    let (_, _, api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(api_key.as_deref(), Some("k-chatfile"));
}

#[test]
fn resolve_llm_defaults_when_nothing_is_set() {
    let config = AppConfig::default();
    let env = env_map(&[]);
    let (endpoint, model, api_key) = resolve_llm(&config, None, None, &env);
    assert_eq!(endpoint, crate::llm::DEFAULT_ENDPOINT);
    assert_eq!(model, DEFAULT_MODEL);
    assert_eq!(api_key, None);
}

// ---------------------------------------------------------------------------
// Output-token limit precedence (CLI > frontmatter > profile)
// ---------------------------------------------------------------------------

#[test]
fn resolve_max_output_tokens_cli_wins_over_frontmatter_and_profile() {
    // The profile carries Some(4096); the CLI flag must win over both it
    // and the frontmatter.
    let config = profiled_config();
    assert_eq!(
        resolve_max_output_tokens(&config, Some(100), Some(200)),
        Some(100)
    );
}

#[test]
fn resolve_max_output_tokens_frontmatter_wins_over_profile() {
    let config = profiled_config();
    assert_eq!(
        resolve_max_output_tokens(&config, None, Some(200)),
        Some(200)
    );
}

#[test]
fn resolve_max_output_tokens_profile_fallback_and_unset() {
    // Nothing above the profile: the profile's value applies.
    let config = profiled_config();
    assert_eq!(resolve_max_output_tokens(&config, None, None), Some(4096));
    // No profile (or no value in it): the field is omitted from requests.
    let empty = AppConfig::default();
    assert_eq!(resolve_max_output_tokens(&empty, None, None), None);
}
