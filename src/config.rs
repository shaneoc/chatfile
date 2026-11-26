//! XDG-compliant TOML configuration for chatfile.
//!
//! Config files are looked up per the FreeDesktop XDG Base Directory
//! Specification, in increasing priority order (lowest first):
//!
//! 1. `<dir>/chatfile/config.toml` for each directory in `$XDG_CONFIG_DIRS`
//!    (colon-separated; the spec's *first* entry has the *highest* priority,
//!    so they are merged in reverse). When `$XDG_CONFIG_DIRS` is unset or
//!    empty, `/etc/xdg` is used.
//! 2. `$XDG_CONFIG_HOME/chatfile/config.toml`, falling back to
//!    `$HOME/.config/chatfile/config.toml` when `$XDG_CONFIG_HOME` is unset
//!    or empty (skipped entirely when neither is set).
//!
//! Every field of every table is optional: a file may name any subset of
//! the keys, and unnamed keys keep the value from the layer below (missing
//! files are skipped, built-in defaults fill in at the bottom). A file that
//! exists but is not valid TOML is a hard error with a descriptive message.
//!
//! Resolution precedence for LLM settings: `.chat.md` frontmatter >
//! `CHATFILE_*` environment variables > the active profile (the
//! `default_profile` entry of `[profiles]`) > standard `OPENAI_*` variables
//! > built-in defaults.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

/// The app's configuration directory name inside each XDG config root.
pub const CONFIG_DIR: &str = "chatfile";

/// The app's configuration file name.
pub const CONFIG_FILE: &str = "config.toml";

/// Fallback system config directory when `$XDG_CONFIG_DIRS` is unset or empty.
pub const DEFAULT_XDG_CONFIG_DIRS: &str = "/etc/xdg";

/// Default model name (used when nothing else supplies one).
pub const DEFAULT_MODEL: &str = "local-model";

/// Endpoint override (frontmatter wins over this).
pub const ENV_ENDPOINT: &str = "CHATFILE_ENDPOINT";
/// Model override (frontmatter wins over this).
pub const ENV_MODEL: &str = "CHATFILE_MODEL";
/// API key override.
pub const ENV_API_KEY: &str = "CHATFILE_API_KEY";
/// Standard OpenAI-compatible endpoint variable, honored as a fallback when
/// the `CHATFILE_*` equivalents are unset.
pub const ENV_OPENAI_BASE: &str = "OPENAI_API_BASE";
/// Standard OpenAI-compatible API key variable, honored as a fallback when
/// the `CHATFILE_*` equivalents are unset.
pub const ENV_OPENAI_KEY: &str = "OPENAI_API_KEY";

// ---------------------------------------------------------------------------
// XDG path lookup
// ---------------------------------------------------------------------------

/// The candidate `config.toml` paths in *increasing* priority order (lowest
/// first): the `$XDG_CONFIG_DIRS` entries in reverse, then the user config.
pub fn config_paths() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME");
    let xdg_config_home = std::env::var_os("XDG_CONFIG_HOME");
    let xdg_config_dirs = std::env::var_os("XDG_CONFIG_DIRS");
    config_paths_from(
        xdg_config_home.as_deref(),
        xdg_config_dirs.as_deref(),
        home.as_deref(),
    )
}

/// Pure form of [`config_paths`] with the environment injected, so tests can
/// exercise the XDG rules without mutating process state.
fn config_paths_from(
    xdg_config_home: Option<&std::ffi::OsStr>,
    xdg_config_dirs: Option<&std::ffi::OsStr>,
    home: Option<&std::ffi::OsStr>,
) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();

    // System directories, lowest priority first (the spec's first entry is
    // the highest priority). Unset or empty falls back to /etc/xdg.
    let dirs: Vec<PathBuf> = xdg_config_dirs
        .filter(|s| !s.is_empty())
        .into_iter()
        .flat_map(std::env::split_paths)
        .collect();
    let dirs = if dirs.is_empty() {
        vec![PathBuf::from(DEFAULT_XDG_CONFIG_DIRS)]
    } else {
        dirs
    };
    for dir in dirs.iter().rev() {
        paths.push(dir.join(CONFIG_DIR).join(CONFIG_FILE));
    }

    // User config, highest priority. Unset or empty XDG_CONFIG_HOME falls
    // back to $HOME/.config; with neither set there is no user layer.
    let user_dir = xdg_config_home
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| home.map(|h| Path::new(h).join(".config")));
    if let Some(dir) = user_dir {
        paths.push(dir.join(CONFIG_DIR).join(CONFIG_FILE));
    }
    paths
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// The fully resolved configuration: built-in defaults with every config
/// file merged on top, lowest priority first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AppConfig {
    /// Name of the `[profiles]` entry used as the default LLM profile.
    pub default_profile: Option<String>,
    /// Named endpoint/model profiles.
    pub profiles: HashMap<String, ProfileConfig>,
    /// Sandbox policy for the `bash` tool.
    pub sandbox: SandboxConfig,
    /// Terminal UI settings.
    pub tui: TuiConfig,
    /// Prompt overrides.
    pub prompts: PromptsConfig,
}

impl<'de> Deserialize<'de> for AppConfig {
    /// A single config file deserializes into defaults with the file's keys
    /// applied on top (an empty document yields pure defaults).
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let overrides = Option::<FileOverrides>::deserialize(deserializer)?;
        let mut config = AppConfig::default();
        if let Some(overrides) = overrides {
            overrides.apply(&mut config);
        }
        Ok(config)
    }
}

impl AppConfig {
    /// Load the effective configuration from the standard XDG locations
    /// (see [`config_paths`]). Missing files are skipped; an existing file
    /// with invalid TOML is an error.
    pub fn load() -> Result<Self> {
        Self::load_from(&config_paths())
    }

    /// Like [`Self::load`] but from an explicit list of config files in
    /// increasing precedence order. Missing files are skipped.
    pub fn load_from(paths: &[PathBuf]) -> Result<Self> {
        let mut config = AppConfig::default();
        for path in paths {
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading config file {}", path.display()))?;
            let overrides: FileOverrides = toml::from_str(&text)
                .with_context(|| format!("parsing config file {}: invalid TOML", path.display()))?;
            overrides.apply(&mut config);
        }
        config.validate()?;
        Ok(config)
    }

    /// The profile used as the default, or `None` when `default_profile` is
    /// unset. (A `default_profile` naming a missing profile is rejected by
    /// [`Self::load_from`] before this is ever called.)
    pub fn active_profile(&self) -> Option<&ProfileConfig> {
        self.default_profile
            .as_deref()
            .and_then(|name| self.profiles.get(name))
    }

    /// Resolve the endpoint, model, and API key for a conversation using the
    /// real process environment (see [`resolve_llm`] for the precedence
    /// rules).
    pub fn resolve_llm(
        &self,
        fm_endpoint: Option<&str>,
        fm_model: Option<&str>,
    ) -> (String, String, Option<String>) {
        resolve_llm(self, fm_endpoint, fm_model, |name| {
            std::env::var(name).ok().filter(|s| !s.is_empty())
        })
    }

    /// Cross-field validation of the merged config.
    fn validate(&self) -> Result<()> {
        if let Some(name) = &self.default_profile {
            if !self.profiles.contains_key(name) {
                anyhow::bail!(
                    "default_profile '{name}' does not match any [profiles.{name}] entry"
                );
            }
        }
        for (name, profile) in &self.profiles {
            if let Some(scope) = &profile.scope {
                if scope != "folder" && scope != "vault" {
                    anyhow::bail!(
                        "profiles.{name}.scope must be 'folder' or 'vault', found '{scope}'"
                    );
                }
            }
        }
        Ok(())
    }
}

/// A named endpoint/model profile under `[profiles.<name>]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// The full `/v1/chat/completions` endpoint URL.
    pub endpoint: Option<String>,
    /// The model name to request.
    pub model: Option<String>,
    /// Name of the environment variable holding the API key (the value is
    /// read at request time, so keys never live in config files).
    pub api_key_env: Option<String>,
    /// Default tool scope: `"folder"` or `"vault"`.
    pub scope: Option<String>,
    /// Built-in tools enabled for conversations that use this profile; names
    /// from `list_files`, `read_file`, `write_file`, `edit_file`,
    /// `search_files`, `fetch_url`, `bash` (`fetch_url` and `bash` still
    /// require the conversation frontmatter's `network`/`allow_shell`).
    pub tools: Option<Vec<String>>,
}

/// Sandbox policy for the `bash` tool (the `[sandbox]` table).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SandboxConfig {
    /// Path to the `bwrap` binary (bare names are resolved via `PATH`).
    pub bwrap_path: String,
    /// Host paths bind-mounted read-only into the sandbox.
    pub ro_bind: Vec<String>,
    /// Host paths bind-mounted read-only only if they exist on the host.
    pub ro_bind_try: Vec<String>,
    /// Host paths recreated inside the sandbox as a symlink (using the
    /// host's link target) when they are symlinks, or read-only bound when
    /// they are directories; missing paths are skipped. This keeps
    /// usr-merged distros (`/bin -> usr/bin`) working without binding `/`.
    pub symlink_or_ro_bind: Vec<String>,
    /// Host paths covered by an empty tmpfs inside the sandbox, only when
    /// the path exists on the host (so masking a missing dir under a
    /// read-only bind can never fail).
    pub mask_tmpfs: Vec<String>,
    /// Host files replaced by a bind of `/dev/null` inside the sandbox,
    /// only when the path is a regular file on the host.
    pub mask_devnull: Vec<String>,
    /// The complete environment of the sandboxed program (`--clearenv` is
    /// always used, so nothing else leaks in).
    pub env: BTreeMap<String, String>,
    /// Extra raw `bwrap` flags appended after all standard flags.
    pub extra_args: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        let mut env = BTreeMap::new();
        env.insert("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into());
        env.insert("HOME".into(), "/tmp".into());
        env.insert("LANG".into(), "C.UTF-8".into());
        env.insert("CARGO_HOME".into(), "/tmp/.cargo".into());
        Self {
            bwrap_path: "bwrap".into(),
            ro_bind: vec!["/usr".into(), "/etc".into()],
            ro_bind_try: vec!["/nix".into(), "/opt".into()],
            symlink_or_ro_bind: vec![
                "/bin".into(),
                "/sbin".into(),
                "/lib".into(),
                "/lib64".into(),
                "/lib32".into(),
            ],
            mask_tmpfs: vec!["/etc/ssh".into()],
            mask_devnull: vec!["/etc/shadow".into(), "/etc/gshadow".into()],
            env,
            extra_args: Vec::new(),
        }
    }
}

impl<'de> Deserialize<'de> for SandboxConfig {
    /// A `SandboxConfig` deserializes from the same partial TOML as a config
    /// file, with missing keys filled in from the built-in defaults (an empty
    /// document yields pure defaults).
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let overrides = Option::<SandboxOverrides>::deserialize(deserializer)?;
        let mut config = SandboxConfig::default();
        if let Some(overrides) = overrides {
            overrides.apply(&mut config);
        }
        Ok(config)
    }
}

/// Terminal UI settings (the `[tui]` table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiConfig {
    /// Capture mouse events in the TUI (scroll wheel scrolls the history).
    #[serde(default = "default_mouse")]
    pub mouse: bool,
}

fn default_mouse() -> bool {
    true
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self { mouse: true }
    }
}

/// Prompt overrides (the `[prompts]` table).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptsConfig {
    /// Override for the `**User** | compact: true` compaction request body
    /// (defaults to [`crate::llm::COMPACT_INSTRUCTION`]).
    pub compact: Option<String>,
}

// ---------------------------------------------------------------------------
// LLM resolution
// ---------------------------------------------------------------------------

/// Resolve the endpoint, model, and API key for a conversation from layered
/// sources. Precedence, highest first:
///
/// 1. `.chat.md` frontmatter (`fm_endpoint`, `fm_model`),
/// 2. `CHATFILE_ENDPOINT` / `CHATFILE_MODEL` environment variables,
/// 3. the active profile's `endpoint` / `model` (and `api_key_env` for the
///    key),
/// 4. standard `OPENAI_API_BASE` / `OPENAI_API_KEY` variables,
/// 5. built-in defaults (`http://localhost:9000/v1/chat/completions`,
///    `local-model`).
///
/// `env` is the environment lookup (injected so tests stay hermetic).
pub fn resolve_llm(
    config: &AppConfig,
    fm_endpoint: Option<&str>,
    fm_model: Option<&str>,
    env: impl Fn(&str) -> Option<String>,
) -> (String, String, Option<String>) {
    let profile = config.active_profile();
    let endpoint = fm_endpoint
        .map(str::to_string)
        .or_else(|| env(ENV_ENDPOINT))
        .or_else(|| profile.and_then(|p| p.endpoint.clone()))
        .or_else(|| env(ENV_OPENAI_BASE))
        .unwrap_or_else(|| crate::llm::DEFAULT_ENDPOINT.to_string());
    let model = fm_model
        .map(str::to_string)
        .or_else(|| env(ENV_MODEL))
        .or_else(|| profile.and_then(|p| p.model.clone()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let api_key = env(ENV_API_KEY)
        .or_else(|| {
            profile
                .and_then(|p| p.api_key_env.as_deref())
                .and_then(&env)
        })
        .or_else(|| env(ENV_OPENAI_KEY));
    (endpoint, model, api_key)
}

// ---------------------------------------------------------------------------
// Wire format (one config file, every field optional)
// ---------------------------------------------------------------------------

/// The wire format of a single `config.toml`: every field optional so a
/// partial file only overrides the keys it names. Mirrors [`AppConfig`].
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileOverrides {
    default_profile: Option<String>,
    profiles: HashMap<String, ProfileOverrides>,
    sandbox: SandboxOverrides,
    tui: TuiOverrides,
    prompts: PromptsOverrides,
}

impl FileOverrides {
    fn apply(self, config: &mut AppConfig) {
        if let Some(name) = self.default_profile {
            config.default_profile = Some(name);
        }
        for (name, overrides) in self.profiles {
            let entry = config.profiles.entry(name).or_default();
            overrides.apply(entry);
        }
        self.sandbox.apply(&mut config.sandbox);
        self.tui.apply(&mut config.tui);
        self.prompts.apply(&mut config.prompts);
    }
}

/// Wire format of one `[profiles.<name>]` table.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ProfileOverrides {
    endpoint: Option<String>,
    model: Option<String>,
    api_key_env: Option<String>,
    scope: Option<String>,
    tools: Option<Vec<String>>,
}

impl ProfileOverrides {
    fn apply(self, profile: &mut ProfileConfig) {
        if let Some(v) = self.endpoint {
            profile.endpoint = Some(v);
        }
        if let Some(v) = self.model {
            profile.model = Some(v);
        }
        if let Some(v) = self.api_key_env {
            profile.api_key_env = Some(v);
        }
        if let Some(v) = self.scope {
            profile.scope = Some(v);
        }
        if let Some(v) = self.tools {
            profile.tools = Some(v);
        }
    }
}

/// Wire format of the `[sandbox]` table.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SandboxOverrides {
    bwrap_path: Option<String>,
    ro_bind: Option<Vec<String>>,
    ro_bind_try: Option<Vec<String>>,
    symlink_or_ro_bind: Option<Vec<String>>,
    mask_tmpfs: Option<Vec<String>>,
    mask_devnull: Option<Vec<String>>,
    env: Option<BTreeMap<String, String>>,
    extra_args: Option<Vec<String>>,
}

impl SandboxOverrides {
    fn apply(self, config: &mut SandboxConfig) {
        if let Some(v) = self.bwrap_path {
            config.bwrap_path = v;
        }
        if let Some(v) = self.ro_bind {
            config.ro_bind = v;
        }
        if let Some(v) = self.ro_bind_try {
            config.ro_bind_try = v;
        }
        if let Some(v) = self.symlink_or_ro_bind {
            config.symlink_or_ro_bind = v;
        }
        if let Some(v) = self.mask_tmpfs {
            config.mask_tmpfs = v;
        }
        if let Some(v) = self.mask_devnull {
            config.mask_devnull = v;
        }
        if let Some(v) = self.env {
            config.env = v;
        }
        if let Some(v) = self.extra_args {
            config.extra_args = v;
        }
    }
}

/// Wire format of the `[tui]` table.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TuiOverrides {
    mouse: Option<bool>,
}

impl TuiOverrides {
    fn apply(self, config: &mut TuiConfig) {
        if let Some(v) = self.mouse {
            config.mouse = v;
        }
    }
}

/// Wire format of the `[prompts]` table.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PromptsOverrides {
    compact: Option<String>,
}

impl PromptsOverrides {
    fn apply(self, config: &mut PromptsConfig) {
        if let Some(v) = self.compact {
            config.compact = Some(v);
        }
    }
}

#[cfg(test)]
mod tests;
