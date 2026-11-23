//! Config-driven rootless [`bubblewrap`](https://github.com/containers/bubblewrap)
//! sandbox for running untrusted commands (e.g. model-generated shell) without
//! host filesystem or network access.
//!
//! The policy is loaded from optional YAML config files —
//! `/etc/mdchat/config.yaml` (system) and `~/.config/mdchat/config.yaml`
//! (user, overrides system) — falling back to built-in distro-agnostic
//! defaults when neither exists. A config file may name any subset of the
//! fields; unnamed fields keep the value from the layer below.
//!
//! Sandbox layout: a fresh mount namespace containing only what the policy
//! binds plus bwrap's own `/proc`, `/dev`, and a private `/tmp`. `work_dir`
//! is the single writable location (bound read-write) and becomes the working
//! directory. The network namespace is unshared unless the caller passes
//! `allow_network` (loopback works inside the fresh namespace either way).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// System-wide sandbox config location.
pub const SYSTEM_CONFIG_PATH: &str = "/etc/mdchat/config.yaml";

/// User-level sandbox config location (`$HOME/.config/mdchat/config.yaml`),
/// or `None` when `$HOME` is unset.
pub fn user_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        Path::new(&home)
            .join(".config")
            .join("mdchat")
            .join("config.yaml")
    })
}

/// Effective sandbox policy: built-in defaults with config-file overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SandboxPolicy {
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

impl Default for SandboxPolicy {
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

/// Wire format for config files: every field optional so a partial file only
/// overrides the keys it names.
#[derive(Debug, Default, Deserialize)]
struct PolicyOverrides {
    bwrap_path: Option<String>,
    ro_bind: Option<Vec<String>>,
    ro_bind_try: Option<Vec<String>>,
    symlink_or_ro_bind: Option<Vec<String>>,
    mask_tmpfs: Option<Vec<String>>,
    mask_devnull: Option<Vec<String>>,
    env: Option<BTreeMap<String, String>>,
    extra_args: Option<Vec<String>>,
}

impl PolicyOverrides {
    fn apply(self, policy: &mut SandboxPolicy) {
        if let Some(v) = self.bwrap_path {
            policy.bwrap_path = v;
        }
        if let Some(v) = self.ro_bind {
            policy.ro_bind = v;
        }
        if let Some(v) = self.ro_bind_try {
            policy.ro_bind_try = v;
        }
        if let Some(v) = self.symlink_or_ro_bind {
            policy.symlink_or_ro_bind = v;
        }
        if let Some(v) = self.mask_tmpfs {
            policy.mask_tmpfs = v;
        }
        if let Some(v) = self.mask_devnull {
            policy.mask_devnull = v;
        }
        if let Some(v) = self.env {
            policy.env = v;
        }
        if let Some(v) = self.extra_args {
            policy.extra_args = v;
        }
    }
}

impl<'de> Deserialize<'de> for SandboxPolicy {
    /// A `SandboxPolicy` deserializes from the same partial YAML as
    /// [`PolicyOverrides`], with missing keys filled in from the built-in
    /// defaults (an empty document yields pure defaults).
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let overrides = Option::<PolicyOverrides>::deserialize(deserializer)?;
        let mut policy = SandboxPolicy::default();
        if let Some(overrides) = overrides {
            overrides.apply(&mut policy);
        }
        Ok(policy)
    }
}

impl SandboxPolicy {
    /// Load the effective policy: built-in defaults, overridden by
    /// `/etc/mdchat/config.yaml` when present, then by
    /// `~/.config/mdchat/config.yaml` when present.
    pub fn load() -> Result<Self> {
        let mut paths = vec![PathBuf::from(SYSTEM_CONFIG_PATH)];
        if let Some(user) = user_config_path() {
            paths.push(user);
        }
        Self::load_from(&paths)
    }

    /// Like [`Self::load`] but from an explicit list of config files in
    /// increasing precedence order. Missing files are skipped.
    fn load_from(paths: &[PathBuf]) -> Result<Self> {
        let mut policy = SandboxPolicy::default();
        for path in paths {
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading sandbox config {}", path.display()))?;
            let overrides: Option<PolicyOverrides> = serde_yaml::from_str(&text)
                .with_context(|| format!("parsing sandbox config {}", path.display()))?;
            if let Some(overrides) = overrides {
                overrides.apply(&mut policy);
            }
        }
        Ok(policy)
    }
}

/// Result of a sandboxed command: exit status plus combined, truncated
/// stdout+stderr.
#[derive(Debug, Clone)]
pub struct SandboxOutput {
    /// Exit code of the program, or `None` if it was killed by a signal.
    pub exit_code: Option<i32>,
    /// Combined stdout and stderr, truncated to the requested limit.
    pub output: String,
}

impl SandboxOutput {
    /// True when the program exited normally with status 0.
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// Locate `program`, resolving bare names against the current `PATH`.
fn find_program(program: &str) -> Option<PathBuf> {
    let path = Path::new(program);
    if path.is_absolute() || path.components().count() > 1 {
        return path.is_file().then(|| path.to_path_buf());
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(program))
            .find(|candidate| candidate.is_file())
    })
}

/// Run `program args...` inside a rootless bubblewrap sandbox configured by
/// `policy`.
///
/// The child is killed if it does not exit within `timeout`, and the combined
/// stdout+stderr is truncated to `max_output_bytes` with a `\n[truncated]`
/// notice when truncation happens.
pub async fn run_sandboxed(
    policy: &SandboxPolicy,
    work_dir: &Path,
    allow_network: bool,
    timeout: Duration,
    max_output_bytes: usize,
    program: &str,
    args: &[&str],
) -> Result<SandboxOutput> {
    let bwrap = find_program(&policy.bwrap_path).ok_or_else(|| {
        anyhow::anyhow!(
            "bubblewrap not found at '{}' (searched PATH). Install it to enable \
             sandboxed command execution, e.g. `apt install bubblewrap`.",
            policy.bwrap_path
        )
    })?;

    let work_dir = work_dir
        .canonicalize()
        .with_context(|| format!("resolving sandbox work directory {}", work_dir.display()))?;

    let mut cmd = Command::new(&bwrap);
    cmd.kill_on_drop(true);

    // Baseline isolation. `--die-with-parent` makes bwrap kill the sandbox
    // init when we go away (e.g. on timeout), and `--new-session` blocks
    // TIOCSTI terminal injection from inside.
    cmd.arg("--unshare-all");
    if allow_network {
        cmd.arg("--share-net");
    }
    cmd.arg("--new-session");
    cmd.arg("--die-with-parent");
    cmd.arg("--clearenv");

    // Sandbox-provided filesystem, then the single writable work dir.
    cmd.arg("--proc").arg("/proc");
    cmd.arg("--dev").arg("/dev");
    cmd.arg("--tmpfs").arg("/tmp");
    cmd.arg("--bind").arg(&work_dir).arg(&work_dir);
    cmd.arg("--chdir").arg(&work_dir);

    // Policy-configured binds. `--ro-bind` takes both a source and a
    // destination path; here they are always the same.
    for path in &policy.ro_bind {
        cmd.arg("--ro-bind").arg(path).arg(path);
    }
    for path in &policy.ro_bind_try {
        if Path::new(path).exists() {
            cmd.arg("--ro-bind").arg(path).arg(path);
        }
    }
    for path in &policy.symlink_or_ro_bind {
        let host_path = Path::new(path);
        match std::fs::symlink_metadata(host_path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let target = std::fs::read_link(host_path)
                    .with_context(|| format!("reading symlink {}", host_path.display()))?;
                cmd.arg("--symlink").arg(&target).arg(path);
            }
            Ok(meta) if meta.is_dir() => {
                cmd.arg("--ro-bind").arg(path).arg(path);
            }
            // Missing (or a regular file, which is not handled): skip.
            _ => {}
        }
    }
    for path in &policy.mask_tmpfs {
        if Path::new(path).exists() {
            cmd.arg("--tmpfs").arg(path);
        }
    }
    for path in &policy.mask_devnull {
        if Path::new(path).is_file() {
            cmd.arg("--ro-bind").arg("/dev/null").arg(path);
        }
    }

    // Environment: exactly what the policy names, nothing else.
    for (key, value) in &policy.env {
        cmd.arg("--setenv").arg(key).arg(value);
    }

    for arg in &policy.extra_args {
        cmd.arg(arg);
    }

    cmd.arg("--");
    cmd.arg(program);
    for arg in args {
        cmd.arg(arg);
    }

    let output = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(result) => result
            .with_context(|| format!("running sandboxed command '{}'", program))?,
        Err(_) => bail!(
            "sandboxed command '{}' timed out after {:?}",
            program,
            timeout
        ),
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    if combined.len() > max_output_bytes {
        let notice = "\n[truncated]";
        let mut cut = max_output_bytes.saturating_sub(notice.len());
        while !combined.is_char_boundary(cut) {
            cut -= 1;
        }
        combined.truncate(cut);
        combined.push_str(notice);
    }

    Ok(SandboxOutput {
        exit_code: output.status.code(),
        output: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tempfile::TempDir;

    const TIMEOUT: Duration = Duration::from_secs(30);
    const MAX_OUTPUT: usize = 16384;

    /// The effective policy when `bwrap` is actually installed, otherwise a
    /// skip notice and `None` (the sandbox needs bubblewrap to run).
    fn sandbox_ready() -> Option<SandboxPolicy> {
        let policy = SandboxPolicy::load().expect("loading effective sandbox policy");
        match find_program(&policy.bwrap_path) {
            Some(_) => Some(policy),
            None => {
                eprintln!("skipping: bwrap not installed (e.g. `apt install bubblewrap`)");
                None
            }
        }
    }

    #[test]
    fn sandbox_config_yaml_parsing() {
        let dir = TempDir::new().unwrap();

        // An empty document yields pure defaults.
        let empty = dir.path().join("empty.yaml");
        std::fs::write(&empty, "").unwrap();
        assert_eq!(SandboxPolicy::load_from(&[empty.to_path_buf()]).unwrap(), SandboxPolicy::default());

        // A partial file overrides only the keys it names.
        let partial = dir.path().join("partial.yaml");
        std::fs::write(
            &partial,
            "bwrap_path: /usr/local/bin/bwrap\nro_bind:\n  - /opt/only\n",
        )
        .unwrap();
        let policy = SandboxPolicy::load_from(&[empty.to_path_buf(), partial.clone()]).unwrap();
        assert_eq!(policy.bwrap_path, "/usr/local/bin/bwrap");
        assert_eq!(policy.ro_bind, vec!["/opt/only".to_string()]);
        let defaults = SandboxPolicy::default();
        assert_eq!(policy.symlink_or_ro_bind, defaults.symlink_or_ro_bind);
        assert_eq!(policy.mask_devnull, defaults.mask_devnull);
        assert_eq!(policy.env.get("PATH").map(String::as_str), Some("/usr/local/bin:/usr/bin:/bin"));

        // A later file overrides an earlier one, key by key.
        let later = dir.path().join("later.yaml");
        std::fs::write(&later, "bwrap_path: bwrap2\n").unwrap();
        let policy = SandboxPolicy::load_from(&[partial, later.clone()]).unwrap();
        assert_eq!(policy.bwrap_path, "bwrap2");
        assert_eq!(policy.ro_bind, vec!["/opt/only".to_string()]);

        // A complete file sets every field.
        let full = dir.path().join("full.yaml");
        std::fs::write(
            &full,
            "bwrap_path: bwrap\n\
             ro_bind:\n  - /usr\n\
             ro_bind_try:\n  - /opt\n\
             symlink_or_ro_bind:\n  - /bin\n\
             mask_tmpfs:\n  - /etc/ssh\n\
             mask_devnull:\n  - /etc/shadow\n\
             env:\n  PATH: /bin\n  HOME: /tmp\n\
             extra_args:\n  - --ro-bind-try\n  - /var\n",
        )
        .unwrap();
        let policy = SandboxPolicy::load_from(&[full]).unwrap();
        assert_eq!(policy.ro_bind, vec!["/usr".to_string()]);
        assert_eq!(policy.ro_bind_try, vec!["/opt".to_string()]);
        assert_eq!(policy.symlink_or_ro_bind, vec!["/bin".to_string()]);
        assert_eq!(policy.mask_tmpfs, vec!["/etc/ssh".to_string()]);
        assert_eq!(policy.mask_devnull, vec!["/etc/shadow".to_string()]);
        assert_eq!(policy.env.get("PATH").map(String::as_str), Some("/bin"));
        assert_eq!(policy.extra_args, vec!["--ro-bind-try".to_string(), "/var".to_string()]);

        // Missing files are skipped; invalid YAML is an error.
        let missing = dir.path().join("nope.yaml");
        assert_eq!(SandboxPolicy::load_from(&[missing]).unwrap(), SandboxPolicy::default());
        let bad = dir.path().join("bad.yaml");
        std::fs::write(&bad, "bwrap_path: [unclosed\n").unwrap();
        assert!(SandboxPolicy::load_from(&[bad]).is_err());
    }

    #[tokio::test]
    async fn sandbox_allows_workdir_and_loopback() {
        let Some(policy) = sandbox_ready() else {
            return
        };
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("work.txt"), "original\n").unwrap();

        // The work dir is readable and writable, and changes reach the host.
        let out = run_sandboxed(
            &policy,
            dir.path(),
            false,
            TIMEOUT,
            MAX_OUTPUT,
            "/bin/sh",
            &["-c", "cat work.txt && echo modified > work.txt && cat work.txt"],
        )
        .await
        .unwrap();
        assert!(out.succeeded(), "output: {}", out.output);
        assert_eq!(out.output.trim(), "original\nmodified");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("work.txt")).unwrap().trim(),
            "modified"
        );

        // Loopback works inside the fresh network namespace (full
        // connect + round-trip).
        let py = r#"
import socket, threading
srv = socket.socket()
srv.bind(("127.0.0.1", 0))
srv.listen(1)
port = srv.getsockname()[1]
def server():
    conn, _ = srv.accept()
    data = conn.recv(4)
    conn.send(b"pong")
t = threading.Thread(target=server); t.start()
cli = socket.socket()
cli.connect(("127.0.0.1", port))
cli.send(b"ping")
print("LOOPBACK-OK", cli.recv(4))
t.join()
"#;
        let out = run_sandboxed(&policy, dir.path(), false, TIMEOUT, MAX_OUTPUT, "python3", &["-c", py])
            .await
            .unwrap();
        assert!(out.succeeded(), "output: {}", out.output);
        assert!(out.output.contains("LOOPBACK-OK b'pong'"), "output: {}", out.output);
    }

    #[tokio::test]
    async fn sandbox_blocks_host_filesystem_and_secrets() {
        let Some(policy) = sandbox_ready() else {
            return
        };
        let base = TempDir::new().unwrap();
        let work = base.path().join("work");
        let secret = base.path().join("secret");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&secret).unwrap();
        std::fs::write(secret.join("file.txt"), "top-secret").unwrap();

        // Writing into a read-only bind fails.
        let out = run_sandboxed(
            &policy,
            &work,
            false,
            TIMEOUT,
            MAX_OUTPUT,
            "/bin/sh",
            &["-c", "echo pwn > /usr/pwned 2>/dev/null && echo PWNED || echo BLOCKED"],
        )
        .await
        .unwrap();
        assert!(out.output.contains("BLOCKED"), "output: {}", out.output);
        assert!(!Path::new("/usr/pwned").exists());

        // Sibling directories outside the work dir are invisible.
        let out = run_sandboxed(
            &policy,
            &work,
            false,
            TIMEOUT,
            MAX_OUTPUT,
            "/bin/sh",
            &["-c", "test -e ../secret/file.txt && echo LEAKED || echo OK"],
        )
        .await
        .unwrap();
        assert!(out.succeeded(), "output: {}", out.output);
        assert!(out.output.contains("OK"), "output: {}", out.output);

        // /etc/ssh is masked with an empty tmpfs (when present on the host).
        if Path::new("/etc/ssh").exists() {
            let out = run_sandboxed(
                &policy,
                &work,
                false,
                TIMEOUT,
                MAX_OUTPUT,
                "/bin/sh",
                &["-c", "ls -A /etc/ssh | wc -l"],
            )
            .await
            .unwrap();
            assert!(out.succeeded(), "output: {}", out.output);
            assert_eq!(out.output.trim(), "0", "output: {}", out.output);
        }

        // Host environment (secrets, cargo variables, ...) never leaks in:
        // only the policy's env vars exist. `--clearenv` strips everything
        // else, so each of these must be empty regardless of the host.
        let out = run_sandboxed(
            &policy,
            &work,
            false,
            TIMEOUT,
            MAX_OUTPUT,
            "/bin/sh",
            &["-c", "for v in ANTHROPIC_API_KEY CARGO_MANIFEST_DIR USER TERM PATH HOME; do \
                    eval \"val=\\${$v:-}\"; printf '%s=[%s]\\n' \"$v\" \"$val\"; done"],
        )
        .await
        .unwrap();
        assert!(out.succeeded(), "output: {}", out.output);
        for empty_var in ["ANTHROPIC_API_KEY", "CARGO_MANIFEST_DIR", "USER", "TERM"] {
            assert!(
                out.output.contains(&format!("{empty_var}=[]")),
                "{empty_var} leaked into the sandbox:\n{}",
                out.output
            );
        }
        assert!(out.output.contains("PATH=[/usr/local/bin:/usr/bin:/bin]"), "output: {}", out.output);
        assert!(out.output.contains("HOME=[/tmp]"), "output: {}", out.output);
    }

    #[tokio::test]
    async fn sandbox_blocks_external_network_and_enforces_timeout() {
        let Some(policy) = sandbox_ready() else {
            return
        };
        let dir = TempDir::new().unwrap();

        // External addresses are unreachable from the fresh namespace, and
        // the failure is immediate rather than a hang.
        let py = r#"
import socket, time
s = socket.socket()
s.settimeout(5)
t0 = time.time()
try:
    s.connect(("1.1.1.1", 80))
    print("CONNECTED-BAD")
except OSError as e:
    dt = time.time() - t0
    print("BLOCKED-OK %.2f %s" % (dt, e))
    assert dt < 4.0, "connect hung: network namespace may be shared"
"#;
        let out = run_sandboxed(&policy, dir.path(), false, TIMEOUT, MAX_OUTPUT, "python3", &["-c", py])
            .await
            .unwrap();
        assert!(out.succeeded(), "output: {}", out.output);
        assert!(
            !out.output.contains("CONNECTED-BAD"),
            "external network reachable inside sandbox: {}",
            out.output
        );
        assert!(out.output.contains("BLOCKED-OK"), "output: {}", out.output);

        // A long-running command is killed at the deadline.
        let started = Instant::now();
        let err = run_sandboxed(&policy, dir.path(), false, Duration::from_millis(200), MAX_OUTPUT, "sleep", &["10"])
            .await
            .unwrap_err();
        let elapsed = started.elapsed();
        assert!(err.to_string().contains("timed out"), "error: {err}");
        assert!(elapsed < Duration::from_secs(5), "kill took {elapsed:?}");
    }
}
