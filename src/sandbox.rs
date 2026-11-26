//! Rootless [`bubblewrap`](https://github.com/containers/bubblewrap) sandbox
//! for running untrusted commands (e.g. model-generated shell) without host
//! filesystem or network access.
//!
//! The policy is a [`SandboxConfig`] passed in by the caller (loaded from the
//! XDG `config.toml` by [`crate::config`] — see its `[sandbox]` table — with
//! built-in distro-agnostic defaults when no config file exists).
//!
//! Sandbox layout: a fresh mount namespace containing only what the policy
//! binds plus bwrap's own `/proc`, `/dev`, and a private `/tmp`. `work_dir`
//! is the single writable location (bound read-write) and becomes the working
//! directory. The network namespace is unshared unless the caller passes
//! `allow_network` (loopback works inside the fresh namespace either way).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::process::Command;

use crate::config::SandboxConfig;

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
    policy: &SandboxConfig,
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
        Ok(result) => result.with_context(|| format!("running sandboxed command '{}'", program))?,
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
mod tests;
