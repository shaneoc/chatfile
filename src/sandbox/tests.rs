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
    assert_eq!(
        SandboxPolicy::load_from(&[empty.to_path_buf()]).unwrap(),
        SandboxPolicy::default()
    );

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
    assert_eq!(
        policy.env.get("PATH").map(String::as_str),
        Some("/usr/local/bin:/usr/bin:/bin")
    );

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
    assert_eq!(
        policy.extra_args,
        vec!["--ro-bind-try".to_string(), "/var".to_string()]
    );

    // Missing files are skipped; invalid YAML is an error.
    let missing = dir.path().join("nope.yaml");
    assert_eq!(
        SandboxPolicy::load_from(&[missing]).unwrap(),
        SandboxPolicy::default()
    );
    let bad = dir.path().join("bad.yaml");
    std::fs::write(&bad, "bwrap_path: [unclosed\n").unwrap();
    assert!(SandboxPolicy::load_from(&[bad]).is_err());
}

#[tokio::test]
async fn sandbox_allows_workdir_and_loopback() {
    let Some(policy) = sandbox_ready() else {
        return;
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
        &[
            "-c",
            "cat work.txt && echo modified > work.txt && cat work.txt",
        ],
    )
    .await
    .unwrap();
    assert!(out.succeeded(), "output: {}", out.output);
    assert_eq!(out.output.trim(), "original\nmodified");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("work.txt"))
            .unwrap()
            .trim(),
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
    let out = run_sandboxed(
        &policy,
        dir.path(),
        false,
        TIMEOUT,
        MAX_OUTPUT,
        "python3",
        &["-c", py],
    )
    .await
    .unwrap();
    assert!(out.succeeded(), "output: {}", out.output);
    assert!(
        out.output.contains("LOOPBACK-OK b'pong'"),
        "output: {}",
        out.output
    );
}

#[tokio::test]
async fn sandbox_blocks_host_filesystem_and_secrets() {
    let Some(policy) = sandbox_ready() else {
        return;
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
        &[
            "-c",
            "echo pwn > /usr/pwned 2>/dev/null && echo PWNED || echo BLOCKED",
        ],
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
        &[
            "-c",
            "for v in ANTHROPIC_API_KEY CARGO_MANIFEST_DIR USER TERM PATH HOME; do \
                eval \"val=\\${$v:-}\"; printf '%s=[%s]\\n' \"$v\" \"$val\"; done",
        ],
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
    assert!(
        out.output.contains("PATH=[/usr/local/bin:/usr/bin:/bin]"),
        "output: {}",
        out.output
    );
    assert!(out.output.contains("HOME=[/tmp]"), "output: {}", out.output);
}

#[tokio::test]
async fn sandbox_blocks_external_network_and_enforces_timeout() {
    let Some(policy) = sandbox_ready() else {
        return;
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
    let out = run_sandboxed(
        &policy,
        dir.path(),
        false,
        TIMEOUT,
        MAX_OUTPUT,
        "python3",
        &["-c", py],
    )
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
    let err = run_sandboxed(
        &policy,
        dir.path(),
        false,
        Duration::from_millis(200),
        MAX_OUTPUT,
        "sleep",
        &["10"],
    )
    .await
    .unwrap_err();
    let elapsed = started.elapsed();
    assert!(err.to_string().contains("timed out"), "error: {err}");
    assert!(elapsed < Duration::from_secs(5), "kill took {elapsed:?}");
}
