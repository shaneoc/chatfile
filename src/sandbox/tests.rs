use super::*;
use std::time::Instant;
use tempfile::TempDir;

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_OUTPUT: usize = 16384;

/// The effective policy when `bwrap` is actually installed, otherwise a
/// skip notice and `None` (the sandbox needs bubblewrap to run). The policy
/// comes from the `[sandbox]` table of the effective XDG config.
fn sandbox_ready() -> Option<SandboxConfig> {
    let policy = crate::config::AppConfig::load()
        .expect("loading effective config")
        .sandbox;
    match find_program(&policy.bwrap_path) {
        Some(_) => Some(policy),
        None => {
            eprintln!("skipping: bwrap not installed (e.g. `apt install bubblewrap`)");
            None
        }
    }
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
