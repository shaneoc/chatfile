//! Hierarchical context: walk upward from a conversation file's directory
//! toward the vault root, collecting `AGENTS.md` files along the way.
//!
//! The returned system-prompt fragment is ordered **root-first** (most
//! general instructions first, most specific last), mirroring how layered
//! configuration is typically read.

use std::path::{Path, PathBuf};

/// Name of the per-directory context file.
pub const AGENTS_FILE: &str = "AGENTS.md";

/// Marker file that designates a non-git vault root.
const VAULT_MARKER: &str = ".chatfile";

/// Directories from `start_dir` up to (and including) the boundary.
///
/// The walk stops at the first directory containing a `.git` entry or a
/// `.chatfile` marker file, or at the filesystem root. Order: start dir first.
pub fn agent_chain(start_dir: &Path) -> Vec<PathBuf> {
    let mut chain: Vec<PathBuf> = Vec::new();
    let mut dir: Option<&Path> = Some(start_dir);
    while let Some(d) = dir {
        chain.push(d.to_path_buf());
        if d.join(".git").exists() || d.join(VAULT_MARKER).exists() {
            break; // boundary reached (inclusive)
        }
        dir = d.parent();
    }
    chain
}

/// Find `AGENTS.md` files on the path from `start_dir` up to the boundary.
/// Returns paths ordered root-first (deepest last).
pub fn find_agents_files(start_dir: &Path) -> Vec<PathBuf> {
    agent_chain(start_dir)
        .into_iter()
        .rev()
        .map(|d| d.join(AGENTS_FILE))
        .filter(|p| p.is_file())
        .collect()
}

/// Concatenate all `AGENTS.md` files found above `start_dir`, root-first,
/// separated by blank lines. Returns an empty string when none exist.
pub fn collect_agents_context(start_dir: &Path) -> String {
    find_agents_files(start_dir)
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Build the full system prompt for a conversation file: hierarchical
/// `AGENTS.md` context followed by the file's `**System**` section content
/// (when present). This is the single system message sent as the first
/// message of every request.
pub fn build_system_prompt(md_file: &Path, system_section: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(dir) = md_file.parent() {
        let ctx = collect_agents_context(dir);
        if !ctx.is_empty() {
            parts.push(ctx);
        }
    }
    if let Some(sp) = system_section.map(str::trim).filter(|s| !s.is_empty()) {
        parts.push(sp.to_string());
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Create a unique temp directory for a test.
    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "chatfile-test-{}-{}-{}",
            tag,
            std::process::id(),
            TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn no_agents_files_returns_empty() {
        let dir = tempdir("empty");
        fs::write(dir.join(VAULT_MARKER), "").unwrap(); // bound the walk here
        assert!(find_agents_files(&dir).is_empty());
        assert_eq!(collect_agents_context(&dir), "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn chain_is_root_first_and_bounded_by_marker() {
        let root = tempdir("chain");
        write(&root.join(AGENTS_FILE), "ROOT RULES");
        fs::write(root.join(VAULT_MARKER), "").unwrap(); // bound the walk here
        let mid = root.join("a").join("b");
        write(&mid.join(AGENTS_FILE), "MID RULES");
        let leaf = mid.join("c");
        write(&leaf.join(AGENTS_FILE), "LEAF RULES");

        let files = find_agents_files(&leaf);
        assert_eq!(files.len(), 3);
        assert_eq!(files[0], root.join(AGENTS_FILE)); // most general first
        assert_eq!(files[1], mid.join(AGENTS_FILE));
        assert_eq!(files[2], leaf.join(AGENTS_FILE)); // most specific last
        assert_eq!(collect_agents_context(&leaf), "ROOT RULES\n\nMID RULES\n\nLEAF RULES");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn unbounded_chain_reaches_filesystem_root() {
        let base = tempdir("unbounded");
        let leaf = base.join("a").join("b");
        fs::create_dir_all(&leaf).unwrap();
        // No .git / .chatfile anywhere below: the chain must end at "/".
        let chain = agent_chain(&leaf);
        assert_eq!(chain[0], leaf);
        assert_eq!(chain.last().map(|p| p.as_path()), Some(Path::new("/")));
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn git_root_stops_the_walk() {
        let base = tempdir("git");
        // Sibling of the vault root: must never be collected.
        write(&base.join("sibling").join(AGENTS_FILE), "SHOULD NOT SEE");
        let root = base.join("vault");
        write(&root.join(AGENTS_FILE), "VAULT RULES");
        fs::create_dir_all(root.join(".git")).unwrap();

        let deep = root.join("x").join("y");
        fs::create_dir_all(&deep).unwrap();
        write(&deep.join(AGENTS_FILE), "DEEP RULES");

        let files = find_agents_files(&deep);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], root.join(AGENTS_FILE)); // root first
        assert_eq!(files[1], deep.join(AGENTS_FILE));
        assert_eq!(collect_agents_context(&deep), "VAULT RULES\n\nDEEP RULES");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn vault_marker_stops_the_walk() {
        let root = tempdir("vault");
        write(&root.join(AGENTS_FILE), "VAULT RULES");
        fs::write(root.join(VAULT_MARKER), "").unwrap();
        let deep = root.join("notes");
        fs::create_dir_all(&deep).unwrap();

        let files = find_agents_files(&deep);
        assert_eq!(files, vec![root.join(AGENTS_FILE)]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_or_empty_agents_files_are_skipped() {
        let root = tempdir("sparse");
        fs::write(root.join(VAULT_MARKER), "").unwrap();
        write(&root.join(AGENTS_FILE), "   \n"); // whitespace only
        let deep = root.join("d");
        fs::create_dir_all(&deep).unwrap();
        assert_eq!(collect_agents_context(&deep), "");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn build_system_prompt_combines_context_and_system_section() {
        let dir = tempdir("sysprompt");
        write(&dir.join(AGENTS_FILE), "Vault rules.");
        fs::write(dir.join(VAULT_MARKER), "").unwrap();
        let md = dir.join("conv.md");
        fs::write(&md, "").unwrap();

        assert_eq!(build_system_prompt(&md, None), "Vault rules.");
        assert_eq!(
            build_system_prompt(&md, Some("Be terse.")),
            "Vault rules.\n\nBe terse."
        );
        assert_eq!(build_system_prompt(&md, Some("  ")), "Vault rules.");

        let _ = fs::remove_dir_all(&dir);
    }
}
