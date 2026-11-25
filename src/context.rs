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
mod tests;
