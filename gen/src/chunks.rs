//! Loading the verbatim fragments and splicing them back together.
//!
//! Milestone 0 has no parameterization at all: `gen/chunks/MANIFEST` names the
//! fragments in order, and `emit --profile verbatim` is `concat`. The whole
//! point is that the output is provably identical to the kernel.asc we are
//! about to start dismantling, so every byte of I/O here is binary. Nothing
//! reads or writes text lines, nothing normalizes a newline, nothing appends a
//! trailing one: the fragments already carry the original line endings (LF)
//! and the original final `\n\n`.

use std::path::{Path, PathBuf};

pub const MANIFEST: &str = "MANIFEST";

/// sha256 of the kernel.asc this generator must reproduce byte for byte.
///
/// Pinned here rather than read from the file so that `check-chunks` is a real
/// check: if someone edits kernel/kernel.asc *and* the fragments in the same
/// way, this constant is what notices.
pub const VERBATIM_SHA256: &str = "5e8298d6a3544cf213d65c3dd8df73a2bc7934be8d5d014ba4710144d79f1bed";
/// Bytes and lines of that same file, as a second, cheaper tripwire.
pub const VERBATIM_BYTES: usize = 319684;
pub const VERBATIM_LINES: usize = 6122;

/// One fragment, in manifest order.
pub struct Fragment {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Where `gen/chunks/` is.
///
/// `GEN_CHUNKS` wins, so the generator can be pointed at an experimental set.
/// Otherwise walk up from the current directory looking for `gen/chunks`,
/// which makes the binary work from anywhere in the workspace, and fall back
/// to the compile-time crate directory for `cargo test` run out of tree.
pub fn chunks_dir() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os("GEN_CHUNKS") {
        let dir = PathBuf::from(dir);
        if !dir.is_dir() {
            return Err(format!("GEN_CHUNKS={} is not a directory", dir.display()));
        }
        return Ok(dir);
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut at: Option<&Path> = Some(cwd.as_path());
        while let Some(dir) = at {
            let candidate = dir.join("gen").join("chunks");
            if candidate.join(MANIFEST).is_file() {
                return Ok(candidate);
            }
            at = dir.parent();
        }
    }
    let baked = Path::new(env!("CARGO_MANIFEST_DIR")).join("chunks");
    if baked.join(MANIFEST).is_file() {
        return Ok(baked);
    }
    Err("no gen/chunks/MANIFEST found above the current directory; set GEN_CHUNKS".to_string())
}

/// The repository root, i.e. the directory that holds `gen/`.
pub fn repo_root() -> Result<PathBuf, String> {
    let dir = chunks_dir()?;
    dir.parent()
        .and_then(|crate_dir| crate_dir.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("{} has no grandparent directory", dir.display()))
}

/// Fragment names in manifest order. `#` starts a comment, blanks are skipped.
pub fn read_manifest(dir: &Path) -> Result<Vec<String>, String> {
    let path = dir.join(MANIFEST);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut names = Vec::new();
    for (no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.contains('/') || line.contains('\\') || line.contains("..") {
            return Err(format!(
                "{}:{}: fragment names are plain file names, got {line:?}",
                path.display(),
                no + 1
            ));
        }
        names.push(line.to_string());
    }
    if names.is_empty() {
        return Err(format!("{}: lists no fragments", path.display()));
    }
    Ok(names)
}

/// Every fragment the manifest names, as raw bytes.
pub fn load(dir: &Path) -> Result<Vec<Fragment>, String> {
    let mut out = Vec::new();
    for name in read_manifest(dir)? {
        let path = dir.join(&name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push(Fragment { name, bytes });
    }
    Ok(out)
}

/// The reassembled translation unit: fragments joined with nothing at all.
pub fn concat(fragments: &[Fragment]) -> Vec<u8> {
    let mut out = Vec::with_capacity(fragments.iter().map(|f| f.bytes.len()).sum());
    for fragment in fragments {
        out.extend_from_slice(&fragment.bytes);
    }
    out
}

/// What a reassembly looks like, pass or fail.
pub struct Report {
    pub fragments: usize,
    pub bytes: usize,
    pub lines: usize,
    pub digest: String,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.digest == VERBATIM_SHA256
            && self.bytes == VERBATIM_BYTES
            && self.lines == VERBATIM_LINES
    }
}

/// Reassemble from the manifest and measure the result.
pub fn verbatim(dir: &Path) -> Result<(Vec<u8>, Report), String> {
    let fragments = load(dir)?;
    let bytes = concat(&fragments);
    let report = Report {
        fragments: fragments.len(),
        bytes: bytes.len(),
        lines: bytes.iter().filter(|&&b| b == b'\n').count(),
        digest: crate::sha256::of_bytes(&bytes),
    };
    Ok((bytes, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same check as `gen check-chunks`, so a broken split cannot be
    /// committed with a green test run.
    #[test]
    fn verbatim_is_byte_exact() {
        let dir = chunks_dir().expect("gen/chunks");
        let (bytes, report) = verbatim(&dir).expect("reassembly");
        assert_eq!(report.digest, VERBATIM_SHA256, "sha256 of the reassembly");
        assert_eq!(report.bytes, VERBATIM_BYTES);
        assert_eq!(report.lines, VERBATIM_LINES);
        // CRLF anywhere means the fragments were renormalized on checkout;
        // see .gitattributes.
        assert!(!bytes.windows(2).any(|w| w == b"\r\n"), "CRLF in the fragments");
        assert!(bytes.ends_with(b"}\n\n"), "lost the original file ending");
    }

    /// And against the reference file itself, when it is where we expect.
    #[test]
    fn verbatim_matches_the_reference_file() {
        let root = repo_root().expect("repo root");
        let path = root.join("kernel").join("kernel.asc");
        let reference = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let (bytes, _) = verbatim(&chunks_dir().expect("gen/chunks")).expect("reassembly");
        assert!(bytes == reference, "reassembly differs from {}", path.display());
    }

    /// No fragment may sit in the directory unlisted, and nothing listed may be
    /// missing: an orphan is a family that silently stopped being emitted.
    #[test]
    fn manifest_covers_the_directory() {
        let dir = chunks_dir().expect("gen/chunks");
        let listed = read_manifest(&dir).expect("manifest");
        let mut present: Vec<String> = std::fs::read_dir(&dir)
            .expect("read gen/chunks")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".inc"))
            .collect();
        present.sort();
        let mut sorted = listed.clone();
        sorted.sort();
        assert_eq!(sorted, present, "MANIFEST and gen/chunks/*.inc disagree");
        // The numeric prefix keeps the directory listing in emission order.
        assert_eq!(listed, sorted, "MANIFEST is not in file-name order");
    }
}
