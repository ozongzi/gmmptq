//! A fixture's identity: every file of every case, by content.
//!
//! Generating a fixture is deterministic -- the RNG is seeded from what each
//! stream is for, not from a counter -- so the same command on another machine
//! writes the same bytes. The manifest is how that stops being a claim: write
//! it once beside a fixture that has been checked against a reference, commit
//! the manifest (a few KB), and `check` proves a freshly generated tree on a
//! new container is the same tree. The data itself never needs to be stored or
//! copied anywhere.
//!
//! The format is one line per file, `<sha256>  <case>/<file>`, sorted. Plain
//! text so a diff says which file moved, and so writing it needs no JSON.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use crate::cases::{Case, Split};
use crate::groups;
use crate::inputs::case_dir;
use crate::sha256;

pub const NAME: &str = "manifest.txt";
/// Files a run produces rather than the fixture defining; never hashed.
const TRANSIENT: [&str; 4] = ["out.bin", "driver.err", "suite_verify.log", NAME];

fn collect(root: &Path, cases: &[Case], splits: &[Split]) -> std::io::Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for case in cases {
        for &split in splits {
            let dir = case_dir(root, case, split);
            if !dir.is_dir() {
                continue;
            }
            let label = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let mut names: Vec<String> = std::fs::read_dir(&dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().is_file())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .filter(|name| !TRANSIENT.contains(&name.as_str()) && !name.ends_with(".tmp"))
                .collect();
            names.sort();
            for name in names {
                let digest = sha256::of_file(&dir.join(&name))?;
                out.insert(format!("{label}/{name}"), digest);
            }
        }
    }
    Ok(out)
}

/// Write `manifest.txt` at the fixture root. Returns the number of files.
pub fn write(root: &Path, cases: &[Case], splits: &[Split]) -> std::io::Result<usize> {
    let entries = collect(root, cases, splits)?;
    if entries.is_empty() {
        return Ok(0);
    }
    let mut text = String::new();
    text.push_str("# fixture manifest: sha256 of every generated file\n");
    text.push_str(&format!("# groups {} sha256 {}\n", groups::SOURCE, groups::SOURCE_SHA256));
    for (path, digest) in &entries {
        text.push_str(&format!("{digest}  {path}\n"));
    }
    let mut file = std::fs::File::create(root.join(NAME))?;
    file.write_all(text.as_bytes())?;
    Ok(entries.len())
}

pub struct Diff {
    pub matched: usize,
    pub changed: Vec<String>,
    pub missing: Vec<String>,
    pub extra: Vec<String>,
}

impl Diff {
    pub fn ok(&self) -> bool {
        self.changed.is_empty() && self.missing.is_empty() && self.extra.is_empty()
    }
}

/// Compare what is on disk against `manifest.txt`.
pub fn check(root: &Path, cases: &[Case], splits: &[Split]) -> std::io::Result<Diff> {
    let text = std::fs::read_to_string(root.join(NAME))?;
    let mut expected = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((digest, path)) = line.split_once("  ") else {
            continue;
        };
        expected.insert(path.to_string(), digest.to_string());
    }
    let actual = collect(root, cases, splits)?;
    let mut diff = Diff { matched: 0, changed: Vec::new(), missing: Vec::new(), extra: Vec::new() };
    for (path, digest) in &expected {
        match actual.get(path) {
            Some(found) if found == digest => diff.matched += 1,
            Some(_) => diff.changed.push(path.clone()),
            None => diff.missing.push(path.clone()),
        }
    }
    for path in actual.keys() {
        if !expected.contains_key(path) {
            diff.extra.push(path.clone());
        }
    }
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CASES;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("fixture-{tag}-{}-{unique}", std::process::id()))
    }

    /// Generate a small case, manifest it, and prove the check notices a
    /// single flipped byte as well as an added or removed file.
    #[test]
    fn a_manifest_pins_every_byte() {
        let root = scratch("manifest");
        let case = *Case::find(3).unwrap();
        let splits = [Split::Real];
        crate::inputs::generate(&root, &case, Split::Real).unwrap();
        let golden = crate::golden::compute(&case_dir(&root, &case, Split::Real), &case, Split::Real).unwrap();
        crate::golden::write(&case_dir(&root, &case, Split::Real), &golden).unwrap();

        let written = write(&root, &[case], &splits).unwrap();
        assert!(written >= 6, "only {written} files");
        assert!(check(&root, &[case], &splits).unwrap().ok());

        // A run's own output must not count as part of the fixture.
        std::fs::write(case_dir(&root, &case, Split::Real).join("out.bin"), b"x").unwrap();
        assert!(check(&root, &[case], &splits).unwrap().ok());

        let x = case_dir(&root, &case, Split::Real).join("x.bin");
        let mut bytes = std::fs::read(&x).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&x, &bytes).unwrap();
        let diff = check(&root, &[case], &splits).unwrap();
        assert_eq!(diff.changed, vec!["tp03_real/x.bin".to_string()]);

        std::fs::remove_file(&x).unwrap();
        assert_eq!(check(&root, &[case], &splits).unwrap().missing, vec!["tp03_real/x.bin".to_string()]);

        std::fs::write(case_dir(&root, &case, Split::Real).join("surprise.bin"), b"?").unwrap();
        let diff = check(&root, &[case], &splits).unwrap();
        assert!(diff.extra.contains(&"tp03_real/surprise.bin".to_string()));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Two independent generations of the same case agree byte for byte. This
    /// is what lets the data be thrown away and regenerated per container.
    #[test]
    fn generation_is_reproducible() {
        let (a, b) = (scratch("repeat-a"), scratch("repeat-b"));
        let case = *Case::find(5).unwrap();
        for root in [&a, &b] {
            crate::inputs::generate(root, &case, Split::Real).unwrap();
            let dir = case_dir(root, &case, Split::Real);
            let golden = crate::golden::compute(&dir, &case, Split::Real).unwrap();
            crate::golden::write(&dir, &golden).unwrap();
        }
        let left = collect(&a, &[case], &[Split::Real]).unwrap();
        let right = collect(&b, &[case], &[Split::Real]).unwrap();
        assert_eq!(left, right);
        assert!(!left.is_empty());
        for root in [a, b] {
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn every_case_has_a_real_grouping_to_generate() {
        for case in CASES.iter() {
            assert!(groups::counts(case.id).is_some(), "tp{}", case.id);
        }
    }
}
