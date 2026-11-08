//! Arbitrary shapes, for bisecting a failure down to the dimension that causes
//! it.
//!
//! The 15 judged points differ in five dimensions at once, so when one fails
//! and its neighbours pass there is no way to tell which dimension matters
//! without being able to generate the shapes in between.
use std::path::{Path, PathBuf};

use crate::cases::{CASES, Case, Split};
use crate::num::Out;

pub fn dir_for(root: &Path, name: &str) -> PathBuf {
    root.join(format!("cu_{name}"))
}

/// A Case carrying an arbitrary shape. Id 0 marks it as not one of the judged
/// points, so it has no recovered grouping and `Split::Real` does not apply.
pub fn make(e: usize, m: usize, k: usize, n: usize, gt: u8, out: Out) -> Case {
    let mut case = CASES[0];
    case.id = 0;
    case.e = e;
    case.m = m;
    case.k = k;
    case.n = n;
    case.gt = gt;
    case.out = out;
    case
}

/// Re-read a shape from a case directory, so verify and dump do not need the
/// dimensions passed again and cannot disagree with what was generated.
pub fn from_dir(dir: &Path) -> Result<Case, String> {
    let text = std::fs::read_to_string(dir.join("shape.txt"))
        .map_err(|e| format!("{}/shape.txt: {e}", dir.display()))?;
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() < 6 {
        return Err(format!("{}/shape.txt is malformed: {text:?}", dir.display()));
    }
    let num = |i: usize| -> Result<usize, String> {
        parts[i].parse().map_err(|_| format!("bad field {i}: {:?}", parts[i]))
    };
    let out = match parts[5] {
        "float16" => Out::F16,
        "bfloat16" => Out::Bf16,
        other => return Err(format!("unknown output dtype {other:?}")),
    };
    Ok(make(num(0)?, num(1)?, num(2)?, num(3)?, num(4)? as u8, out))
}

/// Generate inputs and the reference for one arbitrary shape.
pub fn build(
    root: &Path,
    name: &str,
    case: &Case,
    split: Split,
) -> std::io::Result<(PathBuf, crate::golden::Golden)> {
    let dir = dir_for(root, name);
    std::fs::create_dir_all(&dir)?;
    // inputs::generate names the directory from the case id and split, so the
    // files are produced into a scratch location and moved here.
    let staging = root.join(format!(".cu_staging_{name}"));
    let _ = std::fs::remove_dir_all(&staging);
    let made = crate::inputs::generate(&staging, case, split)?;
    for entry in std::fs::read_dir(&made)? {
        let entry = entry?;
        std::fs::rename(entry.path(), dir.join(entry.file_name()))?;
    }
    let _ = std::fs::remove_dir_all(&staging);
    let golden = crate::golden::compute(&dir, case, split)?;
    crate::golden::write(&dir, &golden)?;
    Ok((dir, golden))
}
