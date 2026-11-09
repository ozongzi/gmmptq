//! Assembles `kernel/src/**` into the single-translation-unit `kernel/kernel.asc`
//! that the judge compiles.
//!
//! The judge accepts one file, so the sources are split for editing only and
//! concatenated here in the order given by `kernel/src/manifest.txt`. Order is
//! load-bearing: the result is one translation unit, so every declaration must
//! precede its uses.
//!
//!   cargo asm           rewrite kernel/kernel.asc from kernel/src
//!   cargo asm --check   exit non-zero if kernel.asc is stale (for CI/pre-commit)

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Repository-relative location of the pieces and of the assembled output.
const SRC_DIR: &str = "kernel/src";
const MANIFEST: &str = "kernel/src/manifest.txt";
const OUTPUT: &str = "kernel/kernel.asc";

fn main() -> ExitCode {
    let check = std::env::args().any(|a| a == "--check");
    let root = match repo_root() {
        Some(root) => root,
        None => {
            eprintln!("asm: run inside the GroupedMatmulPerTokenQuant checkout");
            return ExitCode::FAILURE;
        }
    };
    let assembled = match assemble(&root) {
        Ok(text) => text,
        Err(message) => {
            eprintln!("asm: {message}");
            return ExitCode::FAILURE;
        }
    };
    let output = root.join(OUTPUT);
    let current = std::fs::read_to_string(&output).unwrap_or_default();
    if check {
        if current == assembled {
            println!("asm: {OUTPUT} is up to date");
            return ExitCode::SUCCESS;
        }
        eprintln!("asm: {OUTPUT} is stale; run `cargo asm`");
        return ExitCode::FAILURE;
    }
    if current == assembled {
        println!("asm: {OUTPUT} unchanged ({} lines)", assembled.lines().count());
        return ExitCode::SUCCESS;
    }
    if let Err(error) = std::fs::write(&output, &assembled) {
        eprintln!("asm: cannot write {OUTPUT}: {error}");
        return ExitCode::FAILURE;
    }
    println!("asm: wrote {OUTPUT} ({} lines)", assembled.lines().count());
    ExitCode::SUCCESS
}

/// Walks up from the current directory to the checkout that holds the manifest.
fn repo_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join(MANIFEST).is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Concatenates every manifest entry, each behind a banner naming its source.
fn assemble(root: &Path) -> Result<String, String> {
    let manifest = root.join(MANIFEST);
    let text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("cannot read {MANIFEST}: {error}"))?;
    let mut out = String::with_capacity(512 * 1024);
    let mut seen = 0usize;
    for (index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.contains("..") || line.starts_with('/') {
            return Err(format!("{MANIFEST}:{}: bad path `{line}`", index + 1));
        }
        let path = root.join(SRC_DIR).join(line);
        let piece = std::fs::read_to_string(&path)
            .map_err(|error| format!("{MANIFEST}:{}: {line}: {error}", index + 1))?;
        let _ = writeln!(out, "// ===== {line} {:=<w$}", "", w = 66usize.saturating_sub(line.len()));
        out.push_str(&piece);
        if !piece.ends_with('\n') {
            out.push('\n');
        }
        seen += 1;
    }
    if seen == 0 {
        return Err(format!("{MANIFEST} lists no sources"));
    }
    Ok(out)
}
