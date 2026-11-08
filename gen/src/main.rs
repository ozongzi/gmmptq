//! gen -- the kernel.asc generator (see gen/DESIGN.md).
//!
//! Milestone 0 only. `emit --profile verbatim` splices the fragments in
//! `gen/chunks/` back into the current `kernel/kernel.asc`, byte for byte, and
//! `check-chunks` proves it. Nothing is parameterized yet: this is the safety
//! net that later milestones replace one family at a time, and the sha256 is
//! what says a replacement changed nothing it was not meant to.
//!
//!     cargo run -p gen -- emit --profile verbatim -o kernel/kernel.asc
//!     cargo run -p gen -- check-chunks
//!
//! Zero dependencies, so the argument parsing is by hand -- same as `fixture`.
mod chunks;
mod sha256;
mod tmpl;

use std::io::Write;
use std::path::PathBuf;

fn usage() -> ! {
    eprint!(
        "{}",
        "gen <command>\n\
         \n\
         emit --profile <name> [-o <path>]   generate kernel.asc (stdout without -o)\n\
         check-chunks [-v]                   verify the verbatim reassembly's sha256\n\
         plan <E> <M> <K> <N> <gt> <dtype> <counts...>\n\
         \n\
         \x20                                dry run: which strategy a shape takes\n\
         classes                             the shape classifier's classes and predicates\n\
         \n\
         profiles: verbatim  the milestone 0 safety net, the current kernel as-is\n\
         \x20         final     the compliant submission        (milestone 1+)\n\
         \x20         probe     with the hidden-point probes    (milestone 1+)\n\
         \x20         spin      with the spin-to-a-fixed-time kernels (milestone 1+)\n"
    );
    std::process::exit(2)
}

/// Anything DESIGN.md specifies but milestone 0 does not build. Exiting 3 with
/// a named milestone beats a stub that quietly emits something plausible.
fn not_yet(what: &str, why: &str) -> i32 {
    eprintln!("gen: {what} is not implemented in milestone 0.");
    eprintln!("     {why}");
    eprintln!("     See gen/DESIGN.md, \"分阶段实施\".");
    3
}

#[derive(Clone, Copy, PartialEq)]
enum Profile {
    Verbatim,
    Final,
    Probe,
    Spin,
}

impl Profile {
    fn parse(name: &str) -> Option<Profile> {
        match name {
            "verbatim" => Some(Profile::Verbatim),
            "final" => Some(Profile::Final),
            "probe" => Some(Profile::Probe),
            "spin" => Some(Profile::Spin),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Profile::Verbatim => "verbatim",
            Profile::Final => "final",
            Profile::Probe => "probe",
            Profile::Spin => "spin",
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let code = match argv.get(1).map(String::as_str) {
        Some("emit") => cmd_emit(&argv),
        Some("check-chunks") => cmd_check_chunks(&argv),
        Some("plan") => not_yet(
            "plan",
            "it needs classify.rs, which cannot exist before a single predicate does.",
        ),
        Some("classes") => not_yet(
            "classes",
            "the shape classifier is milestone 1; today's kernel dispatches by equality.",
        ),
        _ => usage(),
    };
    std::process::exit(code)
}

/// Generate a translation unit. Only `verbatim` produces bytes today.
fn cmd_emit(argv: &[String]) -> i32 {
    let mut profile: Option<Profile> = None;
    let mut out: Option<PathBuf> = None;
    let mut at = 2;
    while at < argv.len() {
        match argv[at].as_str() {
            "--profile" | "-p" => {
                let Some(name) = argv.get(at + 1) else { usage() };
                let Some(parsed) = Profile::parse(name) else {
                    eprintln!("gen: unknown profile {name:?}; want verbatim final probe spin");
                    return 2;
                };
                profile = Some(parsed);
                at += 2;
            }
            "-o" | "--output" => {
                let Some(path) = argv.get(at + 1) else { usage() };
                out = Some(PathBuf::from(path));
                at += 2;
            }
            other => {
                eprintln!("gen emit: unexpected argument {other:?}");
                return 2;
            }
        }
    }
    let Some(profile) = profile else {
        eprintln!("gen emit: --profile is required");
        return 2;
    };
    match profile {
        Profile::Verbatim => {}
        Profile::Final => {
            return not_yet(
                "--profile final",
                "it must refuse probe/spin constructs and per-point equality predicates, \
                 and there is nothing to refuse until the emitters exist.",
            );
        }
        Profile::Probe | Profile::Spin => {
            return not_yet(
                &format!("--profile {}", profile.name()),
                "the probe and spin kernels become generator output in milestone 1; \
                 until then use the ones already in the verbatim fragments.",
            );
        }
    }

    let dir = match chunks::chunks_dir() {
        Ok(d) => d,
        Err(error) => {
            eprintln!("gen emit: {error}");
            return 1;
        }
    };
    let (bytes, report) = match chunks::verbatim(&dir) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("gen emit: {error}");
            return 1;
        }
    };
    // The verbatim fragments hold no placeholders; checking anyway means the
    // gate is already wired when the first parameterized family lands.
    if let Ok(text) = std::str::from_utf8(&bytes) {
        if let Err(error) = tmpl::assert_clean(text) {
            eprintln!("gen emit: {error}");
            return 1;
        }
    }
    if !report.passed() {
        eprintln!(
            "gen emit: refusing to write a verbatim profile that is not verbatim\n  \
             sha256 {} ({} bytes, {} lines)\n  want   {} ({} bytes, {} lines)",
            report.digest,
            report.bytes,
            report.lines,
            chunks::VERBATIM_SHA256,
            chunks::VERBATIM_BYTES,
            chunks::VERBATIM_LINES
        );
        return 1;
    }
    match out {
        // Binary write, no newline translation: on this machine core.autocrlf
        // is true and the whole point of the profile is byte equality.
        Some(path) => {
            if let Err(error) = std::fs::write(&path, &bytes) {
                eprintln!("gen emit: {}: {error}", path.display());
                return 1;
            }
            eprintln!(
                "{}  {}  {} bytes from {} fragments",
                report.digest,
                path.display(),
                report.bytes,
                report.fragments
            );
        }
        None => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            if let Err(error) = lock.write_all(&bytes).and_then(|()| lock.flush()) {
                eprintln!("gen emit: stdout: {error}");
                return 1;
            }
        }
    }
    0
}

/// Reassemble from the manifest and compare against the pinned digest.
fn cmd_check_chunks(argv: &[String]) -> i32 {
    let verbose = argv.iter().skip(2).any(|a| a == "-v" || a == "--verbose");
    let dir = match chunks::chunks_dir() {
        Ok(d) => d,
        Err(error) => {
            eprintln!("check-chunks: {error}");
            return 1;
        }
    };
    let fragments = match chunks::load(&dir) {
        Ok(f) => f,
        Err(error) => {
            eprintln!("check-chunks: {error}");
            return 1;
        }
    };
    if verbose {
        let mut line = 1usize;
        for fragment in &fragments {
            let lines = fragment.bytes.iter().filter(|&&b| b == b'\n').count();
            println!(
                "  {:>6} {:>5}L {:>7}B  {}",
                line,
                lines,
                fragment.bytes.len(),
                fragment.name
            );
            line += lines;
        }
    }
    let bytes = chunks::concat(&fragments);
    let digest = sha256::of_bytes(&bytes);
    let lines = bytes.iter().filter(|&&b| b == b'\n').count();
    println!(
        "{} fragments  {} bytes  {} lines  sha256 {digest}",
        fragments.len(),
        bytes.len(),
        lines
    );
    if digest == chunks::VERBATIM_SHA256
        && bytes.len() == chunks::VERBATIM_BYTES
        && lines == chunks::VERBATIM_LINES
    {
        println!("PASS  the reassembly is kernel/kernel.asc byte for byte");
        return 0;
    }
    println!(
        "FAIL  want sha256 {} ({} bytes, {} lines)",
        chunks::VERBATIM_SHA256,
        chunks::VERBATIM_BYTES,
        chunks::VERBATIM_LINES
    );
    // Say *where* it diverged when the reference file is at hand; "the hash
    // differs" is not actionable on a 320 KB file.
    if let Ok(root) = chunks::repo_root() {
        let path = root.join("kernel").join("kernel.asc");
        if let Ok(reference) = std::fs::read(&path) {
            match first_difference(&bytes, &reference) {
                None => println!("      but it equals {} -- update the pinned constants", path.display()),
                Some(at) => {
                    let line = bytes[..at.min(bytes.len())].iter().filter(|&&b| b == b'\n').count() + 1;
                    let (owner, offset) = locate(&fragments, at);
                    println!(
                        "      first difference at byte {at} (reassembled line {line}), \
                         inside {owner} +{offset}B"
                    );
                }
            }
        }
    }
    1
}

fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    let common = got.len().min(want.len());
    for at in 0..common {
        if got[at] != want[at] {
            return Some(at);
        }
    }
    if got.len() == want.len() { None } else { Some(common) }
}

/// Which fragment a byte offset of the reassembly falls in.
fn locate(fragments: &[chunks::Fragment], at: usize) -> (String, usize) {
    let mut base = 0usize;
    for fragment in fragments {
        if at < base + fragment.bytes.len() {
            return (fragment.name.clone(), at - base);
        }
        base += fragment.bytes.len();
    }
    ("past the last fragment".to_string(), at.saturating_sub(base))
}
