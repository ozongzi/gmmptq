//! fixture -- local test data and the reference answer for
//! GroupedMatmulPerTokenQuant.
//!
//! Zero dependencies on purpose: this is cross-compiled to the devenv
//! container, which has no cargo and no crates.io, and it must run there
//! without a project checkout, a database or a network.
//!
//!     cargo fixture cases                      the 15 points and their groupings
//!     cargo fixture gen      <dir> [case] [split]
//!     cargo fixture golden   <dir> [case] [split]
//!     cargo fixture verify   <dir> <out.bin> <case> <split>
//!     cargo fixture suite    <dir> [split]     gen + golden + manifest
//!     cargo fixture manifest <dir> [case] [split]
//!     cargo fixture check    <dir> [case] [split]
//!
//! `real` is the judge's own grouping and the only one to measure on. The other
//! five are correctness stress shapes; timing on them has produced reversed
//! conclusions.
mod cases;
mod custom;
mod examples;
mod golden;
mod groups;
mod inputs;
mod manifest;
mod num;
mod rng;
mod sha256;
mod verify;

use std::path::{Path, PathBuf};

use cases::{CASES, Case, SPLITS, Split};

fn usage() -> ! {
    eprint!(
        "{}",
        "fixture <command>\n\
         \n\
         cases                          the 15 points, shapes and groupings\n\
         gen      <dir> [case] [split]  write the input tensors\n\
         golden   <dir> [case] [split]  compute and store the reference\n\
         verify   <dir> <out.bin> <case> <split>\n\
         dump     <dir> <out.bin> <case> <split> [limit]\n\
         suite    <dir> [split]         gen + golden + manifest for every case\n\
         manifest <dir> [case] [split]  write manifest.txt (sha256 of each file)\n\
         check    <dir> [case] [split]  compare the tree against manifest.txt\n\
         examples <dir>                 the statement's three worked examples\n\
         exverify <dir> <name> <out.bin>\n\
         custom   <dir> <name> E M K N gt float16|bfloat16 [split]\n\
         dverify  <dir> <out.bin>       verify any case dir against its golden\n\
         \n\
         case is 1..15 (default: all); split is one of real even skew empty\n\
         tail head (default: all). Only `real` is the judge's grouping.\n"
    );
    std::process::exit(2)
}

fn pick_case(arg: Option<&String>) -> Vec<Case> {
    match arg {
        None => CASES.to_vec(),
        Some(text) => match text.trim_start_matches("tp").parse::<u32>() {
            Ok(id) => match Case::find(id) {
                Some(c) => vec![*c],
                None => {
                    eprintln!("no case {id}; cases are 1..15");
                    std::process::exit(2)
                }
            },
            Err(_) => {
                eprintln!("case must be a number 1..15, got {text:?}");
                std::process::exit(2)
            }
        },
    }
}

fn pick_split(arg: Option<&String>) -> Vec<Split> {
    match arg {
        None => SPLITS.to_vec(),
        Some(text) => match Split::parse(text) {
            Some(s) => vec![s],
            None => {
                eprintln!("no split {text:?}; splits are real even skew empty tail head");
                std::process::exit(2)
            }
        },
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        usage();
    }
    let code = match argv[1].as_str() {
        "cases" => cmd_cases(),
        "gen" => cmd_gen(&argv),
        "golden" => cmd_golden(&argv),
        "verify" => cmd_verify(&argv),
        "dump" => cmd_dump(&argv),
        "suite" => cmd_suite(&argv),
        "manifest" => cmd_manifest(&argv),
        "check" => cmd_check(&argv),
        "examples" => cmd_examples(&argv),
        "exverify" => cmd_exverify(&argv),
        "custom" => cmd_custom(&argv),
        "dverify" => cmd_dverify(&argv),
        other => {
            eprintln!("unknown command {other:?}");
            usage()
        }
    };
    std::process::exit(code);
}

fn cmd_cases() -> i32 {
    println!(
        "{:>4} {:>4} {:>6} {:>6} {:>6} {:>3} {:>8} {:>9} {:>8} {:>9}",
        "case", "E", "M", "K", "N", "gt", "out", "wscale", "MAC", "io"
    );
    for case in CASES.iter() {
        println!(
            "tp{:<2} {:>4} {:>6} {:>6} {:>6} {:>3} {:>8} {:>9} {:>7.2}G {:>8.1}M",
            case.id,
            case.e,
            case.m,
            case.k,
            case.n,
            case.gt,
            case.out.name(),
            if case.wscale_bytes() == 4 { "float32" } else { "bfloat16" },
            case.macs() as f64 / 1e9,
            case.io_bytes() as f64 / 1e6,
        );
    }
    println!("\ngt=0 是累计结束位置，gt=1 是各专家 token 数。");
    println!("判题器真实分组（{} 的 {}）：", groups::SOURCE, &groups::SOURCE_SHA256[..12]);
    for case in CASES.iter() {
        let counts = cases::counts(case, Split::Real);
        let empty = counts.iter().filter(|&&c| c == 0).count();
        let shown = if counts.len() > 8 {
            format!("{:?}… ({} 个专家)", &counts[..8], counts.len())
        } else {
            format!("{counts:?}")
        };
        println!("  tp{:<3} 空 {:>3}  {}", case.id, empty, shown);
    }
    println!("\n其余分组只用于正确性，不要用来计时：");
    println!("  {}", SPLITS.iter().map(|s| s.name()).collect::<Vec<_>>().join(" "));
    0
}

fn cmd_gen(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    for case in pick_case(argv.get(3)) {
        for split in pick_split(argv.get(4)) {
            match inputs::generate(&root, &case, split) {
                Ok(path) => println!("  tp{:<3} {:<6} -> {}", case.id, split.name(), path.display()),
                Err(error) => {
                    eprintln!("tp{} {}: {error}", case.id, split.name());
                    return 1;
                }
            }
        }
    }
    0
}

fn cmd_golden(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    for case in pick_case(argv.get(3)) {
        for split in pick_split(argv.get(4)) {
            let path = inputs::case_dir(&root, &case, split);
            let started = std::time::Instant::now();
            match golden::compute(&path, &case, split) {
                Ok(g) => {
                    if let Err(error) = golden::write(&path, &g) {
                        eprintln!("tp{} {}: {error}", case.id, split.name());
                        return 1;
                    }
                    println!(
                        "  tp{:<3} {:<6} {:>10} 元素  max|y|={:>10.2}  max|acc|={:>11}  \
                         {:>6.2}s on {} threads",
                        case.id,
                        split.name(),
                        case.m * case.n,
                        g.max_abs,
                        g.max_acc,
                        started.elapsed().as_secs_f64(),
                        g.threads,
                    );
                }
                Err(error) => {
                    eprintln!("tp{} {}: {error}", case.id, split.name());
                    return 1;
                }
            }
        }
    }
    0
}

fn read_golden(path: &Path, case: &Case) -> Result<Vec<u16>, String> {
    let bytes = std::fs::read(path.join("golden.bin"))
        .map_err(|e| format!("{}: {e}; run `golden` first", path.display()))?;
    if bytes.len() != case.m * case.n * 2 {
        return Err(format!("golden.bin is {} bytes, wrong for this case", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn cmd_verify(argv: &[String]) -> i32 {
    let (Some(dir), Some(out), Some(case_arg), Some(split_arg)) =
        (argv.get(2), argv.get(3), argv.get(4), argv.get(5))
    else {
        usage()
    };
    let case = pick_case(Some(case_arg))[0];
    let split = pick_split(Some(split_arg))[0];
    let path = inputs::case_dir(Path::new(dir), &case, split);
    let golden_bits = match read_golden(&path, &case) {
        Ok(g) => g,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let got = match std::fs::read(out) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("{out}: {error}");
            return 1;
        }
    };
    match verify::compare(&case, &got, &golden_bits) {
        Err(error) => {
            eprintln!("{error}");
            1
        }
        Ok(r) => {
            println!("tp{} {} ({})", case.id, split.name(), case.out.name());
            println!("  元素           {}", r.elements);
            println!(
                "  逐位不同       {} ({:.4}%)",
                r.differing,
                r.differing as f64 / r.elements as f64 * 100.0
            );
            println!("  max 绝对误差   {:.6e}", r.max_abs);
            println!("  max 相对误差   {:.6e}", r.max_rel);
            println!("  失败(宽松读法) {}   相对或绝对任一在 1e-3 内即算过", r.fail_either);
            println!("  失败(严格读法) {}   两者都要在 1e-3 内", r.fail_both);
            println!(
                "  最差元素 #{}   得到 {:.6} 期望 {:.6}",
                r.worst_index, r.worst_got, r.worst_want
            );
            println!("  表示误差下限   {:.3e}  (这个输出类型本身的 eps)", case.out.eps());
            if r.passed() {
                println!("  => PASS");
                0
            } else {
                println!("  => FAIL");
                1
            }
        }
    }
}

/// Everything a container needs in one command: inputs, reference, manifest.
fn cmd_suite(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    let splits = pick_split(argv.get(3));
    let started = std::time::Instant::now();
    let mut done = 0usize;
    for case in CASES.iter() {
        for &split in &splits {
            if let Err(error) = inputs::generate(&root, case, split) {
                eprintln!("gen tp{} {}: {error}", case.id, split.name());
                return 1;
            }
            let path = inputs::case_dir(&root, case, split);
            let g = match golden::compute(&path, case, split) {
                Ok(g) => g,
                Err(error) => {
                    eprintln!("golden tp{} {}: {error}", case.id, split.name());
                    return 1;
                }
            };
            if let Err(error) = golden::write(&path, &g) {
                eprintln!("write tp{} {}: {error}", case.id, split.name());
                return 1;
            }
            // The reference compared against itself must be exact; if this
            // ever fails, the comparison itself is broken.
            let bytes: Vec<u8> = g.bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            match verify::compare(case, &bytes, &g.bits) {
                Ok(r) if r.passed() && r.differing == 0 && r.max_abs == 0.0 => {}
                Ok(r) => {
                    eprintln!(
                        "tp{} {}: golden does not match itself ({} differing)",
                        case.id,
                        split.name(),
                        r.differing
                    );
                    return 1;
                }
                Err(error) => {
                    eprintln!("tp{} {}: {error}", case.id, split.name());
                    return 1;
                }
            }
            done += 1;
        }
    }
    match manifest::write(&root, &CASES, &splits) {
        Ok(files) => println!(
            "suite ok: {done} case/grouping pairs, {files} files manifested in {:.1}s -> {}",
            started.elapsed().as_secs_f64(),
            root.join(manifest::NAME).display()
        ),
        Err(error) => {
            eprintln!("manifest: {error}");
            return 1;
        }
    }
    0
}

fn cmd_manifest(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    let cases = pick_case(argv.get(3));
    let splits = pick_split(argv.get(4));
    match manifest::write(&root, &cases, &splits) {
        Ok(0) => {
            eprintln!("nothing to manifest under {}; run `gen` first", root.display());
            1
        }
        Ok(files) => {
            println!("{files} files -> {}", root.join(manifest::NAME).display());
            0
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

fn cmd_check(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    let cases = pick_case(argv.get(3));
    let splits = pick_split(argv.get(4));
    match manifest::check(&root, &cases, &splits) {
        Err(error) => {
            eprintln!("{}: {error}", root.join(manifest::NAME).display());
            1
        }
        Ok(diff) => {
            println!("{} 个文件与清单一致", diff.matched);
            for (label, paths) in
                [("内容不同", &diff.changed), ("缺失", &diff.missing), ("多出", &diff.extra)]
            {
                if !paths.is_empty() {
                    println!("{label} {}：", paths.len());
                    for path in paths.iter().take(20) {
                        println!("  {path}");
                    }
                    if paths.len() > 20 {
                        println!("  … 还有 {} 个", paths.len() - 20);
                    }
                }
            }
            if diff.ok() {
                println!("=> PASS，这棵树和清单记录的那一份逐字节相同");
                0
            } else {
                println!("=> FAIL");
                1
            }
        }
    }
}

/// Side-by-side element dump, for when a verdict is not enough.
fn cmd_dump(argv: &[String]) -> i32 {
    let (Some(dir), Some(out), Some(case_arg), Some(split_arg)) =
        (argv.get(2), argv.get(3), argv.get(4), argv.get(5))
    else {
        usage()
    };
    let limit: usize = argv.get(6).and_then(|s| s.parse().ok()).unwrap_or(16);
    let case = pick_case(Some(case_arg))[0];
    let split = pick_split(Some(split_arg))[0];
    let path = inputs::case_dir(Path::new(dir), &case, split);
    let golden = match read_golden(&path, &case) {
        Ok(g) => g,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let exact = match std::fs::read(path.join("golden_f32.bin")) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("golden_f32.bin: {error}");
            return 1;
        }
    };
    let got = match std::fs::read(out) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("{out}: {error}");
            return 1;
        }
    };
    println!(
        "{:>8} {:>5} {:>5} {:>16} {:>16} {:>16} {:>12}",
        "index", "row", "col", "got", "want", "exact f32", "rel err"
    );
    for index in 0..(case.m * case.n).min(limit) {
        let got_bits = u16::from_le_bytes([got[index * 2], got[index * 2 + 1]]);
        let g = case.out.decode(got_bits);
        let w = case.out.decode(golden[index]);
        let e = f32::from_le_bytes([
            exact[index * 4],
            exact[index * 4 + 1],
            exact[index * 4 + 2],
            exact[index * 4 + 3],
        ]);
        let rel = if w != 0.0 { ((g - w) / w).abs() } else { f32::INFINITY };
        println!(
            "{:>8} {:>5} {:>5} {:>16.6} {:>16.6} {:>16.6} {:>12.3e}",
            index,
            index / case.n,
            index % case.n,
            g,
            w,
            e,
            rel
        );
    }
    0
}

/// Write the problem statement's own worked examples as case directories, so
/// the kernel can be run against ground truth that none of our code produced.
fn cmd_examples(argv: &[String]) -> i32 {
    let Some(dir) = argv.get(2) else { usage() };
    let root = PathBuf::from(dir);
    for ex in examples::EXAMPLES.iter() {
        match ex.write(&root) {
            Ok(path) => println!(
                "  {:<22} E={} M={} K={} N={} {} -> {}",
                ex.name,
                ex.e,
                ex.m,
                ex.k,
                ex.n,
                ex.out.name(),
                path.display()
            ),
            Err(error) => {
                eprintln!("{}: {error}", ex.name);
                return 1;
            }
        }
    }
    0
}

/// Compare a kernel's output for one worked example against the stated answer.
fn cmd_exverify(argv: &[String]) -> i32 {
    let (Some(dir), Some(name), Some(out)) = (argv.get(2), argv.get(3), argv.get(4)) else {
        usage()
    };
    let Some(ex) = examples::EXAMPLES.iter().find(|e| e.name == name.as_str()) else {
        eprintln!(
            "no example {name:?}; have {}",
            examples::EXAMPLES.iter().map(|e| e.name).collect::<Vec<_>>().join(" ")
        );
        return 2;
    };
    let golden: Vec<u16> = ex.expect.iter().map(|v| ex.out.round(*v)).collect();
    let got = match std::fs::read(out) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("{out}: {error}");
            return 1;
        }
    };
    let _ = dir;
    // A synthetic Case, just to carry the shape and output type into compare().
    let case = custom::make(ex.e, ex.m, ex.k, ex.n, ex.gt, ex.out);
    match verify::compare(&case, &got, &golden) {
        Err(error) => {
            eprintln!("{error}");
            1
        }
        Ok(r) => {
            println!(
                "{:<22} {:>4} 元素  max rel {:.3e}  max abs {:.3e}  失败 {}  => {}",
                ex.name,
                r.elements,
                r.max_rel,
                r.max_abs,
                r.fail_either,
                if r.passed() { "PASS" } else { "FAIL" }
            );
            if !r.passed() {
                for index in 0..ex.expect.len() {
                    let bits = u16::from_le_bytes([got[index * 2], got[index * 2 + 1]]);
                    println!(
                        "   [{:>2}] got {:>12.6}   题面 {:>12.6}",
                        index,
                        ex.out.decode(bits),
                        ex.expect[index]
                    );
                }
            }
            if r.passed() { 0 } else { 1 }
        }
    }
}

/// Generate and reference an arbitrary shape, for bisecting which dimension
/// breaks a kernel.
fn cmd_custom(argv: &[String]) -> i32 {
    let need = ["<dir>", "<name>", "E", "M", "K", "N", "gt", "float16|bfloat16"];
    if argv.len() < 2 + need.len() {
        eprintln!("custom {} [split]", need.join(" "));
        return 2;
    }
    let root = PathBuf::from(&argv[2]);
    let name = argv[3].clone();
    let num = |i: usize| -> usize { argv[i].parse().unwrap_or(0) };
    let out = match argv[9].as_str() {
        "float16" => num::Out::F16,
        "bfloat16" => num::Out::Bf16,
        other => {
            eprintln!("output dtype must be float16 or bfloat16, got {other:?}");
            return 2;
        }
    };
    let case = custom::make(num(4), num(5), num(6), num(7), num(8) as u8, out);
    // An invented shape has no judge grouping, so `real` does not apply here.
    let split = match argv.get(10) {
        None => Split::Even,
        Some(text) => match Split::parse(text).filter(|s| cases::SYNTHETIC.contains(s)) {
            Some(s) => s,
            None => {
                eprintln!(
                    "custom shapes take one of {}, got {text:?}",
                    cases::SYNTHETIC.iter().map(|s| s.name()).collect::<Vec<_>>().join(" ")
                );
                return 2;
            }
        },
    };
    match custom::build(&root, &name, &case, split) {
        Ok((dir, g)) => {
            println!(
                "cu_{name}  E={} M={} K={} N={} gt={} {} {}  max|y|={:.2}  -> {}",
                case.e,
                case.m,
                case.k,
                case.n,
                case.gt,
                case.out.name(),
                split.name(),
                g.max_abs,
                dir.display()
            );
            0
        }
        Err(error) => {
            eprintln!("{name}: {error}");
            1
        }
    }
}

/// Verify any case directory against its own golden.bin, taking the shape from
/// the directory rather than from arguments.
fn cmd_dverify(argv: &[String]) -> i32 {
    let (Some(dir), Some(out)) = (argv.get(2), argv.get(3)) else { usage() };
    let path = Path::new(dir);
    let case = match custom::from_dir(path) {
        Ok(c) => c,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let golden = match read_golden(path, &case) {
        Ok(g) => g,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let got = match std::fs::read(out) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("{out}: {error}");
            return 1;
        }
    };
    match verify::compare(&case, &got, &golden) {
        Err(error) => {
            eprintln!("{error}");
            1
        }
        Ok(r) => {
            println!(
                "{:<28} E={} M={} K={} N={} gt={} {:>6}  失败 {:>8} / {:<8} max rel {:.3e}  => {}",
                path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                case.e,
                case.m,
                case.k,
                case.n,
                case.gt,
                case.out.name(),
                r.fail_either,
                r.elements,
                r.max_rel,
                if r.passed() { "PASS" } else { "FAIL" }
            );
            if r.passed() { 0 } else { 1 }
        }
    }
}
