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
mod dispatch;
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
         plan <E> <M> <K> <N> <gt> <fp16|bf16> <counts...>\n\
         plan --all                          the 15 published points, one line each\n\
         \n\
         \x20 plan options\n\
         \x20   --cores <n>       availableCoreNum (default 20)\n\
         \x20   --aic <n>         GetCoreNumAic()  (default = --cores)\n\
         \x20   --ptr <p>=<v>     low 7 bits of a device pointer; p in\n\
         \x20                     x weight xScale weightScale groupList y\n\
         \x20   --misaligned      shorthand for all six pointers = 1\n\
         \x20   --no-l1-guard     drop gate Y's 512*kp<=524032 check (6105), to\n\
         \x20                     demonstrate the p1_half_pair k-field landmine\n\
         \x20   --json            machine-readable output\n\
         \n\
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
        Some("plan") => cmd_plan(&argv),
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

// ---------------------------------------------------------------------------
// plan -- a dry run of run_kernel's dispatch cascade (see dispatch.rs)
// ---------------------------------------------------------------------------

/// `gen plan` arguments. Hand-rolled, same house style as `emit`.
struct PlanArgs {
    env: dispatch::Env,
    all: bool,
    json: bool,
    positional: Vec<String>,
    misaligned: bool,
}

fn parse_plan_args(argv: &[String]) -> Result<PlanArgs, String> {
    let mut args = PlanArgs {
        env: dispatch::Env::default(),
        all: false,
        json: false,
        positional: Vec::new(),
        misaligned: false,
    };
    let mut aic: Option<u32> = None;
    let mut at = 2;
    while at < argv.len() {
        match argv[at].as_str() {
            "--all" => {
                args.all = true;
                at += 1;
            }
            "--json" => {
                args.json = true;
                at += 1;
            }
            "--misaligned" => {
                args.misaligned = true;
                args.env.ptrs = dispatch::Ptrs::misaligned();
                at += 1;
            }
            "--no-l1-guard" => {
                args.env.l1_guard = false;
                at += 1;
            }
            "--cores" => {
                let value = argv.get(at + 1).ok_or("--cores needs a value")?;
                args.env.available_core_num =
                    value.parse().map_err(|_| format!("--cores {value:?} is not a number"))?;
                at += 2;
            }
            "--aic" => {
                let value = argv.get(at + 1).ok_or("--aic needs a value")?;
                aic = Some(value.parse().map_err(|_| format!("--aic {value:?} is not a number"))?);
                at += 2;
            }
            "--ptr" => {
                let spec = argv.get(at + 1).ok_or("--ptr needs <name>=<value>")?;
                let (name, value) = spec.split_once('=').ok_or("--ptr wants <name>=<value>")?;
                let value: u32 = value
                    .parse()
                    .map_err(|_| format!("--ptr {spec:?}: {value:?} is not a number"))?;
                let slot = match name {
                    "x" => &mut args.env.ptrs.x,
                    "weight" | "w" => &mut args.env.ptrs.weight,
                    "xScale" | "xs" => &mut args.env.ptrs.x_scale,
                    "weightScale" | "ws" => &mut args.env.ptrs.w_scale,
                    "groupList" | "gl" => &mut args.env.ptrs.group_list,
                    "y" => &mut args.env.ptrs.y,
                    other => return Err(format!("--ptr: unknown pointer {other:?}")),
                };
                *slot = value & 127;
                at += 2;
            }
            other if other.starts_with('-') => {
                return Err(format!("gen plan: unexpected option {other:?}"));
            }
            other => {
                args.positional.push(other.to_string());
                at += 1;
            }
        }
    }
    args.env.physical_aic = aic.unwrap_or(args.env.available_core_num);
    if args.env.available_core_num == 0 {
        return Err("--cores must be positive (run_kernel throws on availableCoreNum<=0)".into());
    }
    Ok(args)
}

fn cmd_plan(argv: &[String]) -> i32 {
    let args = match parse_plan_args(argv) {
        Ok(a) => a,
        Err(error) => {
            eprintln!("gen plan: {error}");
            return 2;
        }
    };
    if args.all {
        if !args.positional.is_empty() {
            eprintln!("gen plan: --all takes no shape arguments");
            return 2;
        }
        return plan_all(&args);
    }
    if args.positional.len() < 6 {
        eprintln!(
            "gen plan: want <E> <M> <K> <N> <gt> <fp16|bf16> <counts...>, or --all"
        );
        return 2;
    }
    let mut numbers = Vec::new();
    for (index, token) in args.positional.iter().enumerate() {
        if index == 5 {
            continue;
        }
        match token.trim_end_matches(',').parse::<u32>() {
            Ok(value) => numbers.push(value),
            Err(_) => {
                eprintln!("gen plan: {token:?} is not a number");
                return 2;
            }
        }
    }
    let dtype = args.positional[5].as_str();
    let bf = match dtype {
        "bf16" | "bfloat16" => true,
        "fp16" | "float16" | "half" => false,
        other => {
            eprintln!("gen plan: dtype must be fp16 or bf16, not {other:?}");
            return 2;
        }
    };
    let shape = dispatch::Shape {
        e: numbers[0],
        m: numbers[1],
        k: numbers[2],
        n: numbers[3],
        group_list_type: numbers[4],
        bf,
        counts: numbers[5..].to_vec(),
    };
    let routed = match dispatch::route(&shape, &args.env) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("gen plan: run_kernel would throw: {error}");
            return 1;
        }
    };
    if args.json {
        print!("{}", plan_json(&shape, &args, &routed));
    } else {
        print_plan(&shape, &args, &routed);
    }
    if routed.errors.is_empty() { 0 } else { 1 }
}

fn plan_all(args: &PlanArgs) -> i32 {
    let points = dispatch::points();
    let mut failures = 0;
    if args.json {
        let mut rows = Vec::new();
        for point in &points {
            match dispatch::route(&point.shape, &args.env) {
                Ok(r) => {
                    if !r.errors.is_empty() {
                        failures += 1;
                    }
                    rows.push(format!(
                        "    {{\"point\":{},\"gate\":{},\"line\":{},\"kernel\":{},\"grid\":{},\"errors\":{}}}",
                        point.id,
                        quote(&r.gate),
                        r.line,
                        quote(&r.kernel),
                        r.grid,
                        json_strings(&r.errors)
                    ));
                }
                Err(error) => {
                    failures += 1;
                    rows.push(format!(
                        "    {{\"point\":{},\"throws\":{}}}",
                        point.id,
                        quote(&error)
                    ));
                }
            }
        }
        println!(
            "{{\n  \"cores\": {},\n  \"points\": [\n{}\n  ]\n}}",
            args.env.available_core_num,
            rows.join(",\n")
        );
        return if failures == 0 { 0 } else { 1 };
    }
    println!(
        "availableCoreNum={}  GetCoreNumAic()={}  pointers {}",
        args.env.available_core_num,
        args.env.physical_aic,
        if args.misaligned { "misaligned (all &127 == 1)" } else { "64B-aligned (default)" }
    );
    println!(
        "{:>3} {:>4} {:>5} {:>5} {:>5} {:>3} {:<5} {:<5} {:>5} {:<72} {:>5}",
        "pt", "E", "M", "K", "N", "gt", "dt", "gate", "line", "kernel", "grid"
    );
    for point in &points {
        let s = &point.shape;
        match dispatch::route(s, &args.env) {
            Ok(r) => {
                if !r.errors.is_empty() {
                    failures += 1;
                }
                println!(
                    "{:>3} {:>4} {:>5} {:>5} {:>5} {:>3} {:<5} {:<5} {:>5} {:<72} {:>5}{}",
                    point.id,
                    s.e,
                    s.m,
                    s.k,
                    s.n,
                    s.group_list_type,
                    s.dtype(),
                    r.gate,
                    r.line,
                    r.kernel,
                    r.grid,
                    if r.errors.is_empty() { "" } else { "  ERROR" }
                );
                for error in &r.errors {
                    println!("      !! {error}");
                }
            }
            Err(error) => {
                failures += 1;
                println!("{:>3}  throws: {error}", point.id);
            }
        }
    }
    if failures == 0 { 0 } else { 1 }
}

fn print_plan(shape: &dispatch::Shape, args: &PlanArgs, r: &dispatch::Routed) {
    let sum: u32 = shape.counts.iter().sum();
    println!(
        "shape  E={} M={} K={} N={} gt={} {}  counts[{}] sum={}{}",
        shape.e,
        shape.m,
        shape.k,
        shape.n,
        shape.group_list_type,
        shape.dtype(),
        shape.counts.len(),
        sum,
        if shape.counts.len() as u32 != shape.e {
            "  (WARN: counts length != E)"
        } else if sum != shape.m {
            "  (WARN: counts do not sum to M)"
        } else {
            ""
        }
    );
    println!(
        "env    availableCoreNum={}  GetCoreNumAic()={}  L1 budget guards {}",
        args.env.available_core_num,
        args.env.physical_aic,
        if args.env.l1_guard { "on" } else { "OFF" }
    );
    let p = &args.env.ptrs;
    println!(
        "ptr    x&127={} weight&127={} xScale&127={} weightScale&127={} groupList&127={} y&127={}{}",
        p.x,
        p.weight,
        p.x_scale,
        p.w_scale,
        p.group_list,
        p.y,
        if args.misaligned { "  (--misaligned)" } else { "  (default: judge buffers are >=64B aligned)" }
    );
    println!("\npredicate chain (source order, label from gen/CURRENT.md)");
    for step in &r.trace {
        let mark = match step.outcome {
            dispatch::Outcome::Pass => "PASS",
            dispatch::Outcome::Fail => "fail",
            dispatch::Outcome::Note => "note",
        };
        let tags: Vec<&str> = step
            .deps
            .iter()
            .map(|d| match d {
                dispatch::Dep::Ptr => "[pointer]",
                dispatch::Dep::Cores => "[cores]",
                dispatch::Dep::Shape => "",
            })
            .collect();
        println!(
            "  {:<6} {:>4}  {mark}  {}{}",
            step.label,
            step.line,
            step.detail,
            if tags.is_empty() { String::new() } else { format!("  {}", tags.join(" ")) }
        );
    }
    println!("\ngate   {} (kernel/kernel.asc:{})", r.gate, r.line);
    println!("kernel {}", r.kernel);
    if let Some(body) = &r.body {
        println!("body   {body}");
    }
    println!("grid   {}   <- {}", r.grid, r.grid_how);
    if let Some(p) = &r.plan {
        // The AIV arms leave the cube tile fields at zero; printing them would
        // suggest a tiling that does not exist.
        if p.m0 != 0 {
            println!(
                "tiles  kp={} m0={} n0={} m0r={} n0r={} n0c={} k0={} kl={} nTiles={} rv={} wstride={}",
                p.kp, p.m0, p.n0, p.m0r, p.n0r, p.n0c, p.k0, p.kl, p.n_tiles, p.rv, p.wstride
            );
        } else {
            println!("tiles  none (AIV path); kp={} wstride={}", p.kp, p.wstride);
        }
    }
    if let Some(v) = &r.vtiling {
        println!(
            "vtile  m={} k={} n={} e={} listType={} coreNum={} kc={} nb={}",
            v.m, v.k, v.n, v.e, v.list_type, v.core_num, v.kc, v.nb
        );
    }
    if let Some(bytes) = r.staging {
        println!("stage  staging scratch {bytes} B ({:.1} MB)", bytes as f64 / 1.0e6);
    }
    if let Some((what, bytes)) = &r.extra_scratch {
        println!("stage  {what} = {bytes} B ({:.1} MB)", *bytes as f64 / 1.0e6);
    }
    if r.packing.is_empty() {
        println!("pack   none -- parameters travel as a Plan/VTiling by value");
    } else {
        println!("pack   uint32_t dims");
        for field in &r.packing {
            let margin = match field.margin() {
                Some(margin) => format!("gate<={} margin {margin}", field.gate_max.unwrap()),
                None => "gate: UNBOUNDED".to_string(),
            };
            println!(
                "         {:<10} bit {:>2} w{:<2} value {:<6} fieldMax {:<6} {}{}",
                field.name,
                field.shift,
                field.width,
                field.value,
                field.field_max(),
                margin,
                if field.fits() { "" } else { "   OVERFLOW" }
            );
        }
        for field in &r.packing {
            // `gt` is 1 bit against a 1-bit domain that run_kernel validates at
            // 5826; it can never widen, so flagging it is noise.
            if field.margin() == Some(0) && field.width > 1 {
                println!(
                    "warn   pack field {} has ZERO margin: the gate admits exactly what the field holds -- any widening overflows",
                    field.name
                );
            }
            if let Some(why) = field.incidental {
                println!("warn   pack field {}: {why}", field.name);
            }
        }
    }
    for note in &r.notes {
        println!("note   {note}");
    }
    let ptr_gates = r.gates_using_pointers();
    if ptr_gates.is_empty() {
        println!("ptrdep none -- every decision on this path was shape-only");
    } else {
        println!("ptrdep these decisions read device pointer values, not the shape:");
        for step in &ptr_gates {
            println!("         {} ({}) {}", step.label, step.line, step.detail);
        }
        // Do not silently pick an assumption: say what the other one gives.
        let other = if args.misaligned {
            dispatch::Ptrs::default()
        } else {
            dispatch::Ptrs::misaligned()
        };
        let label = if args.misaligned { "all six 64B-aligned" } else { "nothing aligned past a byte" };
        let alt_env = dispatch::Env { ptrs: other, ..args.env };
        match dispatch::route(shape, &alt_env) {
            Ok(alt) => println!(
                "         under the opposite assumption ({label}): gate {} ({}) -> {}, grid {}",
                alt.gate, alt.line, alt.kernel, alt.grid
            ),
            Err(error) => println!("         under the opposite assumption ({label}): throws: {error}"),
        }
    }
    let core_gates = r.gates_using_cores();
    if !core_gates.is_empty() {
        println!("coredep these decisions read availableCoreNum / GetCoreNumAic():");
        for step in &core_gates {
            println!("         {} ({}) {}", step.label, step.line, step.detail);
        }
    }
    for error in &r.errors {
        println!("ERROR  {error}");
    }
}

fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_strings(items: &[String]) -> String {
    format!(
        "[{}]",
        items.iter().map(|s| quote(s)).collect::<Vec<_>>().join(",")
    )
}

fn plan_json(shape: &dispatch::Shape, args: &PlanArgs, r: &dispatch::Routed) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"shape\": {{\"e\":{},\"m\":{},\"k\":{},\"n\":{},\"groupListType\":{},\"dtype\":{},\"counts\":[{}]}},\n",
        shape.e,
        shape.m,
        shape.k,
        shape.n,
        shape.group_list_type,
        quote(shape.dtype()),
        shape.counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
    ));
    out.push_str(&format!(
        "  \"env\": {{\"availableCoreNum\":{},\"physicalAic\":{},\"yL1Guard\":{},\"ptr\":{{\"x\":{},\"weight\":{},\"xScale\":{},\"weightScale\":{},\"groupList\":{},\"y\":{}}}}},\n",
        args.env.available_core_num,
        args.env.physical_aic,
        args.env.l1_guard,
        args.env.ptrs.x,
        args.env.ptrs.weight,
        args.env.ptrs.x_scale,
        args.env.ptrs.w_scale,
        args.env.ptrs.group_list,
        args.env.ptrs.y
    ));
    out.push_str(&format!("  \"gate\": {},\n  \"line\": {},\n", quote(&r.gate), r.line));
    out.push_str(&format!("  \"kernel\": {},\n", quote(&r.kernel)));
    if let Some(body) = &r.body {
        out.push_str(&format!("  \"body\": {},\n", quote(body)));
    }
    out.push_str(&format!("  \"grid\": {},\n  \"gridHow\": {},\n", r.grid, quote(&r.grid_how)));
    if let Some(p) = &r.plan {
        out.push_str(&format!(
            "  \"plan\": {{\"kp\":{},\"m0\":{},\"n0\":{},\"m0r\":{},\"n0r\":{},\"n0c\":{},\"k0\":{},\"kl\":{},\"nTiles\":{},\"rv\":{},\"wstride\":{}}},\n",
            p.kp, p.m0, p.n0, p.m0r, p.n0r, p.n0c, p.k0, p.kl, p.n_tiles, p.rv, p.wstride
        ));
    }
    if let Some(v) = &r.vtiling {
        out.push_str(&format!(
            "  \"vtiling\": {{\"coreNum\":{},\"kc\":{},\"nb\":{}}},\n",
            v.core_num, v.kc, v.nb
        ));
    }
    if let Some(bytes) = r.staging {
        out.push_str(&format!("  \"stagingBytes\": {bytes},\n"));
    }
    let fields: Vec<String> = r
        .packing
        .iter()
        .map(|field| {
            format!(
                "    {{\"name\":{},\"shift\":{},\"width\":{},\"value\":{},\"fieldMax\":{},\"gateMax\":{},\"margin\":{},\"fits\":{}}}",
                quote(field.name),
                field.shift,
                field.width,
                field.value,
                field.field_max(),
                field.gate_max.map(|g| g.to_string()).unwrap_or_else(|| "null".into()),
                field.margin().map(|m| m.to_string()).unwrap_or_else(|| "null".into()),
                field.fits()
            )
        })
        .collect();
    out.push_str(&format!("  \"packing\": [\n{}\n  ],\n", fields.join(",\n")));
    let chain: Vec<String> = r
        .trace
        .iter()
        .map(|step| {
            format!(
                "    {{\"label\":{},\"line\":{},\"outcome\":{},\"detail\":{},\"pointerDependent\":{},\"coreDependent\":{}}}",
                quote(&step.label),
                step.line,
                quote(match step.outcome {
                    dispatch::Outcome::Pass => "pass",
                    dispatch::Outcome::Fail => "fail",
                    dispatch::Outcome::Note => "note",
                }),
                quote(&step.detail),
                step.deps.contains(&dispatch::Dep::Ptr),
                step.deps.contains(&dispatch::Dep::Cores)
            )
        })
        .collect();
    out.push_str(&format!("  \"chain\": [\n{}\n  ],\n", chain.join(",\n")));
    out.push_str(&format!("  \"notes\": {},\n", json_strings(&r.notes)));
    out.push_str(&format!("  \"errors\": {}\n", json_strings(&r.errors)));
    out.push_str("}\n");
    out
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
