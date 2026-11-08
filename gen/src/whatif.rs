//! `gen plan --what-if`: where a new gate would go, and what it would emit.
//!
//! The 14 November question is "five shapes, 90 minutes, what do I add?". This
//! answers the mechanical half of it without touching the table: given a shape
//! and a set of cube tile parameters, it
//!
//!   1. says where the shape routes **today** (usually gate Z, the 2-5x
//!      fallback),
//!   2. synthesizes a gate whose predicate is the smallest round window around
//!      that shape, inserted just before the fallback,
//!   3. prints the C++ that gate would emit, through the real renderer,
//!   4. prints the **Rust table entry** to paste into `cascade::cascade()`, and
//!   5. runs the bit-packing capacity check on the candidate table, so a
//!      launcher that cannot hold the shape's `k` or `n` is caught here rather
//!      than in a wrong answer on the judge.
//!
//! Nothing here mutates the shipped cascade: it builds a local `Vec<Node>`.

use crate::cascade::{self, Cfg, Expr, Gate, Node, Op, Packer, Q, R};
use crate::dispatch::{self, Env, Shape};

/// `--tiles BM,BN,BK,RV[,prefetch,singleL0]`.
#[derive(Clone, Copy, Debug)]
pub struct Tiles {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub rv: u32,
    pub prefetch: bool,
    pub single_l0: bool,
}

impl Tiles {
    pub fn parse(spec: &str) -> Result<Tiles, String> {
        let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
        if parts.len() < 4 || parts.len() > 6 {
            return Err(format!(
                "--tiles wants BM,BN,BK,RV[,prefetch,singleL0], got {spec:?}"
            ));
        }
        let num = |at: usize| -> Result<u32, String> {
            parts[at]
                .parse::<u32>()
                .map_err(|_| format!("--tiles: {:?} is not a number", parts[at]))
        };
        let flag = |at: usize, default: bool| -> Result<bool, String> {
            match parts.get(at) {
                None => Ok(default),
                Some(&"true" | &"1") => Ok(true),
                Some(&"false" | &"0") => Ok(false),
                Some(other) => Err(format!("--tiles: {other:?} is not true/false")),
            }
        };
        Ok(Tiles {
            bm: num(0)?,
            bn: num(1)?,
            bk: num(2)?,
            rv: num(3)?,
            prefetch: flag(4, true)?,
            single_l0: flag(5, false)?,
        })
    }
}

fn up(value: u32, grain: u32) -> u32 {
    (value + grain - 1) / grain * grain
}

/// The largest power of two not exceeding `value`, at least 1.
fn floor_pow2(value: u32) -> u32 {
    if value == 0 {
        return 1;
    }
    1u32 << (31 - value.leading_zeros())
}

/// The predicate for a candidate gate: the smallest round window around this
/// shape, in the shape of the windows the cube family already uses.
///
/// This is the part a human will edit. It is generated conservatively on
/// purpose -- it admits a neighbourhood of the measured shape, not the shape
/// itself, because a gate that matches exactly one shape is both useless on the
/// other four new points and the thing rule 2.8 is about.
fn window(shape: &Shape, tiles: Tiles, target: &'static str) -> Vec<Expr> {
    let rpe = shape.m / shape.e;
    let rpe_lo = floor_pow2(rpe.max(1));
    let k_lo = shape.k.saturating_sub(1) / 128 * 128;
    let k_hi = up(shape.k, 128).max(128);
    let n_lo = shape.n.saturating_sub(1) / 128 * 128;
    let n_hi = up(shape.n, 128).max(128);
    let mut pred = vec![
        Expr::Flag(if shape.bf { Q::Bf } else { Q::NotBf }),
        Expr::Cmp(Q::PCores, Op::Le, R::Const(20)),
        Expr::Cmp(Q::PE, Op::Gr, R::Const(1)),
        Expr::Cmp(Q::PE, Op::Le, R::Const(u64::from(up(shape.e, 16).max(16)))),
        Expr::Cmp(Q::RowsPerExpert, Op::Ge, R::Const(u64::from(rpe_lo))),
        Expr::Cmp(Q::RowsPerExpert, Op::Lt, R::Const(u64::from(rpe_lo * 2))),
        Expr::Cmp(Q::PK, Op::Gr, R::Const(u64::from(k_lo))),
        Expr::Cmp(Q::PK, Op::Le, R::Const(u64::from(k_hi))),
        Expr::Cmp(Q::PN, Op::Gr, R::Const(u64::from(n_lo))),
        Expr::Cmp(Q::PN, Op::Le, R::Const(u64::from(n_hi))),
    ];
    pred.push(Expr::Configure(Cfg {
        bm: tiles.bm,
        bn: tiles.bn,
        bk: tiles.bk,
        rv: tiles.rv,
        prefetch: tiles.prefetch,
        single_l0: tiles.single_l0,
        target,
        // `p` is configured by reference (gate X's shape), so its call belongs
        // in the condition. A `selected` copy is declared by the renderer's
        // nested form; a `packed` one is declared by the launcher's prologue,
        // which the renderer emits ahead of the nested call.
        copy: target == "selected",
        nested: target != "p",
    }));
    pred
}

/// Resolve `--launcher` to one of the launchers the table models a `dims`
/// layout for. The name may be the plain function name or its namespace path.
pub fn launcher(name: &str) -> Result<Packer, String> {
    let all = cascade::packers();
    if let Some(found) = all.iter().find(|p| {
        p.launcher == name || p.launcher.rsplit("::").next() == Some(name)
    }) {
        return Ok(*found);
    }
    Err(format!(
        "--launcher {name:?} is not one the table packs for; known: {}",
        all.iter().map(|p| p.launcher).collect::<Vec<_>>().join(" ")
    ))
}

/// How a launcher is called: the statement, plus any prologue it needs and the
/// name of the `Plan` it configures.
///
/// These signatures are in the verbatim fragments, not in the table, so this is
/// a small hard-coded registry -- six entries, one per cube launcher. Getting
/// one wrong is a compile error on the day, which is survivable; silently
/// packing into the wrong `dims` layout is not, and that part *is* table data.
struct Launch {
    call: String,
    prologue: &'static str,
    target: &'static str,
}

fn launch_for(packer: &Packer, bf: bool) -> Launch {
    let base = packer.launcher.rsplit("::").next().unwrap_or(packer.launcher);
    match base {
        "LaunchAlignedMedium64" => Launch {
            call: "LaunchAlignedMedium64(x,weight,xScale,weightScale,groupList,y,selected,bf,stream);return;"
                .to_string(),
            prologue: "",
            target: "selected",
        },
        // The packed-224 backend needs the packed weight stride and its own
        // GM Scratch, so the candidate gets V's prologue too.
        "LaunchActivePacked224" => Launch {
            call: format!(
                "Scratch packedWeight(uint64_t(p.e)*p.k*packed.wstride);\n\
                 LaunchActivePacked224<{bf}>(x,weight,xScale,weightScale,groupList,y,\n\
                 \x20   (GM_ADDR)packedWeight.p,packed,stream);return;"
            ),
            prologue: "Plan packed=p;\npacked.wstride=UpH(p.n,128u);",
            target: "packed",
        },
        "Launch" if packer.launcher.starts_with("mix11") => Launch {
            call: "mix11::Launch(x,weight,xScale,weightScale,groupList,y,p,bf,stream);\nreturn;"
                .to_string(),
            prologue: "",
            target: "p",
        },
        "Launch" if packer.launcher.starts_with("p1_half_pair") => Launch {
            call: "p1_half_pair::Launch(x,weight,xScale,weightScale,groupList,y,selected,stream);return;"
                .to_string(),
            prologue: "",
            target: "selected",
        },
        other => Launch {
            call: format!(
                "{other}(x,weight,xScale,weightScale,groupList,y,selected,stream);return;"
            ),
            prologue: "",
            target: "selected",
        },
    }
}

pub struct WhatIf {
    pub label: &'static str,
    /// The candidate table: the shipped one plus the new gate.
    pub nodes: Vec<Node>,
    /// Where it went in, by label of the preceding element.
    pub after: String,
    pub cpp: String,
    pub rust: String,
    pub packing: Vec<String>,
}

/// Build the candidate. `after` is the label of the node to insert behind; the
/// default is the last gate, i.e. immediately before the Z fallback.
pub fn build(shape: &Shape, tiles: Tiles, packer: Packer, after: Option<&str>) -> Result<WhatIf, String> {
    let launch = launch_for(&packer, shape.bf);
    let candidate = Gate::new("NEW", 0)
        .c("// CANDIDATE -- generated by `gen plan --what-if`, not yet in the table.")
        .p(window(shape, tiles, launch.target))
        .pro(launch.prologue)
        .l(&launch.call)
        .pack(packer);

    let mut nodes = cascade::cascade();
    let at = match after {
        Some(label) => nodes
            .iter()
            .position(|node| node_label(node) == label)
            .ok_or_else(|| format!("--after {label:?} is not a node in the cascade"))?,
        None => nodes
            .iter()
            .rposition(|node| matches!(node, Node::Gate(_)))
            .ok_or("the cascade holds no gates")?,
    };
    let after_label = node_label(&nodes[at]).to_string();
    nodes.insert(at + 1, Node::Gate(candidate.clone()));

    let mut cpp = String::new();
    cascade::render_one(&candidate, &mut cpp, 1);
    let rust = rust_literal(&candidate);
    let packing = cascade::check_packing(&nodes);
    Ok(WhatIf { label: "NEW", nodes, after: after_label, cpp, rust, packing })
}

fn node_label(node: &Node) -> &str {
    match node {
        Node::Fixed { label, .. } => label,
        Node::Derived(d) => d.name,
        Node::Gate(g) => g.label,
    }
}

/// Print the gate as the Rust the human pastes into `cascade()`.
pub fn rust_literal(gate: &Gate) -> String {
    let mut out = String::new();
    out.push_str("    out.push(Node::Gate(\n");
    out.push_str(&format!("        Gate::new({:?}, {})\n", gate.label, gate.line));
    if !gate.comment.is_empty() {
        out.push_str(&format!("            .c({:?})\n", gate.comment));
    }
    out.push_str("            .p(vec![\n");
    for term in &gate.pred {
        out.push_str(&format!("                {},\n", rust_expr(term)));
    }
    out.push_str("            ])\n");
    if !gate.prologue.is_empty() {
        out.push_str(&format!("            .pro({:?})\n", gate.prologue));
    }
    if !gate.launch.is_empty() {
        out.push_str(&format!("            .l({:?})\n", gate.launch));
    }
    match gate.pack {
        Some(packer) => out.push_str(&format!("            .pack({}),\n", packer.const_name)),
        None => out.push_str("            ,\n"),
    }
    out.push_str("    ));\n");
    out
}

fn rust_expr(term: &Expr) -> String {
    match term {
        Expr::Cmp(q, op, R::Const(v)) => {
            format!("Expr::Cmp(Q::{q:?}, Op::{op:?}, R::Const({v}))")
        }
        Expr::Cmp(q, op, R::Named(name, v)) => {
            format!("Expr::Cmp(Q::{q:?}, Op::{op:?}, R::Named({name:?}, {v}))")
        }
        Expr::Cmp(q, op, R::Qty(r)) => {
            format!("Expr::Cmp(Q::{q:?}, Op::{op:?}, R::Qty(Q::{r:?}))")
        }
        Expr::Flag(q) => format!("Expr::Flag(Q::{q:?})"),
        Expr::Opaque { cpp, .. } => format!("/* opaque: {} */", cpp.replace('\n', " ")),
        Expr::Configure(cfg) => format!(
            "Expr::Configure(Cfg {{ bm: {}, bn: {}, bk: {}, rv: {}, prefetch: {}, \
             single_l0: {}, target: {:?}, copy: {}, nested: {} }})",
            cfg.bm, cfg.bn, cfg.bk, cfg.rv, cfg.prefetch, cfg.single_l0, cfg.target, cfg.copy,
            cfg.nested
        ),
    }
}

/// Print the whole report.
pub fn report(shape: &Shape, env: &Env, what: &WhatIf) {
    let before = dispatch::route(shape, env);
    println!(
        "shape  E={} M={} K={} N={} gt={} {}  rowsPerExpert={}",
        shape.e,
        shape.m,
        shape.k,
        shape.n,
        shape.group_list_type,
        shape.dtype(),
        shape.m / shape.e
    );
    match &before {
        Ok(r) => println!("today  gate {} ({}) -> {}, grid {}", r.gate, r.line, r.kernel, r.grid),
        Err(error) => println!("today  run_kernel would throw: {error}"),
    }
    println!(
        "insert a new gate {} immediately after {} (so before the Z fallback)",
        what.label, what.after
    );
    match dispatch::route_with(&what.nodes, shape, env) {
        Ok(r) => println!(
            "after  gate {} ({}) -> {}, grid {}",
            r.gate, r.line, r.kernel, r.grid
        ),
        Err(error) => println!("after  run_kernel would throw: {error}"),
    }
    if what.packing.is_empty() {
        println!("pack   OK -- every dims field holds what the candidate's window admits");
    } else {
        println!("pack   REFUSED -- `emit --profile final` would not write this:");
        for problem in &what.packing {
            println!("         {problem}");
        }
    }
    println!("\nthe C++ it would emit:\n{}", what.cpp);
    println!("the table entry to paste into gen/src/cascade.rs::cascade():\n{}", what.rust);
    // Does the rest of the table still route the same way? That is the whole
    // reason this exists as a dry run rather than an edit-and-compile.
    let mut moved = Vec::new();
    for point in dispatch::points() {
        let base = dispatch::route(&point.shape, env);
        let alt = dispatch::route_with(&what.nodes, &point.shape, env);
        match (base, alt) {
            (Ok(a), Ok(b)) if a.gate != b.gate => {
                moved.push(format!("point {}: {} -> {}", point.id, a.gate, b.gate));
            }
            _ => {}
        }
    }
    if moved.is_empty() {
        println!("regress none of the fifteen published points reroute");
    } else {
        println!("regress these published points reroute -- check before committing:");
        for line in moved {
            println!("         {line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(e: u32, m: u32, k: u32, n: u32, bf: bool) -> Shape {
        Shape {
            e,
            m,
            k,
            n,
            group_list_type: 1,
            bf,
            counts: vec![m / e; e as usize],
        }
    }

    /// A plausible new point: E8 M2048 K640 N896 bf16. It falls to Z today; the
    /// candidate must catch it, emit, and not move any published point.
    #[test]
    fn a_candidate_gate_catches_a_shape_that_falls_to_the_fallback() {
        let s = shape(8, 2048, 640, 896, true);
        let env = Env::default();
        assert_eq!(dispatch::route(&s, &env).unwrap().gate, "Z");
        let tiles = Tiles { bm: 128, bn: 128, bk: 128, rv: 64, prefetch: true, single_l0: false };
        let what = build(&s, tiles, launcher("LaunchActivePacked224").unwrap(), None).unwrap();
        assert!(what.packing.is_empty(), "{:?}", what.packing);
        assert_eq!(dispatch::route_with(&what.nodes, &s, &env).unwrap().gate, "NEW");
        for point in dispatch::points() {
            let a = dispatch::route(&point.shape, &env).unwrap().gate;
            let b = dispatch::route_with(&what.nodes, &point.shape, &env).unwrap().gate;
            assert_eq!(a, b, "point {} rerouted", point.id);
        }
        assert!(what.cpp.contains("// gate NEW"), "{}", what.cpp);
        assert!(what.cpp.contains("LaunchActivePacked224"), "{}", what.cpp);
        assert!(what.rust.contains("Expr::Configure"), "{}", what.rust);
    }

    /// And the capacity refusal, from the CLI's side: the same shape against a
    /// launcher whose `k` field is two bits wide.
    #[test]
    fn what_if_refuses_a_launcher_that_cannot_hold_the_shape() {
        let s = shape(8, 2048, 640, 896, true);
        let tiles = Tiles { bm: 64, bn: 128, bk: 128, rv: 64, prefetch: true, single_l0: false };
        let what = build(&s, tiles, launcher("LaunchAlignedMedium64").unwrap(), None).unwrap();
        assert!(!what.packing.is_empty(), "should refuse");
        assert!(what.packing.iter().any(|p| p.contains("k/128-1")), "{:?}", what.packing);
    }

    #[test]
    fn the_window_is_round_and_contains_the_shape() {
        let s = shape(7, 3073, 1023, 1025, true);
        let tiles = Tiles { bm: 224, bn: 128, bk: 128, rv: 64, prefetch: true, single_l0: false };
        let terms = window(&s, tiles, "selected");
        let find = |q: Q, op: Op| {
            terms.iter().find_map(|t| match t {
                Expr::Cmp(lhs, o, R::Const(v)) if *lhs == q && *o == op => Some(*v),
                _ => None,
            })
        };
        assert_eq!(find(Q::PK, Op::Gr), Some(896));
        assert_eq!(find(Q::PK, Op::Le), Some(1024));
        assert_eq!(find(Q::PN, Op::Gr), Some(1024));
        assert_eq!(find(Q::PN, Op::Le), Some(1152));
        // rowsPerExpert = 439 -> [256, 512)
        assert_eq!(find(Q::RowsPerExpert, Op::Ge), Some(256));
        assert_eq!(find(Q::RowsPerExpert, Op::Lt), Some(512));
    }

    #[test]
    fn unknown_launchers_are_rejected_with_the_list() {
        let error = launcher("LaunchNothing").unwrap_err();
        assert!(error.contains("LaunchActivePacked224"), "{error}");
    }
}
