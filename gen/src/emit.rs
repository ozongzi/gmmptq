//! `emit --profile final`: the verbatim fragments, with `run_kernel` generated.
//!
//! Milestone 0's `verbatim` profile concatenated 45 fragments. `final` does the
//! same for 44 of them and *generates* the 45th, `5814-run-kernel.inc`, from
//! the gate table in [`crate::cascade`] -- the table `gen plan` evaluates. The
//! bodies stay verbatim; only the routing becomes data.
//!
//! Three things are checked before anything is written:
//!
//!   1. **bit packing** -- a gate that reuses a launcher whose `dims` cannot
//!      hold the dimensions the gate admits is a refusal, naming the field.
//!      `LaunchAlignedMedium64` packs `k/128-1` into *two bits*; a new gate
//!      with `k<=1024` would alias `k=1024` onto `k=128` and compute a wrong
//!      answer with no diagnostic anywhere.
//!   2. **compliance** -- `final` refuses to emit probe or spin constructs
//!      (DESIGN.md, the profile table). The `#if GMM_ASK` hook at 5830 is
//!      therefore absent; since `GMM_ASK` defaults to 0, the preprocessed
//!      translation unit is unchanged for a real build.
//!   3. **placeholders** -- no `「」` may survive into a .asc file.

// `parse` and friends are the acceptance test.s instruments; the binary only
// calls `assemble`.
#![allow(dead_code)]

use std::path::Path;

use crate::cascade::{self, Node};
use crate::chunks;
use crate::tmpl;

/// The one fragment `final` replaces.
pub const RUN_KERNEL_FRAGMENT: &str = "5814-run-kernel.inc";

/// Fragments `final` leaves out entirely, with the reason.
///
/// `kernel/ENDGAME.md`'s pre-submission checklist is
/// `grep -n 'GetSystemCycle|GmmProbeSpin|GMM_PROBE|GMM_ASK' <final kernel>`
/// and requires it to be empty. The probe-ask fragment defines `GMM_ASK`, so
/// leaving it in would make `final` fail that grep even though the namespace
/// compiles away at `GMM_ASK=0`. Its only caller was the hook at 5830, which
/// the generated cascade does not emit, so nothing else refers to it -- the
/// test `nothing_outside_the_omitted_fragment_refers_to_it` keeps that true.
/// Dropping it also drops `probe_ask::Point`'s hard-coded table of the fifteen
/// published shapes (5780..5784), which is the one thing in this file that
/// really would read like per-case matching.
pub const OMITTED_IN_FINAL: &[(&str, &str)] = &[(
    "5731-probe-ask.inc",
    "the GMM_ASK content oracle: a probe construct, and the only place a \
     hard-coded table of the published shapes survives",
)];

/// Constructs `--profile final` must never emit.
const FORBIDDEN: &[(&str, &str)] = &[
    ("GMM_ASK", "the hidden-point probe switch"),
    ("probe_ask::", "a call into the probe namespace"),
    ("AskAllZero", "the probe scan kernel"),
    ("GetSystemCycle", "the spin-to-a-fixed-time loop"),
];

/// Generate the `run_kernel` body for a cascade, with every `final`-profile
/// check applied.
pub fn run_kernel(nodes: &[Node]) -> Result<String, String> {
    let problems = cascade::check_packing(nodes);
    if !problems.is_empty() {
        let mut message = String::from(
            "refusing to emit: a gate's shape does not fit its launcher's packed dims fields\n",
        );
        for problem in &problems {
            message.push_str("  ");
            message.push_str(problem);
            message.push('\n');
        }
        return Err(message);
    }
    let body = cascade::render(nodes);
    for (needle, what) in FORBIDDEN {
        if body.contains(needle) {
            return Err(format!(
                "refusing to emit --profile final: the cascade contains {needle:?} ({what})"
            ));
        }
    }
    tmpl::assert_clean(&body)?;
    Ok(body)
}

/// The whole translation unit: 44 verbatim fragments plus the generated
/// cascade, spliced at the position the manifest gives `5814-run-kernel.inc`.
pub fn assemble(dir: &Path, nodes: &[Node]) -> Result<Vec<u8>, String> {
    let body = run_kernel(nodes)?;
    let fragments = chunks::load(dir)?;
    if !fragments.iter().any(|f| f.name == RUN_KERNEL_FRAGMENT) {
        return Err(format!("{} lists no {RUN_KERNEL_FRAGMENT}", dir.display()));
    }
    let mut out = Vec::new();
    for fragment in &fragments {
        if fragment.name == RUN_KERNEL_FRAGMENT {
            out.extend_from_slice(body.as_bytes());
        } else if OMITTED_IN_FINAL.iter().any(|(name, _)| *name == fragment.name) {
            continue;
        } else {
            out.extend_from_slice(&fragment.bytes);
        }
    }
    if let Ok(text) = std::str::from_utf8(&out) {
        tmpl::assert_clean(text)?;
        // The compliance gate applies to the whole translation unit, not just
        // to the part this generator wrote.
        for (needle, what) in FORBIDDEN {
            if text.contains(needle) {
                return Err(format!(
                    "refusing to emit --profile final: {needle:?} ({what}) survives in the \
                     assembled unit; add its fragment to OMITTED_IN_FINAL"
                ));
            }
        }
    }
    Ok(out)
}

/// `--profile final` for the shipped cascade.
pub fn final_profile(dir: &Path) -> Result<Vec<u8>, String> {
    assemble(dir, &cascade::cascade())
}

/// The hand-written `run_kernel` the generated one replaces, for diffing.
pub fn original_run_kernel(dir: &Path) -> Result<String, String> {
    let path = dir.join(RUN_KERNEL_FRAGMENT);
    std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// reading the emitted cascade back
// ---------------------------------------------------------------------------

/// One gate as it appears in the *emitted text*: its marker and, for a
/// conditional gate, the condition between the `if (` and its closing paren.
#[derive(Clone, Debug, PartialEq)]
pub struct Emitted {
    pub label: String,
    pub line: u32,
    pub cond: Option<String>,
}

/// Recover the gate list from emitted C++. This is deliberately a parse of the
/// *output*, not a second walk of the table: it is what lets a test say the
/// file we are about to compile routes the way `plan` says it does.
pub fn parse(text: &str) -> Vec<Emitted> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("// gate ") {
        let after = &rest[at + "// gate ".len()..];
        let eol = after.find('\n').unwrap_or(after.len());
        let marker = &after[..eol];
        let Some((label, tail)) = marker.split_once(" -- kernel.asc:") else {
            rest = &after[eol..];
            continue;
        };
        let unconditional = tail.contains("(unconditional)");
        let line: u32 = tail.split_whitespace().next().unwrap_or("").parse().unwrap_or(0);
        let body = &after[eol..];
        // The condition, if the marker says there is one. Without the marker's
        // say-so an unconditional arm whose launch begins `if (bf)` would look
        // like a gate with the condition `bf`.
        let cond = if unconditional { None } else { next_if_condition(body) };
        out.push(Emitted { label: label.to_string(), line, cond });
        rest = body;
    }
    out
}

/// The condition of the first `if (` in `text`, if nothing but whitespace
/// precedes it. Returns `None` for an unconditional arm.
fn next_if_condition(text: &str) -> Option<String> {
    let head = text.trim_start();
    let rest = head.strip_prefix("if (")?;
    let mut depth = 1i32;
    for (at, ch) in rest.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(squash(&rest[..at]));
                }
            }
            _ => {}
        }
    }
    None
}

pub fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::{Cfg, Expr, Gate, Op, PackField, Packer, Q, R};
    use crate::dispatch::{self, Env};

    /// Every gate of the table appears in the emitted file exactly once, in
    /// order, carrying exactly the conjuncts the evaluator uses.
    #[test]
    fn the_emitted_file_carries_the_table() {
        let nodes = cascade::cascade();
        let text = run_kernel(&nodes).expect("emits");
        let emitted = parse(&text);
        let table = cascade::gates(&nodes);
        assert_eq!(emitted.len(), table.len(), "gate count");
        for (got, want) in emitted.iter().zip(&table) {
            assert_eq!(got.label, want.label, "gate order");
            assert_eq!(got.line, want.line, "gate {} provenance", want.label);
            let cfg = want.cfg();
            let terms: Vec<String> = want
                .pred
                .iter()
                .filter(|t| !(cfg.is_some_and(|c| c.nested) && matches!(t, Expr::Configure(_))))
                .map(|t| squash(&t.cpp()))
                .collect();
            if terms.is_empty() {
                assert_eq!(got.cond, None, "gate {} should be unconditional", want.label);
            } else {
                assert_eq!(
                    got.cond.as_deref(),
                    Some(terms.join(" && ").as_str()),
                    "gate {}: emitted condition != table condition",
                    want.label
                );
            }
        }
    }

    /// Acceptance (b). Each of the fifteen points' winning gate must be present
    /// in the *emitted* text under the label and line `plan` reports, with the
    /// condition the evaluator used. Taken with
    /// `dispatch::tests::all_fifteen_points_match_current_md` -- which pins the
    /// fifteen rows to CURRENT.md -- that is "the emitted cascade gives the
    /// identical fifteen-point table".
    #[test]
    fn emitted_run_kernel_routes_the_fifteen_points_identically() {
        let nodes = cascade::cascade();
        let text = run_kernel(&nodes).expect("emits");
        let emitted = parse(&text);
        let env = Env::default();
        for point in dispatch::points() {
            let routed = dispatch::route(&point.shape, &env).expect("routes");
            let found = emitted
                .iter()
                .find(|g| g.label == routed.gate)
                .unwrap_or_else(|| panic!("point {}: gate {} not emitted", point.id, routed.gate));
            assert_eq!(found.line, routed.line, "point {} line", point.id);
            // …and the gate's whole enclosing chain is emitted too, in order,
            // because an arm is only reachable through its ancestors.
            let chain = cascade::chain_to(&nodes, &routed.gate).expect("chain");
            let mut at = 0usize;
            for gate in &chain {
                let pos = emitted[at..]
                    .iter()
                    .position(|g| g.label == gate.label)
                    .unwrap_or_else(|| panic!("point {}: {} out of order", point.id, gate.label));
                at += pos;
            }
        }
    }

    /// Requirement 5, end to end: the refusal happens before any bytes, and it
    /// names the field and the overflow.
    #[test]
    fn a_gate_that_overflows_its_launcher_is_refused() {
        let mut nodes = cascade::cascade();
        // A plausible 14 November gate: E8 M2048 K1024 N512, aligned, reusing
        // the aligned-medium backend. `k/128-1` is two bits; 1024/128-1 = 7.
        nodes.push(Node::Gate(
            Gate::new("P16", 9001)
                .p(vec![
                    Expr::Cmp(Q::PCores, Op::Le, R::Const(20)),
                    Expr::Cmp(Q::PE, Op::Le, R::Const(16)),
                    Expr::Cmp(Q::PK, Op::Le, R::Const(1024)),
                    Expr::Cmp(Q::PN, Op::Le, R::Const(512)),
                    Expr::Configure(Cfg {
                        bm: 64, bn: 128, bk: 128, rv: 64,
                        prefetch: true, single_l0: false,
                        target: "selected", copy: true, nested: true,
                    }),
                ])
                .l("LaunchAlignedMedium64(x,weight,xScale,weightScale,groupList,y,selected,bf,stream);return;")
                .pack(ALIGNED_MEDIUM_FOR_TEST),
        ));
        let error = run_kernel(&nodes).expect_err("must refuse");
        assert!(error.contains("k/128-1"), "{error}");
        assert!(error.contains("2 bits"), "{error}");
        assert!(error.contains("P16"), "{error}");
        assert!(error.contains("LaunchAlignedMedium64"), "{error}");
    }

    const ALIGNED_MEDIUM_FOR_TEST: Packer = Packer {
        const_name: "PACK_ALIGNED_MEDIUM_64",
        launcher: "LaunchAlignedMedium64",
        line: 4942,
        fields: &[
            PackField { name: "k/128-1", shift: 0, width: 2, q: Q::PK,
                enc: cascade::Enc::Div128Minus1, bound: cascade::Bound::FromGate },
            PackField { name: "n/128-1", shift: 2, width: 3, q: Q::PN,
                enc: cascade::Enc::Div128Minus1, bound: cascade::Bound::FromGate },
            PackField { name: "e-1", shift: 5, width: 4, q: Q::PE,
                enc: cascade::Enc::Minus1, bound: cascade::Bound::FromGate },
            PackField { name: "cores-1", shift: 9, width: 5, q: Q::PCores,
                enc: cascade::Enc::Minus1, bound: cascade::Bound::FromGate },
            PackField { name: "gt", shift: 14, width: 1, q: Q::Gt,
                enc: cascade::Enc::Plain, bound: cascade::Bound::FromGate },
        ],
    };

    /// `final` is a compliance gate, not just a code path -- and it applies to
    /// the whole file, which is `kernel/ENDGAME.md`'s pre-submission grep.
    #[test]
    fn final_emits_no_probe_or_spin_construct() {
        let text = run_kernel(&cascade::cascade()).expect("emits");
        for (needle, _) in FORBIDDEN {
            assert!(!text.contains(needle), "emitted cascade contains {needle:?}");
        }
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let whole = String::from_utf8(final_profile(&dir).expect("final")).expect("utf8");
        for needle in ["GMM_ASK", "GetSystemCycle", "GmmProbeSpin", "GMM_PROBE"] {
            assert!(!whole.contains(needle), "the assembled unit contains {needle:?}");
        }
    }

    /// Dropping a fragment is only safe while nothing else names it.
    #[test]
    fn nothing_outside_the_omitted_fragment_refers_to_it() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let fragments = chunks::load(&dir).expect("fragments");
        for (omitted, _) in OMITTED_IN_FINAL {
            assert!(
                fragments.iter().any(|f| f.name == *omitted),
                "OMITTED_IN_FINAL names {omitted:?}, which the manifest does not list"
            );
            for fragment in &fragments {
                if fragment.name == *omitted || fragment.name == RUN_KERNEL_FRAGMENT {
                    continue;
                }
                let text = String::from_utf8_lossy(&fragment.bytes);
                assert!(
                    !text.contains("probe_ask"),
                    "{} refers to probe_ask, so it cannot be omitted",
                    fragment.name
                );
            }
        }
    }

    /// Everything but `run_kernel` must come through untouched, so a regression
    /// in the generator can only ever be in the cascade.
    #[test]
    fn only_the_run_kernel_fragment_changes() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let fragments = chunks::load(&dir).expect("fragments");
        let generated = final_profile(&dir).expect("final profile");
        // Rebuild what we expect byte for byte: every fragment verbatim except
        // the generated cascade and the omitted ones. Anything else differing
        // is a bug in the splice, not an intended regeneration.
        let body = run_kernel(&cascade::cascade()).expect("emits");
        let mut want: Vec<u8> = Vec::new();
        for fragment in &fragments {
            if fragment.name == RUN_KERNEL_FRAGMENT {
                want.extend_from_slice(body.as_bytes());
            } else if OMITTED_IN_FINAL.iter().any(|(n, _)| *n == fragment.name) {
                continue;
            } else {
                want.extend_from_slice(&fragment.bytes);
            }
        }
        assert!(generated == want, "the splice is not fragment-for-fragment verbatim");
        // `5814-run-kernel.inc` is last in the manifest, so there is no suffix;
        // assert that rather than silently skipping a comparison.
        assert_eq!(fragments.last().map(|f| f.name.as_str()), Some(RUN_KERNEL_FRAGMENT));
        assert!(generated.ends_with(b"}\n\n"), "lost the file ending");
    }

    /// The generated body must at least be brace-balanced and end the function.
    #[test]
    fn the_generated_body_is_balanced() {
        let text = run_kernel(&cascade::cascade()).expect("emits");
        let mut depth = 0i32;
        for ch in text.chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "closed a brace that was never opened");
        }
        // Braces inside string literals and `<<<…>>>` do not occur in this
        // body, so the count is exact.
        assert_eq!(depth, 0, "unbalanced braces in the generated run_kernel");
        assert!(text.trim_end().ends_with('}'));
    }
}
