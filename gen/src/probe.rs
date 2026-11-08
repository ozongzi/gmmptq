//! `emit --rofile instrument --probe <name>`: deliberately-wrong measurement
//! kernels.
//!
//! On 14 November we get five unknown shapes and 90 minutes, and the first
//! question about each is **what fraction of its time is fixed and what
//! fraction is proportional to the reduction dimension**. `kernel/ENDGAME.md`
//! records that one build plus three runs of a wrong-on-purpose kernel answered
//! that, and killed or confirmed a whole class of change each time:
//!
//!   * halving point 13's K loop: 20.28 -> 13.54 us, so 6.85 us (34%) is
//!     K-independent and the proportional part already runs at 307 GB/s;
//!   * halving point 11's x DMA: 7.040 -> 6.980, i.e. -0.8%, so its 790 KB
//!     input stream is not the bottleneck at all;
//!   * removing point 5's preloads: 1.040 -> 1.130, +0.09 us with the warm min
//!     moving in all five runs.
//!
//! ## Why `instrument` and not `probe`
//!
//! `gen/DESIGN.md` reserves `--profile probe` for the `GMM_ASK` **content
//! oracle** -- the one-bit AC-vs-WA channel that reads a hidden point's data.
//! That is a different thing in every respect: it answers a question about the
//! judge's *inputs*, it must be submitted to be read, and it is the construct
//! `--profile final` refuses. These kernels answer a question about our own
//! *time*, are read from a local `msprof` run, and never need submitting. One
//! name for both would be the kind of confusion that gets the wrong file
//! uploaded at 11:40, so they get separate names.
//!
//! ## They compute wrong answers
//!
//! Every output carries [`BANNER`] at the top of the file and the marker
//! [`MARKER`] on each mutated line, and the CLI prints a warning. A submit path
//! should refuse any kernel containing [`MARKER`] -- see the report.
//!
//! ## Scope
//!
//! The mutations are textual substitutions against the verbatim fragments,
//! because this milestone keeps every kernel body unparameterized. Each edit
//! declares how many times its needle must occur, so a body that changes under
//! it is a hard error rather than a silently-inert probe.

use std::path::Path;

use crate::cascade;
use crate::chunks;
use crate::emit;

/// Grep for this to find a kernel that is wrong on purpose.
pub const MARKER: &str = "GMM_WRONG_ON_PURPOSE";

/// Prepended to every instrumented translation unit.
pub const BANNER: &str = "\
// ===========================================================================
// GMM_WRONG_ON_PURPOSE -- THIS KERNEL COMPUTES WRONG ANSWERS BY DESIGN.
//
// It is a measurement instrument built by `gen emit --profile instrument`.
// One dimension or one DMA extent has been mutated so that the difference
// between this kernel's time and the real one decomposes the cost. It will
// fail every correctness check, and it must NEVER be submitted.
//
// probe: 「probe」
// what:  「what」
// read:  「read」
// ===========================================================================
";

/// One textual mutation. `expect` is how many times `find` must occur; a
/// mismatch is a refusal, so a probe cannot quietly become a no-op after
/// someone edits a kernel body.
pub struct Edit {
    /// A fragment file name, or `"*"` for every fragment.
    pub fragment: &'static str,
    pub find: &'static str,
    pub replace: &'static str,
    pub expect: usize,
}

pub struct Probe {
    pub name: &'static str,
    /// What is changed, one line.
    pub what: &'static str,
    /// What the difference tells you.
    pub read: &'static str,
    pub families: &'static str,
    pub unsupported: &'static str,
    pub edits: &'static [Edit],
}

const HALVE_K: &[Edit] = &[
    // The Mmad accumulation loop of each cube body: identical text in all four.
    Edit {
        fragment: "2025-cube-fullk-body.inc",
        find: "for (uint32_t ks = 0; ks < p.k; ks += BK) {",
        replace: "for (uint32_t ks = 0; ks < p.k/2u; ks += BK) {  // GMM_WRONG_ON_PURPOSE halve-k",
        expect: 1,
    },
    Edit {
        fragment: "2472-cube-fullk-unit-body.inc",
        find: "for (uint32_t ks = 0; ks < p.k; ks += BK) {",
        replace: "for (uint32_t ks = 0; ks < p.k/2u; ks += BK) {  // GMM_WRONG_ON_PURPOSE halve-k",
        expect: 1,
    },
    Edit {
        fragment: "3676-p1-half-pair.inc",
        find: "for (uint32_t ks = 0; ks < p.k; ks += BK) {",
        replace: "for (uint32_t ks = 0; ks < p.k/2u; ks += BK) {  // GMM_WRONG_ON_PURPOSE halve-k",
        expect: 1,
    },
    Edit {
        fragment: "4225-cube-fullk-body-scalar-half.inc",
        find: "for (uint32_t ks = 0; ks < p.k; ks += BK) {",
        replace: "for (uint32_t ks = 0; ks < p.k/2u; ks += BK) {  // GMM_WRONG_ON_PURPOSE halve-k",
        expect: 1,
    },
    // Point 13's family, because that is the measurement ENDGAME.md records.
    Edit {
        fragment: "1571-balanced-short-group.inc",
        find: "for (uint32_t k0 = 0; k0 < t_->k; k0 += kc_) {",
        replace: "for (uint32_t k0 = 0; k0 < t_->k/2u; k0 += kc_) {  // GMM_WRONG_ON_PURPOSE halve-k",
        expect: 1,
    },
];

const HALVE_X_DMA: &[Edit] = &[
    // `ap` is the A (x) Nd2Nz descriptor; dValue is the row length copied, so
    // halving it halves the x byte count and leaves every extent alone.
    // 2025 has two, one per StreamB arm.
    Edit {
        fragment: "2025-cube-fullk-body.inc",
        find: "ap.dValue = static_cast<uint16_t>(p.k);",
        replace: "ap.dValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-x-dma",
        expect: 2,
    },
    Edit {
        fragment: "2472-cube-fullk-unit-body.inc",
        find: "ap.dValue = static_cast<uint16_t>(p.k);",
        replace: "ap.dValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-x-dma",
        expect: 1,
    },
    Edit {
        fragment: "3676-p1-half-pair.inc",
        find: "ap.dValue = static_cast<uint16_t>(p.k);",
        replace: "ap.dValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-x-dma",
        expect: 1,
    },
    Edit {
        fragment: "4225-cube-fullk-body-scalar-half.inc",
        find: "ap.dValue = static_cast<uint16_t>(p.k);",
        replace: "ap.dValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-x-dma",
        expect: 1,
    },
];

const HALVE_W_DMA: &[Edit] = &[
    // `bp` is the B (weight) descriptor. In the cube family this is the big
    // stream -- e*k*n bytes against x's m*k -- so it is the one to halve first.
    Edit {
        fragment: "2025-cube-fullk-body.inc",
        find: "bp.nValue = static_cast<uint16_t>(StreamB ? Min(BK,p.k) : p.k);",
        replace: "bp.nValue = static_cast<uint16_t>((StreamB ? Min(BK,p.k) : p.k)/2u);  // GMM_WRONG_ON_PURPOSE halve-w-dma",
        expect: 1,
    },
    Edit {
        fragment: "2472-cube-fullk-unit-body.inc",
        find: "bp.nValue = static_cast<uint16_t>(p.k);",
        replace: "bp.nValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-w-dma",
        expect: 1,
    },
    Edit {
        fragment: "3676-p1-half-pair.inc",
        find: "bp.nValue = static_cast<uint16_t>(p.k);",
        replace: "bp.nValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-w-dma",
        expect: 1,
    },
    Edit {
        fragment: "4225-cube-fullk-body-scalar-half.inc",
        find: "bp.nValue = static_cast<uint16_t>(p.k);",
        replace: "bp.nValue = static_cast<uint16_t>(p.k/2u);  // GMM_WRONG_ON_PURPOSE halve-w-dma",
        expect: 1,
    },
];

const NO_PRELOAD: &[Edit] = &[Edit {
    fragment: "*",
    find: "DataCachePreload(",
    // `if(false)` rather than deleting the line, so nothing else about the
    // statement -- including the tensor it names -- has to be understood.
    replace: "/*GMM_WRONG_ON_PURPOSE no-preload*/ if(false) DataCachePreload(",
    expect: 57,
}];

pub fn probes() -> &'static [Probe] {
    &[
        Probe {
            name: "halve-k",
            what: "halve the K accumulation loop's upper bound",
            read: "the residue is the K-independent cost. Large fixed share => tile geometry and \
                   sync tuning are pointless, attack launch and prologue. Large proportional \
                   share => compare the achieved bandwidth with 634 GB/s.",
            families: "the four cube bodies, plus balanced-short-group (point 13)",
            unsupported: "the tiny register families (3/5/6) have no K loop to halve; longdot \
                          (7/8), outer (10) and the N==1 matvec (11) have one but their loop text \
                          differs per family -- add an Edit for the family you need",
            edits: HALVE_K,
        },
        Probe {
            name: "halve-x-dma",
            what: "halve the x (A) Nd2Nz row length; every compute extent untouched",
            read: "a near-zero delta means the x stream is not the bottleneck and the arm is \
                   overhead-bound, as it was for point 11 (-0.8%).",
            families: "the four cube bodies",
            unsupported: "everything else; the AIV families copy x with DataCopyPad and each \
                          spells its blockLen differently",
            edits: HALVE_X_DMA,
        },
        Probe {
            name: "halve-w-dma",
            what: "halve the weight (B) Nd2Nz row count; every compute extent untouched",
            read: "in the cube family the weight stream is e*k*n against x's m*k, so this is the \
                   one to halve first. A near-zero delta means weight movement is already hidden \
                   behind the Mmad.",
            families: "the four cube bodies",
            unsupported: "everything else",
            edits: HALVE_W_DMA,
        },
        Probe {
            name: "no-preload",
            what: "disable every DataCachePreload call",
            read: "a slowdown means the preloads are doing real work and are already accounted \
                   for; no change means the prefetch distance is wrong or the line is already hot.",
            families: "every family that preloads -- the tiny and vector paths (13 fragments, \
                       57 calls)",
            unsupported: "the cube bodies issue no DataCachePreload at all, so this probe is inert \
                          for large shapes",
            edits: NO_PRELOAD,
        },
    ]
}

pub fn find(name: &str) -> Option<&'static Probe> {
    probes().iter().find(|p| p.name == name)
}


/// Apply one probe to the `final` translation unit.
pub fn instrument(dir: &Path, probe: &Probe) -> Result<Vec<u8>, String> {
    let body = emit::run_kernel(&cascade::cascade())?;
    let fragments = chunks::load(dir)?;
    let mut out = String::new();

    // The banner, with this probe's own description in it.
    let banner = BANNER
        .replace("「probe」", probe.name)
        .replace("「what」", probe.what)
        .replace("「read」", probe.read);
    out.push_str(&banner);

    // Hits per edit, counted as they are applied. Counting replacements in the
    // finished text instead would conflate edits that share a needle -- the
    // four cube bodies hold the same K loop line.
    let mut hits = vec![0usize; probe.edits.len()];
    for fragment in &fragments {
        let mut text = if fragment.name == emit::RUN_KERNEL_FRAGMENT {
            body.clone()
        } else {
            String::from_utf8(fragment.bytes.clone())
                .map_err(|_| format!("{} is not UTF-8", fragment.name))?
        };
        for (index, edit) in probe.edits.iter().enumerate() {
            if edit.fragment != "*" && edit.fragment != fragment.name {
                continue;
            }
            let found = text.matches(edit.find).count();
            if found == 0 {
                continue;
            }
            hits[index] += found;
            text = text.replace(edit.find, edit.replace);
        }
        out.push_str(&text);
    }

    for (index, edit) in probe.edits.iter().enumerate() {
        if hits[index] != edit.expect {
            return Err(format!(
                "probe {:?}: expected {} occurrence(s) of\n    {}\nin {}, found {}. \
                 The kernel body changed under the probe -- re-read it and update the Edit \
                 rather than measuring an inert probe.",
                probe.name, edit.expect, edit.find, edit.fragment, hits[index]
            ));
        }
    }
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every probe must actually fire, with exactly the arity it declares.
    /// This is the test that fails the day someone edits a cube body.
    #[test]
    fn every_probe_applies_with_its_declared_arity() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        for probe in probes() {
            let bytes = instrument(&dir, probe)
                .unwrap_or_else(|e| panic!("probe {}: {e}", probe.name));
            let text = String::from_utf8(bytes).expect("utf8");
            assert!(text.starts_with("// ==="), "probe {}: no banner", probe.name);
            assert!(text.contains(MARKER), "probe {}: no marker", probe.name);
            let total: usize = probe.edits.iter().map(|e| e.expect).sum();
            // banner mentions the marker once in the headline plus once per
            // mutated site.
            let marks = text.matches(MARKER).count();
            assert!(
                marks >= total,
                "probe {}: {marks} markers for {total} declared sites",
                probe.name
            );
        }
    }

    /// The instrumented unit must still be the `final` cascade, byte for byte,
    /// so a probe measures the kernel we would submit and not some other one.
    #[test]
    fn a_probe_changes_nothing_in_run_kernel() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let body = emit::run_kernel(&cascade::cascade()).expect("emits");
        for probe in probes() {
            let text = String::from_utf8(instrument(&dir, probe).unwrap()).unwrap();
            assert!(
                text.contains(&body),
                "probe {} perturbed the generated run_kernel",
                probe.name
            );
        }
    }

    /// An edit whose needle is gone is a refusal, not a silent no-op.
    #[test]
    fn a_stale_needle_is_refused() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let stale = Probe {
            name: "stale",
            what: "x",
            read: "x",
            families: "x",
            unsupported: "x",
            edits: &[Edit {
                fragment: "2025-cube-fullk-body.inc",
                find: "for (uint32_t ks = 0; ks < p.kNoSuchThing; ks += BK) {",
                replace: "nope  // GMM_WRONG_ON_PURPOSE",
                expect: 1,
            }],
        };
        let error = instrument(&dir, &stale).expect_err("must refuse");
        assert!(error.contains("found 0"), "{error}");
    }

    /// The banner must not collide with the compliance greps in ENDGAME.md,
    /// which look for GMM_ASK / GetSystemCycle / GMM_PROBE / GmmProbeSpin.
    #[test]
    fn the_banner_uses_its_own_marker() {
        for needle in ["GMM_ASK", "GetSystemCycle", "GMM_PROBE", "GmmProbeSpin"] {
            assert!(!BANNER.contains(needle), "banner mentions {needle}");
        }
        assert!(BANNER.contains(MARKER));
    }
}
