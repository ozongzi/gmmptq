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


// ---------------------------------------------------------------------------
// Stage-deletion probes.
//
// These price one stage's contribution to the *cadence*, which is a different
// quantity from its occupancy: on point 4 the AIC pipes all read under 20%
// while MTE2 sat at 77%, and deleting stages showed the whole compute chain
// was 1.47 of 18.13 us. Occupancy cannot measure that, and every point that
// has been settled was settled this way.
//
// A deleted stage leaves its destination holding stale UB or L0, which is why
// these are instruments. Read the *pipe busy* delta, not just the duration:
// `w11gv` moved the clock 0.56 us with every pipe's busy time unchanged, which
// is how pure serialisation was told apart from removed work.
// ---------------------------------------------------------------------------

const NO_FIX: &[Edit] = &[
    // int32 staging (NoQuant) in the two plain bodies, half staging (DEQF16)
    // in the two ScalarHalf ones -- same statement, different accumulator type.
    Edit {
        fragment: "2025-cube-fullk-body.inc",
        find: "Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-fix*/;",
        expect: 1,
    },
    Edit {
        fragment: "2472-cube-fullk-unit-body.inc",
        find: "Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-fix*/;",
        expect: 1,
    },
    Edit {
        fragment: "3676-p1-half-pair.inc",
        find: "Fixpipe<half, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-fix*/;",
        expect: 1,
    },
    Edit {
        fragment: "4225-cube-fullk-body-scalar-half.inc",
        find: "Fixpipe<half, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-fix*/;",
        expect: 1,
    },
];

const NO_MAD: &[Edit] = &[Edit {
    // Identical in all four cube bodies and in CubeMix, hence five.
    fragment: "*",
    find: "Mmad(l0c, a, b, mp);",
    replace: "/*GMM_WRONG_ON_PURPOSE no-mad*/;",
    expect: 5,
}];

/// The whole credit ring, both directions. Deleting one side alone **hangs**:
/// drop the AIC announce and the AIV waits forever on a flag nobody sets, so
/// every site has to go together. Worth -15.4% on point 4 and -19.1% on point
/// 12, and the AIV's per-core busy time collapsed 14.47 -> 5.94 against AIC's
/// 13.35, which is how AIC was established as the long leg.
const NO_RING: &[Edit] = &[
    // AIC: wait for a staging credit, then announce the finished tile.
    Edit {
        fragment: "*",
        find: "CrossCoreWaitFlag(uint16_t(4u+tileId));",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 5,
    },
    Edit {
        fragment: "*",
        find: "CrossCoreSetFlag<2, PIPE_FIX>(tileId);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 5,
    },
    Edit {
        fragment: "*",
        find: "CrossCoreWaitFlag(uint16_t(4u+(q&3u)));",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 5,
    },
    // AIV: the four cube bodies spell both halves on one `if constexpr` line,
    // one per BM arm. These must be replaced before the bare CubeMix forms
    // below would match them, so they are scoped by fragment there, not here.
    Edit {
        fragment: "*",
        find: "if constexpr (BM <= 32 || BM == 208) { CrossCoreWaitFlag(tileId); if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId)); }",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 4,
    },
    Edit {
        fragment: "*",
        find: "if constexpr (BM > 32 && BM != 208) { CrossCoreWaitFlag(tileId); if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId)); }",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 4,
    },
    // CubeMix spells its AIV half on two separate lines and has no BM arms.
    // Scoped to that fragment so the bare strings cannot reach the guarded
    // lines above, whatever order the edits run in.
    Edit {
        fragment: "0134-cubemix-kernel.inc",
        find: "CrossCoreWaitFlag(tileId);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 1,
    },
    Edit {
        fragment: "0134-cubemix-kernel.inc",
        find: "if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId));",
        replace: "/*GMM_WRONG_ON_PURPOSE no-ring*/;",
        expect: 1,
    },
];


/// Both `WholeReduceSum` levels in all three dot-product bodies.
///
/// Prices the ceiling on replacing the primitive. On point 11 that was 1.36 us
/// -- 44% of all vector work, and 7.2x what its element count should cost --
/// and `p11add`'s aligned `Add` fold collected most of it (-8.0%). But the
/// cause was the *stride*, not the primitive: point 11 reads with
/// `srcRepStride = pairs/8 = 50` blocks (64 floats, then jump 400), while
/// `RowDot` and the longdot body read contiguously at `srcRepStride = 8`.
/// Points 7/8's reduce touches 8320 fp32 = 0.16 us at full lane width against
/// an entire AIV vec busy of 1.05 us, so a 7x reduce would not fit inside
/// their total. Expect this probe to show ~nothing on the contiguous bodies.
const NO_REDUCE: &[Edit] = &[
    Edit {
        fragment: "1142-n1-fixed-layout-safe.inc",
        find: "WholeReduceSum(scratch[col/64u],f[col],uint64_t(Min(64u,realPairs-col)),uint8_t(nr),8,1,uint8_t(pairs/8u));",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
    Edit {
        fragment: "1142-n1-fixed-layout-safe.inc",
        find: "WholeReduceSum(spaced,scratch,uint64_t((realPairs+63u)/64u),uint8_t(nr),8,1,1);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
    Edit {
        fragment: "1376-row-dot.inc",
        find: "for (uint32_t r=0;r<have;++r) WholeReduceSum(red1[r*16],xf[r*fp],64,fp/64,1,1,8);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
    Edit {
        fragment: "1376-row-dot.inc",
        find: "WholeReduceSum(red2,red1,16,have,1,1,2);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
    Edit {
        fragment: "1471-long-narrow-dot.inc",
        find: "WholeReduceSum(red1,wf,64,partialCount,1,1,8);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
    Edit {
        fragment: "1471-long-narrow-dot.inc",
        find: "WholeReduceSum(red2,red1,16,groups,1,1,2);",
        replace: "/*GMM_WRONG_ON_PURPOSE no-reduce*/;",
        expect: 1,
    },
];
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
        Probe {
            name: "no-fix",
            what: "delete the Fixpipe that drains L0C to staging",
            read: "its share of the per-tile cadence. On point 4 this was -0.72 us of 18.13 \
                   (-4.0%), the largest of the four compute stages and still only a twentieth \
                   of the point.",
            families: "the four cube bodies",
            unsupported: "CubeMix spells the same call with its own staging type; add an Edit for \
                          it if the Z fallback is ever the arm under test. The AIV-only families \
                          have no Fixpipe.",
            edits: NO_FIX,
        },
        Probe {
            name: "no-mad",
            what: "delete the Mmad accumulation itself",
            read: "the cube's actual arithmetic as a share of the clock. On point 4: -0.48 us, \
                   -2.6%. Check that mte1 did not collapse with it -- if the feeder stopped too, \
                   the reading is two stages, not one.",
            families: "the four cube bodies and CubeMix",
            unsupported: "the AIV-only families, which multiply with vector Mul rather than Mmad",
            edits: NO_MAD,
        },
        Probe {
            name: "no-ring",
            what: "delete the whole AIC<->AIV credit ring, both directions",
            read: "-15.4% on point 4 and -19.1% on point 12, the largest single structural cost \
                   found. But read the aiv column before concluding it is overhead: on point 12 \
                   the saving was 1.61 us of AIV wait-spin against an AIC that moved 0.05, so it \
                   is the data dependency itself, not bookkeeping. Halving the announces moves \
                   that cost rather than removing it (+4.4%).",
            families: "the four cube bodies and CubeMix",
            unsupported: "the AIV-only families have no cross-core ring; point 11's intra-core \
                          GvBarriers are a different mechanism",
            edits: NO_RING,
        },
        Probe {
            name: "no-reduce",
            what: "delete both WholeReduceSum levels in all three dot-product bodies",
            read: "the ceiling on replacing the primitive. 1.36 us on point 11, where the cause \
                   was the strided source rather than the primitive -- so expect near-nothing on \
                   RowDot and the longdot body, which read contiguously. If that holds, the \
                   reusable finding is about strided vector reads, not WholeReduceSum.",
            families: "n1-fixed-layout-safe (11), row-dot (none of the 15), long-narrow-dot (7/8)",
            unsupported: "the tiny reductions (3/5/6) reduce in registers; the cube families \
                          accumulate in L0C",
            edits: NO_REDUCE,
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
