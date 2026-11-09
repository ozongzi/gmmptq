//! `emit --profile spin --isolate <point>`: the score-hiding instrument.
//!
//! The judge reports a time for the five visible points and nothing at all for
//! the ten hidden ones, so the only readout for a hidden point is the team's
//! board total. Appending a fixed spin to every point *but* one turns that
//! single number into that one point's score: with `S` the sum of the spun
//! scores, which an all-spin run measures once and for all,
//!
//! ```text
//!   total_iso = ( S - s_spun(N) + s_N ) / 15
//! ```
//!
//! and `s_N` falls out. The spun points stay Accepted -- the real kernel runs
//! first and the spin is appended after it -- so no `tbest` is disturbed and
//! the board shows a number in the single digits instead of our real total.
//!
//! ## Why this is generated and not spliced
//!
//! The shell version of this instrument (`tools/isolate.sh`) finds its seams
//! by grepping a past submission for `extern "C" void run_kernel(`. That works
//! and it is what measured points 6..15, but it carries a hand-written table
//! of the fifteen published shapes inside `GmmProbePoint`. Here the shape
//! table is **printed from [`crate::dispatch::points`]**, the same data `plan`
//! and `emit --profile final` read, so the instrument cannot come to disagree
//! with the generator about which shape is point 11.
//!
//! `--profile final` refuses to emit any of this (`emit::FORBIDDEN`), which is
//! what keeps a spin kernel out of a real submission.

use std::path::Path;

use crate::cascade;
use crate::chunks;
use crate::dispatch;
use crate::emit;

/// 1 tick of SYS_CNT is 20 ns, so this is 50.00 ms.
pub const DEFAULT_TICKS: u64 = 2_500_000;

/// Prepended to every spin translation unit.
const BANNER: &str = "\
// ===========================================================================
// GMM_PROBE -- SCORE-HIDING INSTRUMENT, NOT A SUBMISSION CANDIDATE.
//
// Built by `gen emit --profile spin`. Every published point except the
// isolated one runs its real kernel and then spins for a fixed time, so the
// judge's board total resolves to the isolated point's score alone. The
// answers are correct and every point is Accepted; the times are not ours.
//
// isolate:    「isolate」
// spin:       「spin」
// own ticks:  「own」
// ===========================================================================
";

/// The device-side spin kernel and the tick table.
fn preamble(ticks: &[u64]) -> String {
    let mut out = String::new();
    out.push_str(
        "\n// --- spin instrument ---------------------------------------------------\n\
         __global__ __aicore__ void GmmProbeSpin(uint64_t ticks) {\n\
         \x20   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);\n\
         \x20   // SYS_CNT runs at exactly 50 MHz: one tick is 20 ns. Touches no\n\
         \x20   // memory, so it adds time and nothing else.\n\
         \x20   const uint64_t begin = uint64_t(AscendC::GetSystemCycle());\n\
         \x20   while (uint64_t(AscendC::GetSystemCycle()) - begin < ticks) {}\n\
         }\n",
    );
    out.push_str("static const uint64_t kGmmProbeTicks[] = {");
    for (at, t) in ticks.iter().enumerate() {
        if at > 0 {
            out.push(',');
        }
        out.push_str(&format!("{t}ull"));
    }
    out.push_str("};\n");
    out.push_str(&format!(
        "static const int kGmmProbePoints = {};\n",
        ticks.len()
    ));
    out
}

/// `GmmProbePoint`: shape -> published point index, printed from the table.
fn point_table() -> String {
    let mut out = String::new();
    out.push_str(
        "// The shape table is printed from the generator's own point list, so it\n\
         // cannot drift from what `gen plan` reports.\n\
         inline int GmmProbePoint(const TensorGroupInfo& ix, const TensorGroupInfo& iw,\n\
         \x20                    int64_t gt) {\n\
         \x20   if (ix.numTensors != 1 || !ix.tensors || ix.tensors[0].numDims != 2 ||\n\
         \x20       !ix.tensors[0].shape || iw.numTensors != 1 || !iw.tensors ||\n\
         \x20       iw.tensors[0].numDims != 3 || !iw.tensors[0].shape)\n\
         \x20       return -1;\n\
         \x20   const int64_t M = ix.tensors[0].shape[0], K = ix.tensors[0].shape[1];\n\
         \x20   const int64_t E = iw.tensors[0].shape[0], N = iw.tensors[0].shape[2];\n\
         \x20   static const struct { int64_t e, m, k, n, gt; } kPoints[] = {\n",
    );
    for p in dispatch::points() {
        let s = &p.shape;
        out.push_str(&format!(
            "        {{{},{},{},{},{}}},  // point {}\n",
            s.e, s.m, s.k, s.n, s.group_list_type, p.id
        ));
    }
    out.push_str(
        "    };\n\
         \x20   for (int h = 0; h < kGmmProbePoints; ++h)\n\
         \x20       if (kPoints[h].e == E && kPoints[h].m == M && kPoints[h].k == K &&\n\
         \x20           kPoints[h].n == N && kPoints[h].gt == gt)\n\
         \x20           return h;\n\
         \x20   return -1;\n\
         }\n",
    );
    out
}

/// The `run_kernel` wrapper: the real dispatch, then the spin.
fn wrapper() -> String {
    "extern \"C\" void run_kernel(\n\
     \x20   GM_ADDR x, const TensorGroupInfo& info_x,\n\
     \x20   GM_ADDR weight, const TensorGroupInfo& info_weight,\n\
     \x20   GM_ADDR xScale, const TensorGroupInfo& info_xScale,\n\
     \x20   GM_ADDR weightScale, const TensorGroupInfo& info_weightScale,\n\
     \x20   GM_ADDR groupList, const TensorGroupInfo& info_groupList,\n\
     \x20   GM_ADDR y, const TensorGroupInfo& info_y,\n\
     \x20   int64_t availableCoreNum, aclrtStream stream,\n\
     \x20   int64_t groupListType, const char* outputDtype) {\n\
     \x20   GmmRunKernelBody(x, info_x, weight, info_weight, xScale, info_xScale,\n\
     \x20                    weightScale, info_weightScale, groupList, info_groupList,\n\
     \x20                    y, info_y, availableCoreNum, stream, groupListType,\n\
     \x20                    outputDtype);\n\
     \x20   const int probePoint = GmmProbePoint(info_x, info_weight, groupListType);\n\
     \x20   if (probePoint >= 0 && kGmmProbeTicks[probePoint]) {\n\
     \x20       GmmProbeSpin<<<1, nullptr, stream>>>(kGmmProbeTicks[probePoint]);\n\
     \x20       gmmptq::Check(aclrtSynchronizeStream(stream));\n\
     \x20   }\n\
     }\n"
        .to_string()
}

/// Tick vector: `ticks` everywhere, `own` on the isolated point. `isolate = 0`
/// spins every point, which is the reusable calibration run.
pub fn tick_vector(isolate: u32, ticks: u64, own: u64) -> Result<Vec<u64>, String> {
    let points = dispatch::points();
    if isolate != 0 && !points.iter().any(|p| p.id == isolate) {
        return Err(format!(
            "--isolate {isolate}: no published point {isolate}; 0 spins every point \
             (the calibration run)"
        ));
    }
    Ok(points
        .iter()
        .map(|p| if p.id == isolate { own } else { ticks })
        .collect())
}

/// The whole spin translation unit.
///
/// The generated `run_kernel` becomes `GmmRunKernelBody` and the wrapper takes
/// its name, so the dispatch itself is still the one `--profile final` emits
/// -- an isolation measures the kernel we would actually submit.
pub fn assemble(
    dir: &Path,
    overrides: &[(String, Vec<u8>)],
    isolate: u32,
    ticks: u64,
    own: u64,
) -> Result<Vec<u8>, String> {
    let vector = tick_vector(isolate, ticks, own)?;
    let dispatch_body = emit::run_kernel(&cascade::cascade())?;
    let renamed = dispatch_body.replacen(
        "extern \"C\" void run_kernel(",
        "static void GmmRunKernelBody(",
        1,
    );
    if renamed == dispatch_body {
        return Err("the generated cascade does not open with `extern \"C\" void run_kernel(`; \
                    the spin wrapper has nothing to rename"
            .to_string());
    }

    let mut out = String::new();
    out.push_str(
        &BANNER
            .replace("「isolate」", &isolate_label(isolate))
            .replace("「spin」", &format!("{ticks} ticks = {:.2} ms", ticks as f64 * 2e-5))
            .replace("「own」", &format!("{own} ticks = {:.2} us", own as f64 * 0.02)),
    );

    let fragments = chunks::load(dir)?;
    for (name, _) in overrides {
        if !fragments.iter().any(|f| &f.name == name) {
            return Err(format!(
                "a selected body replaces fragment {name:?}, which {} does not list",
                dir.display()
            ));
        }
    }
    for fragment in &fragments {
        if fragment.name == emit::RUN_KERNEL_FRAGMENT {
            out.push_str(&preamble(&vector));
            out.push_str(&point_table());
            out.push_str(&renamed);
            out.push_str(&wrapper());
        } else if emit::OMITTED_IN_FINAL.iter().any(|(n, _)| *n == fragment.name) {
            // The spin profile has no use for the GMM_ASK oracle either.
            continue;
        } else if let Some((_, text)) = overrides.iter().find(|(n, _)| *n == fragment.name) {
            out.push_str(
                std::str::from_utf8(text).map_err(|_| format!("{name} is not UTF-8", name = fragment.name))?,
            );
        } else {
            out.push_str(
                std::str::from_utf8(&fragment.bytes)
                    .map_err(|_| format!("{} is not UTF-8", fragment.name))?,
            );
        }
    }
    Ok(out.into_bytes())
}

fn isolate_label(isolate: u32) -> String {
    if isolate == 0 {
        "none -- every point spins (the calibration run)".to_string()
    } else {
        format!("point {isolate}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Isolating a point zeroes exactly that point's entry and nothing else.
    #[test]
    fn the_tick_vector_isolates_exactly_one_point() {
        let v = tick_vector(11, DEFAULT_TICKS, 0).expect("point 11");
        assert_eq!(v.len(), dispatch::points().len());
        assert_eq!(v.iter().filter(|&&t| t == 0).count(), 1);
        assert_eq!(v[10], 0, "point 11 is the eleventh entry");

        let all = tick_vector(0, DEFAULT_TICKS, 0).expect("calibration");
        assert!(all.iter().all(|&t| t == DEFAULT_TICKS), "0 must spin every point");

        let own = tick_vector(11, DEFAULT_TICKS, 250).expect("own ticks");
        assert_eq!(own[10], 250);
        assert_eq!(own.iter().filter(|&&t| t == DEFAULT_TICKS).count(), 14);

        let e = tick_vector(99, DEFAULT_TICKS, 0).expect_err("no point 99");
        assert!(e.contains("no published point 99"), "{e}");
    }

    /// The emitted shape table must be the generator's own point list, in
    /// order -- that is the whole reason this is generated.
    #[test]
    fn the_shape_table_comes_from_the_point_list() {
        let text = point_table();
        for p in dispatch::points() {
            let s = &p.shape;
            let row = format!(
                "{{{},{},{},{},{}}},  // point {}",
                s.e, s.m, s.k, s.n, s.group_list_type, p.id
            );
            assert!(text.contains(&row), "the table is missing {row}");
        }
    }

    /// A spin unit carries the spin construct, keeps the generated dispatch
    /// under its new name, and has exactly one `run_kernel` entry point.
    #[test]
    fn the_spin_unit_renames_the_dispatch_and_wraps_it() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let bytes = assemble(&dir, &[], 11, DEFAULT_TICKS, 0).expect("spin unit");
        let text = String::from_utf8(bytes).expect("utf8");
        assert_eq!(text.matches("extern \"C\" void run_kernel(").count(), 1);
        assert_eq!(text.matches("static void GmmRunKernelBody(").count(), 1);
        assert!(text.contains("GmmProbeSpin<<<1, nullptr, stream>>>"));
        assert!(text.contains("GetSystemCycle"));
        assert!(text.contains("GmmRunKernelBody(x, info_x,"), "the wrapper must call the body");
        // …and it is unmistakably not a submission candidate.
        assert!(text.starts_with("// ="), "the banner must come first");
        assert!(text.contains("NOT A SUBMISSION CANDIDATE"));
    }

    /// `--profile final` must still refuse everything this profile emits, so
    /// the two cannot be confused at the CLI.
    #[test]
    fn final_still_refuses_what_spin_emits() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let spun = String::from_utf8(assemble(&dir, &[], 11, DEFAULT_TICKS, 0).expect("spin"))
            .expect("utf8");
        let clean = String::from_utf8(emit::final_profile(&dir).expect("final")).expect("utf8");
        for needle in ["GetSystemCycle", "GmmProbeSpin", "GMM_PROBE"] {
            assert!(spun.contains(needle), "spin should contain {needle:?}");
            assert!(!clean.contains(needle), "final must not contain {needle:?}");
        }
    }

    /// A body selection reaches the spin profile too, so an isolation can
    /// measure the picked kernel rather than only the default one.
    #[test]
    fn a_picked_body_reaches_the_spin_unit() {
        let dir = chunks::chunks_dir().expect("gen/chunks");
        let env = dispatch::Env::default();
        let picks = crate::body::parse_picks("11=n1-fold", &env).expect("pickable");
        let overrides = crate::body::overrides(&dir, &picks).expect("overrides");
        let text = String::from_utf8(
            assemble(&dir, &overrides, 11, DEFAULT_TICKS, 0).expect("spin unit"),
        )
        .expect("utf8");
        assert!(text.contains("Add(f[c],f[c],f[half+c]"), "the picked fold is not in the unit");
    }
}
