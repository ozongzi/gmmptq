//! Input generation. Writes the five input tensors as raw little-endian
//! binaries, the same bytes a host harness would hand the kernel.
//!
//! The one non-obvious part is the scale ceiling. `problem.md` §3.4 promises
//! two things at once: scales live in [1e-5, 1.0], and every dequantised FP32
//! result is finite with |v| <= 60000. Those are not independent. For random
//! full-range INT8 the accumulator has standard deviation 5461*sqrt(K), so at
//! K=8192 it runs to about half a million; scales anywhere near 1.0 would blow
//! straight past 60000. So the draw is capped per case to keep 8 sigma inside
//! the promise, and `golden` asserts the promise actually held.
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cases::{Case, Split, counts, group_list};
use crate::num::{Out, f32_to_bf16};
use crate::rng::Rng;

/// Standard deviation of one int8*int8 product, for uniform full-range input:
/// var(int8 uniform) = (256^2 - 1)/12 = 5461.25, and the product of two
/// independent zero-mean draws has E[(xw)^2] = 5461.25^2.
const PRODUCT_SIGMA: f64 = 5461.25;
const OUTPUT_LIMIT: f64 = 60_000.0;
const SIGMAS: f64 = 8.0;

/// The largest scale either factor may take so that 8 sigma of accumulator
/// still dequantises inside |v| <= 60000.
pub fn scale_ceiling(k: usize) -> f64 {
    let acc_sigma = PRODUCT_SIGMA * (k as f64).sqrt();
    let product_max = OUTPUT_LIMIT / (SIGMAS * acc_sigma);
    product_max.sqrt().min(1.0)
}

/// Floor is above 1e-5 so that rounding a weightScale to BFLOAT16 cannot drop
/// it under the promised minimum.
const SCALE_FLOOR: f64 = 2e-5;

pub fn case_dir(root: &Path, case: &Case, split: Split) -> PathBuf {
    root.join(format!("tp{:02}_{}", case.id, split.name()))
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    Ok(())
}

pub fn generate(root: &Path, case: &Case, split: Split) -> std::io::Result<PathBuf> {
    let dir = case_dir(root, case, split);
    std::fs::create_dir_all(&dir)?;
    let ceiling = scale_ceiling(case.k);

    // x: (M, K) int8, full range.
    let mut x = vec![0u8; case.m * case.k];
    let mut rng = Rng::new("x", case.id, 0);
    for byte in x.iter_mut() {
        *byte = rng.int8() as u8;
    }
    write_bytes(&dir.join("x.bin"), &x)?;

    // weight: (E, K, N) int8, ND, not transposed.
    let mut weight = vec![0u8; case.e * case.k * case.n];
    let mut rng = Rng::new("weight", case.id, 0);
    for byte in weight.iter_mut() {
        *byte = rng.int8() as u8;
    }
    write_bytes(&dir.join("weight.bin"), &weight)?;

    // xScale: (M,) f32, always FLOAT32.
    let mut xscale = Vec::with_capacity(case.m * 4);
    let mut rng = Rng::new("xscale", case.id, 0);
    for _ in 0..case.m {
        let v = rng.log_uniform(SCALE_FLOOR, ceiling) as f32;
        xscale.extend_from_slice(&v.to_le_bytes());
    }
    write_bytes(&dir.join("xscale.bin"), &xscale)?;

    // weightScale: (E, N), FLOAT32 with an fp16 output, BFLOAT16 with a bf16 one.
    let mut wscale = Vec::with_capacity(case.e * case.n * case.wscale_bytes());
    let mut rng = Rng::new("wscale", case.id, 0);
    for _ in 0..case.e * case.n {
        let v = rng.log_uniform(SCALE_FLOOR, ceiling) as f32;
        match case.out {
            Out::F16 => wscale.extend_from_slice(&v.to_le_bytes()),
            Out::Bf16 => wscale.extend_from_slice(&f32_to_bf16(v).to_le_bytes()),
        }
    }
    write_bytes(&dir.join("wscale.bin"), &wscale)?;

    // groupList: (E,) int64, in this case's own encoding.
    let list = group_list(case, split);
    let mut glist = Vec::with_capacity(case.e * 8);
    for v in &list {
        glist.extend_from_slice(&v.to_le_bytes());
    }
    write_bytes(&dir.join("grouplist.bin"), &glist)?;

    let counts = counts(case, split);
    let meta = format!(
        "case            tp{}\n\
         split           {}\n\
         E M K N         {} {} {} {}\n\
         groupListType   {}  ({})\n\
         outputDtype     {}\n\
         weightScale     {}\n\
         scale range     [{:.3e}, {:.3e}]  (ceiling from K={})\n\
         counts          {:?}\n\
         groupList       {:?}\n\
         empty experts   {} of {}\n\
         io bytes        {}\n",
        case.id,
        split.name(),
        case.e,
        case.m,
        case.k,
        case.n,
        case.gt,
        if case.gt == 0 { "cumulative ends" } else { "per-expert counts" },
        case.out.name(),
        if case.wscale_bytes() == 4 { "float32" } else { "bfloat16" },
        SCALE_FLOOR,
        ceiling,
        case.k,
        counts,
        list,
        counts.iter().filter(|&&c| c == 0).count(),
        case.e,
        case.io_bytes(),
    );
    write_bytes(&dir.join("meta.txt"), meta.as_bytes())?;
    write_shape(&dir, case)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CASES;

    #[test]
    fn ceiling_keeps_outputs_inside_the_promise() {
        for case in CASES.iter() {
            let c = scale_ceiling(case.k);
            assert!(c > 0.0 && c <= 1.0, "tp{} ceiling {c}", case.id);
            let acc_sigma = PRODUCT_SIGMA * (case.k as f64).sqrt();
            assert!(
                SIGMAS * acc_sigma * c * c <= OUTPUT_LIMIT * 1.000_001,
                "tp{} would exceed 60000",
                case.id
            );
        }
    }

    #[test]
    fn small_k_is_not_capped_below_one() {
        // K=1 accumulates a single product, so the full [1e-5, 1] range is
        // safe and the cap must not shrink it for no reason.
        assert_eq!(scale_ceiling(1), 1.0);
        // K=2 is already fractionally capped at 8 sigma -- which is correct,
        // not a bug, but it should still be essentially the whole range.
        assert!(scale_ceiling(2) > 0.98, "{}", scale_ceiling(2));
        // K=8192 must bite hard.
        assert!(scale_ceiling(8192) < 0.2, "{}", scale_ceiling(8192));
    }

    #[test]
    fn bf16_rounding_keeps_scales_above_the_minimum() {
        use crate::num::bf16_to_f32;
        let rounded = bf16_to_f32(f32_to_bf16(SCALE_FLOOR as f32));
        assert!(rounded >= 1e-5, "{rounded}");
    }
}

/// One machine-readable line per case, so the C++ driver does not need shape
/// arguments plumbed through a shell: `E M K N gt outputDtype`.
pub fn write_shape(dir: &Path, case: &Case) -> std::io::Result<()> {
    write_bytes(
        &dir.join("shape.txt"),
        format!(
            "{} {} {} {} {} {}\n",
            case.e,
            case.m,
            case.k,
            case.n,
            case.gt,
            match case.out {
                Out::F16 => "float16",
                Out::Bf16 => "bfloat16",
            }
        )
        .as_bytes(),
    )
}
