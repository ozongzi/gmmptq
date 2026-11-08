//! The reference. Exact INT32 accumulation, then the two FLOAT32 multiplies in
//! the order `problem.md` §3.2 specifies, then round-half-to-even to the output
//! type.
//!
//! The integer part is exact, so the only freedom is the floating-point order,
//! and the spec pins it: weightScale first (widened from BFLOAT16 when that is
//! its type), xScale second. A kernel is allowed to reorder -- §3.2 says so --
//! but the reference must not, or the tolerance is being measured against the
//! wrong number.
use std::path::Path;

use crate::cases::{Case, Split};
use crate::num::Out;

pub struct Golden {
    /// Rounded to the output type: what the kernel's bytes are compared against.
    pub bits: Vec<u16>,
    /// Before rounding, so the bench can separate representation error from
    /// computation error.
    pub exact: Vec<f32>,
    pub max_abs: f32,
    pub max_acc: i32,
    pub threads: usize,
}

fn read(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Cumulative end position per expert, whichever encoding the case uses.
pub fn ends(case: &Case, list: &[i64]) -> Vec<usize> {
    if case.gt == 0 {
        return list.iter().map(|&v| v as usize).collect();
    }
    let mut acc = 0usize;
    list.iter()
        .map(|&v| {
            acc += v as usize;
            acc
        })
        .collect()
}

/// Which expert owns each of the M rows. Empty experts simply never appear.
pub fn row_experts(case: &Case, ends: &[usize]) -> Vec<u32> {
    let mut out = vec![0u32; case.m];
    let mut start = 0usize;
    for (e, &end) in ends.iter().enumerate() {
        for row in start..end.min(case.m) {
            out[row] = e as u32;
        }
        start = start.max(end);
    }
    out
}

pub fn compute(dir: &Path, case: &Case, _split: Split) -> std::io::Result<Golden> {
    let x = read(&dir.join("x.bin"))?;
    let weight = read(&dir.join("weight.bin"))?;
    let xscale_raw = read(&dir.join("xscale.bin"))?;
    let wscale_raw = read(&dir.join("wscale.bin"))?;
    let glist_raw = read(&dir.join("grouplist.bin"))?;

    let (m, k, n, e) = (case.m, case.k, case.n, case.e);
    assert_eq!(x.len(), m * k, "x.bin wrong size");
    assert_eq!(weight.len(), e * k * n, "weight.bin wrong size");

    let xscale: Vec<f32> = xscale_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // weightScale arrives as FLOAT32 or BFLOAT16 and is widened to FLOAT32
    // before use, which is what the reference does.
    let wscale: Vec<f32> = match case.out {
        Out::F16 => wscale_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Out::Bf16 => wscale_raw
            .chunks_exact(2)
            .map(|c| crate::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    };
    let glist: Vec<i64> = glist_raw
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let ends = ends(case, &glist);
    assert_eq!(*ends.last().unwrap_or(&0), m, "groupList does not cover M");
    let owner = row_experts(case, &ends);

    let xs: &[i8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const i8, x.len()) };
    let ws: &[i8] =
        unsafe { std::slice::from_raw_parts(weight.as_ptr() as *const i8, weight.len()) };

    // Shadowed as slices: every worker closure is `move`, and a Vec would be
    // moved into the first one. A shared slice is Copy, so each gets its own.
    let xscale: &[f32] = &xscale;
    let wscale: &[f32] = &wscale;

    let mut bits = vec![0u16; m * n];
    let mut exact = vec![0f32; m * n];

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(m.max(1));
    let rows_per = (m + threads - 1) / threads;

    // One i32 per output column, reused down the K loop. The inner loop walks
    // a contiguous weight row and a contiguous accumulator, which is the only
    // layout that makes a scalar reference fast enough to be usable.
    let stats: Vec<(f32, i32)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let bit_chunks = bits.chunks_mut(rows_per * n);
        let exact_chunks = exact.chunks_mut(rows_per * n);
        for (chunk_index, (bit_chunk, exact_chunk)) in
            bit_chunks.zip(exact_chunks).enumerate()
        {
            let owner = &owner;
            let handle = scope.spawn(move || {
                let first_row = chunk_index * rows_per;
                let mut acc = vec![0i32; n];
                let mut max_abs = 0f32;
                let mut max_acc = 0i32;
                for local_row in 0..(bit_chunk.len() / n) {
                    let row = first_row + local_row;
                    let expert = owner[row] as usize;
                    acc.iter_mut().for_each(|a| *a = 0);
                    let xrow = &xs[row * k..row * k + k];
                    let wbase = expert * k * n;
                    for (kk, &a) in xrow.iter().enumerate() {
                        if a == 0 {
                            continue;
                        }
                        let a = a as i32;
                        let wrow = &ws[wbase + kk * n..wbase + kk * n + n];
                        for (dst, &w) in acc.iter_mut().zip(wrow.iter()) {
                            *dst += a * w as i32;
                        }
                    }
                    let xs_row = xscale[row];
                    let wbase_scale = expert * n;
                    for col in 0..n {
                        max_acc = max_acc.max(acc[col].abs());
                        // weightScale first, then xScale: the specified order.
                        let value = (acc[col] as f32) * wscale[wbase_scale + col] * xs_row;
                        max_abs = max_abs.max(value.abs());
                        exact_chunk[local_row * n + col] = value;
                        bit_chunk[local_row * n + col] = case.out.round(value);
                    }
                }
                (max_abs, max_acc)
            });
            handles.push(handle);
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let max_abs = stats.iter().map(|s| s.0).fold(0f32, f32::max);
    let max_acc = stats.iter().map(|s| s.1).max().unwrap_or(0);

    // The two promises from problem.md §3.4, checked rather than trusted: if
    // the generator's scale ceiling were wrong, every later measurement would
    // be against data the judge would never produce.
    assert!(
        max_abs <= 60_000.0,
        "tp{}: |value| reached {max_abs}, past the 60000 the spec promises",
        case.id
    );
    assert!(
        (max_acc as i64) < i32::MAX as i64,
        "tp{}: accumulator overflowed int32",
        case.id
    );

    Ok(Golden { bits, exact, max_abs, max_acc, threads })
}

pub fn write(dir: &Path, golden: &Golden) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(golden.bits.len() * 2);
    for b in &golden.bits {
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    std::fs::write(dir.join("golden.bin"), &bytes)?;
    let mut exact = Vec::with_capacity(golden.exact.len() * 4);
    for v in &golden.exact {
        exact.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("golden_f32.bin"), &exact)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CASES;

    fn case_by_id(id: u32) -> Case {
        *CASES.iter().find(|c| c.id == id).unwrap()
    }

    #[test]
    fn ends_decodes_both_encodings_to_the_same_thing() {
        // tp7 and tp8 are the same shape with gt 0 and 1, so [1] as ends and
        // [1] as counts must agree.
        let seven = case_by_id(7);
        let eight = case_by_id(8);
        assert_eq!(ends(&seven, &[1]), vec![1]);
        assert_eq!(ends(&eight, &[1]), vec![1]);
        // A three-expert split: ends [2,2,5] == counts [2,0,3].
        let mut c = case_by_id(1);
        c.e = 3;
        c.m = 5;
        c.gt = 0;
        assert_eq!(ends(&c, &[2, 2, 5]), vec![2, 2, 5]);
        c.gt = 1;
        assert_eq!(ends(&c, &[2, 0, 3]), vec![2, 2, 5]);
    }

    #[test]
    fn row_experts_skips_empty_experts() {
        let mut c = case_by_id(1);
        c.e = 4;
        c.m = 5;
        // expert 0 gets rows 0-1, expert 1 empty, expert 2 gets 2-4, expert 3 empty.
        let owner = row_experts(&c, &[2, 2, 5, 5]);
        assert_eq!(owner, vec![0, 0, 2, 2, 2]);
    }

    /// The obvious implementation: one thread, three loops, no skipping, no
    /// slicing. Slow and therefore trustworthy. `compute` has to agree with it
    /// bit for bit, which is what exercises the thread chunking, the row ->
    /// expert mapping across chunk boundaries, and the multiply order.
    fn naive(dir: &Path, case: &Case) -> Vec<u16> {
        let x = std::fs::read(dir.join("x.bin")).unwrap();
        let w = std::fs::read(dir.join("weight.bin")).unwrap();
        let xs_raw = std::fs::read(dir.join("xscale.bin")).unwrap();
        let ws_raw = std::fs::read(dir.join("wscale.bin")).unwrap();
        let g_raw = std::fs::read(dir.join("grouplist.bin")).unwrap();

        let xscale: Vec<f32> = xs_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let wscale: Vec<f32> = match case.out {
            Out::F16 => ws_raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Out::Bf16 => ws_raw
                .chunks_exact(2)
                .map(|c| crate::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        };
        let glist: Vec<i64> = g_raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let ends = ends(case, &glist);

        let (m, k, n) = (case.m, case.k, case.n);
        let mut out = vec![0u16; m * n];
        let mut start = 0usize;
        for (e, &end) in ends.iter().enumerate() {
            for row in start..end {
                for col in 0..n {
                    let mut acc = 0i32;
                    for kk in 0..k {
                        let a = x[row * k + kk] as i8 as i32;
                        let b = w[e * k * n + kk * n + col] as i8 as i32;
                        acc += a * b;
                    }
                    let value = (acc as f32) * wscale[e * n + col] * xscale[row];
                    out[row * n + col] = case.out.round(value);
                }
            }
            start = start.max(end);
        }
        out
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gmmbench-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn threaded_reference_matches_the_obvious_one() {
        // M is deliberately larger than the thread count so the row range is
        // split, and every split shape is covered because the expert boundary
        // landing inside a chunk is exactly where this would break.
        for (e, m, k, n, gt, out) in [
            (3usize, 200usize, 4usize, 3usize, 0u8, Out::F16),
            (3, 200, 4, 3, 1, Out::Bf16),
            (1, 97, 33, 7, 0, Out::F16),
            (7, 61, 5, 9, 1, Out::Bf16),
        ] {
            let mut case = case_by_id(1);
            case.e = e;
            case.m = m;
            case.k = k;
            case.n = n;
            case.gt = gt;
            case.out = out;
            // These are invented shapes, so only the stress groupings apply.
            for split in crate::cases::SYNTHETIC {
                let dir = scratch(&format!("{e}x{m}x{k}x{n}-{gt}-{}", split.name()));
                crate::inputs::generate(&dir, &case, split).unwrap();
                let path = crate::inputs::case_dir(&dir, &case, split);
                let fast = compute(&path, &case, split).unwrap();
                let slow = naive(&path, &case);
                assert_eq!(
                    fast.bits.len(),
                    slow.len(),
                    "{e}x{m}x{k}x{n} {}", split.name()
                );
                let differing = fast
                    .bits
                    .iter()
                    .zip(slow.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    differing, 0,
                    "{e}x{m}x{k}x{n} gt={gt} {} : {differing} of {} elements differ",
                    split.name(),
                    slow.len()
                );
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }

    #[test]
    fn leading_empty_experts_do_not_claim_row_zero() {
        let mut c = case_by_id(1);
        c.e = 3;
        c.m = 2;
        // experts 0 and 1 empty, expert 2 owns everything.
        let owner = row_experts(&c, &[0, 0, 2]);
        assert_eq!(owner, vec![2, 2]);
    }
}
