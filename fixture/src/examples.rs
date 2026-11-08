//! The three worked examples from `problem.md` §6, verbatim.
//!
//! These are the only ground truth that does not come from our own code: the
//! problem statement states the inputs and the exact expected output. They are
//! small enough to check by hand and they cover the three things most likely
//! to be wrong -- two experts with different per-channel scales, empty experts
//! at the head, the middle and the tail with a BFLOAT16 weightScale, and rows
//! whose result is negative or zero.
//!
//! They are written to disk as ordinary case directories, so the kernel can be
//! run against them exactly like a generated case.
use std::path::{Path, PathBuf};

use crate::num::{Out, f32_to_bf16};

pub struct Example {
    pub name: &'static str,
    pub e: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub gt: u8,
    pub out: Out,
    /// Row major, M*K.
    pub x: &'static [i8],
    /// Row major, E*K*N.
    pub weight: &'static [i8],
    pub xscale: &'static [f32],
    /// E*N. Stored as BFLOAT16 when the output is BFLOAT16.
    pub wscale: &'static [f32],
    pub grouplist: &'static [i64],
    /// What §6 says the answer is, M*N row major.
    pub expect: &'static [f32],
}

pub const EXAMPLES: [Example; 3] = [
    // §6 example 1: two experts, different per-token and per-channel scales.
    Example {
        name: "two-experts",
        e: 2, m: 3, k: 2, n: 2, gt: 0, out: Out::F16,
        x: &[1, 2, 3, 4, -1, 2],
        weight: &[1, 2, 3, 4, /* expert 1 */ 2, -1, 1, 3],
        xscale: &[0.5, 0.25, 0.125],
        wscale: &[0.5, 0.25, 0.25, 0.5],
        grouplist: &[2, 3],
        expect: &[1.75, 1.25, 1.875, 1.375, 0.0, 0.4375],
    },
    // §6 example 2: experts 0, 2 and 4 empty -- head, middle and tail at once,
    // with a BFLOAT16 weightScale.
    Example {
        name: "empty-head-mid-tail",
        e: 5, m: 2, k: 2, n: 1, gt: 0, out: Out::Bf16,
        x: &[1, 2, -1, 1],
        weight: &[9, 9, 2, 1, 8, 8, 1, 3, 7, 7],
        xscale: &[0.5, 0.25],
        wscale: &[0.5, 0.5, 0.5, 0.25, 0.5],
        grouplist: &[0, 1, 1, 2, 2],
        expect: &[1.0, 0.125],
    },
    // §6 example 3: a negative result and an all-zero input row, both of which
    // must survive rather than being skipped.
    Example {
        name: "negative-and-zero",
        e: 1, m: 2, k: 2, n: 2, gt: 0, out: Out::F16,
        x: &[-1, 0, 0, 0],
        weight: &[2, -3, 1, 4],
        xscale: &[0.5, 0.25],
        wscale: &[0.25, 0.5],
        grouplist: &[2],
        expect: &[-0.25, 0.75, 0.0, 0.0],
    },
];

impl Example {
    pub fn dir(&self, root: &Path) -> PathBuf {
        root.join(format!("ex_{}", self.name))
    }

    /// Write the example as a case directory the driver can consume, with
    /// golden.bin holding the output §6 states rather than anything we
    /// computed.
    pub fn write(&self, root: &Path) -> std::io::Result<PathBuf> {
        let dir = self.dir(root);
        std::fs::create_dir_all(&dir)?;

        let as_u8 = |v: &[i8]| v.iter().map(|&b| b as u8).collect::<Vec<u8>>();
        std::fs::write(dir.join("x.bin"), as_u8(self.x))?;
        std::fs::write(dir.join("weight.bin"), as_u8(self.weight))?;

        let mut xs = Vec::new();
        for v in self.xscale {
            xs.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("xscale.bin"), &xs)?;

        let mut ws = Vec::new();
        for v in self.wscale {
            match self.out {
                Out::F16 => ws.extend_from_slice(&v.to_le_bytes()),
                Out::Bf16 => ws.extend_from_slice(&f32_to_bf16(*v).to_le_bytes()),
            }
        }
        std::fs::write(dir.join("wscale.bin"), &ws)?;

        let mut gl = Vec::new();
        for v in self.grouplist {
            gl.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("grouplist.bin"), &gl)?;

        let mut golden = Vec::new();
        let mut exact = Vec::new();
        for v in self.expect {
            golden.extend_from_slice(&self.out.round(*v).to_le_bytes());
            exact.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("golden.bin"), &golden)?;
        std::fs::write(dir.join("golden_f32.bin"), &exact)?;

        std::fs::write(
            dir.join("shape.txt"),
            format!(
                "{} {} {} {} {} {}\n",
                self.e, self.m, self.k, self.n, self.gt,
                match self.out { Out::F16 => "float16", Out::Bf16 => "bfloat16" }
            ),
        )?;
        std::fs::write(
            dir.join("meta.txt"),
            format!(
                "problem.md §6 example: {}\nE M K N  {} {} {} {}\ngroupListType {}\n\
                 outputDtype {}\ngroupList {:?}\nexpected {:?}\n",
                self.name, self.e, self.m, self.k, self.n, self.gt,
                self.out.name(), self.grouplist, self.expect
            ),
        )?;
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Our own reference, run on the stated inputs, must produce the stated
    /// output. If this fails, `golden.rs` is wrong, not the problem statement.
    #[test]
    fn our_reference_reproduces_the_stated_answers() {
        for example in EXAMPLES.iter() {
            let (e, m, k, n) = (example.e, example.m, example.k, example.n);
            // Decode the groupList the way the spec says for this type.
            let ends: Vec<usize> = if example.gt == 0 {
                example.grouplist.iter().map(|&v| v as usize).collect()
            } else {
                let mut acc = 0usize;
                example.grouplist.iter().map(|&v| { acc += v as usize; acc }).collect()
            };
            assert_eq!(*ends.last().unwrap(), m, "{}: list does not cover M", example.name);

            // weightScale is widened from its stored type first, exactly as
            // the reference does -- for the bf16 case that rounding matters.
            let wscale: Vec<f32> = match example.out {
                Out::F16 => example.wscale.to_vec(),
                Out::Bf16 => example
                    .wscale
                    .iter()
                    .map(|v| crate::num::bf16_to_f32(f32_to_bf16(*v)))
                    .collect(),
            };

            let mut start = 0usize;
            for (expert, &end) in ends.iter().enumerate() {
                for row in start..end {
                    for col in 0..n {
                        let mut acc = 0i32;
                        for kk in 0..k {
                            acc += example.x[row * k + kk] as i32
                                * example.weight[expert * k * n + kk * n + col] as i32;
                        }
                        let got = (acc as f32) * wscale[expert * n + col] * example.xscale[row];
                        let want = example.expect[row * n + col];
                        assert_eq!(
                            example.out.round(got),
                            example.out.round(want),
                            "{} row {row} col {col}: got {got}, §6 says {want}",
                            example.name
                        );
                    }
                }
                start = start.max(end);
            }
            assert_eq!(e, ends.len());
        }
    }

    #[test]
    fn every_example_is_internally_consistent() {
        for ex in EXAMPLES.iter() {
            assert_eq!(ex.x.len(), ex.m * ex.k, "{}: x", ex.name);
            assert_eq!(ex.weight.len(), ex.e * ex.k * ex.n, "{}: weight", ex.name);
            assert_eq!(ex.xscale.len(), ex.m, "{}: xScale", ex.name);
            assert_eq!(ex.wscale.len(), ex.e * ex.n, "{}: weightScale", ex.name);
            assert_eq!(ex.grouplist.len(), ex.e, "{}: groupList", ex.name);
            assert_eq!(ex.expect.len(), ex.m * ex.n, "{}: expected", ex.name);
        }
    }
}
