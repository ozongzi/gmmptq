//! Comparison against the reference.
//!
//! `problem.md` §5 asks for "相对误差 < 1e-3，绝对误差 < 1e-3" for fp16 and
//! bf16. Read literally -- both must hold for every element -- a value of
//! 50000 would need to be accurate to 1e-3 absolute, which is far finer than
//! the type can even represent. The usual reading, and the one the Ascend
//! tooling applies, is that an element passes when either holds: relative for
//! the large values, absolute for the small ones.
//!
//! Rather than pick, this reports both counts. If the two ever disagree about
//! pass or fail, that is a fact worth seeing rather than hiding behind one
//! chosen interpretation.
use crate::cases::Case;

pub const TOL: f32 = 1e-3;

pub struct Report {
    pub elements: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    /// Elements where neither the relative nor the absolute error is inside
    /// tolerance. These are failures under any reading.
    pub fail_either: usize,
    /// Elements where one of the two is outside tolerance. Failures only under
    /// the strict reading.
    pub fail_both: usize,
    pub worst_index: usize,
    pub worst_got: f32,
    pub worst_want: f32,
    /// Elements that differ at all, bit for bit.
    pub differing: usize,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.fail_either == 0
    }
}

pub fn compare(case: &Case, got_bytes: &[u8], golden_bits: &[u16]) -> Result<Report, String> {
    let want_elements = case.m * case.n;
    if got_bytes.len() != want_elements * 2 {
        return Err(format!(
            "output is {} bytes, expected {} ({} elements x 2)",
            got_bytes.len(),
            want_elements * 2,
            want_elements
        ));
    }
    if golden_bits.len() != want_elements {
        return Err(format!(
            "golden has {} elements, expected {want_elements}",
            golden_bits.len()
        ));
    }

    let mut r = Report {
        elements: want_elements,
        max_abs: 0.0,
        max_rel: 0.0,
        fail_either: 0,
        fail_both: 0,
        worst_index: 0,
        worst_got: 0.0,
        worst_want: 0.0,
        differing: 0,
    };
    let mut worst_score = -1f32;

    for index in 0..want_elements {
        let got_bits = u16::from_le_bytes([got_bytes[index * 2], got_bytes[index * 2 + 1]]);
        let want_bits = golden_bits[index];
        if got_bits != want_bits {
            r.differing += 1;
        }
        let got = case.out.decode(got_bits);
        let want = case.out.decode(want_bits);

        if !got.is_finite() || !want.is_finite() {
            // A non-finite anywhere is a hard failure: §3.4 promises the data
            // keeps every result finite.
            r.fail_either += 1;
            r.fail_both += 1;
            continue;
        }
        let abs = (got - want).abs();
        let rel = if want == 0.0 {
            if abs == 0.0 { 0.0 } else { f32::INFINITY }
        } else {
            abs / want.abs()
        };
        r.max_abs = r.max_abs.max(abs);
        if rel.is_finite() {
            r.max_rel = r.max_rel.max(rel);
        }
        let abs_ok = abs < TOL;
        let rel_ok = rel < TOL;
        if !abs_ok && !rel_ok {
            r.fail_either += 1;
        }
        if !abs_ok || !rel_ok {
            r.fail_both += 1;
        }
        // Rank by whichever error is the smaller one, since that is what
        // decides the lenient verdict.
        let score = abs.min(if rel.is_finite() { rel } else { f32::MAX });
        if score > worst_score {
            worst_score = score;
            r.worst_index = index;
            r.worst_got = got;
            r.worst_want = want;
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CASES;
    use crate::num::Out;

    fn tiny(out: Out) -> Case {
        let mut c = *CASES.iter().find(|c| c.id == 6).unwrap();
        c.m = 1;
        c.n = 2;
        c.out = out;
        c
    }

    #[test]
    fn identical_output_passes_cleanly() {
        let case = tiny(Out::F16);
        let golden = vec![crate::num::f32_to_f16(1.5), crate::num::f32_to_f16(-2.25)];
        let bytes: Vec<u8> = golden.iter().flat_map(|b| b.to_le_bytes()).collect();
        let r = compare(&case, &bytes, &golden).unwrap();
        assert!(r.passed());
        assert_eq!(r.differing, 0);
        assert_eq!(r.fail_both, 0);
        assert_eq!(r.max_abs, 0.0);
    }

    #[test]
    fn one_ulp_off_a_large_value_passes_on_relative_and_fails_on_strict() {
        let case = tiny(Out::F16);
        let want = crate::num::f32_to_f16(50000.0);
        // One representable step away: ~32 absolute at this magnitude, which
        // is 6.4e-4 relative -- inside the relative budget, nowhere near the
        // absolute one.
        let got = want + 1;
        let golden = vec![want, want];
        let bytes: Vec<u8> = [got, want].iter().flat_map(|b| b.to_le_bytes()).collect();
        let r = compare(&case, &bytes, &golden).unwrap();
        assert!(r.passed(), "lenient reading should pass");
        assert_eq!(r.fail_both, 1, "strict reading should flag it");
        assert!(r.max_rel < 1e-3 && r.max_abs > 1.0, "{} {}", r.max_rel, r.max_abs);
    }

    #[test]
    fn a_real_error_fails_under_both_readings() {
        let case = tiny(Out::Bf16);
        let golden = vec![crate::num::f32_to_bf16(1.0), crate::num::f32_to_bf16(1.0)];
        let wrong = crate::num::f32_to_bf16(1.5);
        let bytes: Vec<u8> = [wrong, golden[1]].iter().flat_map(|b| b.to_le_bytes()).collect();
        let r = compare(&case, &bytes, &golden).unwrap();
        assert!(!r.passed());
        assert_eq!(r.fail_either, 1);
    }

    #[test]
    fn size_mismatch_is_an_error_not_a_panic() {
        let case = tiny(Out::F16);
        let golden = vec![0u16; 2];
        assert!(compare(&case, &[0u8; 2], &golden).is_err());
    }

    #[test]
    fn zero_against_nonzero_is_a_failure() {
        let case = tiny(Out::F16);
        let golden = vec![crate::num::f32_to_f16(0.0), crate::num::f32_to_f16(0.0)];
        let got = crate::num::f32_to_f16(1.0);
        let bytes: Vec<u8> = [got, golden[1]].iter().flat_map(|b| b.to_le_bytes()).collect();
        let r = compare(&case, &bytes, &golden).unwrap();
        assert!(!r.passed());
    }
}
