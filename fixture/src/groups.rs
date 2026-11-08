//! The judge's own groupList for each point.
//!
//! These are **not** guesses and not a stress split: they are the distributions
//! the judge actually runs, recovered through the OJ and confirmed element-wise
//! on point 14 by a submission that executed them. `data/recovered.json` is the
//! provenance record and `SOURCE_SHA256` pins the snapshot these numbers came
//! from; the numbers live here, in Rust, so that `counts_cover_m` and
//! `encoding_matches_the_recorded_raw` check them at test time instead of a
//! JSON parser checking them at run time.
//!
//! Only the grouping is the judge's. x, weight and both scales are still our
//! own generated inputs -- the judge never returns those.
//!
//! What is stored is per-expert token counts. The wire encoding (cumulative
//! ends for `groupListType` 0, counts for 1) is derived by `cases::group_list`,
//! so there is one source for it and the two cannot disagree.
pub const SOURCE: &str = "data/recovered.json";
pub const SOURCE_SHA256: &str =
    "b40da251a039e14c337e0b3eb46a98d01f2b1bdf9e5f6476db4849c8730f9871";

/// Point 13 is 63 experts holding one row each, interleaved with empty ones:
/// `[0, 1]` sixty-three times, then `[0, 0]`.
const fn point13() -> [usize; 128] {
    let mut out = [0usize; 128];
    let mut i = 1;
    while i < 126 {
        out[i] = 1;
        i += 2;
    }
    out
}
static POINT13: [usize; 128] = point13();

/// Per-expert token counts, by point id.
pub const RECOVERED: [(u32, &[usize]); 15] = [
    (1, &[1024, 1024, 1024, 1024]),
    (2, &[63, 65, 64, 65]),
    (3, &[2, 1]),
    (4, &[897, 1, 17, 0, 65, 31, 0, 16]),
    (5, &[0, 1, 0, 1, 0]),
    (6, &[2]),
    (7, &[1]),
    (8, &[1]),
    (9, &[1, 257, 0, 255]),
    (10, &[1, 255, 0, 769]),
    (11, &[1, 511, 0, 515]),
    (12, &[96, 96, 96, 96, 96, 96, 96, 96]),
    (13, &POINT13),
    (14, &[383, 385, 511, 0, 769, 1, 1024]),
    (15, &[3073, 1, 0, 1023]),
];

pub fn counts(case_id: u32) -> Option<&'static [usize]> {
    RECOVERED
        .iter()
        .find(|(id, _)| *id == case_id)
        .map(|(_, counts)| *counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::{CASES, Split, group_list};

    #[test]
    fn every_point_has_counts_that_cover_m() {
        for case in CASES.iter() {
            let counts = counts(case.id).unwrap_or_else(|| panic!("tp{} missing", case.id));
            assert_eq!(counts.len(), case.e, "tp{} expert count", case.id);
            assert_eq!(counts.iter().sum::<usize>(), case.m, "tp{} rows", case.id);
        }
    }

    /// The `raw` arrays as `recovered.json` records them, i.e. already in each
    /// point's own wire encoding. Deriving them from counts has to reproduce
    /// these exactly, or one of the two is wrong.
    #[test]
    fn encoding_matches_the_recorded_raw() {
        let recorded: [(u32, &[i64]); 14] = [
            (1, &[1024, 2048, 3072, 4096]),
            (2, &[63, 65, 64, 65]),
            (3, &[2, 3]),
            (4, &[897, 898, 915, 915, 980, 1011, 1011, 1027]),
            (5, &[0, 1, 1, 2, 2]),
            (6, &[2]),
            (7, &[1]),
            (8, &[1]),
            (9, &[1, 258, 258, 513]),
            (10, &[1, 256, 256, 1025]),
            (11, &[1, 511, 0, 515]),
            (12, &[96, 96, 96, 96, 96, 96, 96, 96]),
            (14, &[383, 385, 511, 0, 769, 1, 1024]),
            (15, &[3073, 1, 0, 1023]),
        ];
        for (id, raw) in recorded {
            let case = CASES.iter().find(|c| c.id == id).unwrap();
            assert_eq!(group_list(case, Split::Real), raw, "tp{id}");
        }
        // Point 13: 128 entries, counts encoding, 63 experts with one row.
        let case = CASES.iter().find(|c| c.id == 13).unwrap();
        let thirteen = group_list(case, Split::Real);
        assert_eq!(thirteen.len(), 128);
        assert_eq!(thirteen.iter().filter(|&&v| v == 1).count(), 63);
        assert_eq!(thirteen.iter().sum::<i64>(), 63);
        assert_eq!(thirteen[126], 0);
        assert_eq!(thirteen[127], 0);
    }

    /// The real grouping is nothing like the synthetic ones: tuning on `Even`
    /// or `Skew` has already produced reversed conclusions.
    #[test]
    fn the_real_grouping_is_not_one_of_the_stress_splits() {
        let skewed = CASES.iter().filter(|c| c.e > 1 && c.m > c.e);
        for case in skewed {
            let real = counts(case.id).unwrap();
            let synthetic = crate::cases::SYNTHETIC
                .iter()
                .any(|&split| crate::cases::counts(case, split) == real);
            if case.id == 1 || case.id == 12 {
                assert!(synthetic, "tp{} is uniform, so Even should match", case.id);
            }
        }
    }
}
