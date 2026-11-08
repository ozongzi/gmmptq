//! The judge's scoring formula, in one place.
//!
//!     score = 100 / (1 + ln(t / best) / ln(1.5))
//!
//! clamped to [0, 100]: matching `best` scores 100, and every factor of 1.5
//! slower costs you half of what is left. `store.rs` registers both functions
//! as SQLite UDFs, so the views below and the Rust code above compute the same
//! number by construction.

/// ln(1.5). `f64::ln` is not const, so this is written out and checked by a
/// test rather than recomputed on every call.
pub const LOG15: f64 = 0.405_465_108_108_164_4;

/// Score one case. Zero for a case that did not run or has no recorded best.
///
/// **This differs from the Python it replaces, on purpose.** The formula is
/// only sane for t >= best: its denominator hits zero at exactly t/best = 1/1.5
/// and goes negative below that, so for a result *faster* than the recorded
/// best the Python version raised ZeroDivisionError at the pole and returned 0
/// everywhere past it -- reporting a breakthrough as a total failure. Score is
/// monotonically decreasing in t, so anything at or under `best` is full marks
/// and the formula is only evaluated where it means something.
pub fn case_score(t: f64, best: f64) -> f64 {
    if best <= 0.0 || t <= 0.0 || !t.is_finite() || !best.is_finite() {
        return 0.0;
    }
    if t <= best {
        return 100.0;
    }
    (100.0 / (1.0 + (t / best).ln() / LOG15)).clamp(0.0, 100.0)
}

/// Invert the formula: how many times off the best is a score?
pub fn ratio_from_score(score: f64) -> f64 {
    if score <= 0.0 {
        return f64::INFINITY;
    }
    1.5f64.powf(100.0 / score - 1.0)
}

/// What a total score implies for the cases we are not shown -- the free
/// feedback signal every submission carries. The total is the mean over all
/// `cases`; `visible_scores` must hold every visible one, and there must be
/// at least one case left over for the answer to mean anything.
///
/// How many cases there are and which are visible is not fixed: the judge
/// says so in every ranking reply (`total_testcase_count`, `testcases[].hidden`).
pub fn hidden_mean(visible_scores: &[f64], cases: usize, total_score: f64) -> Option<f64> {
    let shown = visible_scores.len();
    if shown >= cases {
        return None;
    }
    let seen: f64 = visible_scores.iter().sum();
    Some((cases as f64 * total_score - seen) / (cases - shown) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log15_is_exact() {
        assert_eq!(LOG15, 1.5f64.ln());
    }

    #[test]
    fn matching_best_is_full_marks() {
        assert!((case_score(10.0, 10.0) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn one_point_five_times_slower_is_half() {
        assert!((case_score(15.0, 10.0) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn beating_the_best_is_full_marks_not_zero() {
        // t/best == 1/1.5 is the pole the Python version crashed on, and
        // everything past it it scored 0 -- a breakthrough shown as a failure.
        assert_eq!(case_score(10.0 / 1.5, 10.0), 100.0);
        assert_eq!(case_score(1.0, 10.0), 100.0);
        assert_eq!(case_score(0.001, 10.0), 100.0);
    }

    #[test]
    fn score_is_monotone_in_time() {
        let best = 11.72;
        let mut last = f64::INFINITY;
        for step in 0..400 {
            let t = best * (0.1 + step as f64 * 0.05);
            let got = case_score(t, best);
            assert!(got <= last + 1e-12, "t={t} gave {got} after {last}");
            assert!((0.0..=100.0).contains(&got), "t={t} gave {got}");
            last = got;
        }
    }

    #[test]
    fn dead_cases_score_zero() {
        assert_eq!(case_score(0.0, 10.0), 0.0);
        assert_eq!(case_score(10.0, 0.0), 0.0);
    }

    #[test]
    fn ratio_inverts_score() {
        for &(t, best) in &[(15.0, 10.0), (22.5, 10.0), (100.0, 3.0)] {
            let got = ratio_from_score(case_score(t, best));
            assert!((got - t / best).abs() < 1e-9, "{t}/{best} -> {got}");
        }
        assert!(ratio_from_score(0.0).is_infinite());
    }

    #[test]
    fn hidden_mean_follows_the_case_count() {
        // 15 cases, five visible at 60, total 50 => the hidden ten average 45.
        let got = hidden_mean(&[60.0; 5], 15, 50.0).unwrap();
        assert!((got - 45.0).abs() < 1e-12, "{got}");
        // 20 cases, five visible at 60, total 50 => the hidden 15 average 46.67.
        let got = hidden_mean(&[60.0; 5], 20, 50.0).unwrap();
        assert!((got - 140.0 / 3.0).abs() < 1e-12, "{got}");
        // Everything visible: nothing hidden to infer.
        assert_eq!(hidden_mean(&[60.0; 20], 20, 50.0), None);
    }
}
