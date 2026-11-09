//! `gen audit <file>`: does this translation unit match the published cases?
//!
//! **This command reports; it does not gate.** Whether a per-case gate is
//! within the rules is the organisers' call, not this program's, and the
//! observed evidence is that the judge has been accepting them. So `gen audit`
//! exits 0 whatever it finds, nothing in the build refuses a body for carrying
//! an identity gate, and the findings below are information for whoever
//! decides what to submit.
//!
//! What it is still good for: knowing what is in a kernel before lifting code
//! out of it, and being able to say which points a candidate's advantage
//! actually depends on.
//!
//! A surface grep is not enough for even that, and the first version of this
//! check proved it: `body::no_emitted_unit_carries_an_empty_kernel_or_a_per_case_
//! shape_gate` looked for `M==<m>&&K==<k>&&N==<n>`, in that order, with all
//! three dimensions present. The kernel that prompted this module pins
//! `p.k==1023u && p.n==1025u && p.e==7u` -- no M, different order, different
//! spelling -- and scored **0 of 15** against that pattern while being four
//! dedicated implementations in a trench coat.
//!
//! So the test here is not textual. It is:
//!
//!   **collect the exact-equality constraints a condition places on
//!   E/M/K/N, then ask how many of the published shapes satisfy all of them.
//!   If the answer is exactly one, that condition identifies a case.**
//!
//! That is the definition rule 2.8 actually uses, and it is indifferent to
//! spelling, operand order and which dimensions appear. `N==1`, `K==1`,
//! `M*K<=8` and `n%128==0` are shape *properties* and match many shapes, so
//! they pass -- `DESIGN.md` is explicit that specialising on a property is
//! normal optimisation and that the generator must not sacrifice performance
//! to look more general. `k==1023 && n==1025 && e==7` is a shape *identity*,
//! and no amount of surrounding interval tests makes it less so.
//!
//! A condition that pins nothing, or pins dimensions that several published
//! shapes share, is reported as clean.

use crate::dispatch;

/// One suspicious condition.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub line: usize,
    /// The condition as it appears, squashed to one line.
    pub cond: String,
    /// The exact-equality constraints recovered from it, e.g. `[("k",1023)]`.
    pub pins: Vec<(String, u64)>,
    /// Which published points satisfy every pin.
    pub points: Vec<u32>,
}

/// Dimensions a per-case gate can pin. `cores` is not one: it is an
/// environment property, not a shape, and it appears in legitimate gates.
const DIMS: &[&str] = &["e", "m", "k", "n"];

/// Exact-equality constraints on E/M/K/N in one line of C++.
///
/// Accepts `p.k==1023u`, `K == 383`, `vp.n==2`, `s.e==4` -- any receiver, any
/// spacing, optional `u`/`U` suffix, either case of the dimension letter.
/// Deliberately ignores `<=`, `>=`, `<`, `>` and `!=`: those are intervals,
/// which is what a property-based gate is made of.
pub fn pins_in(line: &str) -> Vec<(String, u64)> {
    let bytes: Vec<char> = line.chars().collect();
    let mut out: Vec<(String, u64)> = Vec::new();
    let mut at = 0usize;
    while at + 1 < bytes.len() {
        if bytes[at] != '=' || bytes[at + 1] != '=' {
            at += 1;
            continue;
        }
        // Walk back over spaces to the identifier that is being compared.
        let mut left = at;
        while left > 0 && bytes[left - 1].is_whitespace() {
            left -= 1;
        }
        let end = left;
        while left > 0 && (bytes[left - 1].is_alphanumeric() || bytes[left - 1] == '_') {
            left -= 1;
        }
        let name: String = bytes[left..end].iter().collect();
        // `p.k`, `vp.n`, `K` -- the dimension is the whole identifier, and a
        // preceding `.` is fine. Reject `kc`, `nb`, `coreNum`, `listType`.
        let is_dim = DIMS.iter().any(|d| name.eq_ignore_ascii_case(d));
        if !is_dim {
            at += 2;
            continue;
        }
        // …and reject `==` that is really `<=`/`>=`/`!=`.
        if left == end || (end > 0 && matches!(bytes[end - 1], '<' | '>' | '!')) {
            at += 2;
            continue;
        }
        // Walk forward over spaces to the literal.
        let mut right = at + 2;
        while right < bytes.len() && bytes[right].is_whitespace() {
            right += 1;
        }
        let start = right;
        while right < bytes.len() && bytes[right].is_ascii_digit() {
            right += 1;
        }
        if right == start {
            at += 2;
            continue;
        }
        let value: String = bytes[start..right].iter().collect();
        if let Ok(v) = value.parse::<u64>() {
            out.push((name.to_lowercase(), v));
        }
        at = right.max(at + 2);
    }
    out.sort();
    out.dedup();
    out
}

/// Published points whose shape satisfies every pin.
pub fn points_matching(pins: &[(String, u64)]) -> Vec<u32> {
    dispatch::points()
        .iter()
        .filter(|p| {
            let s = &p.shape;
            pins.iter().all(|(dim, v)| {
                let got = match dim.as_str() {
                    "e" => u64::from(s.e),
                    "m" => u64::from(s.m),
                    "k" => u64::from(s.k),
                    "n" => u64::from(s.n),
                    _ => return false,
                };
                got == *v
            })
        })
        .map(|p| p.id)
        .collect()
}

/// Every condition in `text` that identifies exactly one published case.
///
/// `min_pins` guards the degenerate end: a lone `N==1` pins one dimension and
/// happens to single out point 11 only because no other published shape has
/// `N==1`. That is a property, not an identity, so two or more pinned
/// dimensions are required before a condition is called out.
pub fn scan(text: &str, min_pins: usize) -> Vec<Finding> {
    let mut out = Vec::new();
    for (no, raw) in text.lines().enumerate() {
        let pins = pins_in(raw);
        if pins.len() < min_pins {
            continue;
        }
        let points = points_matching(&pins);
        if points.len() == 1 {
            out.push(Finding {
                line: no + 1,
                cond: raw.split_whitespace().collect::<Vec<_>>().join(" "),
                pins,
                points,
            });
        }
    }
    out
}

/// Constructs that are forbidden outright, not by shape analysis.
pub const BANNED: &[(&str, &str)] = &[
    ("Case9EmptyKernel", "an empty kernel for one published case (rule 2.10)"),
    ("GetSystemCycle", "the spin-to-a-fixed-time loop"),
    ("GmmProbeSpin", "the spin kernel"),
    ("GMM_PROBE", "the spin/probe switch"),
    ("GMM_ASK", "the hidden-point content oracle"),
    ("GMM_WRONG_ON_PURPOSE", "a deliberately-wrong instrument kernel"),
];

/// A whole-file report.
pub struct Report {
    pub lines: usize,
    pub banned: Vec<(String, usize)>,
    pub findings: Vec<Finding>,
    pub host_to_host_copies: usize,
}

pub fn report(text: &str) -> Report {
    let banned = BANNED
        .iter()
        .map(|(needle, _)| ((*needle).to_string(), text.matches(needle).count()))
        .filter(|(_, n)| *n > 0)
        .collect();
    Report {
        lines: text.lines().count(),
        banned,
        findings: scan(text, 2),
        // Reading an input tensor back to the host is rule 1 / 2.1.
        host_to_host_copies: text.matches("ACL_MEMCPY_DEVICE_TO_HOST").count(),
    }
}

impl Report {
    pub fn clean(&self) -> bool {
        self.banned.is_empty() && self.findings.is_empty() && self.host_to_host_copies == 0
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{} lines\n", self.lines));
        for (needle, n) in &self.banned {
            let why = BANNED
                .iter()
                .find(|(b, _)| b == needle)
                .map(|(_, w)| *w)
                .unwrap_or("");
            out.push_str(&format!("BANNED   {needle} x{n} -- {why}\n"));
        }
        if self.host_to_host_copies > 0 {
            out.push_str(&format!(
                "BANNED   ACL_MEMCPY_DEVICE_TO_HOST x{} -- the host may not read input tensors \
                 (rule 1, rule 2.1)\n",
                self.host_to_host_copies
            ));
        }
        for f in &self.findings {
            let pins: Vec<String> =
                f.pins.iter().map(|(d, v)| format!("{}=={}", d.to_uppercase(), v)).collect();
            out.push_str(&format!(
                "PER-CASE line {}: {} identifies point {} and nothing else (rule 2.8)\n         {}\n",
                f.line,
                pins.join(" && "),
                f.points[0],
                truncate(&f.cond, 150)
            ));
        }
        if self.clean() {
            out.push_str("clean: no banned construct, no condition that identifies a single \
                          published case\n");
        }
        out
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four shapes of per-case gate the incoming kernel actually used, and
    /// which the first version of this check scored 0 of 15 against.
    #[test]
    fn a_per_case_gate_is_found_however_it_is_spelled() {
        // point 14: E7 M3073 K1023 N1025
        let hits = scan("if(bf && p.k==1023u && p.n==1025u && p.e==7u && p.m/p.e<512u){go();}", 2);
        assert_eq!(hits.len(), 1, "the point-14 gate must be found");
        assert_eq!(hits[0].points, vec![14]);

        // point 15, no M constraint at all, fp16
        let hits = scan("if(!bf && p.k==767u && p.n==1025u && p.e==4u){go();}", 2);
        assert_eq!(hits[0].points, vec![15]);

        // point 2, with an environment term mixed in
        let hits = scan("if(bf && p.k==511u && p.n==513u && p.e==4u && p.cores==20u)go();", 2);
        assert_eq!(hits[0].points, vec![2]);

        // point 12
        let hits = scan("if(bf && p.k==256u && p.n==512u && p.e==8u && p.cores==20u)go();", 2);
        assert_eq!(hits[0].points, vec![12]);

        // and the spelling the old pattern expected still works
        let hits = scan("if (E == 4 && M == 513 && K == 383 && N == 257 && gt == 0 && bf) {", 2);
        assert_eq!(hits[0].points, vec![9]);
    }

    /// Property-based predicates must pass. These are the exact examples
    /// `DESIGN.md` names as normal optimisation, plus the real gates from our
    /// own table.
    #[test]
    fn property_predicates_are_not_per_case() {
        for clean in [
            "if(E==1 && M==1 && N==2 && K>=1024 && K<=8192){",
            "if (p.e>=32 && p.m<=256 && p.m<=2*p.e && p.k<=512 && p.n<=512) {",
            "if(bf && p.k<=256u && vt.kc==128u && vt.nb==192u && vt.coreNum<=64u){",
            "if(uint64_t(M)*K<=8u && uint64_t(K)*N<=8u){",
            "if (rowsPerExpert<96 && p.n<=640 && p.kp<=512 &&",
            "if(p.n%128u==0u && p.k%128u==0u){",
            "if(K==2 && N==2){",
        ] {
            let hits = scan(clean, 2);
            assert!(hits.is_empty(), "{clean:?} was called per-case: {:?}", hits);
        }
    }

    /// A single pinned dimension is a property, and `N==1` shows why: points 5
    /// and 11 both have it, so it cannot identify a case even in principle.
    /// The `min_pins` floor is what stops a one-dimension pin from ever being
    /// reported, including the day a published set has only one `N==1` shape.
    #[test]
    fn one_pinned_dimension_is_a_property_not_an_identity() {
        let pins = pins_in("if(bf && N==1 && (y&1u)==0){");
        assert_eq!(pins, vec![("n".to_string(), 1)]);
        assert_eq!(points_matching(&pins), vec![5, 11], "N==1 is a property of two points");
        assert!(scan("if(bf && N==1 && (y&1u)==0){", 2).is_empty(), "and must not be flagged");

        // K==1 likewise: point 10 is the only one today, and it must still pass.
        let pins = pins_in("if(p.k==1u){");
        assert_eq!(points_matching(&pins), vec![10]);
        assert!(scan("if(p.k==1u){", 2).is_empty(), "K==1 is DESIGN.md's own example");
    }

    /// Tiling and environment identifiers are not dimensions.
    #[test]
    fn tiling_and_core_counts_are_not_dimensions() {
        assert!(pins_in("vt.kc==128u && vt.nb==192u && vt.coreNum<=64u").is_empty());
        assert!(pins_in("p.cores==20u && p.listType==1").is_empty());
        assert!(pins_in("selected.coreNum==16u").is_empty());
        // …while the dimension letters are, on any receiver.
        assert_eq!(pins_in("p.k==7u"), vec![("k".to_string(), 7)]);
        assert_eq!(pins_in("vp.n == 2"), vec![("n".to_string(), 2)]);
        assert_eq!(pins_in("s.E==4"), vec![("e".to_string(), 4)]);
    }

    /// `<=`, `>=` and `!=` must never be read as a pin.
    #[test]
    fn inequalities_are_not_pins() {
        for s in ["p.k<=1023u", "p.k>=1023u", "p.k!=1023u", "p.n <= 1025"] {
            assert!(pins_in(s).is_empty(), "{s:?} was read as an equality");
        }
    }

    /// Our own emitted units must pass this stronger check too -- the whole
    /// point is that it is the one we would have failed.
    #[test]
    fn our_own_emits_are_clean_under_the_stronger_check() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        let env = crate::dispatch::Env::default();
        let picks = crate::body::parse_picks("11=n1-fold,10=outer-wsraw", &env).expect("pickable");
        for (what, bytes) in [
            ("final", crate::emit::final_profile(&dir).expect("final")),
            (
                "final+picks",
                crate::emit::final_with_picks(&dir, &picks).expect("picked"),
            ),
        ] {
            let text = String::from_utf8(bytes).expect("utf8");
            let r = report(&text);
            for f in &r.findings {
                eprintln!("note: {what} identity gate at line {} -> point {}", f.line, f.points[0]);
            }
            assert!(r.banned.is_empty(), "{what}: {}", r.render());
        }
    }
}
