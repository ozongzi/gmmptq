//! Named body implementations, and `--pick <point>=<body>`.
//!
//! ## The seam
//!
//! `DESIGN.md`'s milestone 1..N replaces one family at a time with a
//! parameterized emitter, and says in as many words that the first version of
//! such an emitter is the verbatim fragment text. That sentence is the seam
//! this module names: a [`Body`] is **one named implementation of one slot**,
//! it owns its own source, and it decides for itself which shapes it will
//! accept. Two bodies in the same slot are two implementations, not a baseline
//! and a patch against it -- nothing here stores a diff, and no body is
//! defined in terms of another.
//!
//! A slot is identified by the **symbol** the body defines, because that is
//! what the router already reports: `dispatch::route` returns `Routed.kernel`,
//! so "which body does point 11 run" is answered by the table rather than by a
//! second map that could drift from it.
//!
//! ## What picks are checked against
//!
//! `--pick 11=n1-fold` is not a string lookup. It resolves point 11 through
//! the cascade to a slot, requires the named body to be registered for *that*
//! slot, and then evaluates the body's own predicate against the point's
//! shape, naming the conjunct that fails. An unknown point, an unknown body
//! name, a body belonging to another slot, a predicate that rejects the shape,
//! or two picks for one point are each a hard error before any byte is
//! written -- never a silently ignored string.
//!
//! ## Deliberately *not* here
//!
//! No `role`-scoped overlap or coverage check. `DESIGN.md`'s dispatch is a
//! single ordered cascade in which **position is the priority** and overlap is
//! resolved by order -- `plan --what-if` exists to report what inserting a
//! gate re-routes. A second disambiguation mechanism living beside gate order
//! would mean two answers to "which gate wins", so bodies compete *inside* one
//! gate's slot and the gate's own predicate decides whether that slot is
//! reached at all.

use std::path::Path;

use crate::cascade::{self, Expr, Op, Q, R};
use crate::dispatch::{self, Env, Shape};

/// Where the standalone body sources live, next to `gen/chunks/`.
pub const BODIES_DIR: &str = "bodies";

/// One body slot: a symbol the cascade can route to, and the verbatim fragment
/// that holds the implementation today.
#[derive(Debug)]
pub struct Slot {
    /// The defining symbol, bare -- no namespace, no template arguments. This
    /// is what `Routed.kernel` reduces to.
    pub symbol: &'static str,
    /// The fragment a selected body's text replaces wholesale.
    pub fragment: &'static str,
    pub what: &'static str,
}

/// One named implementation of one slot.
#[derive(Debug)]
pub struct Body {
    pub name: &'static str,
    pub slot: &'static str,
    /// File under `gen/bodies/`. The milestone-0 emitter is "return these
    /// bytes"; a later milestone replaces `emit` with a real parameterization
    /// without anything else in this file changing.
    pub source: &'static str,
    pub what: &'static str,
    /// The default implementation for the slot. Exactly one per slot.
    pub default: bool,
    /// Conditions this body adds on top of its gate's. Each term both prints
    /// as C++ and evaluates against a shape, so a predicate cannot drift from
    /// its own text -- the same rule `DESIGN.md` sets for the cascade.
    pub pred: &'static [Expr],
    /// Why the predicate is what it is.
    pub why: &'static str,
}

pub const SLOTS: &[Slot] = &[
    Slot {
        symbol: "SearchN1FixedLayoutSafe7100000",
        fragment: "1142-n1-fixed-layout-safe.inc",
        what: "N==1: one dot product per token, reduced inside the vector unit",
    },
    Slot {
        symbol: "SearchP10FixedUb7300000",
        fragment: "3242-fixed-ub-outer.inc",
        what: "K==1 outer product with a fixed UB layout (namespace p10_fixed_ub)",
    },
];

/// The fold needs at least one halving level to exist, i.e. `pairs > 64` where
/// `pairs = Up(k,32)/2`, so `k >= 128` is the honest lower bound. Below it the
/// body is still correct -- the `while` simply never runs and it degenerates
/// to the single `WholeReduceSum` -- but it is then not the implementation
/// anyone chose, so it declines the shape rather than quietly becoming its
/// sibling.
const FOLD_PRED: &[Expr] = &[Expr::Cmp(Q::K, Op::Ge, R::Const(128))];

pub const BODIES: &[Body] = &[
    Body {
        name: "n1-wholereduce",
        slot: "SearchN1FixedLayoutSafe7100000",
        source: "n1-wholereduce.inc",
        what: "two WholeReduceSum levels, 64 lanes wide, then a Gather",
        default: true,
        pred: &[],
        why: "no added condition: this is what the slot's gate already admits",
    },
    Body {
        name: "n1-fold",
        slot: "SearchN1FixedLayoutSafe7100000",
        source: "n1-fold.inc",
        what: "aligned halving Add fold down to <=64, then one WholeReduceSum",
        default: false,
        pred: FOLD_PRED,
        why: "needs pairs>64 for a halving level to exist, i.e. k>=128; the \
              split points are rounded up to 8 floats because an unaligned \
              vector operand start faults the unit (ACL 507035)",
    },
    Body {
        name: "outer-reinterpret",
        slot: "SearchP10FixedUb7300000",
        source: "outer-reinterpret.inc",
        what: "the weight scale aliases wsf_ via ReinterpretCast and is cast in place",
        default: true,
        pred: OUTER_PRED,
        why: OUTER_WHY,
    },
    Body {
        name: "outer-wsraw",
        slot: "SearchP10FixedUb7300000",
        source: "outer-wsraw.inc",
        what: "the weight scale gets its own tensor in the slot wsf_ already reserved, \
               and is multiplied in without a separate cast pass",
        default: false,
        pred: OUTER_PRED,
        why: OUTER_WHY,
    },
];

/// Both `p10_fixed_ub` bodies are the K==1 outer product, and both read the
/// weight scale as `float` -- `outer-wsraw` places a `LocalTensor<WScaleT>`
/// directly in the slot and drops the `CastScale` pass, so a bf16 weight scale
/// would be read as float and silently compute nonsense.
///
/// Gate R already requires `k==1` and gate R.1 already requires `!bf`, so
/// today these two terms restate conditions the cascade enforces. They are
/// here because they are conditions of *this implementation*, not of that
/// gate: the day a new gate reaches this slot -- which is exactly what 14
/// November invites -- these are what refuse it instead of miscompiling.
///
/// Note what is deliberately *not* a term: UB. The two bodies advance `at` by
/// the same `nUp * sizeof(float)`, the default merely leaving that slot
/// aliased, so both fit the same budget and gate R.1's `selectedUb <= 188160`
/// covers them equally.
const OUTER_PRED: &[Expr] = &[Expr::Cmp(Q::K, Op::Eq, R::Const(1)), Expr::Flag(Q::NotBf)];
const OUTER_WHY: &str = "the K==1 outer product, reading the weight scale as float: \
                         k==1 because the body has no reduction loop at all, and !bf \
                         because it does not cast the scale before using it";

pub fn slots() -> &'static [Slot] {
    SLOTS
}

pub fn bodies() -> &'static [Body] {
    BODIES
}

/// Every registered body name, for an error message that lists the options.
pub fn body_names() -> Vec<&'static str> {
    bodies().iter().map(|b| b.name).collect()
}

pub fn slot(symbol: &str) -> Option<&'static Slot> {
    SLOTS.iter().find(|s| s.symbol == symbol)
}

pub fn bodies_for(symbol: &str) -> Vec<&'static Body> {
    BODIES.iter().filter(|b| b.slot == symbol).collect()
}

pub fn body(name: &str) -> Option<&'static Body> {
    BODIES.iter().find(|b| b.name == name)
}

/// The default body of a slot, or an error if the registry has none or two.
pub fn default_body(symbol: &str) -> Result<&'static Body, String> {
    let defaults: Vec<&Body> = bodies_for(symbol).into_iter().filter(|b| b.default).collect();
    match defaults.len() {
        1 => Ok(defaults[0]),
        0 => Err(format!("slot {symbol:?} has no default body")),
        n => Err(format!("slot {symbol:?} has {n} default bodies; exactly one may be default")),
    }
}

/// `p10_fixed_ub::SearchP10FixedUb7300000` -> `SearchP10FixedUb7300000`;
/// `SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800>` -> the stem.
pub fn bare_symbol(kernel: &str) -> &str {
    let no_args = kernel.split('<').next().unwrap_or(kernel);
    no_args.rsplit("::").next().unwrap_or(no_args)
}

/// Which slot a point's shape actually reaches, by routing it.
pub fn slot_of_shape(shape: &Shape, env: &Env) -> Result<Option<&'static Slot>, String> {
    let routed = dispatch::route(shape, env)?;
    Ok(slot(bare_symbol(&routed.kernel)))
}

/// One resolved `--pick` entry.
#[derive(Debug)]
pub struct Pick {
    pub point: u32,
    pub body: &'static Body,
    pub slot: &'static Slot,
}

/// Parse and fully validate `11=n1-fold,4=...`.
///
/// Every failure mode is an error carrying what was wrong, because a pick that
/// is quietly dropped is a measurement of the wrong kernel.
pub fn parse_picks(spec: &str, env: &Env) -> Result<Vec<Pick>, String> {
    let mut out: Vec<Pick> = Vec::new();
    for field in spec.split(',') {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let Some((left, name)) = field.split_once('=') else {
            return Err(format!(
                "--pick {field:?}: want <point>=<body>, for example 11=n1-fold"
            ));
        };
        let point: u32 = left
            .trim()
            .parse()
            .map_err(|_| format!("--pick {field:?}: {:?} is not a point number", left.trim()))?;
        let name = name.trim();

        let points = dispatch::points();
        let Some(published) = points.iter().find(|p| p.id == point) else {
            let known: Vec<String> = points.iter().map(|p| p.id.to_string()).collect();
            return Err(format!(
                "--pick {field:?}: no published point {point}; known points are {}",
                known.join(" ")
            ));
        };
        if out.iter().any(|p| p.point == point) {
            return Err(format!("--pick: point {point} picked twice"));
        }

        let Some(found) = body(name) else {
            let known = body_names();
            return Err(format!(
                "--pick {field:?}: no body named {name:?}; registered bodies are {}",
                known.join(" ")
            ));
        };

        // The point must actually reach a slot this generator knows, and it
        // must be *this* body's slot.
        let reached = slot_of_shape(&published.shape, env)?;
        let Some(reached) = reached else {
            let routed = dispatch::route(&published.shape, env)?;
            return Err(format!(
                "--pick {field:?}: point {point} routes to {}, which is not a registered body \
                 slot. Register a Slot for it before picking a body for that point.",
                bare_symbol(&routed.kernel)
            ));
        };
        if reached.symbol != found.slot {
            return Err(format!(
                "--pick {field:?}: body {name:?} implements slot {:?}, but point {point} routes \
                 to {:?}. Picking it would emit a body the point never executes.",
                found.slot, reached.symbol
            ));
        }

        // …and the body must accept the shape.
        if let Some(failing) = first_rejecting_term(found, &published.shape, env) {
            return Err(format!(
                "--pick {field:?}: body {name:?} declines point {point}'s shape \
                 (E{} M{} K{} N{} gt{} {}): the condition {} does not hold.\n  why: {}",
                published.shape.e,
                published.shape.m,
                published.shape.k,
                published.shape.n,
                published.shape.group_list_type,
                if published.shape.bf { "bf16" } else { "fp16" },
                failing,
                found.why
            ));
        }

        out.push(Pick { point, body: found, slot: reached });
    }
    Ok(out)
}

/// The first conjunct of a body's predicate that the shape fails, as C++ text.
pub fn first_rejecting_term(b: &Body, shape: &Shape, env: &Env) -> Option<String> {
    if b.pred.is_empty() {
        return None;
    }
    let nodes = cascade::cascade();
    let ctx = dispatch::Ctx::new(&nodes, shape, env);
    let mut plan = ctx.plan;
    b.pred
        .iter()
        .find(|term| !term.eval(&ctx, &mut plan))
        .map(|term| term.cpp())
}

/// Read a body's source.
pub fn source_of(chunks_dir: &Path, b: &Body) -> Result<Vec<u8>, String> {
    let dir = chunks_dir
        .parent()
        .map(|crate_dir| crate_dir.join(BODIES_DIR))
        .ok_or_else(|| format!("{} has no parent directory", chunks_dir.display()))?;
    let path = dir.join(b.source);
    std::fs::read(&path).map_err(|e| format!("body {:?}: {}: {e}", b.name, path.display()))
}

/// Fragment-name -> replacement text, for the picks given. Fragments with no
/// pick are left alone by the caller, which is what makes an empty pick list
/// change nothing at all.
pub fn overrides(chunks_dir: &Path, picks: &[Pick]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    for pick in picks {
        // Picking the registered default is a no-op, not an error: it lets a
        // sweep name every point explicitly.
        let text = source_of(chunks_dir, pick.body)?;
        let fragment = pick.slot.fragment.to_string();
        if let Some(existing) = out.iter().find(|(name, _)| *name == fragment) {
            if existing.1 != text {
                return Err(format!(
                    "two picks disagree about fragment {fragment}: point {} wants body {:?}",
                    pick.point, pick.body.name
                ));
            }
            continue;
        }
        out.push((fragment, text));
    }
    Ok(out)
}

/// One line per registered body, for the CLI.
pub fn listing() -> String {
    let mut out = String::new();
    for s in slots() {
        out.push_str(&format!("slot {}  ({})\n", s.symbol, s.what));
        out.push_str(&format!("     fragment {}\n", s.fragment));
        // Through `default_body` rather than the flag, so a registry with two
        // defaults is visible here instead of printing two "[default]" tags.
        let default = default_body(s.symbol);
        for b in bodies_for(s.symbol) {
            let tag = match &default {
                Ok(d) if d.name == b.name => "  [default]",
                Ok(_) => "",
                Err(_) => "  [?]",
            };
            out.push_str(&format!("  {:16} {}{}\n", b.name, b.what, tag));
            if !b.pred.is_empty() {
                let terms: Vec<String> = b.pred.iter().map(|t| t.cpp()).collect();
                out.push_str(&format!("  {:16} needs {}\n", "", terms.join(" && ")));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slot names a fragment the manifest lists, and every body names a
    /// source file that exists. A registry entry pointing at nothing would
    /// turn `--pick` into a runtime surprise.
    #[test]
    fn the_registry_points_at_real_files() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        let listed = crate::chunks::read_manifest(&dir).expect("manifest");
        for s in SLOTS {
            assert!(
                listed.iter().any(|n| n == s.fragment),
                "slot {} names fragment {}, which the manifest does not list",
                s.symbol,
                s.fragment
            );
        }
        for b in BODIES {
            assert!(slot(b.slot).is_some(), "body {} names unregistered slot {}", b.name, b.slot);
            source_of(&dir, b).unwrap_or_else(|e| panic!("body {}: {e}", b.name));
        }
    }

    /// Exactly one default per slot, and no two bodies share a name.
    #[test]
    fn each_slot_has_one_default_and_names_are_unique() {
        for s in SLOTS {
            default_body(s.symbol).unwrap_or_else(|e| panic!("{e}"));
        }
        let mut names: Vec<&str> = BODIES.iter().map(|b| b.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "two bodies share a name");
    }

    /// The default body must be the fragment the manifest already holds, byte
    /// for byte. This is what makes "no pick changes nothing" structural
    /// rather than a promise: the default source and the fragment are the same
    /// bytes, so selecting the default and selecting nothing cannot differ.
    #[test]
    fn the_default_body_is_the_fragment_it_replaces() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        for s in SLOTS {
            let d = default_body(s.symbol).expect("default");
            let body_bytes = source_of(&dir, d).expect("body source");
            let fragment = std::fs::read(dir.join(s.fragment)).expect("fragment");
            assert!(
                body_bytes == fragment,
                "default body {:?} differs from fragment {}",
                d.name,
                s.fragment
            );
        }
    }

    /// The slot symbol has to be one the router actually produces, otherwise
    /// no point could ever reach it and `--pick` would reject everything.
    #[test]
    fn every_slot_is_reachable_by_some_published_point() {
        let env = Env::default();
        let reached: Vec<String> = dispatch::points()
            .iter()
            .filter_map(|p| dispatch::route(&p.shape, &env).ok())
            .map(|r| bare_symbol(&r.kernel).to_string())
            .collect();
        for s in SLOTS {
            assert!(
                reached.iter().any(|k| k == s.symbol),
                "slot {} is not reached by any of the fifteen points; reached: {:?}",
                s.symbol,
                reached
            );
        }
    }

    /// Point 10 is the outer slot's point, and both bodies are pickable.
    #[test]
    fn point_ten_resolves_to_the_outer_slot() {
        let env = Env::default();
        let points = dispatch::points();
        let p10 = points.iter().find(|p| p.id == 10).expect("point 10");
        let s = slot_of_shape(&p10.shape, &env).expect("routes").expect("a registered slot");
        assert_eq!(s.symbol, "SearchP10FixedUb7300000");
        for name in ["outer-reinterpret", "outer-wsraw"] {
            let picks = parse_picks(&format!("10={name}"), &env)
                .unwrap_or_else(|e| panic!("10={name} should be pickable: {e}"));
            assert_eq!(picks.len(), 1);
            assert_eq!(picks[0].body.name, name);
        }
    }

    /// The outer bodies' predicate is load-bearing on both of its terms: a
    /// reduction dimension and a bf16 weight scale are each refused, naming
    /// the term that failed rather than the body.
    #[test]
    fn the_outer_predicate_refuses_a_reduction_and_a_bf16_scale() {
        let env = Env::default();
        let points = dispatch::points();
        let p10 = points.iter().find(|p| p.id == 10).expect("point 10");
        for name in ["outer-reinterpret", "outer-wsraw"] {
            let b = body(name).expect(name);
            assert_eq!(first_rejecting_term(b, &p10.shape, &env), None, "{name} on point 10");

            let mut deep = p10.shape.clone();
            deep.k = 256;
            let failing = first_rejecting_term(b, &deep, &env)
                .unwrap_or_else(|| panic!("{name} must decline K=256"));
            assert!(failing.contains("==1"), "{name}: the failing term is named: {failing}");

            let mut wide = p10.shape.clone();
            wide.bf = true;
            let failing = first_rejecting_term(b, &wide, &env)
                .unwrap_or_else(|| panic!("{name} must decline a bf16 weight scale"));
            assert!(failing.contains("bf"), "{name}: the failing term is named: {failing}");
        }
    }

    /// Point 11 is the slot's point, and both bodies are pickable for it.
    #[test]
    fn point_eleven_resolves_to_the_n1_slot() {
        let env = Env::default();
        let points = dispatch::points();
        let p11 = points.iter().find(|p| p.id == 11).expect("point 11");
        let s = slot_of_shape(&p11.shape, &env).expect("routes").expect("a registered slot");
        assert_eq!(s.symbol, "SearchN1FixedLayoutSafe7100000");
        for name in ["n1-wholereduce", "n1-fold"] {
            let picks = parse_picks(&format!("11={name}"), &env)
                .unwrap_or_else(|e| panic!("11={name} should be pickable: {e}"));
            assert_eq!(picks.len(), 1);
            assert_eq!(picks[0].body.name, name);
        }
    }

    /// Each way a pick can be wrong is an error, and the message says which.
    #[test]
    fn a_bad_pick_is_refused_with_the_reason() {
        let env = Env::default();

        let e = parse_picks("11", &env).expect_err("no '='");
        assert!(e.contains("want <point>=<body>"), "{e}");

        let e = parse_picks("99=n1-fold", &env).expect_err("no such point");
        assert!(e.contains("no published point 99"), "{e}");

        let e = parse_picks("11=no-such-body", &env).expect_err("no such body");
        assert!(e.contains("no body named"), "{e}");
        assert!(e.contains("n1-fold"), "the error lists the registered bodies: {e}");

        // Point 7 routes to the longdot slot, which is not registered, so a
        // pick for it must say so rather than emit a body point 7 never runs.
        let e = parse_picks("7=n1-fold", &env).expect_err("wrong slot");
        assert!(
            e.contains("not a registered body slot") || e.contains("routes to"),
            "{e}"
        );

        let e = parse_picks("11=n1-fold,11=n1-wholereduce", &env).expect_err("twice");
        assert!(e.contains("picked twice"), "{e}");
    }

    /// The predicate is load-bearing: a shape the body declines is refused,
    /// and the message names the failing condition rather than the body.
    #[test]
    fn a_predicate_that_rejects_the_shape_is_refused_naming_the_term() {
        let env = Env::default();
        // Point 11's own shape has K=769, so the fold accepts it...
        let points = dispatch::points();
        let p11 = points.iter().find(|p| p.id == 11).expect("point 11");
        assert_eq!(first_rejecting_term(body("n1-fold").unwrap(), &p11.shape, &env), None);
        // ...while a K below the fold's lower bound does not.
        let mut small = p11.shape.clone();
        small.k = 64;
        let failing = first_rejecting_term(body("n1-fold").unwrap(), &small, &env)
            .expect("the fold must decline K=64");
        assert!(failing.contains(">=128"), "the failing term is named: {failing}");
        // The default body has no predicate, so it accepts what the gate does.
        assert_eq!(first_rejecting_term(body("n1-wholereduce").unwrap(), &small, &env), None);
    }

    /// Picking the default, and picking nothing, must produce the same
    /// override set as far as the bytes are concerned.
    #[test]
    fn picking_the_default_overrides_with_the_fragment_bytes() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        let env = Env::default();
        let picks = parse_picks("11=n1-wholereduce", &env).expect("pickable");
        let got = overrides(&dir, &picks).expect("overrides");
        assert_eq!(got.len(), 1);
        let fragment = std::fs::read(dir.join("1142-n1-fixed-layout-safe.inc")).expect("fragment");
        assert!(got[0].1 == fragment, "picking the default changed bytes");
        assert!(overrides(&dir, &[]).expect("none").is_empty());
    }

    /// No body source may carry the point-9 empty kernel, nor the exact-shape
    /// gate that reached it.
    ///
    /// Three of the five submitted lines hold
    /// `if (E==4 && M==513 && K==383 && N==257 && gt==0 && bf) {
    /// Case9EmptyKernel<<<1,...>>>(); return; }` -- one published shape matched
    /// exactly, answered by a kernel that computes nothing. That is the shape
    /// rule 2.8 forbids and rule 2.10 names outright, and rule 3 reviews the
    /// last submission by hand. Bodies are lifted *from those lines*, so this
    /// is the check that a lift never brings it along.
    #[test]
    fn no_body_source_carries_an_empty_kernel_or_a_per_case_shape_gate() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        for b in BODIES {
            let text = String::from_utf8(source_of(&dir, b).expect("source")).expect("utf8");
            assert!(
                !text.contains("Case9EmptyKernel"),
                "body {} carries Case9EmptyKernel",
                b.name
            );
            assert!(
                !text.to_lowercase().contains("emptykernel"),
                "body {} carries an empty kernel",
                b.name
            );
            let findings = crate::audit::scan(&text, 2);
            assert!(
                findings.is_empty(),
                "body {} carries a per-case shape gate: {:?}",
                b.name,
                findings
            );
        }
    }

    /// …and neither may any assembled translation unit, by either profile.
    /// The body sources are only one way it could get in; this is the check on
    /// the artefact that would actually be submitted.
    #[test]
    fn no_emitted_unit_carries_an_empty_kernel_or_a_per_case_shape_gate() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        let env = Env::default();
        let picks = parse_picks("11=n1-fold,10=outer-wsraw", &env).expect("pickable");
        let units: Vec<(&str, Vec<u8>)> = vec![
            ("final", crate::emit::final_profile(&dir).expect("final")),
            ("final+picks", crate::emit::final_with_picks(&dir, &picks).expect("picked")),
            (
                "spin",
                crate::spin::assemble(&dir, &crate::cascade::cascade(), &[], 11, crate::spin::DEFAULT_TICKS, 0).expect("spin"),
            ),
        ];
        for (what, bytes) in units {
            let text = String::from_utf8(bytes).expect("utf8");
            let report = crate::audit::report(&text);
            // A per-case gate is forbidden in every profile: it is a statement
            // about what the code does, not about what the file is for.
            assert!(
                report.findings.is_empty(),
                "{what} carries a per-case gate:\n{}",
                report.render()
            );
            assert_eq!(report.host_to_host_copies, 0, "{what} reads an input tensor on the host");
            // The banned *constructs* are a submission question, so the spin
            // profile is allowed exactly the two that make it a spin profile
            // -- and nothing else, which is what this spells out.
            let allowed: &[&str] =
                if what == "spin" { &["GetSystemCycle", "GmmProbeSpin"] } else { &[] };
            for (needle, n) in &report.banned {
                assert!(
                    allowed.contains(&needle.as_str()),
                    "{what} carries {needle} x{n}, which that profile may not have:\n{}",
                    report.render()
                );
            }
        }
    }

    /// The fold body must really be a different implementation, and must not
    /// have smuggled in a probe construct.
    #[test]
    fn the_fold_body_is_a_different_implementation_and_is_clean() {
        let dir = crate::chunks::chunks_dir().expect("gen/chunks");
        let fold = source_of(&dir, body("n1-fold").unwrap()).expect("fold");
        let base = source_of(&dir, body("n1-wholereduce").unwrap()).expect("base");
        assert!(fold != base, "the two bodies are the same bytes");
        let text = String::from_utf8(fold).expect("utf8");
        assert!(text.contains("Add(f[c],f[c],f[half+c]"), "the fold's halving Add is missing");
        assert_eq!(
            text.matches("WholeReduceSum").count(),
            1,
            "the fold keeps exactly one reduce level"
        );
        for needle in ["GMM_ASK", "GetSystemCycle", "GmmProbeSpin", "GMM_WRONG_ON_PURPOSE"] {
            assert!(!text.contains(needle), "the fold body contains {needle:?}");
        }
    }
}
