//! The `「」` template engine (DESIGN.md, "占位 sigil").
//!
//! Placeholders are written with full-width Chinese brackets because those code
//! points cannot occur in Ascend C source, so nothing ever needs escaping and
//! nothing collides with `{}`, `<>`, `$` or `#`.
//!
//! | form                | meaning                                           |
//! | ------------------- | ------------------------------------------------- |
//! | `「名字」`          | scalar substitution: the value of `名字`          |
//! | `「#名字」`         | fragment inclusion, expanded recursively          |
//! | `「?条件：甲：乙」` | binary choice on the flag `条件`                  |
//!
//! Branches and fragment bodies are themselves expanded, so the forms nest.
//!
//! Every lookup that fails is an error. That is the whole design rule here: an
//! unresolved placeholder must never reach the Ascend C compiler, because a
//! `「」` in a source file is a confusing parse error a thousand lines away
//! from its cause -- and a *silently empty* substitution is worse, since it
//! compiles and computes the wrong thing.
//!
//! Not exercised by `--profile verbatim` (the milestone 0 fragments contain no
//! placeholders at all). It exists and is tested because milestone 1 is the
//! first family to be emitted through it.

// Only `assert_clean` has a caller in milestone 0; the rest is exercised by
// the tests below and called for real by the first emitter.
#![allow(dead_code)]

use std::collections::BTreeMap;

pub const OPEN: char = '「';
pub const CLOSE: char = '」';
/// Leads a fragment inclusion.
pub const FRAGMENT: char = '#';
/// Leads a binary choice.
pub const CHOICE: char = '?';
/// Separates condition, then-branch and else-branch.
pub const SEP: char = '：';
/// A fragment that includes itself is a bug, not a feature.
const MAX_DEPTH: u32 = 64;

/// Names a scalar, a flag and a fragment by name. Everything the engine can
/// resolve lives here; there is no ambient state and no fallback.
#[derive(Default)]
pub struct Env {
    vars: BTreeMap<String, String>,
    flags: BTreeMap<String, bool>,
    frags: BTreeMap<String, String>,
}

impl Env {
    pub fn new() -> Env {
        Env::default()
    }

    /// Bind a scalar. Values are substituted literally and are NOT re-expanded:
    /// a scalar is a number or an identifier the emitter computed, never
    /// template source.
    pub fn var(mut self, name: &str, value: impl std::fmt::Display) -> Env {
        self.vars.insert(name.to_string(), value.to_string());
        self
    }

    /// Bind a flag for `「?条件：甲：乙」`.
    pub fn flag(mut self, name: &str, value: bool) -> Env {
        self.flags.insert(name.to_string(), value);
        self
    }

    /// Bind a fragment for `「#名字」`. Fragment bodies ARE re-expanded.
    pub fn frag(mut self, name: &str, body: impl Into<String>) -> Env {
        self.frags.insert(name.to_string(), body.into());
        self
    }

    pub fn set_var(&mut self, name: &str, value: impl std::fmt::Display) {
        self.vars.insert(name.to_string(), value.to_string());
    }

    pub fn set_flag(&mut self, name: &str, value: bool) {
        self.flags.insert(name.to_string(), value);
    }

    pub fn set_frag(&mut self, name: &str, body: impl Into<String>) {
        self.frags.insert(name.to_string(), body.into());
    }

    fn known(names: impl Iterator<Item = String>) -> String {
        let mut all: Vec<String> = names.collect();
        all.sort();
        if all.is_empty() { "none bound".to_string() } else { all.join(" ") }
    }
}

/// Expand every placeholder in `source`.
pub fn render(source: &str, env: &Env) -> Result<String, String> {
    let mut out = String::with_capacity(source.len());
    expand(source, env, 0, &mut out)?;
    Ok(out)
}

fn expand(source: &str, env: &Env, depth: u32, out: &mut String) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("placeholder nesting deeper than {MAX_DEPTH}: a fragment cycle?"));
    }
    let mut rest = source;
    while let Some(at) = rest.find(OPEN) {
        out.push_str(&rest[..at]);
        let after = &rest[at + OPEN.len_utf8()..];
        let (body, tail) = split_at_close(after)?;
        expand_one(body, env, depth, out)?;
        rest = tail;
    }
    if let Some(stray) = rest.find(CLOSE) {
        return Err(format!("unmatched {CLOSE} at {:?}", snippet(&rest[stray..])));
    }
    out.push_str(rest);
    Ok(())
}

/// Everything up to the `」` that closes the placeholder we just opened,
/// counting nested `「`, and the remainder after it.
fn split_at_close(after_open: &str) -> Result<(&str, &str), String> {
    let mut depth = 1u32;
    for (offset, ch) in after_open.char_indices() {
        if ch == OPEN {
            depth += 1;
        } else if ch == CLOSE {
            depth -= 1;
            if depth == 0 {
                return Ok((&after_open[..offset], &after_open[offset + CLOSE.len_utf8()..]));
            }
        }
    }
    Err(format!("unterminated {OPEN} at {:?}", snippet(after_open)))
}

fn expand_one(body: &str, env: &Env, depth: u32, out: &mut String) -> Result<(), String> {
    let mut chars = body.chars();
    match chars.next() {
        None => Err(format!("empty placeholder {OPEN}{CLOSE}")),
        Some(FRAGMENT) => {
            let name = chars.as_str();
            check_name(name, "fragment")?;
            let Some(frag) = env.frags.get(name) else {
                return Err(format!(
                    "no fragment named {name:?}; bound: {}",
                    Env::known(env.frags.keys().cloned())
                ));
            };
            expand(frag, env, depth + 1, out)
        }
        Some(CHOICE) => {
            let (cond, yes, no) = split_choice(chars.as_str())?;
            check_name(cond, "condition")?;
            let Some(&flag) = env.flags.get(cond) else {
                return Err(format!(
                    "no condition named {cond:?}; bound: {}",
                    Env::known(env.flags.keys().cloned())
                ));
            };
            expand(if flag { yes } else { no }, env, depth + 1, out)
        }
        Some(_) => {
            check_name(body, "name")?;
            let Some(value) = env.vars.get(body) else {
                return Err(format!(
                    "no value named {body:?}; bound: {}",
                    Env::known(env.vars.keys().cloned())
                ));
            };
            out.push_str(value);
            Ok(())
        }
    }
}

/// `条件：甲：乙`, split on the two top-level `：`. The branches may hold
/// nested placeholders, whose own `：` are inside `「」` and so are skipped.
fn split_choice(body: &str) -> Result<(&str, &str, &str), String> {
    let mut depth = 0u32;
    let mut cuts = Vec::new();
    for (offset, ch) in body.char_indices() {
        match ch {
            OPEN => depth += 1,
            CLOSE => depth = depth.saturating_sub(1),
            SEP if depth == 0 => {
                if cuts.len() == 2 {
                    return Err(format!(
                        "choice {OPEN}{CHOICE}{body}{CLOSE} has more than two branches; \
                         a third {SEP} must sit inside a nested placeholder"
                    ));
                }
                cuts.push(offset);
            }
            _ => {}
        }
    }
    if cuts.len() != 2 {
        return Err(format!(
            "choice wants {OPEN}{CHOICE}条件{SEP}甲{SEP}乙{CLOSE}, got {OPEN}{CHOICE}{body}{CLOSE}"
        ));
    }
    let sep = SEP.len_utf8();
    Ok((
        &body[..cuts[0]],
        &body[cuts[0] + sep..cuts[1]],
        &body[cuts[1] + sep..],
    ))
}

/// Names are identifier-ish on purpose: a mistyped sigil then fails here with
/// a readable message instead of being looked up as a 40-character key.
fn check_name(name: &str, what: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("empty {what} in a placeholder"));
    }
    if let Some(bad) = name.chars().find(|c| !(c.is_alphanumeric() || *c == '_' || *c == '.')) {
        return Err(format!("{what} {name:?} contains {bad:?}; want [A-Za-z0-9_.]"));
    }
    Ok(())
}

fn snippet(text: &str) -> String {
    text.chars().take(24).collect()
}

/// Reject any `「` or `」` left in finished output. The emitters call this on
/// the whole translation unit before it is written, because a placeholder that
/// survives into a .asc file becomes a compiler error far from its cause.
pub fn assert_clean(text: &str) -> Result<(), String> {
    for (no, line) in text.lines().enumerate() {
        if line.contains(OPEN) || line.contains(CLOSE) {
            return Err(format!(
                "line {}: a placeholder survived into the output: {:?}",
                no + 1,
                line.trim()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar() {
        let env = Env::new().var("BM", 224u32).var("ScaleT", "bfloat16_t");
        assert_eq!(
            render("CubeFullKBody<「ScaleT」,「BM」>", &env).unwrap(),
            "CubeFullKBody<bfloat16_t,224>"
        );
    }

    #[test]
    fn fragment_is_recursive() {
        let env = Env::new()
            .frag("tail", "Barrier<「event」>();")
            .frag("body", "step();\n「#tail」")
            .var("event", "HardEvent::MTE3_S");
        assert_eq!(
            render("「#body」\n", &env).unwrap(),
            "step();\nBarrier<HardEvent::MTE3_S>();\n"
        );
    }

    #[test]
    fn choice_picks_a_branch() {
        let yes = Env::new().flag("bf", true).var("n", 2u32);
        let no = Env::new().flag("bf", false).var("n", 2u32);
        let src = "using Out = 「?bf：bfloat16_t：half」; // 「n」";
        assert_eq!(render(src, &yes).unwrap(), "using Out = bfloat16_t; // 2");
        assert_eq!(render(src, &no).unwrap(), "using Out = half; // 2");
    }

    #[test]
    fn choice_branches_may_be_empty_and_may_nest() {
        let env = Env::new()
            .flag("pf", true)
            .flag("inner", false)
            .var("depth", 4u32);
        assert_eq!(render("a「?pf：：b」c", &env).unwrap(), "ac");
        assert_eq!(
            render("「?pf：「?inner：x：y「depth」」：z」", &env).unwrap(),
            "y4"
        );
    }

    #[test]
    fn a_colon_inside_a_nested_placeholder_is_not_a_separator() {
        let env = Env::new().flag("bf", true).var("ns", "AscendC");
        // The nested scalar contributes no top-level separator, so the choice
        // still has exactly two branches.
        assert_eq!(render("「?bf：「ns」::x：y」", &env).unwrap(), "AscendC::x");
    }

    #[test]
    fn unknown_names_are_hard_errors() {
        let env = Env::new().var("a", 1u32).flag("f", true).frag("g", "x");
        for src in ["「b」", "「#h」", "「?nope：x：y」"] {
            let error = render(src, &env).unwrap_err();
            assert!(error.contains("no "), "{src} -> {error}");
        }
    }

    #[test]
    fn malformed_placeholders_are_hard_errors() {
        let env = Env::new().var("a", 1u32).flag("f", true);
        for src in [
            "「a",            // unterminated
            "a」",            // stray close
            "「」",           // empty
            "「?f：only」",   // one branch
            "「?f：a：b：c」", // three
            "「a b」",        // not a name
            "「#」",          // empty fragment name
        ] {
            assert!(render(src, &env).is_err(), "{src} should not render");
        }
    }

    #[test]
    fn a_cycle_is_caught_rather_than_hanging() {
        let env = Env::new().frag("loop", "「#loop」");
        let error = render("「#loop」", &env).unwrap_err();
        assert!(error.contains("nesting deeper"), "{error}");
    }

    #[test]
    fn plain_source_passes_through_untouched() {
        // The milestone 0 fragments are exactly this case: no sigils, so the
        // engine must be the identity, byte for byte, including line endings.
        let src = "template <bool BF>\n__aicore__ inline void f() { x<<<1,nullptr>>>(); }\n\n";
        assert_eq!(render(src, &Env::new()).unwrap(), src);
    }

    #[test]
    fn assert_clean_catches_leaks() {
        assert!(assert_clean("int x = 1;\n").is_ok());
        let leak = assert_clean("int x = 1;\nint y = 「BN」;\n").unwrap_err();
        assert!(leak.contains("line 2"), "{leak}");
    }
}
