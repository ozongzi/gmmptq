//! The `run_kernel` dispatch cascade **as data**.
//!
//! `kernel/kernel.asc:5814..6121` is a flat `if … return` chain. Until now it
//! existed twice: once as C++ in `gen/chunks/5814-run-kernel.inc`, once as Rust
//! control flow in `dispatch.rs`. Two copies of a 26-gate cascade drift, and
//! the drift is invisible until a test point reroutes on the device.
//!
//! So the gates live here, once. Each conjunct is an [`Expr`] — a structured
//! comparison over a named [`Q`]uantity — and an `Expr` knows **both** how to
//! print itself as C++ and how to evaluate itself against a shape. `plan`
//! evaluates this table (via `dispatch::route`); `emit --profile final` prints
//! it. There is no second string to keep in step.
//!
//! What resisted becoming data is marked `Opaque` (pointer-alignment tests,
//! whose C++ text and Rust evaluator *are* two strings) or lives in a
//! [`Node::Fixed`] block (validation, the `Plan` construction, gate Z). For the
//! `Opaque` ones the mechanical cross-check in `tests` compares the integer
//! literals appearing in the C++ against the literals the evaluator declares.

// Several accessors here exist for the anti-drift tests and for `--what-if`;
// the binary itself only ever calls `cascade`, `render` and `check_packing`.
#![allow(dead_code)]

use crate::dispatch::{Ctx, Plan};

// ---------------------------------------------------------------------------
// quantities
// ---------------------------------------------------------------------------

/// A named quantity the cascade tests. One variant = one C++ spelling + one
/// evaluator, so a predicate term cannot say one thing and mean another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Q {
    // the int64 host locals derived from the tensor shapes (5839)
    M,
    K,
    N,
    E,
    Gt,
    // the uint64 product terms
    MN,
    MK,
    KN,
    M4,
    NBytes,
    MN2,
    EKN,
    ENBytes,
    // core budget
    AvailCores,
    Cores2,
    // booleans the preamble derives
    Bf,
    NotBf,
    Sow,
    TinyCacheAligned,
    PackWeights,
    Scalar,
    // Plan fields (only meaningful after the `Plan p` construction at 6006)
    PM,
    PK,
    PN,
    PE,
    PE2,
    Kp,
    Wstride,
    PCores,
    RowsPerExpert,
    KMod128,
    NMod128,
    NMod32,
    /// Gate Y's L1 budget, spelled as the source spells it.
    Kp512,
    /// The packed-224 L1 budget.
    Kp480,
    /// `packWeights`'s L1 budget, whose coefficient depends on rowsPerExpert.
    PackL1,
    // gate R's locals
    SelectedPitch8,
    SelectedUb,
    /// `min(p.cores, GetCoreNumAic())`. Never emitted — it is computed inside
    /// `mix11::Launch` (3001) — but the X-arm packers encode it.
    ClampedCores,
}

impl Q {
    /// The C++ text. This is what `emit` prints.
    pub fn cpp(self) -> &'static str {
        match self {
            Q::M => "M",
            Q::K => "K",
            Q::N => "N",
            Q::E => "E",
            Q::Gt => "groupListType",
            Q::MN => "uint64_t(M)*uint64_t(N)",
            Q::MK => "uint64_t(M)*uint64_t(K)",
            Q::KN => "uint64_t(K)*uint64_t(N)",
            Q::M4 => "uint64_t(M)*4u",
            Q::NBytes => "uint64_t(N)*(bf?2u:4u)",
            Q::MN2 => "uint64_t(M)*uint64_t(N)*2u",
            Q::EKN => "uint64_t(E)*uint64_t(K)*uint64_t(N)",
            Q::ENBytes => "uint64_t(E)*uint64_t(N)*(bf?2u:4u)",
            Q::AvailCores => "availableCoreNum",
            Q::Cores2 => "2*uint64_t(availableCoreNum)",
            Q::Bf => "bf",
            Q::NotBf => "!bf",
            Q::Sow => "scalarOutputWindow",
            Q::TinyCacheAligned => "tinyCacheAligned",
            Q::PackWeights => "packWeights",
            Q::Scalar => "scalar",
            Q::PM => "p.m",
            Q::PK => "p.k",
            Q::PN => "p.n",
            Q::PE => "p.e",
            Q::PE2 => "2*p.e",
            Q::Kp => "p.kp",
            Q::Wstride => "p.wstride",
            Q::PCores => "p.cores",
            Q::RowsPerExpert => "rowsPerExpert",
            Q::KMod128 => "p.k%128u",
            Q::NMod128 => "p.n%128u",
            Q::NMod32 => "p.n%32u",
            Q::Kp512 => "uint64_t(256u+2u*128u)*p.kp",
            Q::Kp480 => "uint64_t(480u)*p.kp",
            Q::PackL1 => "uint64_t(rowsPerExpert>=512u?464u:480u)*p.kp",
            Q::SelectedPitch8 => "selectedPitch/8u",
            Q::SelectedUb => "selectedUb",
            Q::ClampedCores => "std::min(p.cores,GetCoreNumAic())",
        }
    }

    /// The integer literals that appear in this quantity's **evaluator**,
    /// declared by hand so `literals_agree_with_the_cpp_text` can compare them
    /// against the literals parsed out of `cpp()`.
    pub fn consts(self) -> &'static [u64] {
        match self {
            Q::M4 => &[4],
            Q::NBytes | Q::ENBytes => &[2, 4],
            Q::MN2 | Q::Cores2 | Q::PE2 => &[2],
            Q::KMod128 | Q::NMod128 => &[128],
            Q::NMod32 => &[32],
            Q::Kp512 => &[256, 2, 128],
            Q::Kp480 => &[480],
            Q::PackL1 => &[512, 464, 480],
            Q::SelectedPitch8 => &[8],
            _ => &[],
        }
    }

    pub fn value(self, c: &Ctx) -> u64 {
        let s = &c.shape;
        let p = &c.plan;
        let bytes = if s.bf { 2u64 } else { 4 };
        match self {
            Q::M => u64::from(s.m),
            Q::K => u64::from(s.k),
            Q::N => u64::from(s.n),
            Q::E => u64::from(s.e),
            Q::Gt => u64::from(s.group_list_type),
            Q::MN => u64::from(s.m) * u64::from(s.n),
            Q::MK => u64::from(s.m) * u64::from(s.k),
            Q::KN => u64::from(s.k) * u64::from(s.n),
            Q::M4 => u64::from(s.m) * 4,
            Q::NBytes => u64::from(s.n) * bytes,
            Q::MN2 => u64::from(s.m) * u64::from(s.n) * 2,
            Q::EKN => u64::from(s.e) * u64::from(s.k) * u64::from(s.n),
            Q::ENBytes => u64::from(s.e) * u64::from(s.n) * bytes,
            Q::AvailCores => u64::from(c.env.available_core_num),
            Q::Cores2 => 2 * u64::from(c.env.available_core_num),
            Q::Bf => u64::from(s.bf),
            Q::NotBf => u64::from(!s.bf),
            Q::Sow => u64::from(c.scalar_output_window),
            Q::TinyCacheAligned => u64::from(c.tiny_cache_aligned),
            Q::PackWeights => u64::from(c.pack_weights),
            Q::Scalar => u64::from(c.scalar),
            Q::PM => u64::from(p.m),
            Q::PK => u64::from(p.k),
            Q::PN => u64::from(p.n),
            Q::PE => u64::from(p.e),
            Q::PE2 => 2 * u64::from(p.e),
            Q::Kp => u64::from(p.kp),
            Q::Wstride => u64::from(p.wstride),
            Q::PCores => u64::from(p.cores),
            Q::RowsPerExpert => u64::from(p.m / p.e),
            Q::KMod128 => u64::from(p.k % 128),
            Q::NMod128 => u64::from(p.n % 128),
            Q::NMod32 => u64::from(p.n % 32),
            Q::Kp512 => u64::from(256u32 + 2 * 128) * u64::from(p.kp),
            Q::Kp480 => 480 * u64::from(p.kp),
            Q::PackL1 => {
                let coeff: u64 = if p.m / p.e >= 512 { 464 } else { 480 };
                coeff * u64::from(p.kp)
            }
            Q::SelectedPitch8 => u64::from(c.selected_pitch / 8),
            Q::SelectedUb => c.selected_ub,
            Q::ClampedCores => u64::from(p.cores.min(c.env.physical_aic)),
        }
    }

    /// True for the quantities that read a device pointer value rather than a
    /// shape, so `plan` can keep saying which decisions were pointer-dependent.
    pub fn dep(self) -> Dep {
        match self {
            Q::Sow | Q::TinyCacheAligned | Q::Scalar => Dep::Ptr,
            Q::AvailCores | Q::Cores2 | Q::PCores | Q::ClampedCores => Dep::Cores,
            _ => Dep::Shape,
        }
    }
}

/// What a decision reads besides the shape.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dep {
    Shape,
    Ptr,
    Cores,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Le,
    Lt,
    Ge,
    Gr,
    Eq,
    Ne,
}

impl Op {
    fn cpp(self) -> &'static str {
        match self {
            Op::Le => "<=",
            Op::Lt => "<",
            Op::Ge => ">=",
            Op::Gr => ">",
            Op::Eq => "==",
            Op::Ne => "!=",
        }
    }
    fn holds(self, a: u64, b: u64) -> bool {
        match self {
            Op::Le => a <= b,
            Op::Lt => a < b,
            Op::Ge => a >= b,
            Op::Gr => a > b,
            Op::Eq => a == b,
            Op::Ne => a != b,
        }
    }
}

/// The right-hand side of a comparison.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum R {
    Const(u64),
    /// A named constant from the kernel's preamble, e.g. `L1_BYTES`.
    Named(&'static str, u64),
    Qty(Q),
}

impl R {
    fn cpp(self) -> String {
        match self {
            R::Const(v) => format!("{v}u"),
            R::Named(name, _) => name.to_string(),
            R::Qty(q) => q.cpp().to_string(),
        }
    }
    fn value(self, c: &Ctx) -> u64 {
        match self {
            R::Const(v) | R::Named(_, v) => v,
            R::Qty(q) => q.value(c),
        }
    }
    fn consts(self) -> Vec<u64> {
        match self {
            R::Const(v) => vec![v],
            // A named constant prints its name, so there is no literal to
            // compare against. `named_constants_match_the_kernel` checks the
            // value instead.
            R::Named(..) => Vec::new(),
            R::Qty(q) => q.consts().to_vec(),
        }
    }

    /// `(name, value)` for the named constants a predicate refers to.
    pub fn named(self) -> Option<(&'static str, u64)> {
        match self {
            R::Named(name, value) => Some((name, value)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigureCubeFullK, as a term
// ---------------------------------------------------------------------------

/// A `ConfigureCubeFullK<BM,BN,BK,RV,Prefetch,SingleL0>` call.
///
/// `nested` distinguishes the two spellings the source uses. T/U/W/Y write
///
/// ```text
/// if (shape terms) { Plan selected=p; if (Configure(selected)) { launch } }
/// ```
///
/// while V.1/V.2/X put the call inside the condition. The two are *logically*
/// the same conjunction — nothing between the shape terms and the call has an
/// effect, and a failed call falls through to the next gate either way — so
/// both are one `Expr` here and the renderer re-spells whichever the source
/// used. `target` matters though: X passes `p` **by reference** (6097) while
/// everyone else configures a copy, which is why it is data and not a default.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cfg {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub rv: u32,
    pub prefetch: bool,
    pub single_l0: bool,
    /// `"selected"`, `"packed"` or `"p"`.
    pub target: &'static str,
    /// True when `target` is a fresh `Plan <target>=p;` the gate's block
    /// declares; false when the gate configures an outer object by reference.
    pub copy: bool,
    pub nested: bool,
}

impl Cfg {
    pub fn cpp(&self) -> String {
        format!(
            "ConfigureCubeFullK<{},{},{},{},{},{}>({})",
            self.bm,
            self.bn,
            self.bk,
            self.rv,
            self.prefetch,
            self.single_l0,
            self.target
        )
    }
}

// ---------------------------------------------------------------------------
// terms
// ---------------------------------------------------------------------------

/// A conjunct of a gate's predicate.
#[derive(Clone, Debug)]
pub enum Expr {
    Cmp(Q, Op, R),
    /// A bare boolean quantity: `bf`, `!bf`, `scalarOutputWindow`, …
    Flag(Q),
    /// A term whose C++ text and Rust evaluator really are two strings: the
    /// pointer-alignment tests and the two `||` disjunctions. The constants
    /// cross-check in `tests` is what keeps these two honest.
    Opaque {
        cpp: &'static str,
        dep: Dep,
        consts: &'static [u64],
        eval: fn(&Ctx) -> bool,
    },
    Configure(Cfg),
}

impl Expr {
    pub fn cpp(&self) -> String {
        match self {
            Expr::Cmp(q, op, r) => format!("{}{}{}", q.cpp(), op.cpp(), r.cpp()),
            Expr::Flag(q) => q.cpp().to_string(),
            Expr::Opaque { cpp, .. } => (*cpp).to_string(),
            Expr::Configure(cfg) => cfg.cpp(),
        }
    }

    pub fn dep(&self) -> Dep {
        match self {
            Expr::Cmp(q, _, r) => match (q.dep(), r) {
                (Dep::Shape, R::Qty(rq)) => rq.dep(),
                (dep, _) => dep,
            },
            Expr::Flag(q) => q.dep(),
            Expr::Opaque { dep, .. } => *dep,
            Expr::Configure(_) => Dep::Shape,
        }
    }

    /// The literals the evaluator uses, for the cross-check.
    pub fn consts(&self) -> Vec<u64> {
        match self {
            Expr::Cmp(q, _, r) => {
                let mut all = q.consts().to_vec();
                all.extend(r.consts());
                all
            }
            Expr::Flag(q) => q.consts().to_vec(),
            Expr::Opaque { consts, .. } => consts.to_vec(),
            // The template arguments are printed from the same four fields the
            // evaluator passes to `configure_cube_full_k`, so this is a check
            // that the printer reads the struct and not a frozen string.
            Expr::Configure(cfg) => {
                vec![u64::from(cfg.bm), u64::from(cfg.bn), u64::from(cfg.bk), u64::from(cfg.rv)]
            }
        }
    }

    /// Evaluate. `plan` is the `Plan` a `Configure` term would write.
    pub fn eval(&self, c: &Ctx, plan: &mut Plan) -> bool {
        match self {
            // `--no-l1-guard` models a generator that rewrote the budget
            // arithmetic; it drops exactly gate Y's 6105 check.
            Expr::Cmp(Q::Kp512, _, _) if !c.env.l1_guard => true,
            Expr::Cmp(q, op, r) => op.holds(q.value(c), r.value(c)),
            Expr::Flag(q) => q.value(c) != 0,
            Expr::Opaque { eval, .. } => eval(c),
            Expr::Configure(cfg) => crate::dispatch::configure_cube_full_k(
                plan,
                cfg.bm,
                cfg.bn,
                cfg.bk,
                cfg.rv,
                cfg.prefetch,
                cfg.single_l0,
                c.env.l1_guard,
            ),
        }
    }

    /// The human form `plan` prints: the C++ text with the left-hand side's
    /// value spliced in.
    pub fn show(&self, c: &Ctx) -> String {
        match self {
            Expr::Cmp(q, op, r) => {
                let rhs = match r {
                    R::Const(v) => v.to_string(),
                    R::Named(name, v) => format!("{name}({v})"),
                    R::Qty(rq) => format!("{}({})", rq.cpp(), rq.value(c)),
                };
                format!("{}({}){}{}", q.cpp(), q.value(c), op.cpp(), rhs)
            }
            Expr::Flag(q) => q.cpp().to_string(),
            Expr::Opaque { cpp, .. } => short(cpp),
            Expr::Configure(cfg) => cfg.cpp(),
        }
    }
}

/// Collapse a multi-line opaque C++ term for the one-line `plan` trace.
fn short(cpp: &str) -> String {
    let mut out = String::new();
    for part in cpp.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(part);
    }
    if out.chars().count() > 96 {
        out = out.chars().take(93).collect::<String>() + "...";
    }
    out
}

fn le(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Le, R::Const(v))
}
fn lt(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Lt, R::Const(v))
}
fn ge(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Ge, R::Const(v))
}
fn gr(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Gr, R::Const(v))
}
fn eq(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Eq, R::Const(v))
}
fn ne(q: Q, v: u64) -> Expr {
    Expr::Cmp(q, Op::Ne, R::Const(v))
}
fn cmpq(q: Q, op: Op, r: Q) -> Expr {
    Expr::Cmp(q, op, R::Qty(r))
}

// ---------------------------------------------------------------------------
// bit packing: the capacity model
// ---------------------------------------------------------------------------

/// How a dimension is encoded into its field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Enc {
    /// `x-1`
    Minus1,
    /// `x/128-1`
    Div128Minus1,
    /// `x` as-is (the `gt` bit)
    Plain,
}

impl Enc {
    fn encode(self, value: u64) -> u64 {
        match self {
            Enc::Minus1 => value.saturating_sub(1),
            Enc::Div128Minus1 => (value / 128).saturating_sub(1),
            Enc::Plain => value,
        }
    }
}

/// Where a field's upper bound comes from.
#[derive(Clone, Copy, Debug)]
pub enum Bound {
    /// Read off the gate's own predicate (walking into enclosing gates, then
    /// falling back to `run_kernel`'s shape validation at 5840).
    FromGate,
    /// A bound the gate implies but does not state as a single conjunct — a
    /// joint product bound, or a clamp applied inside the launcher.
    Manual(u64, &'static str),
    /// No bound on the packed quantity at all. This is the CURRENT.md
    /// landmine-1 class: the field is kept in range by something that is not
    /// about the field.
    Incidental(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub struct PackField {
    pub name: &'static str,
    pub shift: u32,
    pub width: u32,
    /// The quantity packed, which is also where the *value* comes from.
    pub q: Q,
    pub enc: Enc,
    pub bound: Bound,
}

/// One `uint32_t dims` layout, named by the launcher that builds it.
#[derive(Clone, Copy, Debug)]
pub struct Packer {
    pub launcher: &'static str,
    /// Line of the packing expression in kernel.asc.
    pub line: u32,
    pub fields: &'static [PackField],
    /// The Rust name of this constant, so `plan --what-if` can print a table
    /// entry that compiles when pasted.
    pub const_name: &'static str,
}

/// `run_kernel`'s shape validation (5840) — the only bound several fields have.
pub fn validation_bound(q: Q) -> Option<u64> {
    match q {
        Q::M | Q::PM => Some(32768),
        Q::K | Q::PK | Q::Kp => Some(8192),
        Q::N | Q::PN | Q::Wstride => Some(8192),
        Q::E | Q::PE => Some(128),
        Q::Gt => Some(1),
        _ => None,
    }
}

/// Which `Plan` quantity a shape quantity is the same number as. The packers
/// read `selected.k` while the gates that bound it may say `K` or `p.k`.
fn alias(q: Q) -> &'static [Q] {
    match q {
        Q::K | Q::PK => &[Q::K, Q::PK],
        Q::N | Q::PN => &[Q::N, Q::PN],
        Q::E | Q::PE => &[Q::E, Q::PE],
        Q::M | Q::PM => &[Q::M, Q::PM],
        Q::Gt => &[Q::Gt],
        Q::PCores | Q::AvailCores => &[Q::PCores, Q::AvailCores],
        Q::ClampedCores => &[Q::ClampedCores],
        _ => &[],
    }
}

/// The tightest upper bound on `q` stated by this chain of gates (outermost
/// first), falling back to the shape validation.
pub fn upper_bound(chain: &[&Gate], q: Q) -> Option<u64> {
    let want = alias(q);
    let mut best: Option<u64> = None;
    let mut take = |v: u64| {
        best = Some(best.map_or(v, |b: u64| b.min(v)));
    };
    for gate in chain {
        for term in &gate.pred {
            if let Expr::Cmp(lhs, op, R::Const(v)) = term {
                let hit = want.contains(lhs);
                // `p.kp<=512` bounds k too: kp = UpH(k,32) >= k.
                let via_kp = q == Q::K || q == Q::PK;
                if hit || (via_kp && *lhs == Q::Kp) {
                    match op {
                        Op::Le | Op::Eq => take(*v),
                        Op::Lt => take(v.saturating_sub(1)),
                        _ => {}
                    }
                }
            }
        }
    }
    best.or_else(|| validation_bound(q))
}

impl PackField {
    /// The largest value this field can hold.
    pub fn field_max(&self) -> u64 {
        (1u64 << self.width) - 1
    }
    /// The largest *encoded* value the gate admits, and why.
    pub fn admits(&self, chain: &[&Gate]) -> Result<u64, &'static str> {
        match self.bound {
            Bound::FromGate => match upper_bound(chain, self.q) {
                Some(v) => Ok(self.enc.encode(v)),
                None => Err("no bound in the gate and none from validation"),
            },
            Bound::Manual(v, _) => Ok(self.enc.encode(v)),
            Bound::Incidental(why) => Err(why),
        }
    }
}

// ---------------------------------------------------------------------------
// gates and nodes
// ---------------------------------------------------------------------------

/// A `const <ty> <name> = <terms joined>;` the cascade derives and later gates
/// read. Kept as terms rather than text so `plan` and `emit` share one source.
#[derive(Clone, Debug)]
pub struct Derived {
    pub name: &'static str,
    pub ty: &'static str,
    pub line: u32,
    pub comment: &'static str,
    pub terms: Vec<Expr>,
    /// `"&&"` or `"||"`.
    pub join: &'static str,
}

impl Derived {
    pub fn eval(&self, c: &Ctx) -> bool {
        let mut scratch = Plan::default();
        if self.join == "||" {
            self.terms.iter().any(|t| t.eval(c, &mut scratch))
        } else {
            self.terms.iter().all(|t| t.eval(c, &mut scratch))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Gate {
    /// The label `gen/CURRENT.md` gives this arm.
    pub label: &'static str,
    /// Provenance: the line in the 6122-line kernel.asc this came from.
    pub line: u32,
    /// C++ comment lines emitted above the `if`, verbatim from the source.
    pub comment: &'static str,
    /// The conjunction. Empty means an unconditional arm (R.3, E'', K.2).
    pub pred: Vec<Expr>,
    /// Booleans the block derives before its nested gates.
    pub derived: Vec<Derived>,
    /// Straight-line C++ at the top of the block (the `dims` packers, gate R's
    /// `selectedPitch`/`selectedUb`, `Plan packed=p;`).
    pub prologue: String,
    /// Nested arms, in source order.
    pub inner: Vec<Gate>,
    /// The block's own launch, emitted after `inner`.
    pub launch: String,
    /// The `dims` layout this arm's launcher builds, if it packs.
    pub pack: Option<Packer>,
}

impl Gate {
    pub fn new(label: &'static str, line: u32) -> Gate {
        Gate {
            label,
            line,
            comment: "",
            pred: Vec::new(),
            derived: Vec::new(),
            prologue: String::new(),
            inner: Vec::new(),
            launch: String::new(),
            pack: None,
        }
    }
    pub fn c(mut self, comment: &'static str) -> Gate {
        self.comment = comment;
        self
    }
    pub fn p(mut self, pred: Vec<Expr>) -> Gate {
        self.pred = pred;
        self
    }
    pub fn pro(mut self, prologue: &str) -> Gate {
        self.prologue = prologue.to_string();
        self
    }
    pub fn l(mut self, launch: &str) -> Gate {
        self.launch = launch.to_string();
        self
    }
    pub fn inner(mut self, inner: Vec<Gate>) -> Gate {
        self.inner = inner;
        self
    }
    pub fn derive(mut self, d: Derived) -> Gate {
        self.derived.push(d);
        self
    }
    pub fn pack(mut self, pack: Packer) -> Gate {
        self.pack = Some(pack);
        self
    }
    /// The `Configure` term, if this gate has one.
    pub fn cfg(&self) -> Option<Cfg> {
        self.pred.iter().find_map(|t| match t {
            Expr::Configure(cfg) => Some(*cfg),
            _ => None,
        })
    }
    /// Every gate in this subtree, outermost first.
    pub fn walk<'a>(&'a self, out: &mut Vec<&'a Gate>) {
        out.push(self);
        for inner in &self.inner {
            inner.walk(out);
        }
    }
}

/// One element of the cascade, in source order.
#[derive(Clone, Debug)]
pub enum Node {
    /// Host code the table does not model: the signature and validation, the
    /// `GeneralVectorDims` and `Plan` constructions, the trailing fallback.
    Fixed {
        label: &'static str,
        line: u32,
        cpp: String,
    },
    Derived(Derived),
    Gate(Gate),
}

// ---------------------------------------------------------------------------
// the packers (a checking model only -- the packing code itself stays in the
// verbatim launcher fragments)
// ---------------------------------------------------------------------------

const PACK_TINY_SHORT: Packer = Packer {
    const_name: "PACK_TINY_SHORT",
    launcher: "TinyShortReduction",
    line: 5849,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 6, q: Q::M, enc: Enc::Minus1,
            bound: Bound::Manual(64, "M*N<=64 bounds M jointly, not as a conjunct") },
        PackField { name: "N-1", shift: 6, width: 6, q: Q::N, enc: Enc::Minus1,
            bound: Bound::Manual(64, "M*N<=64 bounds N jointly") },
        PackField { name: "K-1", shift: 12, width: 3, q: Q::K, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "E-1", shift: 15, width: 4, q: Q::E, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 19, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_EXPERT_ROW: Packer = Packer {
    const_name: "PACK_EXPERT_ROW",
    launcher: "expert_output::SearchTinyExpertOutput6400000",
    line: 5864,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 2, q: Q::M, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "N-1", shift: 2, width: 5, q: Q::N, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "K-1", shift: 7, width: 3, q: Q::K, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "E-1", shift: 10, width: 7, q: Q::E, enc: Enc::Minus1,
            bound: Bound::Manual(4, "E<=M && M<=4") },
        PackField { name: "gt", shift: 17, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_TINY_CAPACITY: Packer = Packer {
    const_name: "PACK_TINY_CAPACITY",
    launcher: "tiny_capacity",
    line: 5874,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 2, q: Q::M, enc: Enc::Minus1,
            bound: Bound::Manual(4, "M*4<=16") },
        PackField { name: "K-1", shift: 2, width: 3, q: Q::K, enc: Enc::Minus1,
            bound: Bound::Manual(8, "M*K<=8 with M>=1") },
        PackField { name: "N-1", shift: 5, width: 2, q: Q::N, enc: Enc::Minus1,
            bound: Bound::Manual(4, "N*(bf?2:4)<=8") },
        PackField { name: "E-1", shift: 7, width: 7, q: Q::E, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 14, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

/// The E-less `rowDims` of gate H (5923). CURRENT.md landmine 6: the decoder
/// (4957) still reads bits 10..16, so this layout is correct only because H
/// requires `E==1`.
const PACK_ROW_DIMS_NO_E: Packer = Packer {
    const_name: "PACK_ROW_DIMS_NO_E",
    launcher: "rowDimsNoE",
    line: 5923,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 2, q: Q::M, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "N-1", shift: 2, width: 5, q: Q::N, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "K-1", shift: 7, width: 3, q: Q::K, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 17, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_ROW_DIMS: Packer = Packer {
    const_name: "PACK_ROW_DIMS",
    launcher: "rowDims",
    line: 5953,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 2, q: Q::M, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "N-1", shift: 2, width: 5, q: Q::N, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "K-1", shift: 7, width: 3, q: Q::K, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "E-1", shift: 10, width: 7, q: Q::E, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 17, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_TINY_CACHE: Packer = Packer {
    const_name: "PACK_TINY_CACHE",
    launcher: "TinyCacheReduction",
    line: 5940,
    fields: &[
        PackField { name: "M-1", shift: 0, width: 4, q: Q::M, enc: Enc::Minus1,
            bound: Bound::Manual(4, "M*4<=64 and M*N<=32; neither is a conjunct on M") },
        PackField { name: "N-1", shift: 4, width: 5, q: Q::N, enc: Enc::Minus1,
            bound: Bound::Manual(32, "M*N<=32 with M>=1") },
        PackField { name: "K-1", shift: 9, width: 3, q: Q::K, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "E-1", shift: 12, width: 3, q: Q::E, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 15, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_MEDIUM_SCALAR_HALF: Packer = Packer {
    const_name: "PACK_MEDIUM_SCALAR_HALF",
    launcher: "LaunchMediumScalarHalf",
    line: 4793,
    fields: &[
        PackField { name: "k-1", shift: 0, width: 9, q: Q::PK, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "n-1", shift: 9, width: 10, q: Q::PN, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 19, width: 4, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 23, width: 5, q: Q::PCores, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 28, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_ALIGNED_MEDIUM_64: Packer = Packer {
    const_name: "PACK_ALIGNED_MEDIUM_64",
    launcher: "LaunchAlignedMedium64",
    line: 4942,
    fields: &[
        PackField { name: "k/128-1", shift: 0, width: 2, q: Q::PK, enc: Enc::Div128Minus1, bound: Bound::FromGate },
        PackField { name: "n/128-1", shift: 2, width: 3, q: Q::PN, enc: Enc::Div128Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 5, width: 4, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 9, width: 5, q: Q::PCores, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 14, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_ACTIVE_PACKED_224: Packer = Packer {
    const_name: "PACK_ACTIVE_PACKED_224",
    launcher: "LaunchActivePacked224",
    line: 4914,
    fields: &[
        PackField { name: "k-1", shift: 0, width: 10, q: Q::PK, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "n-1", shift: 10, width: 11, q: Q::PN, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 21, width: 4, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 25, width: 5, q: Q::PCores, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 30, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_NARROW_SINGLE_K: Packer = Packer {
    const_name: "PACK_NARROW_SINGLE_K",
    launcher: "LaunchNarrowSingleKSelected",
    line: 3666,
    fields: &[
        PackField { name: "k-1", shift: 0, width: 9, q: Q::PK, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "n-1", shift: 9, width: 9, q: Q::PN, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 18, width: 4, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 22, width: 5, q: Q::PCores, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 27, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_MIX11: Packer = Packer {
    const_name: "PACK_MIX11",
    launcher: "mix11::Launch",
    line: 3006,
    fields: &[
        PackField { name: "k-1", shift: 0, width: 9, q: Q::PK, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "n-1", shift: 9, width: 10, q: Q::PN, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 19, width: 7, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 26, width: 5, q: Q::ClampedCores, enc: Enc::Minus1,
            bound: Bound::Manual(32, "mix11::Launch demotes to packed32 when cores>32 (3002)") },
        PackField { name: "gt", shift: 31, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

const PACK_P1_HALF_PAIR: Packer = Packer {
    const_name: "PACK_P1_HALF_PAIR",
    launcher: "p1_half_pair::Launch",
    line: 4215,
    fields: &[
        PackField { name: "k/128-1", shift: 0, width: 3, q: Q::PK, enc: Enc::Div128Minus1,
            bound: Bound::Incidental(
                "gate Y states no upper bound on k; only the two L1 budget checks -- \
                 512*kp<=524032 (6105) and ConfigureCubeFullK's (256+2*128)*kp<=512K (2863) -- \
                 keep it <=896. Rewrite that arithmetic and k=1152 aliases to k=128 \
                 [CURRENT.md landmine 1]") },
        PackField { name: "n/128-1", shift: 3, width: 6, q: Q::PN, enc: Enc::Div128Minus1, bound: Bound::FromGate },
        PackField { name: "e-1", shift: 9, width: 7, q: Q::PE, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "cores-1", shift: 16, width: 5, q: Q::PCores, enc: Enc::Minus1, bound: Bound::FromGate },
        PackField { name: "gt", shift: 21, width: 1, q: Q::Gt, enc: Enc::Plain, bound: Bound::FromGate },
    ],
};

/// Every launcher the table knows how to bit-pack for, by name. `--what-if`
/// resolves `--launcher` against this.
pub fn packers() -> Vec<Packer> {
    vec![
        PACK_MEDIUM_SCALAR_HALF,
        PACK_ALIGNED_MEDIUM_64,
        PACK_ACTIVE_PACKED_224,
        PACK_NARROW_SINGLE_K,
        PACK_MIX11,
        PACK_P1_HALF_PAIR,
        PACK_TINY_SHORT,
        PACK_EXPERT_ROW,
        PACK_TINY_CAPACITY,
        PACK_ROW_DIMS,
        PACK_ROW_DIMS_NO_E,
        PACK_TINY_CACHE,
    ]
}

// ---------------------------------------------------------------------------
// opaque terms
// ---------------------------------------------------------------------------

const PTR_SIX_ALIGNED_64: &str = "((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(weight)|\n\
     reinterpret_cast<uintptr_t>(xScale)|reinterpret_cast<uintptr_t>(weightScale)|\n\
     reinterpret_cast<uintptr_t>(groupList)|reinterpret_cast<uintptr_t>(y))&63u)==0u";

fn t_six_aligned_64() -> Expr {
    Expr::Opaque {
        cpp: PTR_SIX_ALIGNED_64,
        dep: Dep::Ptr,
        consts: &[63, 0],
        eval: |c| c.tiny_cache_aligned,
    }
}

fn t_scales_misaligned() -> Expr {
    Expr::Opaque {
        cpp: "(((reinterpret_cast<uintptr_t>(xScale)&3u)!=0u) ||\n\
              ((reinterpret_cast<uintptr_t>(weightScale)&(bf?1u:3u))!=0u) ||\n\
              ((reinterpret_cast<uintptr_t>(groupList)&7u)!=0u))",
        dep: Dep::Ptr,
        consts: &[3, 0, 1, 3, 0, 7, 0],
        eval: |c| {
            let p = &c.env.ptrs;
            (p.x_scale & 3) != 0
                || (p.w_scale & if c.shape.bf { 1 } else { 3 }) != 0
                || (p.group_list & 7) != 0
        },
    }
}

fn t_y_even() -> Expr {
    Expr::Opaque {
        cpp: "(reinterpret_cast<uintptr_t>(y)&1u)==0u",
        dep: Dep::Ptr,
        consts: &[1, 0],
        eval: |c| (c.env.ptrs.y & 1) == 0,
    }
}

fn t_y_row_aligned() -> Expr {
    Expr::Opaque {
        cpp: "(reinterpret_cast<uintptr_t>(y)&uint64_t(N*2-1))==0u",
        dep: Dep::Ptr,
        consts: &[2, 1, 0],
        eval: |c| (c.env.ptrs.y & (c.shape.n * 2 - 1)) == 0,
    }
}

fn t_k_or_n_unaligned_128() -> Expr {
    Expr::Opaque {
        cpp: "(p.k%128u || p.n%128u)",
        dep: Dep::Shape,
        consts: &[128, 128],
        eval: |c| c.plan.k % 128 != 0 || c.plan.n % 128 != 0,
    }
}

fn t_group_list_batch() -> Expr {
    Expr::Opaque {
        cpp: "(reinterpret_cast<uintptr_t>(groupList)&7u)==0u &&\n\
              (reinterpret_cast<uintptr_t>(groupList)&63u)+uint64_t(E)*8u<=64u",
        dep: Dep::Ptr,
        consts: &[7, 0, 63, 8, 64],
        eval: |c| {
            let gl = c.env.ptrs.group_list;
            (gl & 7) == 0 && u64::from(gl & 63) + u64::from(c.shape.e) * 8 <= 64
        },
    }
}

fn t_y_32b() -> Expr {
    Expr::Opaque {
        cpp: "(uint64_t(y)&31u)==0u",
        dep: Dep::Ptr,
        consts: &[31, 0],
        eval: |c| (c.env.ptrs.y & 31) == 0,
    }
}

// ---------------------------------------------------------------------------
// the cascade
// ---------------------------------------------------------------------------

/// The whole of `extern "C" run_kernel`, in source order.
pub fn cascade() -> Vec<Node> {
    let mut out = Vec::new();

    out.push(Node::Fixed { label: "A", line: 5814, cpp: PREAMBLE.to_string() });

    // --- B: non-natural typed input views -> TinyShortReduction (5845) -----
    out.push(Node::Gate(
        Gate::new("B", 5845)
            .c("// Non-natural typed input views use existing bounded all-DMA staging.")
            .p(vec![le(Q::K, 8), le(Q::E, 16), le(Q::MN, 64), t_scales_misaligned()])
            .pro("const uint32_t dims=uint32_t((M-1)|((N-1)<<6)|((K-1)<<12)|((E-1)<<15)|(groupListType<<19));")
            .l("if(bf)TinyShortReduction<true><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,dims);\n\
                else TinyShortReduction<false><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,dims);\n\
                return;")
            .pack(PACK_TINY_SHORT),
    ));

    // --- C: scalarOutputWindow (5854) ---------------------------------------
    out.push(Node::Derived(Derived {
        name: "scalarOutputWindow",
        ty: "bool",
        line: 5854,
        comment: "",
        join: "&&",
        terms: vec![Expr::Opaque {
            cpp: "((reinterpret_cast<uintptr_t>(y)&127u)+uint64_t(M)*uint64_t(N)*2u)<=32u",
            dep: Dep::Ptr,
            consts: &[127, 2, 32],
            eval: |c| {
                u64::from(c.env.ptrs.y & 127) + u64::from(c.shape.m) * u64::from(c.shape.n) * 2 <= 32
            },
        }],
    }));

    // --- D: 2x2 register microtile, expert-parallel (5858) -------------------
    out.push(Node::Gate(
        Gate::new("D", 5858)
            .c("// Register-capacity algorithms consume runtime tensors and group metadata.\n\
                // Fixed 2x2 register microtile; up to four output rows fit two uint64 words\n\
                // per expert. Complete original GT0 expert-parallel resource domain.")
            .p(vec![
                Expr::Flag(Q::NotBf),
                eq(Q::Gt, 0),
                ge(Q::E, 2),
                cmpq(Q::E, Op::Le, Q::M),
                le(Q::M, 4),
                cmpq(Q::E, Op::Le, Q::Cores2),
                eq(Q::K, 2),
                eq(Q::N, 2),
                Expr::Flag(Q::Sow),
                t_six_aligned_64(),
            ])
            .pro("const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|\n\
                  \x20                             ((E-1)<<10)|(groupListType<<17));")
            .l("expert_output::SearchTinyExpertOutput6400000<float,half><<<uint32_t(E),nullptr,stream>>>(\n\
                \x20   x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                return;")
            .pack(PACK_EXPERT_ROW),
    ));

    // --- E: single register-capacity algorithm (5871), with E' and E'' -------
    out.push(Node::Gate(
        Gate::new("E", 5871)
            .c("// Single register-capacity algorithm, every dimension and grouping stays runtime.")
            .p(vec![
                le(Q::MK, 8),
                le(Q::KN, 8),
                le(Q::M4, 16),
                le(Q::NBytes, 8),
                le(Q::MN2, 16),
                Expr::Flag(Q::Sow),
                t_y_even(),
            ])
            .pro("const uint32_t dims=uint32_t((M-1)|((K-1)<<2)|((N-1)<<5)|((E-1)<<7)|(groupListType<<14));")
            .inner(vec![
                Gate::new("E'", 5876)
                    .c("// Complete resource family; only naturally aligned single-line GL batches use independent owner reads.")
                    .p(vec![ge(Q::E, 2), le(Q::E, 8), t_group_list_batch()])
                    .l("if(bf)tiny_capacity_contract::Select<true>(x,weight,xScale,weightScale,groupList,y,dims,stream);\n\
                        else tiny_capacity_contract::Select<false>(x,weight,xScale,weightScale,groupList,y,dims,stream);\n\
                        return;"),
                Gate::new("E''", 5882)
                    .l("if(bf)tiny_capacity_unrolled::Select<true>(x,weight,xScale,weightScale,groupList,y,dims,stream);\n\
                        else tiny_capacity_unrolled::Select<false>(x,weight,xScale,weightScale,groupList,y,dims,stream);\n\
                        return;"),
            ])
            .pack(PACK_TINY_CAPACITY),
    ));

    // --- F: one row, two interleaved channels (5888) + the K ladder ----------
    let mut f = Gate::new("F", 5888)
        .c("// A single row with two interleaved channels; all true K lengths in the\n\
            // capacity domain, both dtypes/GT, runtime inputs and scales remain dynamic.")
        .p(vec![eq(Q::E, 1), eq(Q::M, 1), eq(Q::N, 2), ge(Q::K, 1024), le(Q::K, 8192)])
        .derive(Derived {
            name: "scalar",
            ty: "bool",
            line: 5890,
            comment: "",
            join: "&&",
            terms: vec![t_y_32b(), Expr::Flag(Q::Sow)],
        });
    let mut ladder = Vec::new();
    for cap in [1024u64, 2048, 4096, 8192] {
        ladder.push(
            Gate::new(
                match cap {
                    1024 => "F.1024",
                    2048 => "F.2048",
                    4096 => "F.4096",
                    _ => "F.8192",
                },
                5891 + (cap.trailing_zeros() - 10) * 5,
            )
            .p(vec![le(Q::K, cap)])
            .l(&format!(
                "if(bf)longdot_capacity::Launch<{cap},true>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);\n\
                 else longdot_capacity::Launch<{cap},false>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);\n\
                 return;"
            )),
        );
    }
    f = f.inner(ladder);
    out.push(Node::Gate(f));

    out.push(Node::Fixed {
        label: "vp",
        line: 5911,
        cpp: "GeneralVectorDims vp{uint32_t(M),uint32_t(K),uint32_t(N),uint32_t(E),uint32_t(groupListType)};"
            .to_string(),
    });

    // --- G: tinyCacheAligned (5914) ------------------------------------------
    out.push(Node::Derived(Derived {
        name: "tinyCacheAligned",
        ty: "bool",
        line: 5914,
        comment: "// Every participating tensor fits one aligned scalar-cache line. The\n\
                  // predicate uses only metadata and addresses, never host tensor contents.",
        join: "&&",
        terms: vec![t_six_aligned_64()],
    }));

    // --- H (5919) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("H", 5919)
            .c("// E1 validgroupmetadata implies owner0; retain the boundedcompleteinputfamily.")
            .p(vec![
                Expr::Flag(Q::Sow),
                eq(Q::E, 1),
                le(Q::M, 4),
                cmpq(Q::M, Op::Le, Q::Cores2),
                le(Q::N, 32),
                le(Q::K, 8),
                le(Q::MN, 32),
                le(Q::MK, 64),
                le(Q::KN, 64),
                le(Q::M4, 64),
                le(Q::NBytes, 64),
                le(Q::N, 2),
                t_y_row_aligned(),
            ])
            .pro("const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|(groupListType<<17));")
            .l("if(bf)SearchTinyE1STDev3400000<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                else SearchTinyE1STDev3400000<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                return;")
            .pack(PACK_ROW_DIMS_NO_E),
    ));

    // --- I (5928) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("I", 5928)
            .p(vec![
                eq(Q::E, 1),
                le(Q::M, 4),
                cmpq(Q::M, Op::Le, Q::Cores2),
                le(Q::N, 32),
                le(Q::K, 8),
                le(Q::MN, 32),
                le(Q::MK, 64),
                le(Q::KN, 64),
                le(Q::M4, 64),
                le(Q::NBytes, 64),
            ])
            .pro("const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|(groupListType<<17));")
            .l("if(bf)E1FixedUbRowTiny<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                else E1FixedUbRowTiny<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                return;")
            .pack(PACK_ROW_DIMS_NO_E),
    ));

    // --- J (5937) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("J", 5937)
            .p(vec![
                eq(Q::E, 1),
                le(Q::K, 8),
                le(Q::MN, 32),
                le(Q::MK, 64),
                le(Q::EKN, 64),
                le(Q::M4, 64),
                le(Q::ENBytes, 64),
                Expr::Flag(Q::TinyCacheAligned),
            ])
            .pro("const uint32_t tinyCacheDims=uint32_t((M-1)|((N-1)<<4)|((K-1)<<9)|\n\
                  \x20                                    ((E-1)<<12)|(groupListType<<15));")
            .l("if (bf) TinyCacheReduction<true><<<1,nullptr,stream>>>(\n\
                \x20   x,weight,xScale,weightScale,groupList,y,tinyCacheDims);\n\
                else TinyCacheReduction<false><<<1,nullptr,stream>>>(\n\
                \x20   x,weight,xScale,weightScale,groupList,y,tinyCacheDims);\n\
                return;")
            .pack(PACK_TINY_CACHE),
    ));

    // --- K (5949) with K.1 / K.2 --------------------------------------------
    out.push(Node::Gate(
        Gate::new("K", 5949)
            .c("// GT0 expert intervals; last cumulative endpoint follows from completeMcoverage.")
            .p(vec![
                eq(Q::Gt, 0),
                ge(Q::E, 2),
                cmpq(Q::E, Op::Le, Q::M),
                le(Q::M, 4),
                cmpq(Q::E, Op::Le, Q::Cores2),
                le(Q::K, 8),
                le(Q::N, 32),
                le(Q::MN, 32),
            ])
            .pro("const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));")
            .inner(vec![
                Gate::new("K.2x2", 5951)
                    .p(vec![eq(Q::K, 2), eq(Q::N, 2)])
                    .inner(vec![
                        Gate::new("K.1", 5952)
                            .p(vec![
                                Expr::Flag(Q::NotBf),
                                Expr::Flag(Q::TinyCacheAligned),
                                Expr::Flag(Q::Sow),
                            ])
                            .l("expert_output::SearchTinyExpertOutput6400000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                                return;"),
                        Gate::new("K.2", 5956)
                            .l("if(bf)expert_reuse::SearchTinyExpertReuse6300000<bfloat16_t,bfloat16_t><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                                else expert_reuse::SearchTinyExpertReuse6300000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                                return;"),
                    ]),
            ])
            .l("if(bf)SearchTinyExpert3500000<bfloat16_t,bfloat16_t><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                else SearchTinyExpert3500000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                return;")
            .pack(PACK_ROW_DIMS),
    ));

    // --- L (5965) with L.1 ---------------------------------------------------
    out.push(Node::Gate(
        Gate::new("L", 5965)
            .c("// One vector core per short output row; all dimensions remain inputs.")
            .p(vec![le(Q::M, 4), cmpq(Q::M, Op::Le, Q::Cores2), le(Q::N, 32), le(Q::K, 8)])
            .pro("const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));")
            .inner(vec![
                Gate::new("L.1", 5967)
                    .p(vec![Expr::Flag(Q::Bf), eq(Q::N, 1), t_y_even(), Expr::Flag(Q::Sow)])
                    .l("tiny_row_stdev::SearchTinyRowSTDev3400000<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                        return;"),
                Gate::new("L.d", 5970)
                    .l("if (bf) RowParallelTiny<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                        else RowParallelTiny<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                        return;"),
            ])
            .pack(PACK_ROW_DIMS),
    ));

    // --- M (5976) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("M", 5976)
            .c("// Supplement the same short-row resource family when the core budget is lower.")
            .p(vec![le(Q::M, 4), cmpq(Q::M, Op::Gr, Q::Cores2), le(Q::N, 32), le(Q::K, 8)])
            .pro("const uint32_t blocks=uint32_t(std::min<uint64_t>(uint64_t(M),2*uint64_t(availableCoreNum)));\n\
                  const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));")
            .l("if (bf) BudgetStridedRowTiny<bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                else BudgetStridedRowTiny<float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);\n\
                return;")
            .pack(PACK_ROW_DIMS),
    ));

    // --- N (5983) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("N", 5983)
            .p(vec![le(Q::MN, 64), le(Q::K, 8), le(Q::E, 16)])
            .l("if (bf) TinyShortReduction<true><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,((vp.m-1u)|((vp.n-1u)<<6)|((vp.k-1u)<<12)|((vp.e-1u)<<15)|(vp.listType<<19)));\n\
                else TinyShortReduction<false><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,((vp.m-1u)|((vp.n-1u)<<6)|((vp.k-1u)<<12)|((vp.e-1u)<<15)|(vp.listType<<19)));\n\
                return;")
            .pack(PACK_TINY_SHORT),
    ));

    // --- O (5988) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("O", 5988)
            .p(vec![eq(Q::M, 1), le(Q::N, 2)])
            .pro("const uint32_t blocks=uint32_t(std::min<int64_t>(2*availableCoreNum,N));")
            .l("if (bf) IntegerProductLongNarrowDot<bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);\n\
                else IntegerProductLongNarrowDot<float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);\n\
                return;"),
    ));

    // --- P (5994) with P.1 / P.2 --------------------------------------------
    out.push(Node::Gate(
        Gate::new("P", 5994)
            .p(vec![eq(Q::N, 1)])
            .pro("const uint32_t blocks=uint32_t(std::min<int64_t>(2*availableCoreNum,M));")
            .inner(vec![
                Gate::new("P.k800", 5997)
                    .c("// Compact 800-column staging permits a 32-row batch within UB.")
                    .p(vec![le(Q::K, 800)])
                    .l("if (bf) SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp,blocks);\n\
                        else SearchN1FixedLayoutSafe7100000<float,half,32,800><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp,blocks);\n\
                        return;"),
                Gate::new("P.2", 6002)
                    .l("if (bf) RowDot<16,bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);\n\
                        else RowDot<16,float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);\n\
                        return;"),
            ]),
    ));

    // --- Q: Plan p (6006). rowsPerExpert is hoisted here from 6073 so that
    //     T/U/V and W/X/Y can share one spelling of p.m/p.e. ------------------
    out.push(Node::Fixed {
        label: "Q",
        line: 6006,
        cpp: "Plan p{};\n\
              p.m=uint32_t(M);p.k=uint32_t(K);p.n=uint32_t(N);p.e=uint32_t(E);\n\
              p.listType=uint32_t(groupListType);p.kp=UpH(p.k,32);p.wstride=p.n;\n\
              p.cores=uint32_t(availableCoreNum);p.vcores=p.cores*2;\n\
              // Hoisted from 6073: p.e>=1 is validated above, so this is safe here\n\
              // and lets every m/e predicate below share one spelling.\n\
              const uint32_t rowsPerExpert=p.m/p.e;"
            .to_string(),
    });

    // --- R: K==1 outer product (6010) ---------------------------------------
    out.push(Node::Gate(
        Gate::new("R", 6010)
            .p(vec![eq(Q::PK, 1)])
            .pro("// RR32 single output: runtime UB and stride guard, unchanged Mode1 arithmetic.\n\
                  const uint32_t selectedPitch=UpH(p.n,16u),selectedNUp=UpH(p.n,32u);\n\
                  const uint64_t selectedUb=uint64_t(32u)*selectedPitch*6u+\n\
                  \x20   12u*selectedNUp+80u*32u+selectedNUp+32u+32u+32u;")
            .inner(vec![
                Gate::new("R.1", 6015)
                    .p(vec![
                        Expr::Flag(Q::NotBf),
                        ge(Q::PM, 256),
                        ge(Q::PN, 512),
                        le(Q::PN, 1024),
                        le(Q::SelectedPitch8, 255),
                        le(Q::SelectedUb, 188160),
                    ])
                    .l("VTiling selected{p.m,p.k,p.n,p.e,p.listType,\n\
                        \x20   std::min(p.vcores,(p.m+31u)/32u+p.e),32u,selectedPitch};\n\
                        selected.coreNum=std::min(selected.coreNum,16u);\n\
                        p10_fixed_ub::SearchP10FixedUb7300000<<<selected.coreNum,nullptr,stream>>>(\n\
                        \x20   x,weight,xScale,weightScale,groupList,y,selected);\n\
                        return;"),
                Gate::new("R.2", 6025)
                    .c("// FP16 K1 resource family: full-column UB <=179840 bytes.")
                    .p(vec![Expr::Flag(Q::NotBf), ge(Q::PM, 256), ge(Q::PN, 512), le(Q::PN, 1024)])
                    .l("VTiling selected{p.m,p.k,p.n,p.e,p.listType,\n\
                        \x20   std::min(p.vcores,(p.m+19u)/20u+p.e),20u,UpH(p.n,16u)};\n\
                        PublishedOuter<<<selected.coreNum,nullptr,stream>>>(\n\
                        \x20   x,weight,xScale,weightScale,groupList,y,selected);\n\
                        return;"),
                Gate::new("R.3", 6032)
                    .l("VTiling vt{};BuildOuterTiling(p,vt);\n\
                        if (bf) GmmOuterKernel<bfloat16_t,bfloat16_t><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);\n\
                        else GmmOuterKernel<float,half><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);\n\
                        return;"),
            ]),
    ));

    // --- S (6038) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("S", 6038)
            .c("// Many short experts: do not pay Cube padding and cross-core handshakes.")
            .p(vec![
                ge(Q::PE, 32),
                le(Q::PM, 256),
                cmpq(Q::PM, Op::Le, Q::PE2),
                le(Q::PK, 512),
                le(Q::PN, 512),
            ])
            .l("VTiling vt=BuildBalancedOverlapShortGroup(p);\n\
                if (bf) BalancedOverlapShortGroupMatmulKernel<bfloat16_t,bfloat16_t><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);\n\
                else BalancedOverlapShortGroupMatmulKernel<float,half><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);\n\
                return;"),
    ));

    // --- T (6046) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("T", 6046)
            .c("// Medium short experts: original FullK body, measured on recovered groups.\n\
                // Group boundaries and all numerical tensors remain runtime inputs.\n\
                // The trailing (k%128 || n%128) is a reverse predicate: its only job is to\n\
                // let aligned shapes skip T and reach U [CURRENT.md, unexplained predicates].")
            .p(vec![
                Expr::Flag(Q::NotBf),
                le(Q::PCores, 20),
                gr(Q::PE, 1),
                le(Q::PE, 16),
                ge(Q::RowsPerExpert, 96),
                lt(Q::RowsPerExpert, 256),
                gr(Q::PK, 256),
                le(Q::PK, 512),
                gr(Q::PN, 640),
                le(Q::PN, 1024),
                t_k_or_n_unaligned_128(),
                Expr::Configure(Cfg {
                    bm: 192, bn: 128, bk: 96, rv: 128,
                    prefetch: true, single_l0: false,
                    target: "selected", copy: true, nested: true,
                }),
            ])
            .l("LaunchMediumScalarHalf(x,weight,xScale,weightScale,groupList,y,selected,stream);return;")
            .pack(PACK_MEDIUM_SCALAR_HALF),
    ));

    // --- U (6055) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("U", 6055)
            .c("// Confirmed aligned-medium int32 backend; actual group boundaries stay on device.")
            .p(vec![
                le(Q::PCores, 20),
                gr(Q::PE, 1),
                le(Q::PE, 16),
                ge(Q::RowsPerExpert, 64),
                lt(Q::RowsPerExpert, 256),
                ge(Q::PN, 384),
                le(Q::PN, 640),
                eq(Q::NMod128, 0),
                ge(Q::PK, 128),
                le(Q::PK, 512),
                eq(Q::KMod128, 0),
                Expr::Configure(Cfg {
                    bm: 64, bn: 128, bk: 128, rv: 64,
                    prefetch: true, single_l0: false,
                    target: "selected", copy: true, nested: true,
                }),
            ])
            .l("LaunchAlignedMedium64(x,weight,xScale,weightScale,groupList,y,selected,bf,stream);return;")
            .pack(PACK_ALIGNED_MEDIUM_64),
    ));

    // --- packWeights (6252) --------------------------------------------------
    out.push(Node::Derived(Derived {
        name: "packWeights",
        ty: "bool",
        line: 6065,
        comment: "// Preserve original packed-family metadata predicates; allocate only when\n\
                  // one of the retained active pack routes will actually launch.",
        join: "&&",
        terms: vec![
            ge(Q::RowsPerExpert, 256),
            ge(Q::PN, 512),
            ne(Q::NMod32, 0),
            ge(Q::PK, 128),
            Expr::Cmp(Q::PackL1, Op::Le, R::Named("L1_BYTES", 512 * 1024)),
        ],
    }));

    // --- V (6067) with V.1 / V.2 --------------------------------------------
    out.push(Node::Gate(
        Gate::new("V", 6067)
            .p(vec![Expr::Flag(Q::PackWeights)])
            .pro("Plan packed=p;\n\
                  packed.wstride=UpH(p.n,128u);")
            .inner(vec![
                Gate::new("V.1", 6070)
                    .p(vec![
                        Expr::Flag(Q::Bf),
                        le(Q::PCores, 20),
                        le(Q::PE, 16),
                        lt(Q::RowsPerExpert, 512),
                        gr(Q::PK, 512),
                        le(Q::PK, 1024),
                        le(Q::PN, 1280),
                        le(Q::Kp480, 524032),
                        Expr::Configure(Cfg {
                            bm: 224, bn: 128, bk: 128, rv: 64,
                            prefetch: true, single_l0: false,
                            target: "packed", copy: false, nested: false,
                        }),
                    ])
                    .l("Scratch packedWeight(uint64_t(p.e)*p.k*packed.wstride);\n\
                        LaunchActivePacked224<true>(x,weight,xScale,weightScale,groupList,y,\n\
                        \x20   (GM_ADDR)packedWeight.p,packed,stream);return;")
                    .pack(PACK_ACTIVE_PACKED_224),
                // V.2's term order is the source's: ConfigureCubeFullK comes
                // *second*, ahead of the e/cores/k/n bounds (6078..6080).
                Gate::new("V.2", 6078)
                    .p(vec![
                        Expr::Flag(Q::NotBf),
                        Expr::Configure(Cfg {
                            bm: 224, bn: 128, bk: 128, rv: 64,
                            prefetch: true, single_l0: false,
                            target: "packed", copy: false, nested: false,
                        }),
                        le(Q::PE, 16),
                        le(Q::PCores, 20),
                        le(Q::PK, 1024),
                        le(Q::PN, 1280),
                        le(Q::Kp480, 524032),
                    ])
                    .l("Scratch packedWeight(uint64_t(p.e)*p.k*packed.wstride);\n\
                        LaunchActivePacked224<false>(x,weight,xScale,weightScale,groupList,y,\n\
                        \x20   (GM_ADDR)packedWeight.p,packed,stream);return;")
                    .pack(PACK_ACTIVE_PACKED_224),
            ]),
    ));

    // --- W (6088) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("W", 6088)
            .c("// Narrow BF16 resource family: one K chunk, runtime expert boundaries.\n\
                // LaunchNarrowSingleKSelected instantiates bf16 only (3670), so the leading\n\
                // `bf` is a correctness condition, not a performance one.")
            .p(vec![
                Expr::Flag(Q::Bf),
                le(Q::PCores, 20),
                gr(Q::PE, 1),
                le(Q::PE, 16),
                ge(Q::RowsPerExpert, 96),
                lt(Q::RowsPerExpert, 256),
                ge(Q::PN, 192),
                le(Q::PN, 320),
                gr(Q::PK, 256),
                le(Q::PK, 384),
                Expr::Configure(Cfg {
                    bm: 64, bn: 128, bk: 384, rv: 32,
                    prefetch: true, single_l0: true,
                    target: "selected", copy: true, nested: true,
                }),
            ])
            .l("LaunchNarrowSingleKSelected(x,weight,xScale,weightScale,groupList,y,selected,stream);return;")
            .pack(PACK_NARROW_SINGLE_K),
    ));

    // --- X (6096): the one gate that configures `p` BY REFERENCE -------------
    out.push(Node::Gate(
        Gate::new("X", 6096)
            .c("// Resident expert/N panels, one full-K Mmad and bulk B loader when aligned.\n\
                // NOTE: this is the only arm that passes `p` by reference. It is safe only\n\
                // because both of ConfigureCubeFullK's `return false`s precede every store,\n\
                // so a failure leaves the plan Y and Z go on to use untouched.")
            .p(vec![
                lt(Q::RowsPerExpert, 96),
                le(Q::PN, 640),
                le(Q::Kp, 512),
                Expr::Configure(Cfg {
                    bm: 96, bn: 128, bk: 512, rv: 48,
                    prefetch: false, single_l0: true,
                    target: "p", copy: false, nested: false,
                }),
            ])
            .l("mix11::Launch(x,weight,xScale,weightScale,groupList,y,p,bf,stream);\nreturn;")
            .pack(PACK_MIX11),
    ));

    // --- Y (6103) ------------------------------------------------------------
    out.push(Node::Gate(
        Gate::new("Y", 6103)
            .c("// Aligned FP16 full-K resource family: exact scalar 2^-12 half transport.\n\
                // Host uses metadata only; all expert boundaries and scales remain dynamic.")
            .p(vec![
                Expr::Flag(Q::NotBf),
                le(Q::AvailCores, 20),
                ge(Q::RowsPerExpert, 256),
                ge(Q::PN, 512),
                eq(Q::NMod128, 0),
                eq(Q::KMod128, 0),
                cmpq(Q::Wstride, Op::Eq, Q::PN),
                le(Q::Kp512, 524032),
                Expr::Configure(Cfg {
                    bm: 256, bn: 128, bk: 128, rv: 64,
                    prefetch: true, single_l0: false,
                    target: "selected", copy: true, nested: true,
                }),
            ])
            .l("p1_half_pair::Launch(x,weight,xScale,weightScale,groupList,y,selected,stream);return;")
            .pack(PACK_P1_HALF_PAIR),
    ));

    // --- Z: unconditional fallback (6112) ------------------------------------
    out.push(Node::Fixed { label: "Z", line: 6112, cpp: FALLBACK.to_string() });

    out
}

/// Every gate in the cascade, outermost first, in source order.
pub fn gates(nodes: &[Node]) -> Vec<&Gate> {
    let mut out = Vec::new();
    for node in nodes {
        if let Node::Gate(gate) = node {
            gate.walk(&mut out);
        }
    }
    out
}

/// One gate by label, at any nesting depth.
pub fn find<'a>(nodes: &'a [Node], label: &str) -> Option<&'a Gate> {
    gates(nodes).into_iter().find(|g| g.label == label)
}

/// Replace a gate's tile geometry in place, keeping `target`/`copy`/`nested`.
///
/// This is the *other* axis a point can be moved on, and it is not the body
/// axis. Point 9 is the case that made the difference obvious: the submitted
/// line that runs it 26% faster has the identical gate predicate and the
/// identical `CubeFullKUnitBody` text, and differs only in
/// `ConfigureCubeFullK<64,128,384,32,true,true>` versus
/// `<96,128,384,48,false,true>`. Registering that as a `body::Body` would have
/// put a parameter change in a slot meant for implementations, so tiles get
/// their own override -- which is also what `DESIGN.md`'s M1 ("cube 的 tile
/// 几何") is about.
///
/// A gate with no `Configure` term is an error rather than a silent no-op: the
/// caller asked to retile something that has no tiles.
pub fn set_tiles(nodes: &mut [Node], label: &str, tiles: Tiles) -> Result<Cfg, String> {
    fn in_gate(gate: &mut Gate, label: &str, tiles: Tiles) -> Option<Result<Cfg, String>> {
        if gate.label == label {
            let found = gate.pred.iter_mut().find_map(|term| match term {
                Expr::Configure(cfg) => Some(cfg),
                _ => None,
            });
            return Some(match found {
                Some(cfg) => {
                    let before = *cfg;
                    cfg.bm = tiles.bm;
                    cfg.bn = tiles.bn;
                    cfg.bk = tiles.bk;
                    cfg.rv = tiles.rv;
                    cfg.prefetch = tiles.prefetch;
                    cfg.single_l0 = tiles.single_l0;
                    Ok(before)
                }
                None => Err(format!(
                    "gate {label:?} has no Configure term, so it has no tile geometry to set"
                )),
            });
        }
        gate.inner.iter_mut().find_map(|g| in_gate(g, label, tiles))
    }
    for node in nodes.iter_mut() {
        if let Node::Gate(gate) = node {
            if let Some(result) = in_gate(gate, label, tiles) {
                return result;
            }
        }
    }
    let known: Vec<&str> = gates(nodes).iter().map(|g| g.label).collect();
    Err(format!(
        "no gate labelled {label:?}; the table's gates are {}",
        known.join(" ")
    ))
}

/// The six numbers `ConfigureCubeFullK` is instantiated with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tiles {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub rv: u32,
    pub prefetch: bool,
    pub single_l0: bool,
}

impl Tiles {
    /// `BM,BN,BK,RV[,prefetch,singleL0]`, the spelling `plan --what-if --tiles`
    /// already takes.
    pub fn parse(spec: &str) -> Result<Tiles, String> {
        let f: Vec<&str> = spec.split(',').map(str::trim).collect();
        if f.len() != 4 && f.len() != 6 {
            return Err(format!(
                "--tiles {spec:?}: want BM,BN,BK,RV or BM,BN,BK,RV,prefetch,singleL0"
            ));
        }
        let num = |at: usize, name: &str| -> Result<u32, String> {
            f[at]
                .parse::<u32>()
                .map_err(|_| format!("--tiles {spec:?}: {name} {:?} is not a number", f[at]))
                .and_then(|v| {
                    if v == 0 {
                        Err(format!("--tiles {spec:?}: {name} must not be zero"))
                    } else {
                        Ok(v)
                    }
                })
        };
        let flag = |at: usize, name: &str| -> Result<bool, String> {
            match f[at] {
                "true" | "1" => Ok(true),
                "false" | "0" => Ok(false),
                other => Err(format!("--tiles {spec:?}: {name} {other:?} is not true or false")),
            }
        };
        Ok(Tiles {
            bm: num(0, "BM")?,
            bn: num(1, "BN")?,
            bk: num(2, "BK")?,
            rv: num(3, "RV")?,
            prefetch: if f.len() == 6 { flag(4, "prefetch")? } else { true },
            single_l0: if f.len() == 6 { flag(5, "singleL0")? } else { false },
        })
    }
}

/// The chain from the outermost gate down to `label`, for bound derivation.
pub fn chain_to<'a>(nodes: &'a [Node], label: &str) -> Option<Vec<&'a Gate>> {
    fn descend<'a>(gate: &'a Gate, label: &str, acc: &mut Vec<&'a Gate>) -> bool {
        acc.push(gate);
        if gate.label == label {
            return true;
        }
        for inner in &gate.inner {
            if descend(inner, label, acc) {
                return true;
            }
        }
        acc.pop();
        false
    }
    for node in nodes {
        if let Node::Gate(gate) = node {
            let mut acc = Vec::new();
            if descend(gate, label, &mut acc) {
                return Some(acc);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// the fixed blocks
// ---------------------------------------------------------------------------

/// The signature, attribute validation, tensor views and shape check (5814..5843).
///
/// `--profile final` deliberately omits the `#if GMM_ASK` probe hook that sits
/// at 5830 in the current file: DESIGN.md makes `final` refuse to emit any
/// probe construct. With `GMM_ASK` at its default 0 the preprocessed output is
/// identical either way, so this is not a behaviour change for a real build.
const PREAMBLE: &str = r#"extern "C" void run_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR weight, const TensorGroupInfo& info_weight,
    GM_ADDR xScale, const TensorGroupInfo& info_xScale,
    GM_ADDR weightScale, const TensorGroupInfo& info_weightScale,
    GM_ADDR groupList, const TensorGroupInfo& info_groupList,
    GM_ADDR y, const TensorGroupInfo& info_y,
    int64_t availableCoreNum, aclrtStream stream,
    int64_t groupListType, const char* outputDtype) {
    using namespace gmmptq;
    if (!outputDtype ||
        (std::strcmp(outputDtype, "float16") && std::strcmp(outputDtype, "bfloat16")) ||
        (groupListType != 0 && groupListType != 1) || availableCoreNum <= 0)
        throw std::invalid_argument("GroupedMatmulPerTokenQuant: invalid attributes/core count");
    bool bf = std::strcmp(outputDtype, "bfloat16") == 0;
    const auto& a = Tensor(info_x, 2, 3);
    const auto& b = Tensor(info_weight, 3, 3);
    const auto& xs = Tensor(info_xScale, 1, 0);
    const auto& ws = Tensor(info_weightScale, 2, bf ? 2 : 0);
    const auto& gl = Tensor(info_groupList, 1, 6);
    const auto& out = Tensor(info_y, 2, bf ? 2 : 1);
    int64_t M = a.shape[0], K = a.shape[1], E = b.shape[0], N = b.shape[2];
    if (M < 1 || M > 32768 || K < 1 || K > 8192 || E < 1 || E > 128 || N < 1 || N > 8192 ||
        b.shape[1] != K || xs.shape[0] != M || ws.shape[0] != E || ws.shape[1] != N ||
        gl.shape[0] != E || out.shape[0] != M || out.shape[1] != N)
        throw std::invalid_argument("GroupedMatmulPerTokenQuant: shape mismatch");"#;

/// Gate Z (6112..6120) plus the closing brace, verbatim.
const FALLBACK: &str = r#"// General legal-metadata fallback: original chunked L1 implementation.
p.wstride=p.n;
LeanPlanCube(p);
const size_t maxTiles=(size_t((p.m+p.m0-1)/p.m0)+p.e)*p.nTiles;
Scratch scratch(maxTiles*p.m0*p.n0*sizeof(int32_t));
try {
    if (bf) CubeMix<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,(GM_ADDR)scratch.p,p);
    else CubeMix<float,half><<<p.cores,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,(GM_ADDR)scratch.p,p);
    Check(aclrtSynchronizeStream(stream));
} catch (...) { aclrtSynchronizeStream(stream);throw; }"#;

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

const INDENT: &str = "    ";
const WRAP: usize = 100;

fn push_block(out: &mut String, text: &str, depth: usize) {
    if text.is_empty() {
        return;
    }
    let pad = INDENT.repeat(depth);
    for line in text.lines() {
        if line.trim().is_empty() {
            out.push('\n');
        } else {
            out.push_str(&pad);
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }
}

/// `a && b && c`, wrapped so no line runs past `WRAP` columns.
fn render_cond(terms: &[Expr], depth: usize) -> String {
    let pad = INDENT.repeat(depth);
    let cont = format!("{pad}   ");
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for (index, term) in terms.iter().enumerate() {
        let cpp = term.cpp();
        // A term that is itself multi-line goes on its own lines.
        if cpp.contains('\n') {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            let mut block = String::new();
            for (no, line) in cpp.lines().enumerate() {
                if no > 0 {
                    block.push('\n');
                    block.push_str(&cont);
                }
                block.push_str(line.trim());
            }
            lines.push(block);
        } else {
            let head = current.len() + cpp.len() + 4;
            if !current.is_empty() && head > WRAP.saturating_sub(cont.len()) {
                lines.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&cpp);
        }
        if index + 1 < terms.len() {
            if current.is_empty() {
                let last = lines.len() - 1;
                lines[last].push_str(" &&");
            } else {
                current.push_str(" &&");
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines.join(&format!("\n{cont}"))
}

fn render_gate(gate: &Gate, out: &mut String, depth: usize) {
    let pad = INDENT.repeat(depth);
    if !gate.comment.is_empty() {
        push_block(out, gate.comment, depth);
    }
    let cfg = gate.cfg();
    let nested_cfg = cfg.filter(|c| c.nested);
    // In the nested spelling the Configure call is not part of the `if`.
    let cond_terms: Vec<Expr> = gate
        .pred
        .iter()
        .filter(|t| !(nested_cfg.is_some() && matches!(t, Expr::Configure(_))))
        .cloned()
        .collect();

    // The marker says whether the arm is unconditional, because an
    // unconditional arm's launch often *starts* with `if (bf)` and a reader --
    // or the parser in `emit::parse` -- must not take that for the gate.
    out.push_str(&pad);
    out.push_str(&format!(
        "// gate {} -- kernel.asc:{}{}\n",
        gate.label,
        gate.line,
        if cond_terms.is_empty() { " (unconditional)" } else { "" }
    ));

    let body_depth;
    if cond_terms.is_empty() {
        body_depth = depth;
    } else {
        out.push_str(&pad);
        out.push_str("if (");
        out.push_str(&render_cond(&cond_terms, depth));
        out.push_str(") {\n");
        body_depth = depth + 1;
    }

    // The prologue comes *before* the nested Configure, not inside it: a gate
    // whose launcher wants a prepared plan (`Plan packed=p; packed.wstride=…`)
    // declares it there, and the Configure call then names it. Every shipped
    // nested-Configure gate has an empty prologue, so this changes nothing for
    // T/U/W/Y.
    push_block(out, &gate.prologue, body_depth);

    let mut inner_depth = body_depth;
    if let Some(cfg) = nested_cfg {
        let bpad = INDENT.repeat(body_depth);
        if cfg.copy {
            out.push_str(&bpad);
            out.push_str(&format!("Plan {}=p;\n", cfg.target));
        }
        out.push_str(&bpad);
        out.push_str(&format!("if ({}) {{\n", cfg.cpp()));
        inner_depth = body_depth + 1;
    }

    for derived in &gate.derived {
        render_derived(derived, out, inner_depth);
    }
    for inner in &gate.inner {
        render_gate(inner, out, inner_depth);
    }
    push_block(out, &gate.launch, inner_depth);

    if nested_cfg.is_some() {
        out.push_str(&INDENT.repeat(body_depth));
        out.push_str("}\n");
    }
    if !cond_terms.is_empty() {
        out.push_str(&pad);
        out.push_str("}\n");
    }
}

fn render_derived(d: &Derived, out: &mut String, depth: usize) {
    let pad = INDENT.repeat(depth);
    if !d.comment.is_empty() {
        push_block(out, d.comment, depth);
    }
    let joined: Vec<String> = d.terms.iter().map(|t| t.cpp()).collect();
    let body = joined.join(&format!(" {} ", d.join));
    let decl = format!("const {} {}={};", d.ty, d.name, body);
    if decl.len() + pad.len() <= WRAP && !decl.contains('\n') {
        out.push_str(&pad);
        out.push_str(&decl);
        out.push('\n');
        return;
    }
    // Too long for one line: break on the joins, at the same wrap column the
    // gate conditions use, so the two read alike.
    out.push_str(&pad);
    out.push_str(&format!("const {} {}=\n", d.ty, d.name));
    let cont = format!("{pad}{INDENT}");
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for (index, term) in joined.iter().enumerate() {
        for (no, piece) in term.lines().enumerate() {
            if no > 0 {
                lines.push(std::mem::take(&mut current));
            }
            if !current.is_empty() && current.len() + piece.len() + cont.len() + 4 > WRAP {
                lines.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(piece.trim());
        }
        if index + 1 < joined.len() {
            current.push(' ');
            current.push_str(d.join);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    for line in &lines {
        out.push_str(&cont);
        out.push_str(line);
        out.push('\n');
    }
    let last = out.len() - 1;
    out.truncate(last);
    out.push_str(";\n");
}

/// Render one gate on its own, for `plan --what-if`. Same code path as the
/// whole cascade, so what the candidate prints is what it would emit.
pub fn render_one(gate: &Gate, out: &mut String, depth: usize) {
    render_gate(gate, out, depth);
}

/// Print the cascade as the body of `extern "C" void run_kernel(...)`.
pub fn render(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Fixed { label, line, cpp } => {
                if *label == "A" {
                    // The signature carries its own indentation.
                    out.push_str(cpp);
                    out.push('\n');
                } else {
                    out.push_str(&format!("{INDENT}// {label} -- kernel.asc:{line}\n"));
                    push_block(&mut out, cpp, 1);
                }
            }
            Node::Derived(d) => render_derived(d, &mut out, 1),
            Node::Gate(gate) => render_gate(gate, &mut out, 1),
        }
    }
    out.push_str("}\n\n");
    out
}

// ---------------------------------------------------------------------------
// the bit-packing capacity gate
// ---------------------------------------------------------------------------

/// Does every packed field of every gate fit, given what that gate admits?
///
/// This is the refusal `emit --profile final` runs before writing anything. A
/// new gate that reuses `LaunchAlignedMedium64` with `k<=1024` would alias
/// `k/128-1` into two bits and compute the wrong answer with no diagnostic;
/// here it is a build error that names the field.
pub fn check_packing(nodes: &[Node]) -> Vec<String> {
    let mut problems = Vec::new();
    for gate in gates(nodes) {
        let Some(packer) = gate.pack else { continue };
        let chain = chain_to(nodes, gate.label).unwrap_or_else(|| vec![gate]);
        for field in packer.fields {
            match field.admits(&chain) {
                Ok(max) => {
                    if max > field.field_max() {
                        problems.push(format!(
                            "gate {} ({}) reuses {} ({}): field {} is {} bits (max {}), but the gate \
                             admits {} up to {} which encodes to {}",
                            gate.label,
                            gate.line,
                            packer.launcher,
                            packer.line,
                            field.name,
                            field.width,
                            field.field_max(),
                            field.q.cpp(),
                            upper_bound(&chain, field.q).map_or("unbounded".into(), |b| b.to_string()),
                            max
                        ));
                    }
                }
                Err(why) => {
                    // An unbounded field is a warning, not a refusal: gate Y
                    // ships like this today and the kernel is correct. Refusing
                    // would mean refusing to emit the current cascade.
                    if !matches!(field.bound, Bound::Incidental(_)) {
                        problems.push(format!(
                            "gate {} ({}) reuses {} ({}): field {} has no usable bound -- {}",
                            gate.label, gate.line, packer.launcher, packer.line, field.name, why
                        ));
                    }
                }
            }
        }
    }
    problems
}

/// The fields a gate's packer builds, for `plan`'s report. Values come from the
/// context; the admitted maxima come from the gate's own predicate.
pub fn pack_fields(nodes: &[Node], label: &str, c: &Ctx) -> Vec<crate::dispatch::Field> {
    let Some(chain) = chain_to(nodes, label) else { return Vec::new() };
    let Some(gate) = chain.last() else { return Vec::new() };
    let Some(packer) = gate.pack else { return Vec::new() };
    packer
        .fields
        .iter()
        .map(|field| crate::dispatch::Field {
            name: field.name,
            shift: field.shift,
            width: field.width,
            value: field.enc.encode(field.q.value(c)),
            gate_max: field.admits(&chain).ok(),
            incidental: match field.bound {
                Bound::Incidental(why) => Some(why),
                _ => None,
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Integer literals in a C++ expression: a maximal digit run that is not
    /// part of an identifier (so `uint64_t` contributes nothing).
    fn literals(cpp: &str) -> Vec<u64> {
        let bytes: Vec<char> = cpp.chars().collect();
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            if bytes[at].is_ascii_digit() {
                let start = at;
                while at < bytes.len() && bytes[at].is_ascii_digit() {
                    at += 1;
                }
                let prev = if start == 0 { ' ' } else { bytes[start - 1] };
                if !(prev.is_alphanumeric() || prev == '_') {
                    if let Ok(v) = bytes[start..at].iter().collect::<String>().parse::<u64>() {
                        out.push(v);
                    }
                }
            } else {
                at += 1;
            }
        }
        out
    }

    #[test]
    fn literals_skips_identifier_digits() {
        assert_eq!(literals("uint64_t(M)*4u"), vec![4]);
        assert_eq!(literals("uint64_t(256u+2u*128u)*p.kp"), vec![256, 2, 128]);
        assert_eq!(literals("bfloat16_t"), Vec::<u64>::new());
    }

    /// The mechanical anti-drift check for the terms whose C++ text and Rust
    /// evaluator are separate: every integer constant one of them uses must
    /// appear in the other.
    #[test]
    fn literals_agree_with_the_cpp_text() {
        let nodes = cascade();
        let mut checked = 0;
        let mut check = |term: &Expr| {
            let cpp = term.cpp();
            let mut want = literals(&cpp);
            let mut got = term.consts();
            want.sort_unstable();
            want.dedup();
            got.sort_unstable();
            got.dedup();
            // `Cmp` prints its rhs from the same u64 it compares against, so
            // only the lhs quantity's own literals can disagree. For `Opaque`
            // the whole expression is at stake.
            assert_eq!(got, want, "term {cpp:?}: declared constants vs C++ literals");
            checked += 1;
        };
        for gate in gates(&nodes) {
            for term in &gate.pred {
                check(term);
            }
            for d in &gate.derived {
                for term in &d.terms {
                    check(term);
                }
            }
        }
        for node in &nodes {
            if let Node::Derived(d) = node {
                for term in &d.terms {
                    check(term);
                }
            }
        }
        assert!(checked > 100, "only {checked} terms checked");
    }

    /// A predicate that prints a named kernel constant must evaluate with that
    /// constant's real value (kernel.asc:10..14).
    #[test]
    fn named_constants_match_the_kernel() {
        let kernel: &[(&str, u64)] = &[
            ("L1_BYTES", 512 * 1024),
            ("L0A_BYTES", 64 * 1024),
            ("L0B_BYTES", 64 * 1024),
            ("L0C_BYTES", 128 * 1024),
            ("UB_BYTES", 192 * 1024),
        ];
        let nodes = cascade();
        let mut seen = 0;
        let mut check = |term: &Expr| {
            if let Expr::Cmp(_, _, r) = term {
                if let Some((name, value)) = r.named() {
                    let want = kernel
                        .iter()
                        .find(|(k, _)| *k == name)
                        .unwrap_or_else(|| panic!("unknown kernel constant {name:?}"));
                    assert_eq!(value, want.1, "{name}");
                    seen += 1;
                }
            }
        };
        for gate in gates(&nodes) {
            gate.pred.iter().for_each(&mut check);
        }
        for node in &nodes {
            if let Node::Derived(d) = node {
                d.terms.iter().for_each(&mut check);
            }
        }
        assert!(seen > 0, "no named constant in the cascade -- did the spelling change?");
    }

    /// A packer must belong to the gate that calls it: if the launch text does
    /// not mention the launcher, the capacity model is checking the wrong
    /// layout, which is precisely the silent-aliasing bug it exists to stop.
    #[test]
    fn each_packer_names_the_launcher_its_gate_calls() {
        let nodes = cascade();
        for gate in gates(&nodes) {
            let Some(packer) = gate.pack else { continue };
            // The row-dims layouts are named for the packing expression, not
            // for a launcher function; their `dims` is built in the prologue.
            if packer.launcher.starts_with("rowDims") || packer.launcher == "tiny_capacity" {
                continue;
            }
            let mut text = gate.launch.clone();
            for inner in &gate.inner {
                let mut all = Vec::new();
                inner.walk(&mut all);
                for g in all {
                    text.push_str(&g.launch);
                }
            }
            let base = packer.launcher.rsplit("::").next().unwrap();
            assert!(
                text.contains(base),
                "gate {} carries packer {} but never calls it: {text:?}",
                gate.label,
                packer.launcher
            );
        }
    }

    /// The cascade as shipped must pass its own capacity gate.
    #[test]
    fn the_current_cascade_packs_within_every_field() {
        let problems = check_packing(&cascade());
        assert!(problems.is_empty(), "{problems:#?}");
    }

    /// Requirement 5, in action: a gate that reuses `LaunchAlignedMedium64`
    /// with a wider `k` must be refused, by name, before anything is written.
    #[test]
    fn a_too_large_shape_is_refused_by_name() {
        let mut nodes = cascade();
        nodes.push(Node::Gate(
            Gate::new("NEW", 9999)
                .p(vec![le(Q::PK, 1024), le(Q::PN, 640), le(Q::PE, 16), le(Q::PCores, 20)])
                .l("LaunchAlignedMedium64(x,weight,xScale,weightScale,groupList,y,selected,bf,stream);return;")
                .pack(PACK_ALIGNED_MEDIUM_64),
        ));
        let problems = check_packing(&nodes);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("k/128-1"), "{}", problems[0]);
        assert!(problems[0].contains("2 bits"), "{}", problems[0]);
        assert!(problems[0].contains("1024"), "{}", problems[0]);
    }

    /// Labels must be unique: `plan` reports them and the emitted markers are
    /// how the emitted-vs-modelled test pairs a gate with its C++.
    #[test]
    fn gate_labels_are_unique() {
        let nodes = cascade();
        let mut seen: Vec<&str> = gates(&nodes).iter().map(|g| g.label).collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), before, "duplicate gate label");
    }

    /// Every gate the table holds must appear in the rendered C++, once, in
    /// order, with exactly the conjuncts the table evaluates. This is what
    /// makes "emit prints what plan evaluates" a test rather than a claim.
    #[test]
    fn the_rendered_cascade_carries_every_gate_in_order() {
        let nodes = cascade();
        let text = render(&nodes);
        let mut at = 0usize;
        for gate in gates(&nodes) {
            let marker = format!("// gate {} -- kernel.asc:{}", gate.label, gate.line);
            let found = text[at..].find(&marker).unwrap_or_else(|| {
                panic!("gate {} missing from the rendered cascade (or out of order)", gate.label)
            });
            assert_eq!(
                text.matches(&marker).count(),
                1,
                "gate {} emitted more than once",
                gate.label
            );
            at += found + marker.len();

            // The rendered condition, whitespace-normalized, must be exactly
            // the table's conjuncts joined with `&&`.
            let cfg = gate.cfg();
            let terms: Vec<String> = gate
                .pred
                .iter()
                .filter(|t| !(cfg.is_some_and(|c| c.nested) && matches!(t, Expr::Configure(_))))
                .map(|t| squash(&t.cpp()))
                .collect();
            if terms.is_empty() {
                continue;
            }
            let rest = &text[at..];
            let open = rest.find("if (").expect("an if for a conditional gate");
            let cond = balanced(&rest[open + 4..]);
            assert_eq!(
                squash(&cond),
                terms.join(" && "),
                "gate {}: rendered condition does not match the table",
                gate.label
            );
        }
    }

    fn squash(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The text up to the paren that closes the one we are inside.
    fn balanced(text: &str) -> String {
        let mut depth = 1i32;
        for (at, ch) in text.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return text[..at].to_string();
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced condition");
    }

    /// Gate X is the one that configures `p` by reference; nothing else may.
    #[test]
    fn only_gate_x_configures_the_outer_plan() {
        for gate in gates(&cascade()) {
            if let Some(cfg) = gate.cfg() {
                if cfg.target == "p" {
                    assert_eq!(gate.label, "X", "an arm other than X writes the outer Plan");
                    assert!(!cfg.copy && !cfg.nested);
                }
            }
        }
    }
}
