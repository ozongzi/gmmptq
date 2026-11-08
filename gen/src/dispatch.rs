//! `gen plan` -- a dry run of `extern "C" run_kernel`.
//!
//! `kernel/kernel.asc:5814..6121` is a flat `if ... return` cascade. Widening or
//! narrowing one gate can silently reroute a *different* test point onto a
//! slower arm, and the only check we had was "compile and time all 15 points on
//! the NPU". This module runs that check in a millisecond instead of a device
//! slot.
//!
//! The gates themselves no longer live here: they live in [`crate::cascade`] as
//! data, and this module *evaluates* that data while `emit --profile final`
//! *prints* it. Before, the predicates existed twice -- once as C++ in
//! `gen/chunks/5814-run-kernel.inc`, once as Rust control flow here -- and
//! nothing could tell you the two had drifted.
//!
//! What stayed here is everything a table cannot express:
//!
//!   * the arithmetic the gates call out to (`ConfigureCubeFullK`,
//!     `LeanPlanCube`, `BuildOuterTiling`, `BuildBalancedOverlapShortGroup`);
//!   * the per-arm result -- which kernel symbol, what grid, how much scratch;
//!   * the two capacity `Select` ladders and `mix11::Launch`'s demotion, which
//!     are in other translation-unit fragments, not in `run_kernel`;
//!   * gate X's by-reference `Plan`, which is why `Cfg.target` is data.

use crate::cascade::{self, Cfg, Expr, Gate, Node, Op, Q, R};

pub use crate::cascade::Dep;

// ---------------------------------------------------------------------------
// machine constants (kernel.asc:10..14)
// ---------------------------------------------------------------------------

const L1_BYTES: u64 = 512 * 1024;
const L0A_BYTES: u64 = 64 * 1024;
const L0B_BYTES: u64 = 64 * 1024;
const L0C_BYTES: u64 = 128 * 1024;
const UB_BYTES: u64 = 192 * 1024;

fn up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b * b
}

// ---------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Shape {
    pub e: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub group_list_type: u32,
    /// `bf = strcmp(outputDtype,"bfloat16")==0` (5828).
    pub bf: bool,
    /// Per-expert token counts. The host side of the cascade never reads them --
    /// see `Routed::notes` -- they are carried so `plan` can validate the shape
    /// and so a later milestone's tile-height chooser has them.
    pub counts: Vec<u32>,
}

impl Shape {
    pub fn dtype(&self) -> &'static str {
        if self.bf { "bf16" } else { "fp16" }
    }
}

/// The six device pointers, each reduced to its low 7 bits. Every
/// pointer-dependent predicate in the cascade tests `&1`, `&3`, `&7`, `&31`,
/// `&63` or `&127`, so 7 bits is the whole of what the cascade can observe.
#[derive(Clone, Copy, Debug)]
pub struct Ptrs {
    pub x: u32,
    pub weight: u32,
    pub x_scale: u32,
    pub w_scale: u32,
    pub group_list: u32,
    pub y: u32,
}

impl Default for Ptrs {
    /// Judge buffers are at least 64B aligned in practice (usually 512B), so
    /// the default is "all six congruent to 0 mod 128". CURRENT.md's 15-point
    /// table is the route under exactly this assumption.
    fn default() -> Self {
        Ptrs { x: 0, weight: 0, x_scale: 0, w_scale: 0, group_list: 0, y: 0 }
    }
}

impl Ptrs {
    /// The adversarial counter-assumption: nothing is aligned beyond a byte.
    pub fn misaligned() -> Self {
        Ptrs { x: 1, weight: 1, x_scale: 1, w_scale: 1, group_list: 1, y: 1 }
    }
    fn any_of_six(&self) -> u32 {
        self.x | self.weight | self.x_scale | self.w_scale | self.group_list | self.y
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Env {
    /// `availableCoreNum` (5821). Six predicates hard-assume 20.
    pub available_core_num: u32,
    /// `GetCoreNumAic()`; only `mix11::Launch` / `LaunchPacked32` clamp to it.
    pub physical_aic: u32,
    pub ptrs: Ptrs,
    /// False models a generator that rewrote the L1 budget arithmetic. Two
    /// independent checks compute essentially the same bound: gate Y's
    /// `512*kp <= 524032` (6105) and `ConfigureCubeFullK`'s
    /// `(BM + 2*BN)*kp <= 512K` (2863), i.e. `512*kp <= 524288`. Between them
    /// they are the *only* thing bounding `k` below `p1_half_pair`'s 3-bit
    /// `k/128-1` field. CURRENT.md landmine 1 names only the first, which
    /// understates the protection and overstates the fragility: dropping 6105
    /// alone changes nothing, because 2864 catches the same shape.
    pub l1_guard: bool,
}

impl Default for Env {
    fn default() -> Self {
        Env {
            available_core_num: 20,
            physical_aic: 20,
            ptrs: Ptrs::default(),
            l1_guard: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Plan (kernel.asc's 15xu32 launch parameter)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Plan {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub e: u32,
    pub list_type: u32,
    pub kp: u32,
    pub wstride: u32,
    pub cores: u32,
    pub vcores: u32,
    pub m0: u32,
    pub n0: u32,
    pub m0r: u32,
    pub n0r: u32,
    pub n0c: u32,
    pub k0: u32,
    pub kl: u32,
    pub n_tiles: u32,
    pub rv: u32,
}

/// Everything a predicate term can read. One value, built once per route, so a
/// `Q` has somewhere to look its number up.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub shape: Shape,
    pub env: Env,
    /// The `Plan` as the construction at 6006 leaves it. The gates before that
    /// line read only `M/K/N/E`, so filling it eagerly is harmless.
    pub plan: Plan,
    pub scalar_output_window: bool,
    pub tiny_cache_aligned: bool,
    pub pack_weights: bool,
    /// Gate F's `scalar` (5890).
    pub scalar: bool,
    /// Gate R's locals (6013).
    pub selected_pitch: u32,
    pub selected_ub: u64,
}

impl Ctx {
    pub fn new(nodes: &[Node], shape: &Shape, env: &Env) -> Ctx {
        let (m, n) = (shape.m, shape.n);
        let plan = Plan {
            m,
            k: shape.k,
            n,
            e: shape.e,
            list_type: shape.group_list_type,
            kp: up(shape.k, 32),
            wstride: n,
            cores: env.available_core_num,
            vcores: env.available_core_num * 2,
            ..Default::default()
        };
        let pitch = up(n, 16);
        let nup = up(n, 32);
        let mut ctx = Ctx {
            shape: shape.clone(),
            env: *env,
            plan,
            scalar_output_window: u64::from(env.ptrs.y & 127)
                + u64::from(m) * u64::from(n) * 2
                <= 32,
            tiny_cache_aligned: (env.ptrs.any_of_six() & 63) == 0,
            pack_weights: false,
            scalar: false,
            selected_pitch: pitch,
            selected_ub: 32u64 * u64::from(pitch) * 6
                + 12 * u64::from(nup)
                + 80 * 32
                + u64::from(nup)
                + 32
                + 32
                + 32,
        };
        // The two derived booleans whose own definitions are table data. They
        // never read each other, so one pass in table order is enough.
        for node in nodes {
            if let Node::Derived(d) = node {
                if d.name == "packWeights" {
                    ctx.pack_weights = d.eval(&ctx);
                }
            }
        }
        if let Some(f) = cascade::find(nodes, "F") {
            if let Some(d) = f.derived.iter().find(|d| d.name == "scalar") {
                ctx.scalar = d.eval(&ctx);
            }
        }
        ctx
    }
}

/// `ConfigureCubeFullK<BM,BN,BK,RV,Prefetch,SingleL0>` (2851).
///
/// Returns false -- **before writing anything** -- iff any dim is 0 or the L1
/// footprint `(BM + (Prefetch?2:1)*BN) * kp` exceeds 512 KiB. Keeping those two
/// `return false`s ahead of every store is what makes gate X's by-reference
/// `p` safe today; a generator that reorders them pollutes Y and Z.
pub fn configure_cube_full_k(
    p: &mut Plan,
    bm: u32,
    bn: u32,
    bk: u32,
    rv: u32,
    prefetch: bool,
    single_l0: bool,
    l1_guard: bool,
) -> bool {
    // The static_asserts (2852..2860) are compile-time in the kernel; keep them
    // as debug assertions so a generator-proposed tile is rejected here too.
    debug_assert!(bm > 0 && bm % 16 == 0, "BM alignment");
    debug_assert!(bn > 0 && bn % 32 == 0, "BN alignment");
    debug_assert!(bk > 0 && bk % 32 == 0, "BK alignment");
    debug_assert!(rv > 0 && rv <= bm && rv % 8 == 0, "RV alignment");
    let dup = if single_l0 { 1u64 } else { 2u64 };
    debug_assert!(u64::from(bm) * u64::from(bk) * dup <= L0A_BYTES, "L0A overflow");
    debug_assert!(u64::from(bn) * u64::from(bk) * dup <= L0B_BYTES, "L0B overflow");
    debug_assert!(u64::from(bm) * u64::from(bn) * 4 <= L0C_BYTES, "L0C overflow");
    debug_assert!(
        u64::from(rv) * u64::from(bn) * 10 + u64::from(bn) * 8 + u64::from(rv) * 36
            <= UB_BYTES - 8192 - 256,
        "UB overflow"
    );

    if p.k == 0 || p.n == 0 || p.m == 0 || p.e == 0 || p.cores == 0 {
        return false;
    }
    let kp = (p.k + 31) / 32 * 32;
    let l1 = u64::from(bm + if prefetch { 2 } else { 1 } * bn) * u64::from(kp);
    if l1_guard && l1 > L1_BYTES {
        return false;
    }
    p.kp = kp;
    p.m0 = bm;
    p.n0 = bn;
    p.m0r = bm;
    p.n0r = bn;
    p.n0c = bn;
    p.k0 = bk;
    p.kl = kp;
    p.n_tiles = (p.n + bn - 1) / bn;
    p.rv = rv;
    true
}

/// `CubeFullKStagingBytes` (2880).
fn staging_bytes(p: &Plan) -> u64 {
    let max_tiles = (u64::from((p.m + p.m0 - 1) / p.m0) + u64::from(p.e)) * u64::from(p.n_tiles);
    max_tiles * u64::from(p.m0) * u64::from(p.n0) * 4
}

/// `LeanPlanCube` (2885) -- gate Z's tiling.
fn lean_plan_cube(p: &mut Plan) {
    let mut per_expert = if p.e != 0 { p.m / p.e } else { p.m };
    if per_expert < 1 {
        per_expert = 1;
    }
    let mut m0 = 256u32;
    while m0 > 16 && m0 / 2 >= up(per_expert, 16) {
        m0 /= 2;
    }
    let mut n0 = 32768 / up(m0, 16);
    if n0 > 512 {
        n0 = 512;
    }
    while n0 > 32 && n0 / 2 >= up(p.n, 16) {
        n0 /= 2;
    }
    if n0 < 16 {
        n0 = 16;
    }
    let m0r = up(m0, 16);
    let n0r = up(n0, 16);
    let n0c = up(n0, 32);
    let mut k0 = 128u32;
    while k0 > 32
        && (u64::from(m0r) * u64::from(k0) * 2 > L0A_BYTES
            || u64::from(k0) * u64::from(n0r) * 2 > L0B_BYTES)
    {
        k0 /= 2;
    }
    while k0 > 32 && k0 / 2 >= p.kp {
        k0 /= 2;
    }
    let mut kl = p.kp;
    while kl > k0 && u64::from(m0r) * u64::from(kl) + 2 * u64::from(kl) * u64::from(n0c) > 524032 {
        kl -= k0;
    }
    if kl < k0 {
        kl = k0;
    }
    p.m0 = m0;
    p.n0 = n0;
    p.m0r = m0r;
    p.n0r = n0r;
    p.n0c = n0c;
    p.k0 = k0;
    p.kl = kl;
    p.n_tiles = (p.n + n0 - 1) / n0;
    let pitch = up(n0, 8);
    let mut rv = 64u32;
    while rv > 1 && u64::from(rv) * u64::from(pitch) * 4 * 2 + 16 * 1024 > UB_BYTES {
        rv /= 2;
    }
    if rv > m0 {
        rv = if m0 < 1 { 1 } else { m0 };
    }
    p.rv = rv;
}

// ---------------------------------------------------------------------------
// VTiling (the 8xu32 AIV launch parameter)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
pub struct VTiling {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub e: u32,
    pub list_type: u32,
    pub core_num: u32,
    pub kc: u32,
    pub nb: u32,
}

/// `BalancedOverlapShortGroupUbBytes` (3020).
fn bosg_ub_bytes(max_k: u32, capacity_n: u32, n_pitch: u32) -> u64 {
    let block = u64::from(max_k) * u64::from(capacity_n);
    let k_up = u64::from((max_k + 31) / 32 * 32);
    128 * 8 + 256 * 4 + (k_up + 32) + (block + 32) + 4 * block + 4 * k_up
        + 10 * u64::from(n_pitch)
        + 32
        + 2 * block
        + 2 * k_up
        + 8192
}

/// `BuildBalancedOverlapShortGroup` (3026) -- gate S.
fn build_bosg(p: &Plan) -> VTiling {
    let mut out = VTiling {
        m: p.m,
        k: p.k,
        n: p.n,
        e: p.e,
        list_type: p.list_type,
        core_num: p.vcores.min(p.m),
        kc: 128,
        nb: 0,
    };
    let n_pitch = (p.n + 31) / 32 * 32;
    out.nb = n_pitch.min(192);
    let k_up = (p.k + 7) / 8 * 8;
    while out.kc > 8 && (bosg_ub_bytes(out.kc, out.nb, n_pitch) > 188160 || out.kc / 2 >= k_up) {
        out.kc /= 2;
    }
    let min_width = ((8 * n_pitch + out.kc - 1) / out.kc + 31) / 32 * 32;
    out.nb = out.nb.max(min_width);
    out
}

/// `BuildOuterTiling` (938) -- gate R.3.
fn build_outer_tiling(p: &Plan) -> VTiling {
    const GRAIN: u32 = 32; // vec::GRAIN
    const MAX_ROWS: u32 = 32; // outer::MAX_ROWS
    let pitch = up(p.n, 16);
    let n_up = up(p.n, GRAIN);
    let mut rows = if pitch / 8 > 255 { 1 } else { MAX_ROWS };
    let ub_bytes = |r: u32| -> u64 {
        let r_up = up(r, GRAIN);
        u64::from(r) * u64::from(pitch) * 6
            + 12 * u64::from(n_up)
            + 80 * u64::from(r_up)
            + u64::from(n_up + GRAIN)
            + u64::from(r_up + GRAIN)
    };
    while rows > 1 && ub_bytes(rows) > 188160 {
        rows -= 1;
    }
    let items = (p.m + rows - 1) / rows + p.e;
    VTiling {
        m: p.m,
        k: p.k,
        n: p.n,
        e: p.e,
        list_type: p.list_type,
        core_num: if items < p.vcores { items.max(1) } else { p.vcores },
        kc: rows,
        nb: pitch,
    }
}

// ---------------------------------------------------------------------------
// bit packing
// ---------------------------------------------------------------------------

/// One field of a packed `uint32_t dims`, as `plan` reports it. Built from the
/// gate's [`cascade::Packer`]: `value` from the shape, `gate_max` from the
/// gate's own predicate. `None` means the gate does not bound the packed
/// quantity at all, which is the CURRENT.md landmine-1 class of bug.
#[derive(Clone, Debug)]
pub struct Field {
    pub name: &'static str,
    pub shift: u32,
    pub width: u32,
    pub value: u64,
    pub gate_max: Option<u64>,
    /// Set when the gate's bound is incidental -- a budget check that happens to
    /// keep the quantity in range -- rather than a bound on the field itself.
    pub incidental: Option<&'static str>,
}

impl Field {
    pub fn field_max(&self) -> u64 {
        (1u64 << self.width) - 1
    }
    pub fn fits(&self) -> bool {
        self.value <= self.field_max()
    }
    /// Headroom between what the gate admits and what the field can hold.
    pub fn margin(&self) -> Option<i64> {
        self.gate_max.map(|g| self.field_max() as i64 - g as i64)
    }
}

/// The `dims` word a field list encodes, so `plan` can print the same number
/// the host computes instead of a second copy of the packing expression.
pub fn pack_value(fields: &[Field]) -> u64 {
    fields.iter().fold(0u64, |acc, f| acc | (f.value << f.shift))
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    Pass,
    Fail,
    /// A derived value, not a branch.
    Note,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub label: String,
    pub line: u32,
    pub outcome: Outcome,
    /// For a Fail: the first conjunct that failed. For a Pass/Note: a summary.
    pub detail: String,
    pub deps: Vec<Dep>,
}

/// Evaluate one gate's conjunction from the table, short-circuiting exactly as
/// `&&` does, and report which term decided it.
///
/// `plan` is what a `ConfigureCubeFullK` term writes: a fresh copy for T/U/W/Y,
/// the caller's own `p` for X.
fn eval_gate(gate: &Gate, c: &Ctx, plan: &mut Plan) -> (bool, Step) {
    let mut ok = true;
    let mut first_fail = String::new();
    let mut deps: Vec<Dep> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    for term in &gate.pred {
        if !ok {
            // `&&` never reaches this term, so it contributes no dependency:
            // claiming one would be noise in exactly the report that matters.
            break;
        }
        let dep = term.dep();
        if dep != Dep::Shape && !deps.contains(&dep) {
            deps.push(dep);
        }
        if term.eval(c, plan) {
            summary.push(term.show(c));
        } else {
            first_fail = term.show(c);
            ok = false;
        }
    }
    let step = Step {
        label: gate.label.to_string(),
        line: gate.line,
        outcome: if ok { Outcome::Pass } else { Outcome::Fail },
        detail: if ok {
            if summary.is_empty() {
                "unconditional".to_string()
            } else {
                summary.join(" && ")
            }
        } else {
            first_fail
        },
        deps,
    };
    (ok, step)
}

fn derived_step(d: &cascade::Derived, c: &Ctx) -> Step {
    let value = d.eval(c);
    let shown: Vec<String> = d.terms.iter().map(|t| t.show(c)).collect();
    Step {
        label: d.name.to_string(),
        line: d.line,
        outcome: Outcome::Note,
        detail: format!("{} = {value}  <- {}", d.name, shown.join(&format!(" {} ", d.join))),
        deps: d.terms.iter().map(|t| t.dep()).filter(|dep| *dep != Dep::Shape).collect(),
    }
}

// ---------------------------------------------------------------------------
// result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Routed {
    pub gate: String,
    pub line: u32,
    pub kernel: String,
    /// What the inner body resolves to, when the launch is a wrapper.
    pub body: Option<String>,
    pub grid: u32,
    pub grid_how: String,
    pub plan: Option<Plan>,
    pub vtiling: Option<VTiling>,
    pub staging: Option<u64>,
    pub extra_scratch: Option<(String, u64)>,
    pub packing: Vec<Field>,
    pub notes: Vec<String>,
    /// Fatal: a bit-packing overflow, or a capacity ladder with no matching arm.
    pub errors: Vec<String>,
    pub trace: Vec<Step>,
}

impl Routed {
    fn new(trace: Vec<Step>, gate: &str, line: u32, kernel: &str, grid: u32, how: &str) -> Routed {
        Routed {
            gate: gate.to_string(),
            line,
            kernel: kernel.to_string(),
            body: None,
            grid,
            grid_how: how.to_string(),
            plan: None,
            vtiling: None,
            staging: None,
            extra_scratch: None,
            packing: Vec::new(),
            notes: Vec::new(),
            errors: Vec::new(),
            trace,
        }
    }
    fn with_pack(mut self, fields: Vec<Field>) -> Routed {
        for field in &fields {
            if !field.fits() {
                self.errors.push(format!(
                    "bit-packing overflow: {} = {} does not fit {} bits (max {})",
                    field.name,
                    field.value,
                    field.width,
                    field.field_max()
                ));
            }
        }
        self.packing = fields;
        self
    }
    pub fn gates_using_pointers(&self) -> Vec<&Step> {
        self.trace.iter().filter(|s| s.deps.contains(&Dep::Ptr)).collect()
    }
    pub fn gates_using_cores(&self) -> Vec<&Step> {
        self.trace.iter().filter(|s| s.deps.contains(&Dep::Cores)).collect()
    }
}

// ---------------------------------------------------------------------------
// the cascade, evaluated
// ---------------------------------------------------------------------------

/// Route a shape through the shipped cascade.
pub fn route(s: &Shape, env: &Env) -> Result<Routed, String> {
    route_with(&cascade::cascade(), s, env)
}

/// Route a shape through *a* cascade -- the shipped one, or a candidate that
/// `plan --what-if` built without touching the table.
///
/// This is a **walk of the table**, not a hand-written chain: the loop below
/// visits the nodes in table order, and a gate the evaluator has no bespoke arm
/// for still gets evaluated, in its table position, through `generic_arm`. That
/// is what makes adding a gate a data change -- insert a `Node::Gate` and both
/// `plan` and `emit` pick it up, at the position you inserted it.
pub fn route_with(nodes: &[Node], s: &Shape, env: &Env) -> Result<Routed, String> {
    let mut tr: Vec<Step> = Vec::new();
    let (m, k, n, e) = (s.m, s.k, s.n, s.e);
    let bf = s.bf;
    let gt = s.group_list_type;

    // --- A: host validation (5824, 5840) -------------------------------------
    if gt > 1 {
        return Err("groupListType must be 0 or 1 (5826)".into());
    }
    if env.available_core_num == 0 {
        return Err("availableCoreNum <= 0 (5826)".into());
    }
    if m < 1 || m > 32768 || k < 1 || k > 8192 || e < 1 || e > 128 || n < 1 || n > 8192 {
        return Err(format!(
            "shape mismatch (5840): need 1<=M<=32768, 1<=K<=8192, 1<=E<=128, 1<=N<=8192; got E{e} M{m} K{k} N{n}"
        ));
    }
    tr.push(Step {
        label: "A".into(),
        line: 5824,
        outcome: Outcome::Pass,
        detail: "attrs and tensor shapes consistent".into(),
        deps: vec![],
    });

    let ctx = Ctx::new(nodes, s, env);
    // The `Plan p` the construction at 6006 leaves. Gate X configures *this*
    // object by reference (6097); every other arm configures a copy.
    let mut p = ctx.plan;

    for node in nodes {
        match node {
            Node::Fixed { label, line, .. } => {
                if *label == "Q" {
                    tr.push(Step {
                        label: "Q".into(),
                        line: *line,
                        outcome: Outcome::Note,
                        detail: format!(
                            "Plan p: kp=UpH(k,32)={} wstride=n={} cores={} vcores={} rowsPerExpert={}",
                            p.kp,
                            p.wstride,
                            p.cores,
                            p.vcores,
                            p.m / p.e
                        ),
                        deps: vec![Dep::Cores],
                    });
                }
            }
            Node::Derived(d) => tr.push(derived_step(d, &ctx)),
            Node::Gate(gate) => {
                if let Some(result) = try_gate(nodes, gate, &ctx, &mut p, &mut tr) {
                    return result;
                }
            }
        }
    }

    // --- Z: unconditional fallback (6112) --------------------------------------
    tr.push(Step {
        label: "Z".into(),
        line: 6112,
        outcome: Outcome::Pass,
        detail: "unconditional: p.wstride=p.n; LeanPlanCube(p); CubeMix<ScaleT,OutT>".into(),
        deps: vec![],
    });
    p.wstride = p.n;
    lean_plan_cube(&mut p);
    let max_tiles = (u64::from((p.m + p.m0 - 1) / p.m0) + u64::from(p.e)) * u64::from(p.n_tiles);
    let kern = if bf { "CubeMix<bfloat16_t,bfloat16_t>" } else { "CubeMix<float,half>" };
    let mut r = Routed::new(tr, "Z", 6112, kern, p.cores, "p.cores");
    r.plan = Some(p);
    r.staging = Some(max_tiles * u64::from(p.m0) * u64::from(p.n0) * 4);
    r.notes
        .push("Plan by value, no bit packing -- Z cannot overflow a dims field".into());
    r.notes.push("correct for any new shape, but 2-5x slower than the specialized families".into());
    Ok(r)
}

/// One top-level gate. `None` means it did not fire and the walk continues.
///
/// Everything in here is what a table *cannot* express: which kernel symbol the
/// arm launches, what its grid is, how much GM scratch it allocates, and the
/// sub-dispatch that happens in another fragment (the two capacity `Select`
/// ladders, `mix11::Launch`'s demotion to `packed32`).
fn try_gate(
    nodes: &[Node],
    gate: &Gate,
    ctx: &Ctx,
    p: &mut Plan,
    tr: &mut Vec<Step>,
) -> Option<Result<Routed, String>> {
    let s = &ctx.shape;
    let env = &ctx.env;
    let (m, k, n, e) = (s.m, s.k, s.n, s.e);
    let bf = s.bf;
    let cores64 = u64::from(env.available_core_num);
    let mut scratch = Plan::default();
    let inner = |label: &str| -> &Gate {
        cascade::find(nodes, label).unwrap_or_else(|| panic!("cascade has no gate {label:?}"))
    };
    let pack = |label: &str| cascade::pack_fields(nodes, label, ctx);

    // The `Plan` this gate's `ConfigureCubeFullK` writes: a fresh copy, except
    // for gate X, which configures the caller's `p` by reference.
    let mut selected = *p;
    let by_ref = gate.cfg().is_some_and(|c| c.target == "p");
    let before = *p;
    let (ok, step) = if by_ref {
        eval_gate(gate, ctx, p)
    } else {
        eval_gate(gate, ctx, &mut selected)
    };
    tr.push(step);
    if by_ref && !ok {
        debug_assert_eq!(*p, before, "a failed by-reference gate must not write the outer Plan");
    }
    if !ok {
        return None;
    }

    Some(Ok(match gate.label {
        // --- B (5845) --------------------------------------------------------
        "B" => Routed::new(
            tr.clone(),
            "B",
            gate.line,
            &format!("TinyShortReduction<{bf}>"),
            1,
            "<<<1>>>",
        )
        .with_pack(pack("B")),

        // --- D (5858) --------------------------------------------------------
        "D" => {
            let fields = pack("D");
            let row_dims = pack_value(&fields);
            let mut r = Routed::new(
                tr.clone(),
                "D",
                gate.line,
                "expert_output::SearchTinyExpertOutput6400000<float,half>",
                e,
                "<<<E>>>",
            )
            .with_pack(fields);
            r.notes.push(format!("rowDims = {row_dims}"));
            r.notes.push("AIV path: launches asynchronously, no aclrtSynchronizeStream".into());
            r
        }

        // --- E (5871), with E' and E'' ---------------------------------------
        "E" => {
            let fields = pack("E");
            let dims = pack_value(&fields);
            let (contract, step) = eval_gate(inner("E'"), ctx, &mut scratch);
            tr.push(step);
            // The Select ladders decode from `dims`, so they see truncated m/k/n.
            let dm = (dims & 3) as u32 + 1;
            let dk = ((dims >> 2) & 7) as u32 + 1;
            let dn = ((dims >> 5) & 3) as u32 + 1;
            if contract {
                let mut r = match contract_select(dm, dk, dn, bf) {
                    Some((cm, ck, cn)) => Routed::new(
                        tr.clone(),
                        "E'",
                        5876,
                        &format!(
                            "tiny_capacity_contract::SearchTinyCapacityContract11850017<{bf},{cm},{ck},{cn}>"
                        ),
                        1,
                        "<<<1>>>",
                    ),
                    None => {
                        let mut r = Routed::new(tr.clone(), "E'", 5876, "tiny_capacity_contract::Select -- no arm", 0, "none");
                        r.errors.push(format!(
                            "tiny_capacity_contract::Select (5546) has no arm for (m={dm},k={dk},n={dn},bf={bf}); \
                             it throws std::invalid_argument at 5727 -> RE"
                        ));
                        r
                    }
                }
                .with_pack(fields);
                r.notes.push(format!("dims = {dims}; Select decodes m={dm} k={dk} n={dn}"));
                return Some(Ok(r));
            }
            let mut r = match unrolled_select(dk, dn, bf) {
                Some((ck, cn)) => Routed::new(
                    tr.clone(),
                    "E''",
                    5882,
                    &format!(
                        "tiny_capacity_unrolled::SearchTinyCapacityUnrolled11850012<{bf},{ck},{cn}>"
                    ),
                    1,
                    "<<<1>>>",
                ),
                None => {
                    let mut r = Routed::new(tr.clone(), "E''", 5882, "tiny_capacity_unrolled::Select -- no arm", 0, "none");
                    r.errors.push(format!(
                        "tiny_capacity_unrolled::Select (5337) has no arm for (k={dk},n={dn},bf={bf}); \
                         it throws std::invalid_argument at 5414 -> RE"
                    ));
                    r
                }
            }
            .with_pack(fields);
            r.notes.push(format!("dims = {dims}; Select decodes k={dk} n={dn}"));
            r
        }

        // --- F (5888), with the K ladder -------------------------------------
        "F" => {
            let scalar = ctx.scalar;
            // The capacity comes from the ladder arm's own predicate, so the
            // 1024/2048/4096/8192 staircase exists once, in the table.
            let mut cap: Option<u64> = None;
            for arm in &gate.inner {
                let (hit, step) = eval_gate(arm, ctx, &mut scratch);
                tr.push(step);
                if hit {
                    cap = arm.pred.iter().find_map(|t| match t {
                        Expr::Cmp(Q::K, Op::Le, R::Const(v)) => Some(*v),
                        _ => None,
                    });
                    break;
                }
            }
            let cap = cap.expect("gate F bounds K<=8192, so some ladder arm must fire");
            // longdot_capacity::Launch (3635): fp16 with misaligned scales takes
            // the DMA variant; bf16 has no DMA variant.
            let ptr = &env.ptrs;
            let dma = !bf && ((ptr.x_scale | ptr.w_scale) & 3) != 0;
            let kernel = if dma {
                format!("longdot_capacity::SearchLongDotCapacityDMAScales<{cap},{scalar}>")
            } else {
                format!("longdot_capacity::SearchLongDotCapacity11840033<{cap},{bf},{scalar}>")
            };
            let mut r = Routed::new(tr.clone(), "F", gate.line, &kernel, 2, "<<<2>>>");
            r.notes.push(format!(
                "K ladder: K={k} -> Capacity={cap}; scalar={scalar} (y&31==0 && scalarOutputWindow)"
            ));
            if dma {
                r.notes.push("DMA-scales variant chosen: (xScale|weightScale)&3 != 0".into());
            } else if !bf {
                r.notes
                    .push("scalar-scales variant: (xScale|weightScale)&3 == 0 [pointer-dependent]".into());
            }
            r.notes.push("K is a runtime argument, not packed -- no bit fields".into());
            r
        }

        // --- H (5919) --------------------------------------------------------
        "H" => {
            let kern = if bf {
                "SearchTinyE1STDev3400000<bfloat16_t,bfloat16_t>"
            } else {
                "SearchTinyE1STDev3400000<float,half>"
            };
            let mut r =
                Routed::new(tr.clone(), "H", gate.line, kern, m, "<<<M>>>").with_pack(pack("H"));
            r.notes.push(
                "rowDims omits the E field (5923) while the decoder reads bits 10..16 (4957); \
                 correct only because H requires E==1 [CURRENT.md landmine 6]"
                    .into(),
            );
            r
        }

        // --- I (5928) --------------------------------------------------------
        "I" => {
            let kern = if bf {
                "E1FixedUbRowTiny<bfloat16_t,bfloat16_t>"
            } else {
                "E1FixedUbRowTiny<float,half>"
            };
            Routed::new(tr.clone(), "I", gate.line, kern, m, "<<<M>>>").with_pack(pack("I"))
        }

        // --- J (5937) --------------------------------------------------------
        "J" => {
            let mut r = Routed::new(
                tr.clone(),
                "J",
                gate.line,
                &format!("TinyCacheReduction<{bf}>"),
                1,
                "<<<1>>>",
            )
            .with_pack(pack("J"));
            r.notes.push(
                "tinyCacheDims packs M-1 in 4 bits and E-1 in 3 bits (5940); both overflows are \
                 blocked only by unrelated clauses [CURRENT.md landmine 6]"
                    .into(),
            );
            r
        }

        // --- K (5949), with K.1 / K.2 ----------------------------------------
        "K" => {
            let fields = pack("K");
            let (two_by_two, step) = eval_gate(inner("K.2x2"), ctx, &mut scratch);
            tr.push(step);
            if two_by_two {
                let (one, step) = eval_gate(inner("K.1"), ctx, &mut scratch);
                tr.push(step);
                if one {
                    return Some(Ok(Routed::new(
                        tr.clone(),
                        "K.1",
                        5951,
                        "expert_output::SearchTinyExpertOutput6400000<float,half>",
                        e,
                        "<<<E>>>",
                    )
                    .with_pack(fields)));
                }
                let kern = if bf {
                    "expert_reuse::SearchTinyExpertReuse6300000<bfloat16_t,bfloat16_t>"
                } else {
                    "expert_reuse::SearchTinyExpertReuse6300000<float,half>"
                };
                return Some(Ok(
                    Routed::new(tr.clone(), "K.2", 5956, kern, e, "<<<E>>>").with_pack(fields)
                ));
            }
            let kern = if bf {
                "SearchTinyExpert3500000<bfloat16_t,bfloat16_t>"
            } else {
                "SearchTinyExpert3500000<float,half>"
            };
            Routed::new(tr.clone(), "K", gate.line, kern, e, "<<<E>>>").with_pack(fields)
        }

        // --- L (5965), with L.1 ----------------------------------------------
        "L" => {
            let fields = pack("L");
            let (one, step) = eval_gate(inner("L.1"), ctx, &mut scratch);
            tr.push(step);
            if one {
                return Some(Ok(Routed::new(
                    tr.clone(),
                    "L.1",
                    5967,
                    "tiny_row_stdev::SearchTinyRowSTDev3400000<bfloat16_t,bfloat16_t>",
                    m,
                    "<<<M>>>",
                )
                .with_pack(fields)));
            }
            let kern = if bf {
                "RowParallelTiny<bfloat16_t,bfloat16_t>"
            } else {
                "RowParallelTiny<float,half>"
            };
            Routed::new(tr.clone(), "L", gate.line, kern, m, "<<<M>>>").with_pack(fields)
        }

        // --- M (5976) --------------------------------------------------------
        "M" => {
            let blocks = u64::from(m).min(2 * cores64) as u32;
            let kern = if bf {
                "BudgetStridedRowTiny<bfloat16_t,bfloat16_t>"
            } else {
                "BudgetStridedRowTiny<float,half>"
            };
            Routed::new(tr.clone(), "M", gate.line, kern, blocks, "min(M, 2*cores)")
                .with_pack(pack("M"))
        }

        // --- N (5983) --------------------------------------------------------
        "N" => Routed::new(
            tr.clone(),
            "N",
            gate.line,
            &format!("TinyShortReduction<{bf}>"),
            1,
            "<<<1>>>",
        )
        .with_pack(pack("N")),

        // --- O (5988) --------------------------------------------------------
        "O" => {
            let blocks = (2 * cores64).min(u64::from(n)) as u32;
            let kern = if bf {
                "IntegerProductLongNarrowDot<bfloat16_t,bfloat16_t>"
            } else {
                "IntegerProductLongNarrowDot<float,half>"
            };
            let mut r = Routed::new(tr.clone(), "O", gate.line, kern, blocks, "min(2*cores, N)");
            r.notes.push("GeneralVectorDims passed by value -- no bit packing".into());
            r
        }

        // --- P (5994), with P.1 / P.2 ----------------------------------------
        "P" => {
            let blocks = (2 * cores64).min(u64::from(m)) as u32;
            let (small_k, step) = eval_gate(inner("P.k800"), ctx, &mut scratch);
            tr.push(step);
            if small_k {
                let kern = if bf {
                    "SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800>"
                } else {
                    "SearchN1FixedLayoutSafe7100000<float,half,32,800>"
                };
                let mut r =
                    Routed::new(tr.clone(), "P", gate.line, kern, blocks, "min(2*cores, M)");
                r.notes.push(format!("K({k})<=800 (5997): 800 is the UB column capacity C"));
                r.notes.push("GeneralVectorDims by value -- no bit packing; fixed UB 184672 B".into());
                return Some(Ok(r));
            }
            let kern = if bf { "RowDot<16,bfloat16_t,bfloat16_t>" } else { "RowDot<16,float,half>" };
            let mut r = Routed::new(tr.clone(), "P.2", 6002, kern, blocks, "min(2*cores, M)");
            r.notes.push(format!("K({k})>800 -> RowDot<16>"));
            r
        }

        // --- R: K==1 outer product (6010) ------------------------------------
        "R" => {
            let (one, step) = eval_gate(inner("R.1"), ctx, &mut scratch);
            tr.push(step);
            if one {
                let core_num = p.vcores.min((p.m + 31) / 32 + p.e).min(16);
                let vt = VTiling {
                    m: p.m,
                    k: p.k,
                    n: p.n,
                    e: p.e,
                    list_type: p.list_type,
                    core_num,
                    kc: 32,
                    nb: ctx.selected_pitch,
                };
                let mut r = Routed::new(
                    tr.clone(),
                    "R.1",
                    6015,
                    "p10_fixed_ub::SearchP10FixedUb7300000",
                    core_num,
                    "min(min(vcores, ceil(m/32)+e), 16)",
                );
                r.vtiling = Some(vt);
                r.plan = Some(*p);
                r.notes.push(format!("selectedUb = {} B", ctx.selected_ub));
                r.notes.push("VTiling by value -- no bit packing".into());
                return Some(Ok(r));
            }
            let (two, step) = eval_gate(inner("R.2"), ctx, &mut scratch);
            tr.push(step);
            if two {
                let core_num = p.vcores.min((p.m + 19) / 20 + p.e);
                let vt = VTiling {
                    m: p.m,
                    k: p.k,
                    n: p.n,
                    e: p.e,
                    list_type: p.list_type,
                    core_num,
                    kc: 20,
                    nb: up(p.n, 16),
                };
                let mut r = Routed::new(
                    tr.clone(),
                    "R.2",
                    6025,
                    "PublishedOuter",
                    core_num,
                    "min(vcores, ceil(m/20)+e)",
                );
                r.vtiling = Some(vt);
                r.plan = Some(*p);
                return Some(Ok(r));
            }
            let (_, step) = eval_gate(inner("R.3"), ctx, &mut scratch);
            tr.push(step);
            let vt = build_outer_tiling(p);
            let kern = if bf {
                "GmmOuterKernel<bfloat16_t,bfloat16_t>"
            } else {
                "GmmOuterKernel<float,half>"
            };
            let mut r =
                Routed::new(tr.clone(), "R.3", 6032, kern, vt.core_num, "BuildOuterTiling.coreNum");
            r.vtiling = Some(vt);
            r.plan = Some(*p);
            r.notes.push(
                "BuildOuterTiling (947) never checks ubBytes(1) <= 188160; its loop condition is \
                 rows>1 [CURRENT.md landmine 7]"
                    .into(),
            );
            r
        }

        // --- S (6038) --------------------------------------------------------
        "S" => {
            let vt = build_bosg(p);
            let kern = if bf {
                "BalancedOverlapShortGroupMatmulKernel<bfloat16_t,bfloat16_t>"
            } else {
                "BalancedOverlapShortGroupMatmulKernel<float,half>"
            };
            let mut r = Routed::new(tr.clone(), "S", gate.line, kern, vt.core_num, "min(vcores, m)");
            r.notes.push(format!("nPitch = UpH(n,32) = {}", (p.n + 31) / 32 * 32));
            r.notes.push("pure AIV; VTiling by value -- no bit packing".into());
            r.vtiling = Some(vt);
            r.plan = Some(*p);
            r
        }

        // --- T (6046) --------------------------------------------------------
        "T" => {
            let mut r = Routed::new(
                tr.clone(),
                "T",
                gate.line,
                "SearchCubeCompact4700001<float,half>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some(cfg_body(gate, "CubeFullKBodyScalarHalf<float,half>"));
            r.staging = Some(staging_bytes(&selected) / 2);
            r.plan = Some(selected);
            r.with_pack(pack("T"))
        }

        // --- U (6055) --------------------------------------------------------
        "U" => {
            let kern = if bf {
                "SearchCubeCompact4700003<bfloat16_t,bfloat16_t>"
            } else {
                "SearchCubeCompact4700003<float,half>"
            };
            let mut r = Routed::new(
                tr.clone(),
                "U",
                gate.line,
                kern,
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some(format!("{}, 2u, true", cfg_body(gate, "CubeFullKBody<...>")));
            r.staging = Some(staging_bytes(&selected));
            r.plan = Some(selected);
            r.with_pack(pack("U"))
        }

        // --- V (6067), with V.1 / V.2 ----------------------------------------
        "V" => {
            let mut packed = *p;
            packed.wstride = up(p.n, 128);
            let v1 = inner("V.1");
            let (one, step) = eval_gate(v1, ctx, &mut packed);
            tr.push(step);
            if one {
                let mut r = Routed::new(
                    tr.clone(),
                    "V.1",
                    v1.line,
                    "SearchCubeCompact4700004<true>",
                    packed.cores,
                    "p.cores (not clamped to GetCoreNumAic)",
                );
                r.body = Some(format!(
                    "PackActiveWeightRowsInline + {}",
                    cfg_body(v1, "CubeFullKUnitBody<...>")
                ));
                r.staging = Some(staging_bytes(&packed));
                r.extra_scratch = Some((
                    "packedWeight = e*k*wstride".into(),
                    u64::from(packed.e) * u64::from(packed.k) * u64::from(packed.wstride),
                ));
                r.plan = Some(packed);
                r.notes.push(format!("packed.wstride = UpH(n,128) = {}", packed.wstride));
                return Some(Ok(r.with_pack(pack("V.1"))));
            }
            let v2 = inner("V.2");
            let (two, step) = eval_gate(v2, ctx, &mut packed);
            tr.push(step);
            if !two {
                // V's block falls through to W/X/Y, like any other failed arm.
                return None;
            }
            let mut r = Routed::new(
                tr.clone(),
                "V.2",
                v2.line,
                "SearchCubeCompact4700005<false>",
                packed.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some(format!(
                "PackActiveWeightRowsInline + {}",
                cfg_body(v2, "CubeFullKBodyScalarHalf<...>")
            ));
            r.staging = Some(staging_bytes(&packed) / 2);
            r.extra_scratch = Some((
                "packedWeight = e*k*wstride".into(),
                u64::from(packed.e) * u64::from(packed.k) * u64::from(packed.wstride),
            ));
            r.plan = Some(packed);
            r.notes.push(format!("packed.wstride = UpH(n,128) = {}", packed.wstride));
            r.with_pack(pack("V.2"))
        }

        // --- W (6088) --------------------------------------------------------
        "W" => {
            let mut r = Routed::new(
                tr.clone(),
                "W",
                gate.line,
                "SearchCubeCompact4700002<bfloat16_t,bfloat16_t>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some(cfg_body(gate, "CubeFullKUnitBody<bf16,bf16>"));
            r.staging = Some(staging_bytes(&selected));
            r.plan = Some(selected);
            r.notes.push(
                "LaunchNarrowSingleKSelected instantiates bf16 only (3670); relaxing W's leading \
                 `bf &&` loses fp16 correctness with no compile error [CURRENT.md landmine 3]"
                    .into(),
            );
            r.with_pack(pack("W"))
        }

        // --- X (6096): `p` was configured BY REFERENCE ------------------------
        "X" => {
            let clamped = p.cores.min(env.physical_aic);
            if clamped == 0 {
                return Some(Err("mix11::Launch: no physical AIC cores (3001)".into()));
            }
            let upper =
                (u64::from((p.m + 95) / 96) + u64::from(p.e) - 1) * u64::from((p.n + 127) / 128);
            let demote = clamped > 32 || upper > 4 * u64::from(clamped);
            if !demote {
                let mut q = *p;
                q.cores = clamped;
                q.vcores = clamped; // 1:1 MIX (3003)
                let kern = if bf {
                    "mix11::SearchP2Mix115300000<bfloat16_t,bfloat16_t>"
                } else {
                    "mix11::SearchP2Mix115300000<float,half>"
                };
                let mut r = Routed::new(
                    tr.clone(),
                    "X",
                    gate.line,
                    kern,
                    q.cores,
                    "min(p.cores, GetCoreNumAic())",
                );
                r.body = Some("CubeFullKBody<...,96,128,256,96,...> (MIX_AIC_1_1)".into());
                r.staging = Some(staging_bytes(&q));
                r.plan = Some(q);
                r.notes.push(format!(
                    "mix11::Upper(p) = {upper} <= 4*cores = {} -> stays in mix11 (3002)",
                    4 * clamped
                ));
                r.notes.push("p.vcores = p.cores (1:1 MIX), unlike the 2*cores everywhere else".into());
                return Some(Ok(r.with_pack(pack("X"))));
            }
            // packed32::LaunchPacked32 receives `incoming` -- the un-clamped Plan.
            let mut q = *p;
            q.cores = clamped;
            q.vcores = q.cores * 2;
            tr.push(Step {
                label: "X.p32".into(),
                line: 3002,
                outcome: Outcome::Pass,
                detail: format!(
                    "cores({clamped})>32 || Upper({upper})>4*cores({}) -> packed32::LaunchPacked32",
                    4 * clamped
                ),
                deps: vec![Dep::Cores],
            });
            if q.cores <= 32 {
                let kern = if bf {
                    "packed32::SearchP2Packed324300000<bfloat16_t,bfloat16_t>"
                } else {
                    "packed32::SearchP2Packed324300000<float,half>"
                };
                let mut r = Routed::new(
                    tr.clone(),
                    "X.p32",
                    2971,
                    kern,
                    q.cores,
                    "min(p.cores, GetCoreNumAic())",
                );
                r.staging = Some(staging_bytes(&q));
                r.plan = Some(q);
                return Some(Ok(r.with_pack(pack("X"))));
            }
            let kern = if bf {
                "packed32::SearchP2Fixed964100000<bfloat16_t,bfloat16_t>"
            } else {
                "packed32::SearchP2Fixed964100000<float,half>"
            };
            let mut r =
                Routed::new(tr.clone(), "X.f96", 2975, kern, q.cores, "min(p.cores, GetCoreNumAic())");
            r.staging = Some(staging_bytes(&q));
            r.plan = Some(q);
            r.notes.push("Plan by value -- no bit packing on this arm".into());
            r
        }

        // --- Y (6103) --------------------------------------------------------
        "Y" => {
            let mut r = Routed::new(
                tr.clone(),
                "Y",
                gate.line,
                "p1_half_pair::SearchP1HalfPairOutput11840034<float,half>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some(cfg_body(gate, "CubeFullKUnitBodyScalarHalf<float,half>"));
            r.staging = Some(staging_bytes(&selected) / 2);
            r.plan = Some(selected);
            r.notes.push("staging = CubeFullKStagingBytes(p)/2 (half-precision backend, 4217)".into());
            r.with_pack(pack("Y"))
        }

        // --- a gate the table has but this evaluator has no bespoke arm for ---
        _ => generic_arm(nodes, gate, ctx, &selected, tr),
    }))
}

/// The arm for a gate added as data -- a `--what-if` candidate, or a new gate
/// pasted into `cascade()` on 14 November.
///
/// Everything it reports comes from the table: the launcher name from the
/// gate's `Packer`, the tiling from its `ConfigureCubeFullK` term, the packed
/// fields and their admitted maxima from its own predicate. A new cube gate
/// therefore needs no Rust here at all; one that wants a different *grid* or a
/// second `Scratch` does, and the note says so.
fn generic_arm(
    nodes: &[Node],
    gate: &Gate,
    ctx: &Ctx,
    selected: &Plan,
    tr: &mut Vec<Step>,
) -> Routed {
    let launcher = gate.pack.map(|p| p.launcher).unwrap_or("<no launcher in the table>");
    let mut r = Routed::new(
        tr.clone(),
        gate.label,
        gate.line,
        launcher,
        selected.cores,
        "p.cores (generic arm: not clamped to GetCoreNumAic)",
    );
    if gate.cfg().is_some() {
        r.body = Some(cfg_body(gate, "CubeFullK*"));
        r.staging = Some(staging_bytes(selected));
    }
    r.plan = Some(*selected);
    r.notes.push(
        "generic arm: grid, dtype split and extra Scratch allocations are not table data -- \
         check them against the launcher before trusting the timing"
            .into(),
    );
    r.with_pack(cascade::pack_fields(nodes, gate.label, ctx))
}

/// `Name<...,BM,BN,BK,RV>` spelled from the gate's own `ConfigureCubeFullK`
/// template arguments, so the reported body cannot name a tile the gate does
/// not configure.
fn cfg_body(gate: &Gate, name: &str) -> String {
    match gate.cfg() {
        Some(Cfg { bm, bn, bk, rv, .. }) => format!("{name},{bm},{bn},{bk},{rv}"),
        None => name.to_string(),
    }
}

/// `tiny_capacity_unrolled::Select` (5337). `None` == the terminating
/// `throw std::invalid_argument` at 5414.
fn unrolled_select(k: u32, n: u32, bf: bool) -> Option<(u32, u32)> {
    if k <= 1 {
        if n <= 1 {
            return Some((1, 1));
        }
        if n <= 2 {
            return Some((1, 2));
        }
        if bf && n <= 4 {
            return Some((1, 4));
        }
    }
    if k <= 2 {
        if n <= 1 {
            return Some((2, 1));
        }
        if n <= 2 {
            return Some((2, 2));
        }
        if bf && n <= 4 {
            return Some((2, 4));
        }
    }
    if k <= 4 {
        if n <= 1 {
            return Some((4, 1));
        }
        if n <= 2 {
            return Some((4, 2));
        }
    }
    if k <= 8 && n <= 1 {
        return Some((8, 1));
    }
    None
}

/// `tiny_capacity_contract::Select` (5546). `None` == the terminating
/// `throw std::invalid_argument` at 5727.
fn contract_select(m: u32, k: u32, n: u32, bf: bool) -> Option<(u32, u32, u32)> {
    if m <= 1 {
        if let Some((ck, cn)) = unrolled_select(k, n, bf) {
            return Some((1, ck, cn));
        }
    }
    if m <= 2 {
        if k <= 1 {
            if n <= 1 {
                return Some((2, 1, 1));
            }
            if n <= 2 {
                return Some((2, 1, 2));
            }
            if bf && n <= 4 {
                return Some((2, 1, 4));
            }
        }
        if k <= 2 {
            if n <= 1 {
                return Some((2, 2, 1));
            }
            if n <= 2 {
                return Some((2, 2, 2));
            }
            if bf && n <= 4 {
                return Some((2, 2, 4));
            }
        }
        if k <= 4 {
            if n <= 1 {
                return Some((2, 4, 1));
            }
            if n <= 2 {
                return Some((2, 4, 2));
            }
        }
    }
    if m <= 4 {
        if k <= 1 {
            if n <= 1 {
                return Some((4, 1, 1));
            }
            if n <= 2 {
                return Some((4, 1, 2));
            }
        }
        if k <= 2 {
            if n <= 1 {
                return Some((4, 2, 1));
            }
            if n <= 2 {
                return Some((4, 2, 2));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// the 15 published points
// ---------------------------------------------------------------------------

pub struct Point {
    pub id: u32,
    pub shape: Shape,
}

pub fn points() -> Vec<Point> {
    let p = |id, e, m, k, n, gt, bf, counts: &[u32]| Point {
        id,
        shape: Shape {
            e,
            m,
            k,
            n,
            group_list_type: gt,
            bf,
            counts: counts.to_vec(),
        },
    };
    let mut p13 = Vec::new();
    for _ in 0..63 {
        p13.push(0);
        p13.push(1);
    }
    p13.push(0);
    p13.push(0);
    vec![
        p(1, 4, 4096, 768, 1024, 0, false, &[1024, 1024, 1024, 1024]),
        p(2, 4, 257, 511, 513, 1, true, &[63, 65, 64, 65]),
        p(3, 2, 3, 2, 2, 0, false, &[2, 1]),
        p(4, 8, 1027, 511, 769, 0, false, &[897, 1, 17, 0, 65, 31, 0, 16]),
        p(5, 5, 2, 2, 1, 0, true, &[0, 1, 0, 1, 0]),
        p(6, 1, 2, 2, 2, 0, false, &[2]),
        p(7, 1, 1, 8192, 2, 0, false, &[1]),
        p(8, 1, 1, 8192, 2, 1, true, &[1]),
        p(9, 4, 513, 383, 257, 1, true, &[1, 257, 0, 255]),
        p(10, 4, 1025, 1, 769, 0, false, &[1, 255, 0, 769]),
        p(11, 4, 1027, 769, 1, 1, true, &[1, 511, 0, 515]),
        p(12, 8, 768, 256, 512, 1, true, &[96; 8]),
        p(13, 128, 63, 255, 257, 1, true, &p13),
        p(14, 7, 3073, 1023, 1025, 1, true, &[383, 385, 511, 0, 769, 1, 1024]),
        p(15, 4, 4097, 767, 1025, 1, false, &[3073, 1, 0, 1023]),
    ]
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `gen/CURRENT.md`'s 15-point routing table, verbatim. If this ever fails,
    /// the discrepancy is a transcription bug here or an error in CURRENT.md --
    /// say which, do not edit the expectation.
    pub const TABLE: &[(u32, &str, u32, &str, u32)] = &[
        (1, "Y", 6103, "p1_half_pair::SearchP1HalfPairOutput11840034<float,half>", 20),
        (2, "X", 6096, "mix11::SearchP2Mix115300000<bfloat16_t,bfloat16_t>", 20),
        (3, "D", 5858, "expert_output::SearchTinyExpertOutput6400000<float,half>", 2),
        (4, "T", 6046, "SearchCubeCompact4700001<float,half>", 20),
        (5, "E'", 5876, "tiny_capacity_contract::SearchTinyCapacityContract11850017<true,2,2,1>", 1),
        (6, "E''", 5882, "tiny_capacity_unrolled::SearchTinyCapacityUnrolled11850012<false,2,2>", 1),
        (7, "F", 5888, "longdot_capacity::SearchLongDotCapacity11840033<8192,false,true>", 2),
        (8, "F", 5888, "longdot_capacity::SearchLongDotCapacity11840033<8192,true,true>", 2),
        (9, "W", 6088, "SearchCubeCompact4700002<bfloat16_t,bfloat16_t>", 20),
        (10, "R.1", 6015, "p10_fixed_ub::SearchP10FixedUb7300000", 16),
        (11, "P", 5994, "SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800>", 40),
        (12, "U", 6055, "SearchCubeCompact4700003<bfloat16_t,bfloat16_t>", 20),
        (13, "S", 6038, "BalancedOverlapShortGroupMatmulKernel<bfloat16_t,bfloat16_t>", 40),
        (14, "V.1", 6070, "SearchCubeCompact4700004<true>", 20),
        (15, "V.2", 6078, "SearchCubeCompact4700005<false>", 20),
    ];

    #[test]
    fn all_fifteen_points_match_current_md() {
        let env = Env::default();
        for (point, want) in points().iter().zip(TABLE) {
            assert_eq!(point.id, want.0, "point order");
            let r = route(&point.shape, &env).expect("routes");
            assert!(r.errors.is_empty(), "point {}: {:?}", point.id, r.errors);
            assert_eq!(r.gate, want.1, "point {} gate", point.id);
            assert_eq!(r.line, want.2, "point {} line", point.id);
            assert_eq!(r.kernel, want.3, "point {} kernel", point.id);
            assert_eq!(r.grid, want.4, "point {} grid", point.id);
        }
    }

    /// CURRENT.md's "关键参数" column, for the points where it states one.
    #[test]
    fn key_parameters_match_current_md() {
        let env = Env::default();
        let pts = points();
        let r1 = route(&pts[0].shape, &env).unwrap();
        let p1 = r1.plan.unwrap();
        assert_eq!((p1.kp, p1.n_tiles), (768, 8), "point 1 kp/nTiles");
        let r4 = route(&pts[3].shape, &env).unwrap();
        assert_eq!(r4.plan.unwrap().n_tiles, 7, "point 4 nTiles");
        let r9 = route(&pts[8].shape, &env).unwrap();
        assert_eq!(r9.plan.unwrap().n_tiles, 3, "point 9 nTiles");
        let r10 = route(&pts[9].shape, &env).unwrap();
        let v10 = r10.vtiling.unwrap();
        assert_eq!((v10.kc, v10.nb), (32, 784), "point 10 kc/nb");
        let r12 = route(&pts[11].shape, &env).unwrap();
        assert_eq!(r12.plan.unwrap().n_tiles, 4, "point 12 nTiles");
        let r13 = route(&pts[12].shape, &env).unwrap();
        let v13 = r13.vtiling.unwrap();
        assert_eq!((v13.kc, v13.nb), (128, 192), "point 13 kc/nb");
        for (idx, point) in [(13usize, 14u32), (14, 15)] {
            let r = route(&pts[idx].shape, &env).unwrap();
            assert_eq!(r.plan.unwrap().wstride, 1152, "point {point} wstride");
        }
        // Point 3's rowDims, as CURRENT.md states it -- now computed from the
        // packer's field list rather than from a second copy of the expression.
        let r3 = route(&pts[2].shape, &env).unwrap();
        assert!(r3.notes.iter().any(|n| n == "rowDims = 1158"), "{:?}", r3.notes);
    }

    #[test]
    fn point_two_stays_in_mix11() {
        let r = route(&points()[1].shape, &Env::default()).unwrap();
        assert!(r.notes.iter().any(|n| n.contains("Upper(p) = 30")), "{:?}", r.notes);
    }

    /// Six gates hard-assume cores<=20; CURRENT.md says a different core count
    /// silently demotes 1/4/9/12/14/15.
    #[test]
    fn a_different_core_count_demotes_six_points() {
        let env = Env { available_core_num: 24, physical_aic: 24, ..Env::default() };
        for (idx, want_demoted) in [(3usize, 4u32), (8, 9), (11, 12), (13, 14), (14, 15)] {
            let r = route(&points()[idx].shape, &env).unwrap();
            assert_eq!(r.gate, "Z", "point {want_demoted} should fall to Z at cores=24");
        }
        // Point 1 (gate Y) too -- its guard is `availableCoreNum<=20` (6103).
        assert_eq!(route(&points()[0].shape, &env).unwrap().gate, "Z");
    }

    /// CURRENT.md landmine 1: `p1_half_pair` packs `k/128-1` into 3 bits, and
    /// gate Y bounds k only through L1 budget arithmetic. Rewrite that
    /// arithmetic -- the thing a generator does -- and k=1152 must come back a
    /// hard error, not a route.
    ///
    /// Reading the real code turned up a correction to CURRENT.md: the bound is
    /// enforced *twice*, at 6105 and again inside `ConfigureCubeFullK` (2863),
    /// and the two agree to within 256 bytes. Dropping 6105 alone is harmless.
    #[test]
    fn dropping_the_l1_guards_overflows_the_k_field() {
        let shape = Shape {
            e: 4,
            m: 4096,
            k: 1152,
            n: 1024,
            group_list_type: 0,
            bf: false,
            counts: vec![1024; 4],
        };
        // With the guards: 512*1152 = 589824 > 524032 and > 524288, so Y fails
        // on both counts and Z catches the shape.
        let kept = route(&shape, &Env::default()).unwrap();
        assert_eq!(kept.gate, "Z");
        // Without them: Y fires and the 3-bit field cannot hold 1152/128-1 = 8.
        let dropped = route(&shape, &Env { l1_guard: false, ..Env::default() }).unwrap();
        assert_eq!(dropped.gate, "Y");
        assert!(
            dropped.errors.iter().any(|e| e.contains("k/128-1") && e.contains("overflow")),
            "{:?}",
            dropped.errors
        );
    }

    /// Points 3, 5 and 6 are the pointer-dependent ones.
    #[test]
    fn misaligned_pointers_reroute_the_tiny_points() {
        let env = Env { ptrs: Ptrs::misaligned(), ..Env::default() };
        let pts = points();
        // Point 3: gate D needs all six 64B aligned -> falls to B (misaligned
        // scales take the all-DMA staging path).
        assert_eq!(route(&pts[2].shape, &env).unwrap().gate, "B");
        // Point 5: B's misalignment disjunct now fires too.
        assert_eq!(route(&pts[4].shape, &env).unwrap().gate, "B");
        assert_eq!(route(&pts[5].shape, &env).unwrap().gate, "B");
        // Points 7 and 8 keep gate F but change kernel *variant* -- the pointer
        // dependence is inside longdot_capacity::Launch (3635), not in the gate.
        let p7 = route(&pts[6].shape, &env).unwrap();
        assert_eq!(p7.gate, "F");
        assert!(p7.kernel.contains("DMAScales"), "{}", p7.kernel);
        assert_eq!(
            route(&pts[7].shape, &env).unwrap().kernel,
            "longdot_capacity::SearchLongDotCapacity11840033<8192,true,false>"
        );
        // Points with no pointer-dependent gate on their path do not move.
        for idx in [0usize, 1, 8, 9, 10, 11, 12, 13, 14] {
            let base = route(&pts[idx].shape, &Env::default()).unwrap();
            let alt = route(&pts[idx].shape, &env).unwrap();
            assert_eq!(base.gate, alt.gate, "point {} moved", pts[idx].id);
        }
    }

    /// The capacity ladders terminate in a `throw`, so an out-of-domain tiny
    /// shape is an RE, not the silent WA CURRENT.md landmine 2 describes.
    #[test]
    fn tiny_ladders_have_a_terminating_throw() {
        assert_eq!(unrolled_select(8, 2, false), None);
        assert_eq!(contract_select(4, 4, 1, false), None);
        assert_eq!(contract_select(2, 2, 1, true), Some((2, 2, 1)));
    }

    #[test]
    fn every_live_point_packs_without_overflow() {
        let env = Env::default();
        for point in points() {
            let r = route(&point.shape, &env).unwrap();
            for field in &r.packing {
                assert!(field.fits(), "point {} field {}", point.id, field.name);
            }
        }
    }

    #[test]
    fn counts_sum_to_m_for_every_point() {
        for point in points() {
            assert_eq!(point.shape.counts.len() as u32, point.shape.e, "point {}", point.id);
            let sum: u32 = point.shape.counts.iter().sum();
            assert_eq!(sum, point.shape.m, "point {}", point.id);
        }
    }

    /// Acceptance (b): the table `emit` prints is the table `plan` evaluates,
    /// so routing against the emitted cascade is routing against this one. The
    /// check that makes that more than a sentence is in `emit.rs`
    /// (`emitted_run_kernel_routes_the_fifteen_points_identically`), which
    /// re-derives the fifteen rows from the rendered C++ markers.
    #[test]
    fn every_gate_the_evaluator_asks_for_exists_in_the_table() {
        // `route` panics on a missing label, so routing every point plus a few
        // shapes chosen to fall through to each tail arm covers the lookups.
        let env = Env::default();
        for point in points() {
            route(&point.shape, &env).unwrap();
        }
        let probes = [
            // K==1 but not R.1/R.2 -> R.3; M<=4 small -> L/M; N<=2, M==1 -> O.
            (4u32, 64u32, 1u32, 64u32, 0u32, false),
            (2, 4, 2, 2, 0, true),
            (1, 1, 64, 1, 0, false),
            (2, 3, 2, 4, 0, true),
            (1, 2, 4, 8, 0, false),
            (64, 64, 300, 300, 1, true),
        ];
        for (e, m, k, n, gt, bf) in probes {
            let shape = Shape { e, m, k, n, group_list_type: gt, bf, counts: vec![0; e as usize] };
            route(&shape, &env).unwrap();
        }
    }
}
