//! `gen plan` -- a dry run of `extern "C" run_kernel`.
//!
//! `kernel/kernel.asc:5814..6121` is a flat `if ... return` cascade. Widening or
//! narrowing one gate can silently reroute a *different* test point onto a
//! slower arm, and the only check we had was "compile and time all 15 points on
//! the NPU". This module transcribes the cascade so that check costs a
//! millisecond instead of a device slot.
//!
//! Faithfulness rules I held myself to:
//!
//!   * source order, one arm at a time, each tagged with the label
//!     `gen/CURRENT.md` gives it and the kernel.asc line number;
//!   * `ConfigureCubeFullK` is a function with the real arithmetic (2851), not a
//!     summary, because gates T/U/V/W/X/Y all branch on whether it succeeds;
//!   * gate X passes `p` **by reference** (6097) while T/U/V/W/Y operate on a
//!     `Plan` copy -- preserved, because it decides what Y and Z see;
//!   * the predicates that read *pointer values* rather than shapes are explicit
//!     inputs with documented defaults, and `plan` says which decisions used
//!     them and what the route would be under the opposite assumption.
//!
//! Label scheme (ours, per CURRENT.md): letters run in source order over every
//! numbered element of the cascade, so three letters land on derived booleans
//! rather than on branches -- C = `scalarOutputWindow` (5854), G =
//! `tinyCacheAligned` (5914), Q = the `Plan p` construction (6006), and V = the
//! `packWeights` bool (6065). That is the only assignment consistent with every
//! anchor CURRENT.md states (B=5845, D=5858, E'=5876, F=5888, H=5919, K.1=5951,
//! L.1=5967, N=5983, R.1=6015, V=6065, X=6096, Y=6103, Z=6112).

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

/// One field of a packed `uint32_t dims`. `gate_max` is the largest *encoded*
/// value the gate that reaches this packer permits; `None` means the gate does
/// not bound the packed quantity at all, which is the CURRENT.md landmine-1
/// class of bug.
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

fn f(
    name: &'static str,
    shift: u32,
    width: u32,
    value: u64,
    gate_max: Option<u64>,
) -> Field {
    Field { name, shift, width, value, gate_max, incidental: None }
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dep {
    Shape,
    /// Reads a device pointer value.
    Ptr,
    /// Reads `availableCoreNum` or `GetCoreNumAic()`.
    Cores,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    Pass,
    Fail,
    /// A derived value, not a branch (labels C, G, Q, V).
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

/// Builds one gate's conjunction, remembering the first failing term.
struct G {
    label: String,
    line: u32,
    ok: bool,
    first_fail: Option<String>,
    deps: Vec<Dep>,
    summary: Vec<String>,
}

impl G {
    fn new(label: &str, line: u32) -> G {
        G {
            label: label.to_string(),
            line,
            ok: true,
            first_fail: None,
            deps: Vec::new(),
            summary: Vec::new(),
        }
    }
    fn term(&mut self, expr: impl Into<String>, ok: bool, dep: Dep) -> &mut G {
        let expr = expr.into();
        // Record the dependency only if `&&` actually reaches this term. A gate
        // that fails on a shape term never reads a pointer, so claiming it as
        // pointer-dependent would be noise in exactly the report that matters.
        if self.ok && dep != Dep::Shape && !self.deps.contains(&dep) {
            self.deps.push(dep);
        }
        if !ok && self.first_fail.is_none() {
            self.first_fail = Some(expr.clone());
            self.ok = false;
        }
        if self.ok {
            self.summary.push(expr);
        }
        self
    }
    fn t(&mut self, expr: impl Into<String>, ok: bool) -> &mut G {
        self.term(expr, ok, Dep::Shape)
    }
    fn p(&mut self, expr: impl Into<String>, ok: bool) -> &mut G {
        self.term(expr, ok, Dep::Ptr)
    }
    fn c(&mut self, expr: impl Into<String>, ok: bool) -> &mut G {
        self.term(expr, ok, Dep::Cores)
    }
    fn passed(&self) -> bool {
        self.ok
    }
    fn step(&self) -> Step {
        Step {
            label: self.label.clone(),
            line: self.line,
            outcome: if self.ok { Outcome::Pass } else { Outcome::Fail },
            detail: if self.ok {
                self.summary.join(" && ")
            } else {
                self.first_fail.clone().unwrap_or_default()
            },
            deps: self.deps.clone(),
        }
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
// the cascade
// ---------------------------------------------------------------------------

pub fn route(s: &Shape, env: &Env) -> Result<Routed, String> {
    let mut tr: Vec<Step> = Vec::new();
    let (m, k, n, e) = (s.m, s.k, s.n, s.e);
    let bf = s.bf;
    let gt = s.group_list_type;
    let cores64 = u64::from(env.available_core_num);
    let ptr = &env.ptrs;

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

    // --- B: non-natural typed input views -> TinyShortReduction (5845) --------
    {
        let mut g = G::new("B", 5845);
        g.t(format!("K({k})<=8"), k <= 8)
            .t(format!("E({e})<=16"), e <= 16)
            .t(format!("M*N({})<=64", u64::from(m) * u64::from(n)), u64::from(m) * u64::from(n) <= 64)
            .p(
                "any-of(xScale&3, weightScale&(bf?1:3), groupList&7) != 0",
                (ptr.x_scale & 3) != 0
                    || (ptr.w_scale & if bf { 1 } else { 3 }) != 0
                    || (ptr.group_list & 7) != 0,
            );
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            return Ok(Routed::new(
                tr,
                "B",
                5845,
                &format!("TinyShortReduction<{}>", bf),
                1,
                "<<<1>>>",
            )
            .with_pack(vec![
                f("M-1", 0, 6, u64::from(m - 1), Some(63)),
                f("N-1", 6, 6, u64::from(n - 1), Some(63)),
                f("K-1", 12, 3, u64::from(k - 1), Some(7)),
                f("E-1", 15, 4, u64::from(e - 1), Some(15)),
                f("gt", 19, 1, u64::from(gt), Some(1)),
            ]));
        }
    }

    // --- C: scalarOutputWindow (5854) ----------------------------------------
    let scalar_output_window = u64::from(ptr.y & 127) + u64::from(m) * u64::from(n) * 2 <= 32;
    tr.push(Step {
        label: "C".into(),
        line: 5854,
        outcome: Outcome::Note,
        detail: format!(
            "scalarOutputWindow = (y&127={}) + M*N*2={} <= 32 -> {scalar_output_window}",
            ptr.y & 127,
            u64::from(m) * u64::from(n) * 2
        ),
        deps: vec![Dep::Ptr],
    });

    let six_aligned_64 = (ptr.any_of_six() & 63) == 0;

    // --- D: 2x2 register microtile, expert-parallel (5858) --------------------
    {
        let mut g = G::new("D", 5858);
        g.t("!bf", !bf)
            .t("groupListType==0", gt == 0)
            .t(format!("E({e})>=2"), e >= 2)
            .t(format!("E({e})<=M({m})"), e <= m)
            .t(format!("M({m})<=4"), m <= 4)
            .c(format!("E<=2*cores({})", 2 * cores64), u64::from(e) <= 2 * cores64)
            .t(format!("K({k})==2"), k == 2)
            .t(format!("N({n})==2"), n == 2)
            .p("scalarOutputWindow", scalar_output_window)
            .p("all six pointers 64B aligned", six_aligned_64);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let row_dims = u64::from(m - 1)
                | (u64::from(n - 1) << 2)
                | (u64::from(k - 1) << 7)
                | (u64::from(e - 1) << 10)
                | (u64::from(gt) << 17);
            let mut r = Routed::new(
                tr,
                "D",
                5858,
                "expert_output::SearchTinyExpertOutput6400000<float,half>",
                e,
                "<<<E>>>",
            )
            .with_pack(vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(1)),
                f("K-1", 7, 3, u64::from(k - 1), Some(1)),
                f("E-1", 10, 7, u64::from(e - 1), Some(3)),
                f("gt", 17, 1, 0, Some(0)),
            ]);
            r.notes.push(format!("rowDims = {row_dims}"));
            r.notes.push("AIV path: launches asynchronously, no aclrtSynchronizeStream".into());
            return Ok(r);
        }
    }

    // --- E: single register-capacity algorithm (5871) --------------------------
    {
        let mut g = G::new("E", 5871);
        g.t(format!("M*K({})<=8", u64::from(m) * u64::from(k)), u64::from(m) * u64::from(k) <= 8)
            .t(format!("K*N({})<=8", u64::from(k) * u64::from(n)), u64::from(k) * u64::from(n) <= 8)
            .t(format!("M*4({})<=16", u64::from(m) * 4), u64::from(m) * 4 <= 16)
            .t(
                format!("N*(bf?2:4)({})<=8", u64::from(n) * if bf { 2 } else { 4 }),
                u64::from(n) * if bf { 2 } else { 4 } <= 8,
            )
            .t(
                format!("M*N*2({})<=16", u64::from(m) * u64::from(n) * 2),
                u64::from(m) * u64::from(n) * 2 <= 16,
            )
            .p("scalarOutputWindow", scalar_output_window)
            .p("y&1==0", (ptr.y & 1) == 0);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let dims = u64::from(m - 1)
                | (u64::from(k - 1) << 2)
                | (u64::from(n - 1) << 5)
                | (u64::from(e - 1) << 7)
                | (u64::from(gt) << 14);
            let pack = vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("K-1", 2, 3, u64::from(k - 1), Some(7)),
                f("N-1", 5, 2, u64::from(n - 1), Some(3)),
                f("E-1", 7, 7, u64::from(e - 1), Some(127)),
                f("gt", 14, 1, u64::from(gt), Some(1)),
            ];
            // E': naturally aligned single-line groupList batch -> contract.
            let mut g2 = G::new("E'", 5876);
            g2.t(format!("E({e})>=2"), e >= 2)
                .t(format!("E({e})<=8"), e <= 8)
                .p("groupList&7==0", (ptr.group_list & 7) == 0)
                .p(
                    format!("(groupList&63={})+E*8<=64", ptr.group_list & 63),
                    u64::from(ptr.group_list & 63) + u64::from(e) * 8 <= 64,
                );
            let contract = g2.passed();
            tr.push(g2.step());
            // The Select ladders decode from `dims`, so they see truncated m/k/n.
            let dm = (dims & 3) as u32 + 1;
            let dk = ((dims >> 2) & 7) as u32 + 1;
            let dn = ((dims >> 5) & 3) as u32 + 1;
            if contract {
                let arm = contract_select(dm, dk, dn, bf);
                let mut r = match arm {
                    Some((cm, ck, cn)) => Routed::new(
                        tr,
                        "E'",
                        5876,
                        &format!(
                            "tiny_capacity_contract::SearchTinyCapacityContract11850017<{bf},{cm},{ck},{cn}>"
                        ),
                        1,
                        "<<<1>>>",
                    ),
                    None => {
                        let mut r = Routed::new(tr, "E'", 5876, "tiny_capacity_contract::Select -- no arm", 0, "none");
                        r.errors.push(format!(
                            "tiny_capacity_contract::Select (5546) has no arm for (m={dm},k={dk},n={dn},bf={bf}); \
                             it throws std::invalid_argument at 5727 -> RE"
                        ));
                        r
                    }
                }
                .with_pack(pack);
                r.notes.push(format!("dims = {dims}; Select decodes m={dm} k={dk} n={dn}"));
                return Ok(r);
            }
            let arm = unrolled_select(dk, dn, bf);
            let mut r = match arm {
                Some((ck, cn)) => Routed::new(
                    tr,
                    "E''",
                    5882,
                    &format!(
                        "tiny_capacity_unrolled::SearchTinyCapacityUnrolled11850012<{bf},{ck},{cn}>"
                    ),
                    1,
                    "<<<1>>>",
                ),
                None => {
                    let mut r = Routed::new(tr, "E''", 5882, "tiny_capacity_unrolled::Select -- no arm", 0, "none");
                    r.errors.push(format!(
                        "tiny_capacity_unrolled::Select (5337) has no arm for (k={dk},n={dn},bf={bf}); \
                         it throws std::invalid_argument at 5414 -> RE"
                    ));
                    r
                }
            }
            .with_pack(pack);
            r.notes.push(format!("dims = {dims}; Select decodes k={dk} n={dn}"));
            return Ok(r);
        }
    }

    // --- F: one row, two interleaved channels (5888) --------------------------
    {
        let mut g = G::new("F", 5888);
        g.t(format!("E({e})==1"), e == 1)
            .t(format!("M({m})==1"), m == 1)
            .t(format!("N({n})==2"), n == 2)
            .t(format!("K({k})>=1024"), k >= 1024)
            .t(format!("K({k})<=8192"), k <= 8192);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let scalar = (ptr.y & 31) == 0 && scalar_output_window;
            let cap: u32 = if k <= 1024 {
                1024
            } else if k <= 2048 {
                2048
            } else if k <= 4096 {
                4096
            } else {
                8192
            };
            // longdot_capacity::Launch (3635): fp16 with misaligned scales takes
            // the DMA variant; bf16 has no DMA variant.
            let dma = !bf && ((ptr.x_scale | ptr.w_scale) & 3) != 0;
            let kernel = if dma {
                format!("longdot_capacity::SearchLongDotCapacityDMAScales<{cap},{scalar}>")
            } else {
                format!("longdot_capacity::SearchLongDotCapacity11840033<{cap},{bf},{scalar}>")
            };
            let mut r = Routed::new(tr, "F", 5888, &kernel, 2, "<<<2>>>");
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
            return Ok(r);
        }
    }

    // --- G: tinyCacheAligned (5914) -------------------------------------------
    tr.push(Step {
        label: "G".into(),
        line: 5914,
        outcome: Outcome::Note,
        detail: format!("tinyCacheAligned = all six pointers 64B aligned -> {six_aligned_64}"),
        deps: vec![Dep::Ptr],
    });

    let m_n = u64::from(m) * u64::from(n);
    let m_k = u64::from(m) * u64::from(k);
    let k_n = u64::from(k) * u64::from(n);
    let n_bytes = u64::from(n) * if bf { 2 } else { 4 };

    // --- H: SearchTinyE1STDev3400000 (5919) -----------------------------------
    {
        let mut g = G::new("H", 5919);
        g.p("scalarOutputWindow", scalar_output_window)
            .t(format!("E({e})==1"), e == 1)
            .t(format!("M({m})<=4"), m <= 4)
            .c(format!("M<=2*cores({})", 2 * cores64), u64::from(m) <= 2 * cores64)
            .t(format!("N({n})<=32"), n <= 32)
            .t(format!("K({k})<=8"), k <= 8)
            .t(format!("M*N({m_n})<=32"), m_n <= 32)
            .t(format!("M*K({m_k})<=64"), m_k <= 64)
            .t(format!("K*N({k_n})<=64"), k_n <= 64)
            .t(format!("M*4({})<=64", u64::from(m) * 4), u64::from(m) * 4 <= 64)
            .t(format!("N*(bf?2:4)({n_bytes})<=64"), n_bytes <= 64)
            .t(format!("N({n})<=2"), n <= 2);
        if g.passed() {
            g.p(
                format!("y&(2N-1)=={}", ptr.y & (n * 2 - 1)),
                (ptr.y & (n * 2 - 1)) == 0,
            );
        }
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let kern = if bf {
                "SearchTinyE1STDev3400000<bfloat16_t,bfloat16_t>"
            } else {
                "SearchTinyE1STDev3400000<float,half>"
            };
            let mut r = Routed::new(tr, "H", 5919, kern, m, "<<<M>>>").with_pack(vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(1)),
                f("K-1", 7, 3, u64::from(k - 1), Some(7)),
                f("gt", 17, 1, u64::from(gt), Some(1)),
            ]);
            r.notes.push(
                "rowDims omits the E field (5923) while the decoder reads bits 10..16 (4957); \
                 correct only because H requires E==1 [CURRENT.md landmine 6]"
                    .into(),
            );
            return Ok(r);
        }
    }

    // --- I: E1FixedUbRowTiny (5928) -------------------------------------------
    {
        let mut g = G::new("I", 5928);
        g.t(format!("E({e})==1"), e == 1)
            .t(format!("M({m})<=4"), m <= 4)
            .c(format!("M<=2*cores({})", 2 * cores64), u64::from(m) <= 2 * cores64)
            .t(format!("N({n})<=32"), n <= 32)
            .t(format!("K({k})<=8"), k <= 8)
            .t(format!("M*N({m_n})<=32"), m_n <= 32)
            .t(format!("M*K({m_k})<=64"), m_k <= 64)
            .t(format!("K*N({k_n})<=64"), k_n <= 64)
            .t(format!("M*4({})<=64", u64::from(m) * 4), u64::from(m) * 4 <= 64)
            .t(format!("N*(bf?2:4)({n_bytes})<=64"), n_bytes <= 64);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let kern = if bf {
                "E1FixedUbRowTiny<bfloat16_t,bfloat16_t>"
            } else {
                "E1FixedUbRowTiny<float,half>"
            };
            return Ok(Routed::new(tr, "I", 5928, kern, m, "<<<M>>>").with_pack(vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(31)),
                f("K-1", 7, 3, u64::from(k - 1), Some(7)),
                f("gt", 17, 1, u64::from(gt), Some(1)),
            ]));
        }
    }

    // --- J: TinyCacheReduction (5937) -----------------------------------------
    {
        let mut g = G::new("J", 5937);
        g.t(format!("E({e})==1"), e == 1)
            .t(format!("K({k})<=8"), k <= 8)
            .t(format!("M*N({m_n})<=32"), m_n <= 32)
            .t(format!("M*K({m_k})<=64"), m_k <= 64)
            .t(
                format!("E*K*N({})<=64", u64::from(e) * k_n),
                u64::from(e) * k_n <= 64,
            )
            .t(format!("M*4({})<=64", u64::from(m) * 4), u64::from(m) * 4 <= 64)
            .t(
                format!("E*N*(bf?2:4)({})<=64", u64::from(e) * n_bytes),
                u64::from(e) * n_bytes <= 64,
            )
            .p("tinyCacheAligned", six_aligned_64);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let mut r = Routed::new(tr, "J", 5937, &format!("TinyCacheReduction<{bf}>"), 1, "<<<1>>>")
                .with_pack(vec![
                    f("M-1", 0, 4, u64::from(m - 1), Some(3)),
                    f("N-1", 4, 5, u64::from(n - 1), Some(31)),
                    f("K-1", 9, 3, u64::from(k - 1), Some(7)),
                    f("E-1", 12, 3, u64::from(e - 1), Some(0)),
                    f("gt", 15, 1, u64::from(gt), Some(1)),
                ]);
            r.notes.push(
                "tinyCacheDims packs M-1 in 4 bits and E-1 in 3 bits (5940); both overflows are \
                 blocked only by unrelated clauses [CURRENT.md landmine 6]"
                    .into(),
            );
            return Ok(r);
        }
    }

    // --- K: GT0 expert intervals (5949) ---------------------------------------
    {
        let mut g = G::new("K", 5949);
        g.t("groupListType==0", gt == 0)
            .t(format!("E({e})>=2"), e >= 2)
            .t(format!("E({e})<=M({m})"), e <= m)
            .t(format!("M({m})<=4"), m <= 4)
            .c(format!("E<=2*cores({})", 2 * cores64), u64::from(e) <= 2 * cores64)
            .t(format!("K({k})<=8"), k <= 8)
            .t(format!("N({n})<=32"), n <= 32)
            .t(format!("M*N({m_n})<=32"), m_n <= 32);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let pack = vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(31)),
                f("K-1", 7, 3, u64::from(k - 1), Some(7)),
                f("E-1", 10, 7, u64::from(e - 1), Some(3)),
                f("gt", 17, 1, 0, Some(0)),
            ];
            if k == 2 && n == 2 {
                let mut g1 = G::new("K.1", 5951);
                g1.t("K==2 && N==2", true)
                    .t("!bf", !bf)
                    .p("tinyCacheAligned", six_aligned_64)
                    .p("scalarOutputWindow", scalar_output_window);
                let one = g1.passed();
                tr.push(g1.step());
                if one {
                    return Ok(Routed::new(
                        tr,
                        "K.1",
                        5951,
                        "expert_output::SearchTinyExpertOutput6400000<float,half>",
                        e,
                        "<<<E>>>",
                    )
                    .with_pack(pack));
                }
                let kern = if bf {
                    "expert_reuse::SearchTinyExpertReuse6300000<bfloat16_t,bfloat16_t>"
                } else {
                    "expert_reuse::SearchTinyExpertReuse6300000<float,half>"
                };
                return Ok(Routed::new(tr, "K.2", 5956, kern, e, "<<<E>>>").with_pack(pack));
            }
            let kern = if bf {
                "SearchTinyExpert3500000<bfloat16_t,bfloat16_t>"
            } else {
                "SearchTinyExpert3500000<float,half>"
            };
            return Ok(Routed::new(tr, "K", 5949, kern, e, "<<<E>>>").with_pack(pack));
        }
    }

    // --- L: one vector core per short output row (5965) -----------------------
    {
        let mut g = G::new("L", 5965);
        g.t(format!("M({m})<=4"), m <= 4)
            .c(format!("M<=2*cores({})", 2 * cores64), u64::from(m) <= 2 * cores64)
            .t(format!("N({n})<=32"), n <= 32)
            .t(format!("K({k})<=8"), k <= 8);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let pack = vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(31)),
                f("K-1", 7, 3, u64::from(k - 1), Some(7)),
                f("E-1", 10, 7, u64::from(e - 1), Some(127)),
                f("gt", 17, 1, u64::from(gt), Some(1)),
            ];
            let mut g1 = G::new("L.1", 5967);
            g1.t("bf", bf)
                .t(format!("N({n})==1"), n == 1)
                .p("y&1==0", (ptr.y & 1) == 0)
                .p("scalarOutputWindow", scalar_output_window);
            let one = g1.passed();
            tr.push(g1.step());
            if one {
                return Ok(Routed::new(
                    tr,
                    "L.1",
                    5967,
                    "tiny_row_stdev::SearchTinyRowSTDev3400000<bfloat16_t,bfloat16_t>",
                    m,
                    "<<<M>>>",
                )
                .with_pack(pack));
            }
            let kern = if bf {
                "RowParallelTiny<bfloat16_t,bfloat16_t>"
            } else {
                "RowParallelTiny<float,half>"
            };
            return Ok(Routed::new(tr, "L", 5965, kern, m, "<<<M>>>").with_pack(pack));
        }
    }

    // --- M: same family under a lower core budget (5976) ----------------------
    {
        let mut g = G::new("M", 5976);
        g.t(format!("M({m})<=4"), m <= 4)
            .c(format!("M>2*cores({})", 2 * cores64), u64::from(m) > 2 * cores64)
            .t(format!("N({n})<=32"), n <= 32)
            .t(format!("K({k})<=8"), k <= 8);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let blocks = u64::from(m).min(2 * cores64) as u32;
            let kern = if bf {
                "BudgetStridedRowTiny<bfloat16_t,bfloat16_t>"
            } else {
                "BudgetStridedRowTiny<float,half>"
            };
            return Ok(Routed::new(tr, "M", 5976, kern, blocks, "min(M, 2*cores)").with_pack(vec![
                f("M-1", 0, 2, u64::from(m - 1), Some(3)),
                f("N-1", 2, 5, u64::from(n - 1), Some(31)),
                f("K-1", 7, 3, u64::from(k - 1), Some(7)),
                f("E-1", 10, 7, u64::from(e - 1), Some(127)),
                f("gt", 17, 1, u64::from(gt), Some(1)),
            ]));
        }
    }

    // --- N: TinyShortReduction fallback (5983) --------------------------------
    {
        let mut g = G::new("N", 5983);
        g.t(format!("M*N({m_n})<=64"), m_n <= 64)
            .t(format!("K({k})<=8"), k <= 8)
            .t(format!("E({e})<=16"), e <= 16);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            return Ok(Routed::new(
                tr,
                "N",
                5983,
                &format!("TinyShortReduction<{bf}>"),
                1,
                "<<<1>>>",
            )
            .with_pack(vec![
                f("M-1", 0, 6, u64::from(m - 1), Some(63)),
                f("N-1", 6, 6, u64::from(n - 1), Some(63)),
                f("K-1", 12, 3, u64::from(k - 1), Some(7)),
                f("E-1", 15, 4, u64::from(e - 1), Some(15)),
                f("gt", 19, 1, u64::from(gt), Some(1)),
            ]));
        }
    }

    // --- O: M==1, only K parallel (5988) --------------------------------------
    {
        let mut g = G::new("O", 5988);
        g.t(format!("M({m})==1"), m == 1).t(format!("N({n})<=2"), n <= 2);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let blocks = (2 * cores64).min(u64::from(n)) as u32;
            let kern = if bf {
                "IntegerProductLongNarrowDot<bfloat16_t,bfloat16_t>"
            } else {
                "IntegerProductLongNarrowDot<float,half>"
            };
            let mut r = Routed::new(tr, "O", 5988, kern, blocks, "min(2*cores, N)");
            r.notes.push("GeneralVectorDims passed by value -- no bit packing".into());
            return Ok(r);
        }
    }

    // --- P: N==1 matvec (5994) -------------------------------------------------
    {
        let mut g = G::new("P", 5994);
        g.t(format!("N({n})==1"), n == 1);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let blocks = (2 * cores64).min(u64::from(m)) as u32;
            if k <= 800 {
                let kern = if bf {
                    "SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800>"
                } else {
                    "SearchN1FixedLayoutSafe7100000<float,half,32,800>"
                };
                let mut r = Routed::new(tr, "P", 5994, kern, blocks, "min(2*cores, M)");
                r.notes.push(format!("K({k})<=800 (5997): 800 is the UB column capacity C"));
                r.notes.push("GeneralVectorDims by value -- no bit packing; fixed UB 184672 B".into());
                return Ok(r);
            }
            let kern = if bf { "RowDot<16,bfloat16_t,bfloat16_t>" } else { "RowDot<16,float,half>" };
            let mut r = Routed::new(tr, "P.2", 6002, kern, blocks, "min(2*cores, M)");
            r.notes.push(format!("K({k})>800 -> RowDot<16>"));
            return Ok(r);
        }
    }

    // --- Q: Plan p (6006) ------------------------------------------------------
    let mut p = Plan {
        m,
        k,
        n,
        e,
        list_type: gt,
        kp: up(k, 32),
        wstride: n,
        cores: env.available_core_num,
        vcores: env.available_core_num * 2,
        ..Default::default()
    };
    tr.push(Step {
        label: "Q".into(),
        line: 6006,
        outcome: Outcome::Note,
        detail: format!(
            "Plan p: kp=UpH(k,32)={} wstride=n={} cores={} vcores={}",
            p.kp, p.wstride, p.cores, p.vcores
        ),
        deps: vec![Dep::Cores],
    });

    // --- R: K==1 outer product (6010) -----------------------------------------
    {
        let mut g = G::new("R", 6010);
        g.t(format!("k({})==1", p.k), p.k == 1);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let selected_pitch = up(p.n, 16);
            let selected_nup = up(p.n, 32);
            let selected_ub = 32u64 * u64::from(selected_pitch) * 6
                + 12 * u64::from(selected_nup)
                + 80 * 32
                + u64::from(selected_nup)
                + 32
                + 32
                + 32;
            let mut g1 = G::new("R.1", 6015);
            g1.t("!bf", !bf)
                .t(format!("m({})>=256", p.m), p.m >= 256)
                .t(format!("n({})>=512", p.n), p.n >= 512)
                .t(format!("n({})<=1024", p.n), p.n <= 1024)
                .t(format!("selectedPitch/8({})<=255", selected_pitch / 8), selected_pitch / 8 <= 255)
                .t(format!("selectedUb({selected_ub})<=188160"), selected_ub <= 188160);
            let one = g1.passed();
            tr.push(g1.step());
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
                    nb: selected_pitch,
                };
                let mut r = Routed::new(
                    tr,
                    "R.1",
                    6015,
                    "p10_fixed_ub::SearchP10FixedUb7300000",
                    core_num,
                    "min(min(vcores, ceil(m/32)+e), 16)",
                );
                r.vtiling = Some(vt);
                r.plan = Some(p);
                r.notes.push(format!("selectedUb = {selected_ub} B"));
                r.notes.push("VTiling by value -- no bit packing".into());
                return Ok(r);
            }
            let mut g2 = G::new("R.2", 6025);
            g2.t("!bf", !bf)
                .t(format!("m({})>=256", p.m), p.m >= 256)
                .t(format!("n({})>=512", p.n), p.n >= 512)
                .t(format!("n({})<=1024", p.n), p.n <= 1024);
            let two = g2.passed();
            tr.push(g2.step());
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
                    tr,
                    "R.2",
                    6025,
                    "PublishedOuter",
                    core_num,
                    "min(vcores, ceil(m/20)+e)",
                );
                r.vtiling = Some(vt);
                r.plan = Some(p);
                return Ok(r);
            }
            tr.push(Step {
                label: "R.3".into(),
                line: 6032,
                outcome: Outcome::Pass,
                detail: "unconditional K==1 fallback: BuildOuterTiling + GmmOuterKernel".into(),
                deps: vec![],
            });
            let vt = build_outer_tiling(&p);
            let kern = if bf {
                "GmmOuterKernel<bfloat16_t,bfloat16_t>"
            } else {
                "GmmOuterKernel<float,half>"
            };
            let mut r = Routed::new(tr, "R.3", 6032, kern, vt.core_num, "BuildOuterTiling.coreNum");
            r.vtiling = Some(vt);
            r.plan = Some(p);
            r.notes.push(
                "BuildOuterTiling (947) never checks ubBytes(1) <= 188160; its loop condition is \
                 rows>1 [CURRENT.md landmine 7]"
                    .into(),
            );
            return Ok(r);
        }
    }

    // --- S: many short experts (6038) ------------------------------------------
    {
        let mut g = G::new("S", 6038);
        g.t(format!("e({})>=32", p.e), p.e >= 32)
            .t(format!("m({})<=256", p.m), p.m <= 256)
            .t(format!("m({})<=2*e({})", p.m, 2 * p.e), p.m <= 2 * p.e)
            .t(format!("k({})<=512", p.k), p.k <= 512)
            .t(format!("n({})<=512", p.n), p.n <= 512);
        let pass = g.passed();
        tr.push(g.step());
        if pass {
            let vt = build_bosg(&p);
            let kern = if bf {
                "BalancedOverlapShortGroupMatmulKernel<bfloat16_t,bfloat16_t>"
            } else {
                "BalancedOverlapShortGroupMatmulKernel<float,half>"
            };
            let mut r = Routed::new(tr, "S", 6038, kern, vt.core_num, "min(vcores, m)");
            r.notes.push(format!("nPitch = UpH(n,32) = {}", (p.n + 31) / 32 * 32));
            r.notes.push("pure AIV; VTiling by value -- no bit packing".into());
            r.vtiling = Some(vt);
            r.plan = Some(p);
            return Ok(r);
        }
    }

    let rows_per_expert = p.m / p.e;

    // --- T: medium short experts, FP16 scalar-half (6046) ----------------------
    {
        let mut g = G::new("T", 6046);
        g.t("!bf", !bf)
            .c(format!("cores({})<=20", p.cores), p.cores <= 20)
            .t(format!("e({})>1", p.e), p.e > 1)
            .t(format!("e({})<=16", p.e), p.e <= 16)
            .t(format!("m/e({rows_per_expert})>=96"), rows_per_expert >= 96)
            .t(format!("m/e({rows_per_expert})<256"), rows_per_expert < 256)
            .t(format!("k({})>256", p.k), p.k > 256)
            .t(format!("k({})<=512", p.k), p.k <= 512)
            .t(format!("n({})>640", p.n), p.n > 640)
            .t(format!("n({})<=1024", p.n), p.n <= 1024)
            .t(
                format!("(k%128({}) || n%128({}))", p.k % 128, p.n % 128),
                p.k % 128 != 0 || p.n % 128 != 0,
            );
        let mut pass = g.passed();
        let mut selected = p; // copy, not a reference
        if pass {
            let ok = configure_cube_full_k(&mut selected, 192, 128, 96, 128, true, false, env.l1_guard);
            g.t("ConfigureCubeFullK<192,128,96,128,Prefetch,->", ok);
            pass = ok;
        }
        tr.push(g.step());
        if pass {
            let mut r = Routed::new(
                tr,
                "T",
                6046,
                "SearchCubeCompact4700001<float,half>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some("CubeFullKBodyScalarHalf<float,half,192,128,96,128>".into());
            r.staging = Some(staging_bytes(&selected) / 2);
            r.plan = Some(selected);
            return Ok(r.with_pack(vec![
                f("k-1", 0, 9, u64::from(selected.k - 1), Some(511)),
                f("n-1", 9, 10, u64::from(selected.n - 1), Some(1023)),
                f("e-1", 19, 4, u64::from(selected.e - 1), Some(15)),
                f("cores-1", 23, 5, u64::from(selected.cores - 1), Some(19)),
                f("gt", 28, 1, u64::from(selected.list_type), Some(1)),
            ]));
        }
    }

    // --- U: aligned medium int32 backend (6055) --------------------------------
    {
        let mut g = G::new("U", 6055);
        g.c(format!("cores({})<=20", p.cores), p.cores <= 20)
            .t(format!("e({})>1", p.e), p.e > 1)
            .t(format!("e({})<=16", p.e), p.e <= 16)
            .t(format!("m/e({rows_per_expert})>=64"), rows_per_expert >= 64)
            .t(format!("m/e({rows_per_expert})<256"), rows_per_expert < 256)
            .t(format!("n({})>=384", p.n), p.n >= 384)
            .t(format!("n({})<=640", p.n), p.n <= 640)
            .t(format!("n%128({})==0", p.n % 128), p.n % 128 == 0)
            .t(format!("k({})>=128", p.k), p.k >= 128)
            .t(format!("k({})<=512", p.k), p.k <= 512)
            .t(format!("k%128({})==0", p.k % 128), p.k % 128 == 0);
        let mut pass = g.passed();
        let mut selected = p;
        if pass {
            let ok = configure_cube_full_k(&mut selected, 64, 128, 128, 64, true, false, env.l1_guard);
            g.t("ConfigureCubeFullK<64,128,128,64,Prefetch,->", ok);
            pass = ok;
        }
        tr.push(g.step());
        if pass {
            let kern = if bf {
                "SearchCubeCompact4700003<bfloat16_t,bfloat16_t>"
            } else {
                "SearchCubeCompact4700003<float,half>"
            };
            let mut r = Routed::new(
                tr,
                "U",
                6055,
                kern,
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some("CubeFullKBody<...,64,128,128,64,2u,true>".into());
            r.staging = Some(staging_bytes(&selected));
            r.plan = Some(selected);
            return Ok(r.with_pack(vec![
                f("k/128-1", 0, 2, u64::from(selected.k / 128 - 1), Some(3)),
                f("n/128-1", 2, 3, u64::from(selected.n / 128 - 1), Some(4)),
                f("e-1", 5, 4, u64::from(selected.e - 1), Some(15)),
                f("cores-1", 9, 5, u64::from(selected.cores - 1), Some(19)),
                f("gt", 14, 1, u64::from(selected.list_type), Some(1)),
            ]));
        }
    }

    // --- V: packWeights (6065), then V.1 (6070) / V.2 (6078) -------------------
    let pack_l1_coeff: u64 = if rows_per_expert >= 512 { 464 } else { 480 };
    let pack_weights = rows_per_expert >= 256
        && p.n >= 512
        && p.n % 32 != 0
        && p.k >= 128
        && pack_l1_coeff * u64::from(p.kp) <= L1_BYTES;
    {
        let mut g = G::new("V", 6065);
        g.t(format!("m/e({rows_per_expert})>=256"), rows_per_expert >= 256)
            .t(format!("n({})>=512", p.n), p.n >= 512)
            .t(format!("n%32({})!=0", p.n % 32), p.n % 32 != 0)
            .t(format!("k({})>=128", p.k), p.k >= 128)
            .t(
                format!("{pack_l1_coeff}*kp({})={}<=524288", p.kp, pack_l1_coeff * u64::from(p.kp)),
                pack_l1_coeff * u64::from(p.kp) <= L1_BYTES,
            );
        let mut step = g.step();
        step.outcome = Outcome::Note;
        step.detail = if pack_weights {
            format!("packWeights = true: {}", step.detail)
        } else {
            format!("packWeights = false, first failing term: {}", step.detail)
        };
        tr.push(step);
    }
    if pack_weights {
        let mut packed = p;
        packed.wstride = up(p.n, 128);
        // V.1 -- bf16
        {
            let mut g = G::new("V.1", 6070);
            g.t("bf", bf)
                .c(format!("cores({})<=20", p.cores), p.cores <= 20)
                .t(format!("e({})<=16", p.e), p.e <= 16)
                .t(format!("m/e({rows_per_expert})<512"), rows_per_expert < 512)
                .t(format!("k({})>512", p.k), p.k > 512)
                .t(format!("k({})<=1024", p.k), p.k <= 1024)
                .t(format!("n({})<=1280", p.n), p.n <= 1280)
                .t(
                    format!("480*kp({})={}<=524032", p.kp, 480 * u64::from(p.kp)),
                    480 * u64::from(p.kp) <= 524032,
                );
            let mut pass = g.passed();
            if pass {
                let ok = configure_cube_full_k(&mut packed, 224, 128, 128, 64, true, false, env.l1_guard);
                g.t("ConfigureCubeFullK<224,128,128,64,Prefetch,->", ok);
                pass = ok;
            }
            tr.push(g.step());
            if pass {
                let mut r = Routed::new(
                    tr,
                    "V.1",
                    6070,
                    "SearchCubeCompact4700004<true>",
                    packed.cores,
                    "p.cores (not clamped to GetCoreNumAic)",
                );
                r.body = Some("PackActiveWeightRowsInline + CubeFullKUnitBody<...,224,128,128,64>".into());
                r.staging = Some(staging_bytes(&packed));
                r.extra_scratch = Some((
                    "packedWeight = e*k*wstride".into(),
                    u64::from(packed.e) * u64::from(packed.k) * u64::from(packed.wstride),
                ));
                r.plan = Some(packed);
                r.notes.push(format!("packed.wstride = UpH(n,128) = {}", packed.wstride));
                return Ok(r.with_pack(vec![
                    f("k-1", 0, 10, u64::from(packed.k - 1), Some(1023)),
                    f("n-1", 10, 11, u64::from(packed.n - 1), Some(1279)),
                    f("e-1", 21, 4, u64::from(packed.e - 1), Some(15)),
                    f("cores-1", 25, 5, u64::from(packed.cores - 1), Some(19)),
                    f("gt", 30, 1, u64::from(packed.list_type), Some(1)),
                ]));
            }
        }
        // V.2 -- fp16. Note the term order: ConfigureCubeFullK comes *second*,
        // before the e/cores/k/n bounds (6078..6080).
        {
            let mut g = G::new("V.2", 6078);
            g.t("!bf", !bf);
            let mut pass = g.passed();
            if pass {
                let ok = configure_cube_full_k(&mut packed, 224, 128, 128, 64, true, false, env.l1_guard);
                g.t("ConfigureCubeFullK<224,128,128,64,Prefetch,->", ok);
                pass = ok;
            }
            if pass {
                g.t(format!("e({})<=16", p.e), p.e <= 16)
                    .c(format!("cores({})<=20", p.cores), p.cores <= 20)
                    .t(format!("k({})<=1024", p.k), p.k <= 1024)
                    .t(format!("n({})<=1280", p.n), p.n <= 1280)
                    .t(
                        format!("480*kp({})={}<=524032", p.kp, 480 * u64::from(p.kp)),
                        480 * u64::from(p.kp) <= 524032,
                    );
                pass = g.passed();
            }
            tr.push(g.step());
            if pass {
                let mut r = Routed::new(
                    tr,
                    "V.2",
                    6078,
                    "SearchCubeCompact4700005<false>",
                    packed.cores,
                    "p.cores (not clamped to GetCoreNumAic)",
                );
                r.body =
                    Some("PackActiveWeightRowsInline + CubeFullKBodyScalarHalf<...,224,128,128,64>".into());
                r.staging = Some(staging_bytes(&packed) / 2);
                r.extra_scratch = Some((
                    "packedWeight = e*k*wstride".into(),
                    u64::from(packed.e) * u64::from(packed.k) * u64::from(packed.wstride),
                ));
                r.plan = Some(packed);
                r.notes.push(format!("packed.wstride = UpH(n,128) = {}", packed.wstride));
                return Ok(r.with_pack(vec![
                    f("k-1", 0, 10, u64::from(packed.k - 1), Some(1023)),
                    f("n-1", 10, 11, u64::from(packed.n - 1), Some(1279)),
                    f("e-1", 21, 4, u64::from(packed.e - 1), Some(15)),
                    f("cores-1", 25, 5, u64::from(packed.cores - 1), Some(19)),
                    f("gt", 30, 1, u64::from(packed.list_type), Some(1)),
                ]));
            }
        }
    }

    // --- W: narrow bf16, one K chunk (6088) ------------------------------------
    {
        let mut g = G::new("W", 6088);
        g.t("bf", bf)
            .c(format!("cores({})<=20", p.cores), p.cores <= 20)
            .t(format!("e({})>1", p.e), p.e > 1)
            .t(format!("e({})<=16", p.e), p.e <= 16)
            .t(format!("m/e({rows_per_expert})>=96"), rows_per_expert >= 96)
            .t(format!("m/e({rows_per_expert})<256"), rows_per_expert < 256)
            .t(format!("n({})>=192", p.n), p.n >= 192)
            .t(format!("n({})<=320", p.n), p.n <= 320)
            .t(format!("k({})>256", p.k), p.k > 256)
            .t(format!("k({})<=384", p.k), p.k <= 384);
        let mut pass = g.passed();
        let mut selected = p;
        if pass {
            let ok = configure_cube_full_k(&mut selected, 64, 128, 384, 32, true, true, env.l1_guard);
            g.t("ConfigureCubeFullK<64,128,384,32,Prefetch,SingleL0>", ok);
            pass = ok;
        }
        tr.push(g.step());
        if pass {
            let mut r = Routed::new(
                tr,
                "W",
                6088,
                "SearchCubeCompact4700002<bfloat16_t,bfloat16_t>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some("CubeFullKUnitBody<bf16,bf16,64,128,384,32>".into());
            r.staging = Some(staging_bytes(&selected));
            r.plan = Some(selected);
            r.notes.push(
                "LaunchNarrowSingleKSelected instantiates bf16 only (3670); relaxing W's leading \
                 `bf &&` loses fp16 correctness with no compile error [CURRENT.md landmine 3]"
                    .into(),
            );
            return Ok(r.with_pack(vec![
                f("k-1", 0, 9, u64::from(selected.k - 1), Some(383)),
                f("n-1", 9, 9, u64::from(selected.n - 1), Some(319)),
                f("e-1", 18, 4, u64::from(selected.e - 1), Some(15)),
                f("cores-1", 22, 5, u64::from(selected.cores - 1), Some(19)),
                f("gt", 27, 1, u64::from(selected.list_type), Some(1)),
            ]));
        }
    }

    // --- X: resident expert/N panels, mix11 (6096) -----------------------------
    // The one gate that passes `p` BY REFERENCE. Safe only because both
    // `return false`s in ConfigureCubeFullK precede every write.
    {
        let mut g = G::new("X", 6096);
        g.t(format!("rowsPerExpert({rows_per_expert})<96"), rows_per_expert < 96)
            .t(format!("n({})<=640", p.n), p.n <= 640)
            .t(format!("kp({})<=512", p.kp), p.kp <= 512);
        let mut pass = g.passed();
        if pass {
            // by reference -- p itself is mutated
            let ok = configure_cube_full_k(&mut p, 96, 128, 512, 48, false, true, env.l1_guard);
            g.t("ConfigureCubeFullK<96,128,512,48,-,SingleL0>(p)  [p BY REFERENCE]", ok);
            pass = ok;
        }
        tr.push(g.step());
        if pass {
            let clamped = p.cores.min(env.physical_aic);
            if clamped == 0 {
                return Err("mix11::Launch: no physical AIC cores (3001)".into());
            }
            let upper = (u64::from((p.m + 95) / 96) + u64::from(p.e) - 1)
                * u64::from((p.n + 127) / 128);
            let demote = clamped > 32 || upper > 4 * u64::from(clamped);
            if !demote {
                let mut q = p;
                q.cores = clamped;
                q.vcores = clamped; // 1:1 MIX (3003)
                let kern = if bf {
                    "mix11::SearchP2Mix115300000<bfloat16_t,bfloat16_t>"
                } else {
                    "mix11::SearchP2Mix115300000<float,half>"
                };
                let mut r = Routed::new(
                    tr,
                    "X",
                    6096,
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
                return Ok(r.with_pack(vec![
                    f("k-1", 0, 9, u64::from(q.k - 1), Some(511)),
                    f("n-1", 9, 10, u64::from(q.n - 1), Some(639)),
                    f("e-1", 19, 7, u64::from(q.e - 1), Some(127)),
                    f("cores-1", 26, 5, u64::from(q.cores - 1), Some(31)),
                    f("gt", 31, 1, u64::from(q.list_type), Some(1)),
                ]));
            }
            // packed32::LaunchPacked32 receives `incoming` -- the un-clamped Plan.
            let mut q = p;
            q.cores = p.cores.min(env.physical_aic);
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
                let mut r = Routed::new(tr, "X.p32", 2971, kern, q.cores, "min(p.cores, GetCoreNumAic())");
                r.staging = Some(staging_bytes(&q));
                r.plan = Some(q);
                return Ok(r.with_pack(vec![
                    f("k-1", 0, 9, u64::from(q.k - 1), Some(511)),
                    f("n-1", 9, 10, u64::from(q.n - 1), Some(639)),
                    f("e-1", 19, 7, u64::from(q.e - 1), Some(127)),
                    f("cores-1", 26, 5, u64::from(q.cores - 1), Some(31)),
                    f("gt", 31, 1, u64::from(q.list_type), Some(1)),
                ]));
            }
            let kern = if bf {
                "packed32::SearchP2Fixed964100000<bfloat16_t,bfloat16_t>"
            } else {
                "packed32::SearchP2Fixed964100000<float,half>"
            };
            let mut r = Routed::new(tr, "X.f96", 2975, kern, q.cores, "min(p.cores, GetCoreNumAic())");
            r.staging = Some(staging_bytes(&q));
            r.plan = Some(q);
            r.notes.push("Plan by value -- no bit packing on this arm".into());
            return Ok(r);
        }
    }

    // --- Y: aligned FP16 full-K, half-pair output (6103) -----------------------
    {
        let mut g = G::new("Y", 6103);
        g.t("!bf", !bf)
            .c(format!("availableCoreNum({})<=20", env.available_core_num), env.available_core_num <= 20)
            .t(format!("rowsPerExpert({rows_per_expert})>=256"), rows_per_expert >= 256)
            .t(format!("n({})>=512", p.n), p.n >= 512)
            .t(format!("n%128({})==0", p.n % 128), p.n % 128 == 0)
            .t(format!("k%128({})==0", p.k % 128), p.k % 128 == 0)
            .t(format!("wstride({})==n({})", p.wstride, p.n), p.wstride == p.n);
        if env.l1_guard {
            g.t(
                format!("512*kp({})={}<=524032", p.kp, 512 * u64::from(p.kp)),
                512 * u64::from(p.kp) <= 524032,
            );
        } else {
            g.t("512*kp<=524032  [L1 GUARDS DISABLED by --no-l1-guard]", true);
        }
        let mut pass = g.passed();
        let mut selected = p;
        if pass {
            let ok = configure_cube_full_k(&mut selected, 256, 128, 128, 64, true, false, env.l1_guard);
            g.t("ConfigureCubeFullK<256,128,128,64,Prefetch,->", ok);
            pass = ok;
        }
        tr.push(g.step());
        if pass {
            let mut r = Routed::new(
                tr,
                "Y",
                6103,
                "p1_half_pair::SearchP1HalfPairOutput11840034<float,half>",
                selected.cores,
                "p.cores (not clamped to GetCoreNumAic)",
            );
            r.body = Some("CubeFullKUnitBodyScalarHalf<float,half,256,128,128,64>".into());
            r.staging = Some(staging_bytes(&selected) / 2);
            r.plan = Some(selected);
            // The landmine: the field holds k<=1024 but gate Y bounds k only
            // incidentally, through the L1 budget line at 6105.
            let mut kf = f("k/128-1", 0, 3, u64::from(selected.k / 128 - 1), if env.l1_guard { Some(6) } else { None });
            kf.incidental = Some(
                "gate Y has no explicit upper bound on k; only the two L1 budget checks -- \
                 512*kp<=524032 (6105) and ConfigureCubeFullK's (256+2*128)*kp<=512K (2863) -- \
                 keep it <= 896. Rewrite that arithmetic and k=1152 aliases to k=128, a silent \
                 wrong answer [CURRENT.md landmine 1, which names only 6105]",
            );
            r.notes.push("staging = CubeFullKStagingBytes(p)/2 (half-precision backend, 4217)".into());
            return Ok(r.with_pack(vec![
                kf,
                f("n/128-1", 3, 6, u64::from(selected.n / 128 - 1), Some(63)),
                f("e-1", 9, 7, u64::from(selected.e - 1), Some(127)),
                f("cores-1", 16, 5, u64::from(selected.cores - 1), Some(19)),
                f("gt", 21, 1, u64::from(selected.list_type), Some(1)),
            ]));
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
    const TABLE: &[(u32, &str, u32, &str, u32)] = &[
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
        // Point 3's rowDims, as CURRENT.md states it.
        let r3 = route(&pts[2].shape, &env).unwrap();
        assert!(r3.notes.iter().any(|n| n == "rowDims = 1158"), "{:?}", r3.notes);
    }

    #[test]
    fn point_two_stays_in_mix11() {
        let r = route(&points()[1].shape, &Env::default()).unwrap();
        assert!(r.notes.iter().any(|n| n.contains("Upper(p) = 30")), "{:?}", r.notes);
    }

    /// Six gates hard-assume cores<=20; CURRENT.md says a different core count
    /// silently demotes 4/9/12/14/15.
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
        // 7 (fp16) takes the DMA-scales kernel; 8 (bf16) has no DMA variant, so
        // only its ScalarOutput template argument flips (y&31 != 0).
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
}
