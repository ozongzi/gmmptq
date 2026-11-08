inline void Check(aclError status) {
    if (status != ACL_SUCCESS)
        throw std::runtime_error("GroupedMatmulPerTokenQuant ACL failure: " +
                                 std::to_string(status));
}
struct Scratch {
    void* p = nullptr;
    explicit Scratch(size_t bytes) {
        if (bytes) Check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Scratch() { if (p) aclrtFree(p); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};
inline uint32_t UpH(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }
inline const TensorInfo& Tensor(const TensorGroupInfo& g, int rank, int dtype) {
    if (g.numTensors != 1 || !g.tensors || g.tensors[0].numDims != rank ||
        g.tensors[0].dtype != dtype || !g.tensors[0].shape)
        throw std::invalid_argument("GroupedMatmulPerTokenQuant: tensor rank/dtype mismatch");
    return g.tensors[0];
}
inline void PlanCube(Plan& p) {
    uint32_t perExpert = p.e ? p.m / p.e : p.m;
    if (perExpert < 1) perExpert = 1;
    uint32_t m0 = 256;
    while (m0 > 16 && m0 / 2 >= UpH(perExpert, 16)) m0 /= 2;
    uint32_t n0 = 32768 / UpH(m0, 16);
    if (n0 > 512) n0 = 512;
    while (n0 > 32 && n0 / 2 >= UpH(p.n, 16)) n0 /= 2;
    if (n0 < 16) n0 = 16;
    uint32_t m0r = UpH(m0, 16);
    uint32_t n0r = UpH(n0, 16);
    uint32_t n0c = UpH(n0, 32);
    uint32_t k0 = 128;
    while (k0 > 32 && (m0r * k0 * 2 > L0A_BYTES || k0 * n0r * 2 > L0B_BYTES)) k0 /= 2;
    while (k0 > 32 && k0 / 2 >= p.kp) k0 /= 2;
    uint32_t kl = p.kp;
    while (kl > k0 && m0r * kl + 2 * kl * n0c > L1_BYTES) kl -= k0;
    if (kl < k0) kl = k0;
    p.m0 = m0;
    p.n0 = n0;
    p.m0r = m0r;
    p.n0r = n0r;
    p.n0c = n0c;
    p.k0 = k0;
    p.kl = kl;
    p.nTiles = (p.n + n0 - 1) / n0;
    uint32_t pitch = UpH(n0, 8);
    uint32_t rv = 64;
    while (rv > 1 && rv * pitch * 4 * 2 + 16 * 1024 > UB_BYTES) rv /= 2;
    if (rv > m0) rv = m0 < 1 ? 1 : m0;
    p.rv = rv;
}
inline void PlanVector(Plan& p) {
    if (p.form == FORM_SMALLN) {
        uint32_t pitch = p.n;
        uint32_t perK = pitch * 15 + 8;
        uint32_t vk = (UB_BYTES - 16 * 1024) / perK;
        if (vk > 1024) vk = 1024;
        if (vk < 2) vk = 2;
        p.vk = vk & ~1u;
        p.vn = pitch;
        p.vm = 1;
        return;
    }
    uint32_t vn = 256;
    while (vn > 64 && vn / 2 >= UpH(p.n, 8)) vn /= 2;
    uint32_t vm;
    if (p.form == FORM_DOT) {
        uint32_t kr = UpH(p.k, 64);
        if (kr > 1024) kr = 1024;
        vm = (UB_BYTES - 32 * 1024) / (kr * 11);
        if (vm > 8) vm = 8;
        if (vm < 1) vm = 1;
    } else {
        vm = 32;
        while (vm > 1 && size_t(vm) * vn * 4 * 3 > 120 * 1024) vm /= 2;
    }
    p.vn = vn;
    p.vm = vm;
}

// --- host tiling for the two added forms -----------------------------------

/// Chunk shape for FORM_VDOT.
///
/// nb is NOT rounded up to a vector block: the shapes that reach this form
/// have N of 1 or 2, and widening the column block to 16 would do eight times
/// the arithmetic for nothing. weight's ND layout already makes
/// w[e, k, n0..n0+nb) contiguous per k.
inline void BuildVdotTiling(const Plan& p, VTiling& out) {
    uint32_t nb = p.n < 512 ? p.n : 512u;
    uint32_t want = 1;
    while (want < p.k) want *= 2;
    if (want < 8) want = 8;
    const uint32_t room = [](uint32_t v) {
        uint32_t q = 1;
        while (q * 2 <= v) q *= 2;
        return q;
    }(vec::BLOCK_ELEMS / (nb ? nb : 1u));
    uint32_t kc = want < room ? want : room;
    if (!kc) kc = 1;
    out.kc = kc;
    out.nb = nb;
    out.m = p.m;
    out.k = p.k;
    out.n = p.n;
    out.e = p.e;
    out.listType = p.listType;
    // One item per (row, column block).
    const uint32_t items = p.m * UpH(p.n, nb ? nb : 1u) / (nb ? nb : 1u);
    out.coreNum = items < p.vcores ? (items ? items : 1u) : p.vcores;
}

/// Tiling for FORM_OUTER. The row block rides in the kc slot.
///
/// Up(n, 16) is the UB row pitch: it keeps every row start 32-byte aligned for
/// both the f32 and the output copy, and it is exactly the step a store with
/// srcStride 0 takes. See the comment block above GmmOuter.
inline void BuildOuterTiling(const Plan& p, VTiling& out) {
    const uint32_t pitch = UpH(p.n, 16);
    const uint32_t nUp=UpH(p.n,vec::GRAIN);
    uint32_t rows=(pitch/8>255)?1u:outer::MAX_ROWS;
    const auto ubBytes=[&](uint32_t r)->uint32_t {
        const uint32_t rUp=UpH(r,vec::GRAIN);
        return r*pitch*6+12*nUp+80*rUp+(nUp+vec::GRAIN)+(rUp+vec::GRAIN);
    };
    while(rows>1 && ubBytes(rows)>188160) --rows;
    out.kc = rows;
    out.nb = pitch;
    out.m = p.m;
    out.k = p.k;
    out.n = p.n;
    out.e = p.e;
    out.listType = p.listType;
    // One item per (expert, row block); the expert term covers the worst case
    // of every expert holding at least one row.
    const uint32_t items = (p.m + rows - 1) / rows + p.e;
    out.coreNum = items < p.vcores ? (items ? items : 1u) : p.vcores;
}

