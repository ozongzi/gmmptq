// Include after Plan and the memory-size constants. No data-dependent host work
// or exact logical-shape matching is required to configure the Cube engine.

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t RV,
          bool Prefetch = false, bool SingleL0 = false>
inline bool ConfigureCubeFullK(Plan& p) {
    static_assert(BM > 0 && BM % 16 == 0, "BM alignment");
    static_assert(BN > 0 && BN % 32 == 0, "BN alignment");
    static_assert(BK > 0 && BK % 32 == 0, "BK alignment");
    static_assert(RV > 0 && RV <= BM && RV % 8 == 0, "RV alignment");
    static_assert(BM * BK * (SingleL0 ? 1u : 2u) <= L0A_BYTES, "L0A overflow");
    static_assert(BN * BK * (SingleL0 ? 1u : 2u) <= L0B_BYTES, "L0B overflow");
    static_assert(BM * BN * sizeof(int32_t) <= L0C_BYTES, "L0C overflow");
    constexpr uint32_t ub = RV * BN * 10u + BN * 8u + RV * 36u;
    static_assert(ub <= 192u * 1024u - 8192u - 256u, "UB overflow");
    if (!p.k || !p.n || !p.m || !p.e || !p.cores) return false;
    const uint32_t kp = (p.k + 31u) / 32u * 32u;
    const uint64_t l1 = uint64_t(BM + (Prefetch ? 2u : 1u) * BN) * kp;
    if (l1 > L1_BYTES) return false;
    p.kp = kp;
    p.m0 = BM;
    p.n0 = BN;
    p.m0r = BM;
    p.n0r = BN;
    p.n0c = BN;
    p.k0 = BK;
    p.kl = kp;
    p.nTiles = (p.n + BN - 1u) / BN;
    p.rv = RV;
    return true;
}

// This bound covers arbitrary valid groupList distributions, including empty
// experts. The AIC/AIV traversal agrees on identical linear tile identifiers.
inline uint64_t CubeFullKStagingBytes(const Plan& p) {
    const uint64_t maxTiles = (uint64_t((p.m + p.m0 - 1u) / p.m0) + p.e) * p.nTiles;
    return maxTiles * p.m0 * p.n0 * sizeof(int32_t);
}

inline void LeanPlanCube(Plan& p) {
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
    while (kl > k0 && m0r * kl + 2 * kl * n0c > 524032u) kl -= k0;
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

} // namespace gmmptq
