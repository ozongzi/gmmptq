namespace gmmptq {
constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L0A_BYTES = 64 * 1024;
constexpr uint32_t L0B_BYTES = 64 * 1024;
constexpr uint32_t L0C_BYTES = 128 * 1024;
constexpr uint32_t UB_BYTES = 192 * 1024;
constexpr uint32_t C0_I8 = 32;
constexpr uint32_t FRAC_I8 = 512;
constexpr uint32_t ROW_FRAC = 16;
constexpr uint32_t BLK = 32;
constexpr uint32_t F32_PER_BLK = 8;
constexpr uint32_t F32_PER_REP = 64;
constexpr uint32_t AIV_PER_AIC = 2;
constexpr uint16_t FLAG_TILE = 0;
constexpr uint32_t FORM_CUBE = 0;
constexpr uint32_t FORM_AXPY = 1;
constexpr uint32_t FORM_DOT = 2;
constexpr uint32_t FORM_TINY = 3;
constexpr uint32_t FORM_SMALLN = 4;
// Few outputs, long reduction: a vector dot product.
constexpr uint32_t FORM_VDOT = 5;
// K == 1: no reduction at all, so the matmul is an outer product.
constexpr uint32_t FORM_OUTER = 6;
constexpr uint32_t EXACT_K = 1024;
struct Plan {
    uint32_t m, k, n, e;
    uint32_t wstride;
    uint32_t listType;
    uint32_t kp;
    uint32_t m0, n0;
    uint32_t k0;
    uint32_t kl;
    uint32_t m0r;
    uint32_t n0r;
    uint32_t n0c;
    uint32_t nTiles;
    uint32_t cores;
    uint32_t vcores;
    uint32_t rv;
    uint32_t form;
    uint32_t vm, vn;
    uint32_t vk;
};
