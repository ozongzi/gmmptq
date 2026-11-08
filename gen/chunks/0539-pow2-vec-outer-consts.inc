inline uint32_t Pow2Below(uint32_t v) {
    uint32_t p = 1;
    while (p * 2 <= v) p *= 2;
    return p;
}

/// Smallest power of two at least v.
inline uint32_t Pow2AtLeast(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p *= 2;
    return p;
}

namespace vec {
/// f32 elements per 32 B block; a vector operand must start on one.
constexpr uint32_t LANES = 8;
/// Elements that make one 32 B block at the NARROWEST type in play. The rule
/// is in bytes, not elements: a Cast whose destination is half needs 16
/// elements to fill a block, and one whose source is int8 needs 32. Rounding
/// every count to 32 satisfies all of them at once.
constexpr uint32_t GRAIN = 32;
/// Cap on the product block, which dominates the UB budget at two f32 copies.
constexpr uint32_t BLOCK_ELEMS = 8192;
/// BroadCast and the widening casts want scratch that does not scale down
/// with the block: a tiny case still issues whole vector repeats, and a
/// 64-byte temporary is not enough for one.
constexpr uint32_t SCRATCH_BYTES = 8 * 1024;
} // namespace vec


// ---------------------------------------------------------------------------
// FORM_OUTER: K == 1, so there is no reduction and the matmul is an outer
// product.
//
// tp10 is E=4, M=1025, K=1, N=769, and y[m][n] = x[m] * w[e][n] * xs[m] *
// ws[n]. One MAC per output, 20 KB of input, 1.5 MB of output: the cost is
// per-row overhead and store bandwidth, nothing else. tbest 9.59 us is
// 164 GB/s of stores.
//
// The AXPY form this shape used to take runs 129.5 us, which is 12 GB/s. It
// loops per row and per 256-column block, and each of those 4100 iterations
// pays its own DataCopyPad of 512 bytes, a Barrier<MTE3_V> after it, and two
// full pipeline barriers around a one-float scalar read of xScale -- about
// 1900 cycles per core to produce 256 outputs.
//
// The UB row pitch here is Up(N, 16), which is what makes both halves work at
// once:
//
//   * A vector op at row r starts at r * Up(N,16) * 4 bytes for the f32 copy
//     and r * Up(N,16) * 2 for the output copy. Both are multiples of 32,
//     because Up(N,16)*4 = 64*ceil(N/16) and Up(N,16)*2 = 32*ceil(N/16). An
//     unaligned vector operand start hangs the unit (design.md), and the
//     obvious pitch of N does not survive: with N = 769, row 1 would start at
//     byte 3076.
//
//   * The whole block still leaves as ONE store. DataCopyPad's srcStride is
//     the gap from the end of one block to the start of the next in units of
//     32 bytes, so with blockLen = N*sizeof(OutT) = 1538 and srcStride = 0 the
//     next source row is taken from Up(1538, 32) = 1568 bytes on, which is
//     exactly Up(N,16)*sizeof(OutT). The destination is packed, so y comes out
//     row major and contiguous.
//
// That combination is why there is no BroadCast in here. BroadCast would have
// expressed cf (x) spread across the columns and wq (w) tiled down the rows as
// one Mul over the whole block, but with dim=2 and axis=0 the API requires
// srcShape[1] to be 32-byte aligned, and N = 769 is not. Tried: it hangs.
// Padding to Up(N,8) satisfies BroadCast and then breaks the single store,
// since the gap would have to be a multiple of 32 bytes and pitch - N a
// multiple of 16 at the same time, which for N = 769 has no solution at all
// (769 is 1 mod 16, so every candidate pitch stays 1 mod 16).
//
// What is left is one Muls per row, which is a single instruction over N
// elements, plus one Cast and one store for the entire block.
//
// Only K == 1 is handled. The same structure covers small K with one
// accumulate per k, but extracting column k from x[R][K] needs a strided read
// that none of the cheap primitives express, and no shape in play wants it.
// ---------------------------------------------------------------------------

namespace outer {
/// f32 plus OutT staging is six bytes per block element. 24576 of them is
/// about 147 KB, which leaves UB room for the per-expert vectors.
constexpr uint32_t BUDGET_ELEMS = 25088;
constexpr uint32_t MAX_ROWS = 32;
}  // namespace outer

