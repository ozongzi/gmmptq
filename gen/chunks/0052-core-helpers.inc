__aicore__ inline uint32_t Up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
template <AscendC::HardEvent E>
__aicore__ inline void Barrier() {
    AscendC::SetFlag<E>(0);
    AscendC::WaitFlag<E>(0);
}
struct Schedule {
    AscendC::GlobalTensor<int64_t> list;
    uint32_t listType;
    uint32_t experts;
    uint32_t m0;
    uint32_t nTiles;
    __aicore__ inline uint32_t Total() const {
        uint32_t total = 0, start = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            uint32_t v = static_cast<uint32_t>(list.GetValue(e));
            uint32_t end = listType == 0 ? v : start + v;
            // Guarded: end - start is unsigned, and one non-monotonic read of
            // the list underflows it into billions, which makes `total` huge
            // and sends tile indices off the end of the workspace. That is
            // ACL 507015, "the DDR address of the MTE instruction is out of
            // range", and it showed up about once in 60 runs of tp1 -- which
            // is also what the two Runtime Errors in our judge history look
            // like. groupList is read from Global Memory with GetValue, by
            // both AIC and AIV independently, so a stale or racy read is all
            // it takes; the arithmetic must not amplify one into an
            // out-of-range address.
            total += (end > start ? (end - start + m0 - 1) / m0 : 0u) * nTiles;
            start = end;
        }
        return total;
    }
};
template <typename ScaleT>
__aicore__ inline void ToFloat(AscendC::LocalTensor<float> dst,
                               AscendC::LocalTensor<ScaleT> src, uint32_t count) {
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}
template <>
__aicore__ inline void ToFloat<float>(AscendC::LocalTensor<float> dst,
                                      AscendC::LocalTensor<float> src, uint32_t count) {
    AscendC::Adds(dst, src, 0.0f, count);
}
__aicore__ inline void MulByRow(AscendC::LocalTensor<float> dst,
                                AscendC::LocalTensor<float> src,
                                AscendC::LocalTensor<float> col,
                                uint32_t rows, uint32_t cols, uint32_t pitch) {
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<uint16_t>(pitch / F32_PER_BLK);
    rp.src0RepStride = static_cast<uint16_t>(pitch / F32_PER_BLK);
    rp.src1RepStride = 0;
    for (uint32_t r = 0; r < rows; r += 255) {
        uint8_t rep = static_cast<uint8_t>(Min(255, rows - r));
        for (uint32_t c = 0; c < cols; c += F32_PER_REP) {
            uint64_t mask = Min(F32_PER_REP, cols - c);
            AscendC::Mul(dst[r * pitch + c], src[r * pitch + c], col[c], mask, rep, rp);
        }
    }
}
__aicore__ inline void MulByCol(AscendC::LocalTensor<float> value,
                                AscendC::LocalTensor<float> brcb,
                                uint32_t rows, uint32_t cols, uint32_t pitch) {
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = static_cast<uint16_t>(pitch / F32_PER_BLK);
    rp.src0BlkStride = static_cast<uint16_t>(pitch / F32_PER_BLK);
    rp.src1BlkStride = 1;
    rp.dstRepStride = 1;
    rp.src0RepStride = 1;
    rp.src1RepStride = 0;
    for (uint32_t r = 0; r < rows; r += F32_PER_BLK) {
        uint64_t mask = Min(F32_PER_BLK, rows - r) * F32_PER_BLK;
        for (uint32_t c = 0; c < cols; c += F32_PER_BLK * 255) {
            uint8_t rep = static_cast<uint8_t>(Min(F32_PER_BLK * 255, cols - c) / F32_PER_BLK);
            AscendC::Mul(value[r * pitch + c], value[r * pitch + c], brcb[r * F32_PER_BLK],
                         mask, rep, rp);
        }
    }
}
