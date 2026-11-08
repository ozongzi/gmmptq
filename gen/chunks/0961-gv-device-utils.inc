// Shared device utilities. Include inside namespace gmmptq, once.

struct GeneralVectorDims {
    uint32_t m,k,n,e,listType;
};
__aicore__ inline uint32_t GvUp(uint32_t n,uint32_t a) { return (n+a-1)/a*a; }
__aicore__ inline uint32_t GvMin(uint32_t a,uint32_t b) { return a<b?a:b; }
template <AscendC::HardEvent Event>
__aicore__ inline void GvBarrier() { AscendC::SetFlag<Event>(0);AscendC::WaitFlag<Event>(0); }
__aicore__ inline float GvFloat(uint32_t bits) {
    union { uint32_t u;float f; } value;value.u=bits;return value.f;
}
__aicore__ inline uint16_t GvRound16(float value,bool bf) {
    union { float f;uint32_t u; } v;v.f=value;
    if (bf) return uint16_t((v.u+0x7fffu+((v.u>>16)&1u))>>16);
    const uint32_t sign=(v.u>>16)&0x8000u,a=v.u&0x7fffffffu;
    if (a>=0x47800000u) return uint16_t(sign|0x7c00u);
    if (a>=0x38800000u) return uint16_t(sign|((a+0xfffu+((a>>13)&1u)-0x38000000u)>>13));
    uint32_t e=a>>23;
    if (e<102u) return uint16_t(sign);
    uint32_t shift=126u-e,mant=(a&0x7fffffu)|0x800000u;
    uint32_t base=mant>>shift,remainder=mant&((1u<<shift)-1u),midpoint=1u<<(shift-1u);
    return uint16_t(sign|(base+uint32_t(remainder>midpoint || (remainder==midpoint && (base&1u)))));
}
template <bool BF>
__aicore__ inline uint16_t GvEncode(float value) {
    if constexpr (BF) return GvRound16(value,true);
    else { union { half h;uint16_t u; } out;out.h=half(value);return out.u; }
}
__aicore__ inline void GvMulRows(AscendC::LocalTensor<float> dst,
                                AscendC::LocalTensor<float> weights,
                                uint32_t rows,uint32_t pitch) {
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride=1;rp.src0BlkStride=1;rp.src1BlkStride=1;
    rp.dstRepStride=uint16_t(pitch/8);rp.src0RepStride=uint16_t(pitch/8);rp.src1RepStride=0;
    for (uint32_t c=0;c<pitch;c+=64)
        AscendC::Mul(dst[c],dst[c],weights[c],uint64_t(GvMin(64,pitch-c)),uint8_t(rows),rp);
}

