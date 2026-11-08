namespace gmmptq {
// Resource-selected short-group GEMV family; no evaluated-shape matching.
inline bool UseBalancedOverlapShortGroup(const Plan& p) {
    return p.e>=32 && p.e<=128 && p.m>0 && p.m<=256 && p.m<=2*p.e &&
           p.k>0 && p.k<=512 && p.n>0 && p.n<=512 && p.vcores>0 && p.listType<=1;
}
inline uint32_t BalancedOverlapShortGroupUbBytes(uint32_t maxK,uint32_t capacityN,uint32_t nPitch) {
    const uint32_t block=maxK*capacityN;
    const uint32_t kUp=(maxK+31)/32*32;
    return 128*8+256*4+(kUp+32)+(block+32)+
           4*block+4*kUp+10*nPitch+32+2*block+2*kUp+8192;
}
inline VTiling BuildBalancedOverlapShortGroup(const Plan& p) {
    VTiling out{};
    out.m=p.m; out.k=p.k; out.n=p.n; out.e=p.e; out.listType=p.listType;
    out.coreNum=std::min(p.vcores,p.m);
    const uint32_t nPitch=(p.n+31)/32*32;
    out.nb=std::min(nPitch,192u);  // capacity width, not actual DMA pitch
    out.kc=128;
    const uint32_t kUp=(p.k+7)/8*8;
    while (out.kc>8 &&
           (BalancedOverlapShortGroupUbBytes(out.kc,out.nb,nPitch)>188160 || out.kc/2>=kUp))
        out.kc/=2;
    // Guarantee the widest row piece can retain at least eight K rows.
    // Brcb processes eight scalar inputs per repeat; a zero repeat is invalid.
    const uint32_t minWidth=((8*nPitch+out.kc-1)/out.kc+31)/32*32;
    out.nb=std::max(out.nb,minWidth);
    return out;
}

}

