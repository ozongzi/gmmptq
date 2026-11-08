namespace gmmptq {
// Independent fixed-bias implementation; not received Misty source.
// Cost and targets are u32 under Host M<4096,E<=16,N<=1024,C<=20.
struct P4WeightedOwnership384 {
    __aicore__ static inline uint32_t ExpertWeight(uint32_t rows,uint32_t nt) {
        const uint32_t full=rows/192u,tail=rows%192u;
        return (full*576u+(tail?384u+tail:0u))*nt;
    }
    __aicore__ static inline uint32_t Inverse(uint32_t rows,uint32_t nt,uint32_t target) {
        const uint32_t full=rows/192u,tail=rows%192u;
        const uint32_t strips=full+(tail!=0u),fullWeight=576u*nt;
        const uint32_t strip=Min(target/fullWeight,full);
        if(strip==strips)return strips*nt;
        const uint32_t width=strip<full?576u:384u+tail;
        const uint32_t local=target-strip*fullWeight;
        uint32_t tile=local/width;
        const uint32_t remainder=local-tile*width;
        if(tile<nt && remainder*2u>width)++tile; // exactly nearest, tie lower
        return strip*nt+tile;
    }
    template<class List>
    __aicore__ static inline void Own(List list,const Plan& p,uint32_t core,
                                    uint32_t& first,uint32_t& last) {
        const uint32_t nt=(p.n+127u)/128u;
        uint32_t start=0,weight=0;
        // First GL scan: total cost; the original Schedule.Total is compiled out.
        for(uint32_t e=0;e<p.e;++e) {
            const uint32_t value=uint32_t(list.GetValue(e));
            const uint32_t end=p.listType?start+value:value;
            const uint32_t rows=end>start?end-start:0u;
            weight+=ExpertWeight(rows,nt);start=end;
        }
        first=last=0;
        if(weight==0u)return;
        const uint32_t target0=weight*core/p.cores;
        const uint32_t target1=weight*(core+1u)/p.cores;
        bool got0=false,got1=false;uint32_t prefix=0,tiles=0;start=0;
        // Second GL scan: locate both boundaries in full-strip/tail segments.
        for(uint32_t e=0;e<p.e && !(got0&&got1);++e) {
            const uint32_t value=uint32_t(list.GetValue(e));
            const uint32_t end=p.listType?start+value:value;
            const uint32_t rows=end>start?end-start:0u;
            const uint32_t w=ExpertWeight(rows,nt);
            if(w) {
                if(!got0 && target0<=prefix+w) {first=tiles+Inverse(rows,nt,target0-prefix);got0=true;}
                if(!got1 && target1<=prefix+w) {last=tiles+Inverse(rows,nt,target1-prefix);got1=true;}
            }
            prefix+=w;tiles+=((rows+191u)/192u)*nt;start=end;
        }
        // The unchanged compute body then makes its original third expert scan.
    }
};

template <typename ScaleT, typename OutT,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t RV,
          bool NMajor = false, bool Prefetch = false, bool SingleL0 = false, bool StaticNoPipe = false, bool NativeM1 = false, bool CacheXS = false, bool WSSpan = false, bool Weighted = false, bool SingleBlockStage = false, bool P4ContiguousScale = false, bool PrecombinedScale = false>
__aicore__ inline void CubeFullKBodyScalarHalf(
    GM_ADDR xGm, GM_ADDR wGm, GM_ADDR xsGm, GM_ADDR wsGm,
    GM_ADDR glGm, GM_ADDR yGm, GM_ADDR stGm, Plan p) {
    using namespace AscendC;
    static_assert(BM > 0 && BM % 16 == 0, "BM must align to the M fractile");
    static_assert(BN > 0 && BN % 32 == 0, "BN must align to int8 transpose pairs");
    static_assert(BK > 0 && BK % 32 == 0, "BK must align to int8 K fractiles");
    static_assert(RV > 0 && RV <= BM && RV % 8 == 0, "RV must align Brcb rows");
    static_assert(sizeof(OutT) == 2, "The output must be fp16 or bf16");
    static_assert(!CacheXS || (!NMajor && RV >= (BM+1u)/2u),"one existing XS chunk per AV");
    static_assert(!(NMajor && Prefetch), "B prefetch is for M-major traversal");
    static_assert(BM * BK * (SingleL0 ? 1u : 2u) <= L0A_BYTES, "L0A overflow");
    static_assert(BN * BK * (SingleL0 ? 1u : 2u) <= L0B_BYTES, "L0B overflow");
    static_assert(BM * BN * sizeof(int32_t) <= L0C_BYTES, "L0C overflow");
    static_assert(!WSSpan || (BM==192 && BN==128 && RV==128 && !NMajor && CacheXS && sizeof(ScaleT)==4), "P4 WS span resource contract");
    static_assert(!Weighted || (BM==192 && BN==128 && BK==96 && RV==128 && !NMajor && Prefetch && StaticNoPipe && NativeM1 && CacheXS && WSSpan), "fixed P4 weighted ownership contract");
    static_assert(!P4ContiguousScale || (BM==192u && BN==128u && BK==96u && RV==128u &&
        !NMajor && Prefetch && !SingleL0 && StaticNoPipe && NativeM1 && CacheXS && WSSpan &&
        Weighted && SingleBlockStage && sizeof(ScaleT)==4u && sizeof(OutT)==2u),
        "P4 fixed contiguous precombined-scale policy");
    static_assert(!(P4ContiguousScale && PrecombinedScale), "P4 and P15 scale policies are mutually exclusive");
    constexpr uint32_t PITCH = BN;
    static_assert(!PrecombinedScale || (BM==224u && BN==128u && BK==128u && RV==64u &&
        !NMajor && Prefetch && !SingleL0 && StaticNoPipe && !NativeM1 && !CacheXS &&
        !WSSpan && !Weighted && SingleBlockStage && sizeof(ScaleT)==4u && sizeof(OutT)==2u),
        "precombined scale only for the existing FP16 active-packed stage family");
    constexpr uint32_t UB_BASE = RV * PITCH * 10u + PITCH * 8u + RV * 36u;
    constexpr uint32_t SCALE_ROWS = ((BM+1u)/2u+7u)/8u*8u;
    constexpr uint32_t UB_USED = UB_BASE + (PrecombinedScale ? SCALE_ROWS*(PITCH+1u)*4u : 0u);
    static_assert(UB_USED <= 192u * 1024u - 8192u - 256u, "UB overflow");

    const uint32_t nTiles = (p.n + BN - 1u) / BN;
    GlobalTensor<int64_t> list;
    list.SetGlobalBuffer((__gm__ int64_t*)glGm);
    Schedule sched{list, p.listType, p.e, BM, nTiles};
    uint32_t total=0;
    if constexpr (!Weighted) total=sched.Total();
    typename FullKPipeType<StaticNoPipe>::Type pipe;
    if ASCEND_IS_AIC {
        const uint32_t aic = GetBlockIdx();
        uint32_t first,last;
        if constexpr (Weighted) P4WeightedOwnership384::Own(list,p,aic,first,last);
        else {
            first=static_cast<uint32_t>(uint64_t(total)*aic/p.cores);
            last=static_cast<uint32_t>(uint64_t(total)*(aic+1u)/p.cores);
        }
        GlobalTensor<int8_t> xg, wg;
        GlobalTensor<half> stg;
        xg.SetGlobalBuffer((__gm__ int8_t*)xGm);
        wg.SetGlobalBuffer((__gm__ int8_t*)wGm);
        stg.SetGlobalBuffer((__gm__ half*)stGm);
        typename FullKBufferType<StaticNoPipe, TPosition::A1>::Type l1Buf;
        typename FullKBufferType<StaticNoPipe, TPosition::A2>::Type l0aBuf;
        typename FullKBufferType<StaticNoPipe, TPosition::B2>::Type l0bBuf;
        typename FullKBufferType<StaticNoPipe, TPosition::CO1>::Type l0cBuf;
        if constexpr (StaticNoPipe) l1Buf.bytes = L1_BYTES;
        else GetTPipePtr()->InitBuffer(l1Buf, L1_BYTES);
        if constexpr (StaticNoPipe) l0aBuf.bytes = L0A_BYTES;
        else GetTPipePtr()->InitBuffer(l0aBuf, L0A_BYTES);
        if constexpr (StaticNoPipe) l0bBuf.bytes = L0B_BYTES;
        else GetTPipePtr()->InitBuffer(l0bBuf, L0B_BYTES);
        if constexpr (StaticNoPipe) l0cBuf.bytes = L0C_BYTES;
        else GetTPipePtr()->InitBuffer(l0cBuf, L0C_BYTES);
        pipe.Destroy();
        LocalTensor<int8_t> l1 = l1Buf.template Get<int8_t>();
        LocalTensor<int8_t> l0a = l0aBuf.template Get<int8_t>();
        LocalTensor<int8_t> l0b = l0bBuf.template Get<int8_t>();
        LocalTensor<int32_t> l0c = l0cBuf.template Get<int32_t>();
        // p.kl is the complete padded K, never a source tensor dimension.
        const uint32_t aBytes = BM * p.kl;
        const uint32_t bBytes = p.kl * BN;
        LocalTensor<int8_t> l1a = l1;
        LocalTensor<int8_t> l1b0 = l1[aBytes];
        LocalTensor<int8_t> l1b1 = l1[aBytes + (Prefetch ? bBytes : 0u)];
        if constexpr (!SingleL0) {
            SetFlag<HardEvent::M_MTE1>(1);
            SetFlag<HardEvent::M_MTE1>(2);
        }
        uint32_t heldAExpert = 0xffffffffu, heldM = 0xffffffffu;
        uint32_t heldBExpert = 0xffffffffu, heldN = 0xffffffffu;
        uint32_t acc = 0, start = 0;
        for (uint32_t e = 0; e < p.e && acc < last; ++e) {
            const uint32_t value = static_cast<uint32_t>(list.GetValue(e));
            const uint32_t end = p.listType == 0 ? value : start + value;
            const uint32_t rows = end > start ? end - start : 0u;
            const uint32_t mBlocks = (rows + BM - 1u) / BM;
            const uint32_t count = mBlocks * nTiles;
            if (acc + count > first) {
                const uint32_t lo = first > acc ? first - acc : 0u;
                const uint32_t hi = Min(count, last - acc);
                for (uint32_t t = lo; t < hi; ++t) {
                    const uint16_t tileId=uint16_t((acc+t-first)&3u);
                    if(acc+t-first>=4u) {
                        CrossCoreWaitFlag(uint16_t(4u+tileId));
                    }
                    uint32_t mIdx, nIdx;
                    if constexpr (NMajor) {
                        mIdx = t % mBlocks;
                        nIdx = t / mBlocks;
                    } else {
                        mIdx = t / nTiles;
                        nIdx = t % nTiles;
                    }
                    const uint32_t gRow = start + mIdx * BM;
                    const uint32_t mt = Min(BM, rows - mIdx * BM);
                    const uint32_t nOff = nIdx * BN;
                    const uint32_t nt = Min(BN, p.n - nOff);
                    const bool reuseA = e == heldAExpert && mIdx == heldM;
                    const bool reuseB = e == heldBExpert && nIdx == heldN;
                    if (!reuseA) {
                        Nd2NzParams ap;
                        ap.ndNum = 1;
                        ap.nValue = static_cast<uint16_t>(mt);
                        ap.dValue = static_cast<uint16_t>(p.k);
                        ap.srcNdMatrixStride = 0;
                        ap.srcDValue = static_cast<uint16_t>(p.k);
                        ap.dstNzC0Stride = static_cast<uint16_t>(NativeM1 && mt==1u?1u:BM);
                        ap.dstNzNStride = 1;
                        ap.dstNzMatrixStride = 0;
                        DataCopy(l1a, xg[uint64_t(gRow) * p.k], ap);
                    }
                    LocalTensor<int8_t> l1b = l1b0;
                    if constexpr (Prefetch)
                        l1b = ((t - lo) & 1u) ? l1b1 : l1b0;
                    Nd2NzParams bp;
                    bp.ndNum = 1;
                    bp.nValue = static_cast<uint16_t>(p.k);
                    bp.dValue = static_cast<uint16_t>(nt);
                    bp.srcNdMatrixStride = 0;
                    bp.srcDValue = static_cast<uint16_t>(p.wstride);
                    bp.dstNzC0Stride = static_cast<uint16_t>(p.kl);
                    bp.dstNzNStride = 1;
                    bp.dstNzMatrixStride = 0;
                    if constexpr (Prefetch) {
                        if (t == lo)
                            DataCopy(l1b, wg[uint64_t(e) * p.k * p.wstride + nOff], bp);
                    } else {
                        if (!reuseB)
                            DataCopy(l1b, wg[uint64_t(e) * p.k * p.wstride + nOff], bp);
                    }
                    Barrier<HardEvent::MTE2_MTE1>();
                    if constexpr (Prefetch) {
                        if (t + 1u < hi) {
                            LocalTensor<int8_t> nextB = ((t - lo) & 1u) ? l1b0 : l1b1;
                            const uint32_t nextNOff = ((t + 1u) % nTiles) * BN;
                            Nd2NzParams nextBp = bp;
                            nextBp.dValue = static_cast<uint16_t>(Min(BN, p.n - nextNOff));
                            DataCopy(nextB, wg[uint64_t(e) * p.k * p.wstride + nextNOff], nextBp);
                        }
                    }
                    if(NativeM1 && mt==1u) {
                        // Both existing K-bank credits retire before whole-L0 reuse.
                        WaitFlag<HardEvent::M_MTE1>(1);
                        WaitFlag<HardEvent::M_MTE1>(2);
                        LoadData2DParams na;
                        na.startIndex=0;na.repeatTimes=1;na.srcStride=1;
                        na.dstGap=0;na.ifTranspose=false;na.sid=0;na.addrMode=0;
                        LoadData(l0a,l1a,na); // contiguousND X, 512B A2; trueK<=512
                        LoadData2dTransposeParams nb;
                        nb.startIndex=0;nb.repeatTimes=uint8_t(Up(nt,32u)/32u);
                        nb.srcStride=uint16_t(p.kl/32u);nb.dstGap=1;nb.dstFracGap=0;nb.addrMode=0;
                        const uint32_t nativeStride=Up(nt,32u)*32u;
                        for(uint32_t k32=0;k32<p.kl/32u;++k32)
                            LoadDataWithTranspose(l0b[k32*nativeStride],l1b[k32*1024u],nb);
                        Barrier<HardEvent::MTE1_M>();
                        Barrier<HardEvent::FIX_M>();
                        MmadParams nm;
                        nm.m=1;nm.n=uint16_t(Up(nt,32u));nm.k=uint16_t(p.k);
                        nm.cmatrixInitVal=true;nm.unitFlag=0;
                        Mmad(l0c,l0a,l0b,nm);
                        PipeBarrier<PIPE_M>();
                        SetFlag<HardEvent::M_MTE1>(1);
                        SetFlag<HardEvent::M_MTE1>(2);
                    } else {
                    bool firstK = true;
                    for (uint32_t ks = 0; ks < p.k; ks += BK) {
                        const uint32_t kstep = Min(BK, p.k - ks);
                        const uint32_t half = SingleL0 ? 0u : ((ks / BK) & 1u);
                        if constexpr (!SingleL0)
                            WaitFlag<HardEvent::M_MTE1>(half + 1u);
                        LocalTensor<int8_t> a = l0a[half * BM * BK];
                        LocalTensor<int8_t> b = l0b[half * BK * BN];
                        const uint32_t kFrac = Up(kstep, C0_I8) / C0_I8;
                        const uint32_t mFrac = Up(mt, ROW_FRAC) / ROW_FRAC;
                        // int8 transpose emits a 32-column pair. N16 stride
                        // aliases the pair's destination when nt is a small tail.
                        const uint32_t bRowStride = Up(nt, C0_I8) * C0_I8;
                        LoadData2DParams la;
                        la.startIndex = 0;
                        la.repeatTimes = static_cast<uint8_t>(mFrac);
                        la.srcStride = 1;
                        la.dstGap = static_cast<uint16_t>(kFrac - 1u);
                        la.ifTranspose = false;
                        la.sid = 0;
                        la.addrMode = 0;
                        for (uint32_t j = 0; j < kFrac; ++j)
                            LoadData(a[j * FRAC_I8], l1a[ks * BM + j * BM * C0_I8], la);
                        LoadData2dTransposeParams lb;
                        lb.startIndex = 0;
                        lb.repeatTimes = static_cast<uint8_t>(Up(nt, C0_I8) / C0_I8);
                        lb.srcStride = static_cast<uint16_t>(p.kl / C0_I8);
                        lb.dstGap = 1;
                        lb.dstFracGap = 0;
                        lb.addrMode = 0;
                        for (uint32_t i = 0; i < kFrac; ++i)
                            LoadDataWithTranspose(b[i * bRowStride],
                                l1b[ks * C0_I8 + i * 2u * FRAC_I8], lb);
                        Barrier<HardEvent::MTE1_M>();
                        MmadParams mp;
                        mp.m = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                        mp.n = static_cast<uint16_t>(Up(nt, C0_I8));
                        mp.k = static_cast<uint16_t>(kstep);
                        mp.cmatrixInitVal = firstK;
                        mp.unitFlag = 0;
                        if (firstK) Barrier<HardEvent::FIX_M>();
                        Mmad(l0c, a, b, mp);
                        // Required small-L0C read-after-write dependency.
                        if ((uint32_t(mp.m) / 16u) * (uint32_t(mp.n) / 16u) < 10u)
                            PipeBarrier<PIPE_M>();
                        if constexpr (SingleL0)
                            Barrier<HardEvent::M_MTE1>();
                        else
                            SetFlag<HardEvent::M_MTE1>(half + 1u);
                        firstK = false;
                    }
                    }
                    // Future A/B DMA may recycle L1 only after the current
                    // operand loads finish; Matrix may still consume L0.
                    Barrier<HardEvent::MTE1_MTE2>();
                    heldAExpert = e; heldM = mIdx;
                    heldBExpert = e; heldN = nIdx;
                    Barrier<HardEvent::M_FIX>();
                    FixpipeParamsV220 fp;
                    fp.nSize = static_cast<uint16_t>(nt);
                    fp.mSize = static_cast<uint16_t>(mt);
                    fp.srcStride = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                    fp.dstStride = BN;
                    fp.quantPre = QuantMode_t::DEQF16;
                    fp.deqScalar = 0x39800000ull; // FP32 2^-12, exact scalar encoding
                    fp.reluEn = false;
                    fp.unitFlag = 0;
                    const uint64_t base = uint64_t(acc + t) * BM * BN;
                    Fixpipe<half, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);
                    CrossCoreSetFlag<2, PIPE_FIX>(tileId);
                }
            }
            acc += count;
            start = end;
        }
        // Drain acknowledgements for the final four notification slots.
        const uint32_t owned=last-first;
        for(uint32_t q=owned>4u?owned-4u:owned;q<owned;++q) {
            CrossCoreWaitFlag(uint16_t(4u+(q&3u)));
        }
        if constexpr (!SingleL0) {
            WaitFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_MTE1>(2);
        }
    } else {
        const uint32_t aic = GetBlockIdx() / GetTaskRation();
        const uint32_t sub = GetSubBlockIdx();
        uint32_t first,last;
        if constexpr (Weighted) P4WeightedOwnership384::Own(list,p,aic,first,last);
        else {
            first=static_cast<uint32_t>(uint64_t(total)*aic/p.cores);
            last=static_cast<uint32_t>(uint64_t(total)*(aic+1u)/p.cores);
        }
        GlobalTensor<half> stg;
        GlobalTensor<float> xsg;
        GlobalTensor<ScaleT> wsg;
        GlobalTensor<OutT> yg;
        stg.SetGlobalBuffer((__gm__ half*)stGm);
        xsg.SetGlobalBuffer((__gm__ float*)xsGm);
        wsg.SetGlobalBuffer((__gm__ ScaleT*)wsGm);
        yg.SetGlobalBuffer((__gm__ OutT*)yGm);
        typename FullKBufferType<StaticNoPipe, TPosition::VECCALC>::Type ub;
        if constexpr (StaticNoPipe) ub.bytes = UB_USED;
        else GetTPipePtr()->InitBuffer(ub, UB_USED);
        pipe.Destroy();
        LocalTensor<uint8_t> raw = ub.template Get<uint8_t>();
        uint32_t off = 0;
        LocalTensor<half> accI = raw[off].ReinterpretCast<half>();
        off += P4ContiguousScale ? 81920u : RV * PITCH * 4u;
        LocalTensor<float> accF = raw[off].ReinterpretCast<float>();
        off += P4ContiguousScale ? 49152u : RV * PITCH * 4u;
        LocalTensor<OutT> out = raw[off].ReinterpretCast<OutT>();
        off += RV * PITCH * 2u;
        LocalTensor<ScaleT> wsl = raw[off].ReinterpretCast<ScaleT>();
        off += PITCH * 4u;
        LocalTensor<float> wsf = raw[off].ReinterpretCast<float>();
        off += PITCH * 4u;
        LocalTensor<float> xsl = raw[off].ReinterpretCast<float>();
        off += RV * 4u;
        LocalTensor<float> xsb = raw[off].ReinterpretCast<float>();
        LocalTensor<float> scaleProduct = raw[UB_BASE].ReinterpretCast<float>();
        LocalTensor<float> scaleXS = raw[UB_BASE+SCALE_ROWS*PITCH*4u].ReinterpretCast<float>();
        uint32_t heldScaleExpert=0xffffffffu,heldScaleN=0xffffffffu;
        uint32_t heldXSRow=0xffffffffu,heldXSRows=0xffffffffu;
        // accI reserves 64KiB but its half input occupies at most32KiB.
        LocalTensor<ScaleT> spanWSRaw = raw[P4ContiguousScale ? 24576u : RV*PITCH*2u].ReinterpretCast<ScaleT>();
        LocalTensor<float> spanWSScaled = raw[(P4ContiguousScale ? 24576u : RV*PITCH*2u)+4096u].ReinterpretCast<float>();
        uint32_t spanExpert=0xffffffffu,spanM=0xffffffffu,spanFirstN=0u,spanEndN=0u;
        uint32_t acc = 0, start = 0;
        for (uint32_t e = 0; e < p.e && acc < last; ++e) {
            const uint32_t value = static_cast<uint32_t>(list.GetValue(e));
            const uint32_t end = p.listType == 0 ? value : start + value;
            const uint32_t rows = end > start ? end - start : 0u;
            const uint32_t mBlocks = (rows + BM - 1u) / BM;
            const uint32_t count = mBlocks * nTiles;
            if (acc + count > first) {
                const uint32_t lo = first > acc ? first - acc : 0u;
                const uint32_t hi = Min(count, last - acc);
                for (uint32_t t = lo; t < hi; ++t) {
                    const uint16_t tileId=uint16_t((acc+t-first)&3u);
                    uint32_t mIdx, nIdx;
                    if constexpr (NMajor) {
                        mIdx = t % mBlocks;
                        nIdx = t / mBlocks;
                    } else {
                        mIdx = t / nTiles;
                        nIdx = t % nTiles;
                    }
                    const uint32_t gRow = start + mIdx * BM;
                    const uint32_t mt = Min(BM, rows - mIdx * BM);
                    const uint32_t nOff = nIdx * BN;
                    const uint32_t nt = Min(BN, p.n - nOff);
                    const uint64_t base = uint64_t(acc + t) * BM * BN;
                    if constexpr (BM <= 32 || BM == 208) { CrossCoreWaitFlag(tileId); if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId)); }
                    const uint32_t half = (mt + AIV_PER_AIC - 1u) / AIV_PER_AIC;
                    const uint32_t r0 = Min(sub * half, mt);
                    const uint32_t r1 = Min((sub + 1u) * half, mt);
                    LocalTensor<float> tileWS = wsf;
                    if constexpr (WSSpan) {
                        if (r0 < r1) {
                            // hi/lo are this expert's slice on this core, not global N.
                            if (spanExpert!=e || spanM!=mIdx || nIdx<spanFirstN || nIdx>=spanEndN) {
                                const uint32_t runEnd=Min(hi,(mIdx+1u)*nTiles);
                                spanFirstN=nIdx;spanEndN=runEnd-mIdx*nTiles;
                                const uint32_t spanNt=Min(p.n,spanEndN*BN)-nOff;
                                DataCopyPad(spanWSRaw,wsg[uint64_t(e)*p.n+nOff],
                                    DataCopyExtParams{1,spanNt*uint32_t(sizeof(ScaleT)),0,0,0},
                                    DataCopyPadExtParams<ScaleT>{false,0,0,0});
                                Barrier<HardEvent::MTE2_V>();
                                Muls(spanWSScaled,spanWSRaw,4096.0f,spanNt);
                                PipeBarrier<PIPE_V>();
                                spanExpert=e;spanM=mIdx;
                            }
                            tileWS=spanWSScaled[(nIdx-spanFirstN)*BN];
                        }
                    } else {
                    if (r0 < r1 && (!NMajor || heldScaleExpert!=e || heldScaleN!=nIdx)) {
                        heldScaleExpert=e;heldScaleN=nIdx;
                        DataCopyPad(wsl, wsg[uint64_t(e) * p.n + nOff],
                            DataCopyExtParams{1, uint32_t(nt * sizeof(ScaleT)), 0, 0, 0},
                            DataCopyPadExtParams<ScaleT>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        Muls(wsf, wsl, 4096.0f, nt); // WS*2^12 exact; replaces the old FP32 copy
                        PipeBarrier<PIPE_V>();
                    }
                    }
                    if constexpr(P4ContiguousScale) {
                        if(r0<r1) {
                            const uint32_t ownedRows=r1-r0;
                            const bool loadXS=heldXSRow!=gRow+r0 || heldXSRows!=ownedRows;
                            if(loadXS) {
                                DataCopyPad(xsl,xsg[gRow+r0],
                                    DataCopyExtParams{1,ownedRows*4u,0,0,0},
                                    DataCopyPadExtParams<float>{false,0,0,0});
                                Barrier<HardEvent::MTE2_V>();
                                BrcbRepeatParams br;br.dstBlkStride=1;br.dstRepStride=8;
                                Brcb(xsb,xsl,uint8_t((ownedRows+7u)/8u),br);
                                PipeBarrier<PIPE_V>();
                                heldXSRow=gRow+r0;heldXSRows=ownedRows;
                            }
                            // Stage24KiB + WS8KiB + WM48KiB + accF48KiB; output unchanged.
                            LocalTensor<float> wm=raw[32768u].ReinterpretCast<float>();
                            BinaryRepeatParams outer;
                            outer.dstBlkStride=1;outer.src0BlkStride=1;outer.src1BlkStride=0;
                            outer.dstRepStride=PITCH/8u;outer.src0RepStride=0;outer.src1RepStride=1;
                            for(uint32_t c=0;c<nt;c+=64u)
                                Mul(wm[c],tileWS[c],xsb,uint64_t(Min(64u,nt-c)),uint8_t(ownedRows),outer);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    if constexpr(PrecombinedScale) {
                        if(r0<r1) {
                            DataCopyPad(scaleXS,xsg[gRow+r0],
                                DataCopyExtParams{1,(r1-r0)*4u,0,0,0},
                                DataCopyPadExtParams<float>{false,0,0,0});
                            Barrier<HardEvent::MTE2_V>();
                            for(uint32_t rr=0;rr<r1-r0;rr+=RV) {
                                const uint32_t nr=Min(RV,r1-r0-rr);
                                BrcbRepeatParams br;br.dstBlkStride=1;br.dstRepStride=8;
                                Brcb(xsb,scaleXS[rr],uint8_t((nr+7u)/8u),br);
                                PipeBarrier<PIPE_V>();
                                BinaryRepeatParams outer;
                                outer.dstBlkStride=1;outer.src0BlkStride=1;outer.src1BlkStride=0;
                                outer.dstRepStride=PITCH/8u;outer.src0RepStride=0;outer.src1RepStride=1;
                                for(uint32_t c=0;c<nt;c+=64u)
                                    Mul(scaleProduct[rr*PITCH+c],tileWS[c],xsb,
                                        uint64_t(Min(64u,nt-c)),uint8_t(nr),outer);
                                PipeBarrier<PIPE_V>();
                            }
                        }
                    }
                    if constexpr (BM > 32 && BM != 208) { CrossCoreWaitFlag(tileId); if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId)); }
                    const uint32_t ntr = Up(nt, 16u);
                    for (uint32_t r = r0; r < r1; r += RV) {
                        const uint32_t nr = Min(RV, r1 - r);
                        DataCopyExtParams stageCopy{uint16_t(nr), nt * 2u, (BN - nt) * 2u,
                                                    (PITCH - ntr) / 16u, 0};
                        if constexpr (SingleBlockStage) {
                            if (nt == BN) stageCopy = DataCopyExtParams{1, nr * BN * 2u, 0, 0, 0};
                        }
                        DataCopyPad(accI, stg[base + uint64_t(r) * BN], stageCopy,
                            DataCopyPadExtParams<OutT>{false, 0, 0, 0});
                        if constexpr(P4ContiguousScale || PrecombinedScale) {
                            Barrier<HardEvent::MTE2_V>();
                            Cast(accF,accI,RoundMode::CAST_NONE,nr*PITCH);
                            PipeBarrier<PIPE_V>();
                        } else if constexpr (CacheXS) {
                        const bool loadXS=heldXSRow!=gRow+r || heldXSRows!=nr;
                        if(loadXS) {
                            DataCopyPad(xsl, xsg[gRow + r],
                                DataCopyExtParams{1, nr * 4u, 0, 0, 0},
                                DataCopyPadExtParams<float>{false, 0, 0, 0});
                        }
                        Barrier<HardEvent::MTE2_V>();
                        Cast(accF, accI, RoundMode::CAST_NONE, nr * PITCH);
                        if(loadXS) {
                            BrcbRepeatParams br;
                            br.dstBlkStride = 1;
                            br.dstRepStride = 8;
                            Brcb(xsb, xsl, static_cast<uint8_t>((nr + 7u) / 8u), br);
                            heldXSRow=gRow+r;heldXSRows=nr;
                        }
                        PipeBarrier<PIPE_V>();
                        } else {
                        DataCopyPad(xsl, xsg[gRow + r],
                            DataCopyExtParams{1, nr * 4u, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        Cast(accF, accI, RoundMode::CAST_NONE, nr * PITCH);
                        BrcbRepeatParams br;
                        br.dstBlkStride = 1;
                        br.dstRepStride = 8;
                        Brcb(xsb, xsl, static_cast<uint8_t>((nr + 7u) / 8u), br);
                        PipeBarrier<PIPE_V>();
                        }
                        if constexpr(P4ContiguousScale) {
                            LocalTensor<float> wm=raw[32768u].ReinterpretCast<float>();
                            BinaryRepeatParams apply;
                            apply.dstBlkStride=1;apply.src0BlkStride=1;apply.src1BlkStride=1;
                            apply.dstRepStride=8;apply.src0RepStride=8;apply.src1RepStride=8;
                            // Each flat product stays in its column; dead padding is never stored.
                            Mul(accF,accF,wm,uint64_t(64),uint8_t(nr*2u),apply);
                            PipeBarrier<PIPE_V>();
                        } else if constexpr(PrecombinedScale) {
                            BinaryRepeatParams apply;
                            apply.dstBlkStride=1;apply.src0BlkStride=1;apply.src1BlkStride=1;
                            apply.dstRepStride=PITCH/8u;apply.src0RepStride=PITCH/8u;apply.src1RepStride=PITCH/8u;
                            for(uint32_t c=0;c<nt;c+=64u)
                                Mul(accF[c],accF[c],scaleProduct[(r-r0)*PITCH+c],
                                    uint64_t(Min(64u,nt-c)),uint8_t(nr),apply);
                            PipeBarrier<PIPE_V>();
                        } else {
                        MulByRow(accF, accF, tileWS, nr, nt, PITCH);
                        PipeBarrier<PIPE_V>();
                        BinaryRepeatParams rowScale;
                        rowScale.dstBlkStride=1;rowScale.src0BlkStride=1;rowScale.src1BlkStride=0;
                        rowScale.dstRepStride=PITCH/8;rowScale.src0RepStride=PITCH/8;rowScale.src1RepStride=1;
                        for (uint32_t c=0;c<nt;c+=64)
                            Mul(accF[c],accF[c],xsb,uint64_t(Min(64u,nt-c)),uint8_t(nr),rowScale);
                        PipeBarrier<PIPE_V>();
                        }
                        Barrier<HardEvent::MTE3_V>();
                        Cast(out, accF, RoundMode::CAST_RINT, nr * PITCH);
                        Barrier<HardEvent::V_MTE3>();
                        DataCopyPad(yg[uint64_t(gRow + r) * p.n + nOff], out,
                            DataCopyExtParams{uint16_t(nr), nt * 2u,
                                              (PITCH * 2u - Up(nt * 2u, BLK)) / BLK,
                                              (p.n - nt) * 2u, 0});
                        Barrier<HardEvent::V_MTE2>();
                    }
                }
            }
            acc += count;
            start = end;
        }
        Barrier<HardEvent::MTE3_S>();
    }
}

}


