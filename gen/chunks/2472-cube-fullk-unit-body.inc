template <typename ScaleT, typename OutT,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t RV,
          bool NMajor = false, bool Prefetch = false, bool SingleL0 = false, bool StaticNoPipe = false, bool CacheXS = false>
__aicore__ inline void CubeFullKUnitBody(
    GM_ADDR xGm, GM_ADDR wGm, GM_ADDR xsGm, GM_ADDR wsGm,
    GM_ADDR glGm, GM_ADDR yGm, GM_ADDR stGm, Plan p) {
    using namespace AscendC;
    static_assert(BM > 0 && BM % 16 == 0, "BM must align to the M fractile");
    static_assert(BN > 0 && BN % 32 == 0, "BN must align to int8 transpose pairs");
    static_assert(BK > 0 && BK % 32 == 0, "BK must align to int8 K fractiles");
    static_assert(RV > 0 && RV <= BM && RV % 8 == 0, "RV must align Brcb rows");
    static_assert(sizeof(OutT) == 2, "The output must be fp16 or bf16");
    static_assert(!(NMajor && Prefetch), "B prefetch is for M-major traversal");
    static_assert(BM * BK * (SingleL0 ? 1u : 2u) <= L0A_BYTES, "L0A overflow");
    static_assert(BN * BK * (SingleL0 ? 1u : 2u) <= L0B_BYTES, "L0B overflow");
    static_assert(BM * BN * sizeof(int32_t) <= L0C_BYTES, "L0C overflow");
    constexpr uint32_t PITCH = BN;
    constexpr uint32_t OLD_UB_USED = RV * PITCH * 10u + PITCH * 8u + RV * 36u;
    constexpr uint32_t UB_USED = OLD_UB_USED + (CacheXS ? RV * 36u : 0u);
    static_assert(!CacheXS || (BM==224u && BN==128u && BK==128u && RV==64u && !NMajor && Prefetch && !SingleL0 && StaticNoPipe && sizeof(ScaleT)==2u),"CacheXS only original P14 BF family");
    static_assert(UB_USED <= 192u * 1024u - 8192u - 256u, "UB overflow");

    const uint32_t nTiles = (p.n + BN - 1u) / BN;
    GlobalTensor<int64_t> list;
    list.SetGlobalBuffer((__gm__ int64_t*)glGm);
    Schedule sched{list, p.listType, p.e, BM, nTiles};
    const uint32_t total = sched.Total();
    typename FullKPipeType<StaticNoPipe>::Type pipe;
    if ASCEND_IS_AIC {
        const uint32_t aic = GetBlockIdx();
        const uint32_t first = static_cast<uint32_t>(uint64_t(total) * aic / p.cores);
        const uint32_t last = static_cast<uint32_t>(uint64_t(total) * (aic + 1u) / p.cores);
        GlobalTensor<int8_t> xg, wg;
        GlobalTensor<int32_t> stg;
        xg.SetGlobalBuffer((__gm__ int8_t*)xGm);
        wg.SetGlobalBuffer((__gm__ int8_t*)wGm);
        stg.SetGlobalBuffer((__gm__ int32_t*)stGm);
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
        // Official CANN9 example uses CO1 tensor solely for unit-flag clearing.
        // NoQuant ignores the FPC quantization-pointer fields.
        SetFixPipeConfig<uint64_t>(l0c.ReinterpretCast<uint64_t>(), true);
        Barrier<HardEvent::FIX_M>();
        SetMMRowMajor();

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
                        ap.dstNzC0Stride = static_cast<uint16_t>(BM);
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
                        mp.unitFlag = (ks + BK >= p.k) ? 3u : 2u;
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
                    // Future A/B DMA may recycle L1 only after the current
                    // operand loads finish; Matrix may still consume L0.
                    Barrier<HardEvent::MTE1_MTE2>();
                    heldAExpert = e; heldM = mIdx;
                    heldBExpert = e; heldN = nIdx;
                    FixpipeParamsV220 fp;
                    fp.nSize = static_cast<uint16_t>(Up(nt, C0_I8));
                    fp.mSize = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                    fp.srcStride = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                    fp.dstStride = BN;
                    fp.quantPre = QuantMode_t::NoQuant;
                    fp.reluEn = false;
                    fp.unitFlag = 3;
                    const uint64_t base = uint64_t(acc + t) * BM * BN;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);
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
        // Drain the final complete consumer before changing MMAD direction.
        Barrier<HardEvent::FIX_M>();
        SetMMColumnMajor();
    } else {
        const uint32_t aic = GetBlockIdx() / GetTaskRation();
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t first = static_cast<uint32_t>(uint64_t(total) * aic / p.cores);
        const uint32_t last = static_cast<uint32_t>(uint64_t(total) * (aic + 1u) / p.cores);
        GlobalTensor<int32_t> stg;
        GlobalTensor<float> xsg;
        GlobalTensor<ScaleT> wsg;
        GlobalTensor<OutT> yg;
        stg.SetGlobalBuffer((__gm__ int32_t*)stGm);
        xsg.SetGlobalBuffer((__gm__ float*)xsGm);
        wsg.SetGlobalBuffer((__gm__ ScaleT*)wsGm);
        yg.SetGlobalBuffer((__gm__ OutT*)yGm);
        typename FullKBufferType<StaticNoPipe, TPosition::VECCALC>::Type ub;
        if constexpr (StaticNoPipe) ub.bytes = UB_USED;
        else GetTPipePtr()->InitBuffer(ub, UB_USED);
        pipe.Destroy();
        LocalTensor<uint8_t> raw = ub.template Get<uint8_t>();
        uint32_t off = 0;
        LocalTensor<int32_t> accI = raw[off].ReinterpretCast<int32_t>();
        off += RV * PITCH * 4u;
        LocalTensor<float> accF = raw[off].ReinterpretCast<float>();
        off += RV * PITCH * 4u;
        LocalTensor<OutT> out = raw[off].ReinterpretCast<OutT>();
        off += RV * PITCH * 2u;
        LocalTensor<ScaleT> wsl = raw[off].ReinterpretCast<ScaleT>();
        off += PITCH * 4u;
        LocalTensor<float> wsf = raw[off].ReinterpretCast<float>();
        off += PITCH * 4u;
        LocalTensor<float> xsl = raw[off].ReinterpretCast<float>();
        off += RV * 4u;
        LocalTensor<float> xsb = raw[off].ReinterpretCast<float>();
        // CacheXS=false leaves these scalar views unused and every original UB address intact.
        LocalTensor<float> xsl1 = raw[CacheXS ? OLD_UB_USED + RV*32u : 0u].ReinterpretCast<float>();
        uint32_t heldXSExpert=0xffffffffu,heldXSM=0xffffffffu;
        uint32_t heldScaleExpert=0xffffffffu,heldScaleN=0xffffffffu;
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
                    if constexpr (CacheXS) {
                    const bool reloadXS=r0<r1 && (heldXSExpert!=e || heldXSM!=mIdx);
                    const uint32_t xsRows0=Min(RV,r1-r0),xsRows1=(r1-r0)-xsRows0;
                    if(reloadXS) {
                        DataCopyPad(xsl,xsg[gRow+r0],DataCopyExtParams{1,xsRows0*4u,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                        if(xsRows1)DataCopyPad(xsl1,xsg[gRow+r0+RV],DataCopyExtParams{1,xsRows1*4u,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    }
                    if (r0 < r1 && (!NMajor || heldScaleExpert!=e || heldScaleN!=nIdx)) {
                        heldScaleExpert=e;heldScaleN=nIdx;
                        DataCopyPad(wsl, wsg[uint64_t(e) * p.n + nOff],
                            DataCopyExtParams{1, uint32_t(nt * sizeof(ScaleT)), 0, 0, 0},
                            DataCopyPadExtParams<ScaleT>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        ToFloat<ScaleT>(wsf, wsl, nt);
                        if(reloadXS) {
                            BrcbRepeatParams br;br.dstBlkStride=1;br.dstRepStride=8;
                            Brcb(xsb,xsl,uint8_t((xsRows0+7u)/8u),br);
                            if(xsRows1)Brcb(xsb[RV*8u],xsl1,uint8_t((xsRows1+7u)/8u),br);
                            heldXSExpert=e;heldXSM=mIdx;
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    } else {
                    if (r0 < r1 && (!NMajor || heldScaleExpert!=e || heldScaleN!=nIdx)) {
                        heldScaleExpert=e;heldScaleN=nIdx;
                        DataCopyPad(wsl, wsg[uint64_t(e) * p.n + nOff],
                            DataCopyExtParams{1, uint32_t(nt * sizeof(ScaleT)), 0, 0, 0},
                            DataCopyPadExtParams<ScaleT>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        ToFloat<ScaleT>(wsf, wsl, nt);
                        PipeBarrier<PIPE_V>();
                    }
                    }
                    if constexpr (BM > 32 && BM != 208) { CrossCoreWaitFlag(tileId); if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId)); }
                    const uint32_t ntr = Up(nt, F32_PER_BLK);
                    for (uint32_t r = r0; r < r1; r += RV) {
                        const uint32_t nr = Min(RV, r1 - r);
                        DataCopyPad(accI, stg[base + uint64_t(r) * BN],
                            DataCopyExtParams{uint16_t(nr), nt * 4u, (BN - nt) * 4u,
                                              (PITCH - ntr) / F32_PER_BLK, 0},
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                        if constexpr (!CacheXS) {
                        DataCopyPad(xsl, xsg[gRow + r],
                            DataCopyExtParams{1, nr * 4u, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                        }
                        Barrier<HardEvent::MTE2_V>();
                        Cast(accF, accI, RoundMode::CAST_RINT, nr * PITCH);
                        if constexpr (!CacheXS) {
                        BrcbRepeatParams br;
                        br.dstBlkStride = 1;
                        br.dstRepStride = 8;
                        Brcb(xsb, xsl, static_cast<uint8_t>((nr + 7u) / 8u), br);
                        }
                        PipeBarrier<PIPE_V>();
                        MulByRow(accF, accF, wsf, nr, nt, PITCH);
                        PipeBarrier<PIPE_V>();
                        BinaryRepeatParams rowScale;
                        rowScale.dstBlkStride=1;rowScale.src0BlkStride=1;rowScale.src1BlkStride=0;
                        rowScale.dstRepStride=PITCH/8;rowScale.src0RepStride=PITCH/8;rowScale.src1RepStride=1;
                        for (uint32_t c=0;c<nt;c+=64)
                            if constexpr (CacheXS)
                                Mul(accF[c],accF[c],xsb[(r-r0)*8u],uint64_t(Min(64u,nt-c)),uint8_t(nr),rowScale);
                            else
                            Mul(accF[c],accF[c],xsb,uint64_t(Min(64u,nt-c)),uint8_t(nr),rowScale);
                        PipeBarrier<PIPE_V>();
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

