
template <typename ScaleT, typename OutT>
__global__ __aicore__ void CubeMix(GM_ADDR xGm, GM_ADDR wGm, GM_ADDR xsGm, GM_ADDR wsGm,
                                   GM_ADDR glGm, GM_ADDR yGm, GM_ADDR stGm, Plan p) {
    using namespace AscendC;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GlobalTensor<int64_t> list;
    list.SetGlobalBuffer((__gm__ int64_t*)glGm);
    Schedule sched{list, p.listType, p.e, p.m0, p.nTiles};
    uint32_t total = sched.Total();
    uint32_t aic, sub = 0, first, last;
    TPipe pipe;
    if ASCEND_IS_AIC {
        aic = GetBlockIdx();
        first = static_cast<uint32_t>(uint64_t(total) * aic / p.cores);
        last = static_cast<uint32_t>(uint64_t(total) * (aic + 1) / p.cores);
        GlobalTensor<int8_t> xg, wg;
        GlobalTensor<int32_t> stg;
        xg.SetGlobalBuffer((__gm__ int8_t*)xGm);
        wg.SetGlobalBuffer((__gm__ int8_t*)wGm);
        stg.SetGlobalBuffer((__gm__ int32_t*)stGm);
        TBuf<TPosition::A1> l1Buf;
        TBuf<TPosition::A2> l0aBuf;
        TBuf<TPosition::B2> l0bBuf;
        TBuf<TPosition::CO1> l0cBuf;
        GetTPipePtr()->InitBuffer(l1Buf, L1_BYTES);
        GetTPipePtr()->InitBuffer(l0aBuf, L0A_BYTES);
        GetTPipePtr()->InitBuffer(l0bBuf, L0B_BYTES);
        GetTPipePtr()->InitBuffer(l0cBuf, L0C_BYTES);
        pipe.Destroy();
        LocalTensor<int8_t> l1 = l1Buf.Get<int8_t>();
        LocalTensor<int8_t> l0a = l0aBuf.Get<int8_t>();
        LocalTensor<int8_t> l0b = l0bBuf.Get<int8_t>();
        LocalTensor<int32_t> l0c = l0cBuf.Get<int32_t>();
        const uint32_t aBytes = p.m0r * p.kl;
        const uint32_t bBytes = p.kl * p.n0c;
        LocalTensor<int8_t> l1a = l1;
        LocalTensor<int8_t> l1b0 = l1[aBytes];
        LocalTensor<int8_t> l1b1 = l1[aBytes + bBytes];
        const uint32_t l0aHalf = p.m0r * p.k0;
        const uint32_t l0bHalf = p.k0 * p.n0r;
        uint32_t heldA = 0xffffffffu;
        uint32_t heldM = 0xffffffffu;
        uint32_t acc = 0, start = 0;
        for (uint32_t e = 0; e < p.e && acc < last; ++e) {
            uint32_t v = static_cast<uint32_t>(list.GetValue(e));
            uint32_t end = p.listType == 0 ? v : start + v;
            // See Schedule::Total: unsigned, so it must not underflow.
            uint32_t rows = (end > start) ? (end - start) : 0u;
            uint32_t mBlocks = (rows + p.m0 - 1) / p.m0;
            uint32_t count = mBlocks * p.nTiles;
            if (acc + count > first) {
                uint32_t lo = first > acc ? first - acc : 0;
                uint32_t hi = Min(count, last - acc);
                for (uint32_t t = lo; t < hi; ++t) {
                    const uint32_t relative=acc+t-first;
                    const uint16_t tileId=uint16_t(relative&3u);
                    if(relative>=4u) CrossCoreWaitFlag(uint16_t(4u+tileId));
                    uint32_t mIdx = t / p.nTiles;
                    uint32_t nIdx = t % p.nTiles;
                    uint32_t gRow = start + mIdx * p.m0;
                    uint32_t mt = Min(p.m0, rows - mIdx * p.m0);
                    uint32_t nOff = nIdx * p.n0;
                    uint32_t nt = Min(p.n0, p.n - nOff);
                    bool reuseA = (e == heldA && mIdx == heldM && p.kl >= p.k);
                    bool firstK = true;
                    for (uint32_t kc = 0; kc < p.k; kc += p.kl) {
                        uint32_t klc = Min(p.kl, p.k - kc);
                        uint32_t klr = Up(klc, C0_I8);
                        if (!reuseA) {
                            Nd2NzParams ap;
                            ap.ndNum = 1;
                            ap.nValue = static_cast<uint16_t>(mt);
                            ap.dValue = static_cast<uint16_t>(klc);
                            ap.srcNdMatrixStride = 0;
                            ap.srcDValue = static_cast<uint16_t>(p.k);
                            ap.dstNzC0Stride = static_cast<uint16_t>(p.m0r);
                            ap.dstNzNStride = 1;
                            ap.dstNzMatrixStride = 0;
                            DataCopy(l1a, xg[uint64_t(gRow) * p.k + kc], ap);
                        }
                        LocalTensor<int8_t> l1b = ((kc / p.kl) & 1) ? l1b1 : l1b0;
                        Nd2NzParams bp;
                        bp.ndNum = 1;
                        bp.nValue = static_cast<uint16_t>(klc);
                        bp.dValue = static_cast<uint16_t>(nt);
                        bp.srcNdMatrixStride = 0;
                        bp.srcDValue = static_cast<uint16_t>(p.n);
                        bp.dstNzC0Stride = static_cast<uint16_t>(p.kl);
                        bp.dstNzNStride = 1;
                        bp.dstNzMatrixStride = 0;
                        DataCopy(l1b, wg[(uint64_t(e) * p.k + kc) * p.n + nOff], bp);
                        Barrier<HardEvent::MTE2_MTE1>();
                        for (uint32_t ks = 0; ks < klr; ks += p.k0) {
                            uint32_t kstep = Min(p.k0, klc - Min(klc, ks));
                            if (kstep == 0) break;
                            uint32_t half = ((ks / p.k0) & 1);
                            LocalTensor<int8_t> a = l0a[half * l0aHalf];
                            LocalTensor<int8_t> b = l0b[half * l0bHalf];
                            const uint32_t kFrac = Up(kstep, C0_I8) / C0_I8;
                            const uint32_t aRowStride = kFrac * C0_I8 * ROW_FRAC;
                            const uint32_t bRowStride = Up(nt, C0_I8) * C0_I8;
                            const uint32_t mFrac = Up(mt, ROW_FRAC) / ROW_FRAC;
                            LoadData2DParams la;
                            la.startIndex = 0;
                            la.repeatTimes = static_cast<uint8_t>(mFrac);
                            la.srcStride = 1;
                            la.dstGap = static_cast<uint16_t>(kFrac - 1);
                            la.ifTranspose = false;
                            la.sid = 0;
                            la.addrMode = 0;
                            for (uint32_t j = 0; j < kFrac; ++j)
                                LoadData(a[j * FRAC_I8],
                                         l1a[ks * p.m0r + j * p.m0r * C0_I8], la);
                            LoadData2dTransposeParams lb;
                            lb.startIndex = 0;
                            lb.repeatTimes = static_cast<uint8_t>(Up(nt, C0_I8) / C0_I8);
                            lb.srcStride = static_cast<uint16_t>(p.kl / C0_I8);
                            lb.dstGap = 1;
                            lb.dstFracGap = 0;
                            lb.addrMode = 0;
                            for (uint32_t i = 0; i < kFrac; ++i)
                                LoadDataWithTranspose(b[i * bRowStride],
                                                      l1b[ks * C0_I8 + i * 2 * FRAC_I8], lb);
                            Barrier<HardEvent::MTE1_M>();
                            MmadParams mp;
                            mp.m = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                            mp.n = static_cast<uint16_t>(Up(nt, C0_I8));
                            mp.k = static_cast<uint16_t>(kstep);
                            mp.cmatrixInitVal = firstK;
                            mp.unitFlag = 0;
                            if (firstK) Barrier<HardEvent::FIX_M>();
                            Mmad(l0c, a, b, mp);
                            Barrier<HardEvent::M_MTE1>();
                            firstK = false;
                        }
                        Barrier<HardEvent::MTE1_MTE2>();
                    }
                    heldA = e;
                    heldM = mIdx;
                    Barrier<HardEvent::M_FIX>();
                    FixpipeParamsV220 fp;
                    fp.nSize = static_cast<uint16_t>(nt);
                    fp.mSize = static_cast<uint16_t>(mt);
                    fp.srcStride = static_cast<uint16_t>(Up(mt, ROW_FRAC));
                    fp.dstStride = p.n0;
                    fp.quantPre = QuantMode_t::NoQuant;
                    fp.reluEn = false;
                    fp.unitFlag = 0;
                    uint64_t base = uint64_t(acc + t) * p.m0 * p.n0;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(stg[base], l0c, fp);
                    CrossCoreSetFlag<2, PIPE_FIX>(tileId);
                }
            }
            acc += count;
            start = end;
        }
        const uint32_t owned=last-first;
        for(uint32_t q=owned>4u?owned-4u:owned;q<owned;++q)
            CrossCoreWaitFlag(uint16_t(4u+(q&3u)));
    } else {
        aic = GetBlockIdx() / GetTaskRation();
        sub = GetSubBlockIdx();
        first = static_cast<uint32_t>(uint64_t(total) * aic / p.cores);
        last = static_cast<uint32_t>(uint64_t(total) * (aic + 1) / p.cores);
        GlobalTensor<int32_t> stg;
        GlobalTensor<float> xsg;
        GlobalTensor<ScaleT> wsg;
        GlobalTensor<OutT> yg;
        stg.SetGlobalBuffer((__gm__ int32_t*)stGm);
        xsg.SetGlobalBuffer((__gm__ float*)xsGm);
        wsg.SetGlobalBuffer((__gm__ ScaleT*)wsGm);
        yg.SetGlobalBuffer((__gm__ OutT*)yGm);
        TBuf<TPosition::VECCALC> ub;
        GetTPipePtr()->InitBuffer(ub, UB_BYTES);
        pipe.Destroy();
        LocalTensor<uint8_t> raw = ub.Get<uint8_t>();
        const uint32_t pitch = Up(p.n0, F32_PER_BLK);
        uint32_t off = 0;
        LocalTensor<int32_t> accI = raw[off].ReinterpretCast<int32_t>();
        off += p.rv * pitch * 4;
        LocalTensor<float> accF = raw[off].ReinterpretCast<float>();
        off += p.rv * pitch * 4;
        LocalTensor<OutT> out = raw[off].ReinterpretCast<OutT>();
        off += Up(p.rv * pitch * 2, BLK);
        LocalTensor<ScaleT> wsl = raw[off].ReinterpretCast<ScaleT>();
        off += Up(pitch * 4, BLK);
        LocalTensor<float> wsf = raw[off].ReinterpretCast<float>();
        off += pitch * 4;
        LocalTensor<float> xsl = raw[off].ReinterpretCast<float>();
        off += Up(p.rv * 4, BLK);
        LocalTensor<float> xsb = raw[off].ReinterpretCast<float>();
        uint32_t acc = 0, start = 0;
        for (uint32_t e = 0; e < p.e && acc < last; ++e) {
            uint32_t v = static_cast<uint32_t>(list.GetValue(e));
            uint32_t end = p.listType == 0 ? v : start + v;
            // See Schedule::Total: unsigned, so it must not underflow.
            uint32_t rows = (end > start) ? (end - start) : 0u;
            uint32_t mBlocks = (rows + p.m0 - 1) / p.m0;
            uint32_t count = mBlocks * p.nTiles;
            if (acc + count > first) {
                uint32_t lo = first > acc ? first - acc : 0;
                uint32_t hi = Min(count, last - acc);
                for (uint32_t t = lo; t < hi; ++t) {
                    const uint32_t relative=acc+t-first;
                    const uint16_t tileId=uint16_t(relative&3u);
                    uint32_t mIdx = t / p.nTiles;
                    uint32_t nIdx = t % p.nTiles;
                    uint32_t gRow = start + mIdx * p.m0;
                    uint32_t mt = Min(p.m0, rows - mIdx * p.m0);
                    uint32_t nOff = nIdx * p.n0;
                    uint32_t nt = Min(p.n0, p.n - nOff);
                    uint64_t base = uint64_t(acc + t) * p.m0 * p.n0;
                    CrossCoreWaitFlag(tileId);
                    if(last-first>4u) CrossCoreSetFlag<2, PIPE_MTE3>(uint16_t(4u+tileId));
                    uint32_t half = (mt + AIV_PER_AIC - 1) / AIV_PER_AIC;
                    uint32_t r0 = Min(sub * half, mt);
                    uint32_t r1 = Min((sub + 1) * half, mt);
                    if (r0 < r1) {
                        DataCopyPad(wsl, wsg[uint64_t(e) * p.n + nOff],
                                    DataCopyExtParams{1, uint32_t(nt * sizeof(ScaleT)), 0, 0, 0},
                                    DataCopyPadExtParams<ScaleT>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        ToFloat<ScaleT>(wsf, wsl, nt);
                        PipeBarrier<PIPE_V>();
                    }
                    const uint32_t ntr = Up(nt, F32_PER_BLK);
                    for (uint32_t r = r0; r < r1; r += p.rv) {
                        uint32_t nr = Min(p.rv, r1 - r);
                        DataCopyPad(accI, stg[base + uint64_t(r) * p.n0],
                                    DataCopyExtParams{uint16_t(nr), nt * 4,
                                                      (p.n0 - nt) * 4,
                                                      (pitch - ntr) / F32_PER_BLK, 0},
                                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                        DataCopyPad(xsl, xsg[gRow + r],
                                    DataCopyExtParams{1, nr * 4, 0, 0, 0},
                                    DataCopyPadExtParams<float>{false, 0, 0, 0});
                        Barrier<HardEvent::MTE2_V>();
                        Cast(accF, accI, RoundMode::CAST_RINT, nr * pitch);
                        BrcbRepeatParams br;
                        br.dstBlkStride = 1;
                        br.dstRepStride = 8;
                        Brcb(xsb, xsl, static_cast<uint8_t>((nr + 7) / 8), br);
                        PipeBarrier<PIPE_V>();
                        MulByRow(accF, accF, wsf, nr, nt, pitch);
                        PipeBarrier<PIPE_V>();
                        MulByCol(accF, xsb, nr, ntr, pitch);
                        PipeBarrier<PIPE_V>();
                        Cast(out, accF, RoundMode::CAST_RINT, nr * pitch);
                        Barrier<HardEvent::V_MTE3>();
                        DataCopyPad(yg[uint64_t(gRow + r) * p.n + nOff], out,
                                    DataCopyExtParams{uint16_t(nr), nt * 2,
                                                      (pitch * 2 - Up(nt * 2, BLK)) / BLK,
                                                      (p.n - nt) * 2, 0});
                        Barrier<HardEvent::MTE3_MTE2>();
                    }
                }
            }
            acc += count;
            start = end;
        }
        Barrier<HardEvent::MTE3_S>();
    }
}

