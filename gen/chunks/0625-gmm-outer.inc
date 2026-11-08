template <typename WScaleT, typename OutT>
class GmmOuter {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR wScale,
                                GM_ADDR groupList, GM_ADDR y, const VTiling* t,
                                AscendC::TPipe* pipe) {
        t_ = t;
        xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
        wGm_.SetGlobalBuffer((__gm__ int8_t*)weight);
        xsGm_.SetGlobalBuffer((__gm__ float*)xScale);
        wsGm_.SetGlobalBuffer((__gm__ WScaleT*)wScale);
        glGm_.SetGlobalBuffer((__gm__ int64_t*)groupList);
        yGm_.SetGlobalBuffer((__gm__ OutT*)y);

        core_ = AscendC::GetBlockIdx();
        blocks_ = t_->coreNum;
        n_ = t_->n;
        rows_ = t_->kc;    // the row block rides in the tiling's kc slot
        pitch_ = t_->nb;   // Up(n, 16), computed on the host
        held_ = 0xFFFFFFFFu;  // no expert's weights are loaded yet

        const uint32_t block = rows_ * pitch_;
        const uint32_t nUp = Up(n_, vec::GRAIN);
        const uint32_t rUp = Up(rows_, vec::GRAIN);

        pipe->InitBuffer(wq_, 1, nUp + vec::GRAIN);
        pipe->InitBuffer(xq_, 1, rUp + vec::GRAIN);
        pipe->InitBuffer(work_, block * sizeof(float) + block * sizeof(OutT) +
                                    3 * nUp * sizeof(float) + 4 * rUp * sizeof(float) + 16 * rUp * sizeof(float));
        uint32_t at = 0;
        outF_ = work_.GetWithOffset<float>(block, at);   at += block * sizeof(float);
        outH_ = work_.GetWithOffset<OutT>(block, at);    at += block * sizeof(OutT);
        wf_ = work_.GetWithOffset<float>(nUp, at);       at += nUp * sizeof(float);
        wsf_ = work_.GetWithOffset<float>(nUp, at);      at += nUp * sizeof(float);
        wsRaw_ = work_.GetWithOffset<WScaleT>(nUp, at);  at += nUp * sizeof(float);
        xf_ = work_.GetWithOffset<float>(rUp, at);       at += rUp * sizeof(float);
        xsf_ = work_.GetWithOffset<float>(rUp, at);      at += rUp * sizeof(float);
        xh_ = work_.GetWithOffset<half>(rUp, at); at += rUp * sizeof(half);
        xbr_ = work_.GetWithOffset<float>(rUp * 8, at); at += rUp * 8 * sizeof(float);
        xsbr_ = work_.GetWithOffset<float>(rUp * 8, at);
    }

    __aicore__ inline void Process() {
        Walk walk;
        walk.Init(glGm_, t_->listType, t_->e, rows_, 1u);
        uint32_t item = 0;
        while (!walk.done) {
            if (item % blocks_ == core_) {
                const uint32_t taken = walk.mIdx * rows_;
                Block(walk.expert, walk.rowStart + taken, MinU(rows_, walk.rows - taken));
            }
            ++item;
            walk.Next();
        }
    }

private:
    /// wf_[n] = f32(w[e][0][n]) and wsf_[n] = f32(ws[e][n]), kept apart.
    ///
    /// They are deliberately NOT pre-multiplied into one vector. The reference
    /// dequantises as (acc * wscale) * xscale, left to right (golden.rs), and
    /// with K == 1 the accumulator is x[r] * w[n], which is exact in f32 since
    /// both factors are integers under 128. Folding w and ws together first
    /// would round that product before x ever multiplied it, and the result
    /// stops being bit-identical: the earlier version here computed
    /// (x*xs) * (w*ws) and 3807 of tp10's 788225 outputs came out one ulp
    /// off, with a worst relative error of 1.12e-3 -- past the 1e-3 the strict
    /// reading of the tolerance allows.
    ///
    /// Loaded once per expert: items go out round robin, so a core sees runs
    /// of the same expert and skips the reload most of the time.
    __aicore__ inline void LoadExpert(uint32_t expert) {
        if (expert == held_) return;
        held_ = expert;
        const uint32_t nUp = Up(n_, vec::GRAIN);

        LocalTensor<int8_t> wr = wq_.template AllocTensor<int8_t>();
        AscendC::DataCopyPad(wr, wGm_[uint64_t(expert) * n_],
                             AscendC::DataCopyExtParams{1, n_, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int8_t>{true, 0, 0, 0});
        AscendC::DataCopyPad(wsRaw_, wsGm_[uint64_t(expert) * n_],
                             AscendC::DataCopyExtParams{
                                 1, uint32_t(n_ * sizeof(WScaleT)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<WScaleT>{true, 0, 0, 0});
        wq_.EnQue(wr);
        wr = wq_.template DeQue<int8_t>();
        // int8 widens in two hops: the hardware has s82f16 and f162f32, not
        // s82f32. wsRaw_ rides in on the same MTE2 run the queue waits for.
        AscendC::Cast(whBuf(), wr, AscendC::RoundMode::CAST_NONE, nUp);
        AscendC::PipeBarrier<PIPE_V>();
        wq_.FreeTensor(wr);
        AscendC::Cast(wf_, whBuf(), AscendC::RoundMode::CAST_NONE, nUp);
        CastScale(wsf_, nUp);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// The half staging for w borrows outF_, which is not written until the
    /// row loop below.
    __aicore__ inline LocalTensor<half> whBuf() {
        return outF_.template ReinterpretCast<half>();
    }

    /// FLOAT32 weightScale is already usable; BFLOAT16 needs one widen.
    __aicore__ inline void CastScale(const LocalTensor<float>& dst, uint32_t count) {
        if constexpr (AscendC::IsSameType<WScaleT, float>::value) {
            AscendC::DataCopy(dst, wsRaw_.template ReinterpretCast<float>(), count);
        } else {
            AscendC::Cast(dst, wsRaw_, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void Block(uint32_t expert, uint32_t row0, uint32_t have) {
        LoadExpert(expert);
        const uint32_t rUp = Up(have, vec::GRAIN);

        LocalTensor<int8_t> xr = xq_.template AllocTensor<int8_t>();
        AscendC::DataCopyPad(xr, xGm_[row0], AscendC::DataCopyExtParams{1, have, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int8_t>{true, 0, 0, 0});
        AscendC::DataCopyPad(xsf_, xsGm_[row0],
                             AscendC::DataCopyExtParams{1, have * 4u, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{true, 0, 0, 0});
        xq_.EnQue(xr);
        xr = xq_.template DeQue<int8_t>();
        AscendC::Cast(xh_, xr, AscendC::RoundMode::CAST_NONE, rUp);
        AscendC::PipeBarrier<PIPE_V>();
        xq_.FreeTensor(xr);
        AscendC::Cast(xf_, xh_, AscendC::RoundMode::CAST_NONE, rUp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BrcbRepeatParams br;
        br.dstBlkStride = 1;
        br.dstRepStride = 8;
        AscendC::Brcb(xbr_, xf_, uint8_t((have+7)/8), br);
        AscendC::Brcb(xsbr_, xsf_, uint8_t((have+7)/8), br);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams rp;
        rp.dstBlkStride=1;
        rp.src0BlkStride=1;
        rp.src1BlkStride=0;
        rp.dstRepStride=uint16_t(pitch_/8);
        rp.src0RepStride=0;
        rp.src1RepStride=1;
        // All rows consume one weight vector and one repeated row scalar.
        for (uint32_t c=0;c<n_;c+=64)
            AscendC::Mul(outF_[c],wf_[c],xbr_,uint64_t(MinU(64,n_-c)),uint8_t(have),rp);
        AscendC::PipeBarrier<PIPE_V>();
        // Preserve exact (integer x*w)*weightScale*xScale ordering.
        rp.src0RepStride=uint16_t(pitch_/8);
        rp.src1BlkStride=1;
        rp.src1RepStride=0;
        for (uint32_t c=0;c<n_;c+=64)
            AscendC::Mul(outF_[c],outF_[c],wsf_[c],uint64_t(MinU(64,n_-c)),uint8_t(have),rp);
        AscendC::PipeBarrier<PIPE_V>();
        rp.src1BlkStride=0;
        rp.src1RepStride=1;
        for (uint32_t c=0;c<n_;c+=64)
            AscendC::Mul(outF_[c],outF_[c],xsbr_,uint64_t(MinU(64,n_-c)),uint8_t(have),rp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outH_, outF_, AscendC::RoundMode::CAST_RINT, have * pitch_);
        Barrier<AscendC::HardEvent::V_MTE3>();
        // One store for the whole block. srcStride 0 walks the UB rows at
        // Up(blockLen, 32) = pitch_ * sizeof(OutT); the destination is packed.
        AscendC::DataCopyPad(yGm_[uint64_t(row0) * n_], outH_,
                             AscendC::DataCopyExtParams{
                                 uint16_t(have), uint32_t(n_ * sizeof(OutT)), 0, 0, 0});
        Barrier<AscendC::HardEvent::MTE3_V>();
    }

    GlobalTensor<int8_t> xGm_, wGm_;
    GlobalTensor<float> xsGm_;
    GlobalTensor<WScaleT> wsGm_;
    GlobalTensor<int64_t> glGm_;
    GlobalTensor<OutT> yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wq_, xq_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_;
    LocalTensor<float> outF_, wf_, wsf_, xf_, xsf_;
    LocalTensor<float> xbr_, xsbr_;
    LocalTensor<WScaleT> wsRaw_;
    LocalTensor<half> xh_;
    LocalTensor<OutT> outH_;
    const VTiling* t_;
    uint32_t core_, blocks_, n_, rows_, pitch_, held_;
};

template <typename WScaleT, typename OutT>
__global__ __aicore__ void GmmOuterKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale,
                                             GM_ADDR wScale, GM_ADDR groupList,
                                             GM_ADDR y, VTiling tiling) {
    // See GmmKernel: the task type belongs here, not in the attribute.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GmmOuter<WScaleT, OutT> op;
    op.Init(x, weight, xScale, wScale, groupList, y, &tiling, &pipe);
    op.Process();
}

