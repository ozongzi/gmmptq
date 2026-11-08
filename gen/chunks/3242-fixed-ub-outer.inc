namespace p10_fixed_ub{using namespace gmmptq;
template <typename WScaleT, typename OutT,int Mode,uint32_t Rows,bool EarlyInput,uint32_t Outputs>
class FixedUbOuter {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR wScale,
                                GM_ADDR groupList, GM_ADDR y, const VTiling* t) {
        static_assert(Mode==1 && Rows==32 && EarlyInput && Outputs==1,"one fixed RR32 family");
        static_assert(sizeof(WScaleT)==4 && sizeof(OutT)==2,"selected FP16 family only");
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
        rows_ = Rows;    // the row block rides in the tiling's kc slot
        pitch_ = t_->nb;   // Up(n, 16), computed on the host
        held_ = 0xFFFFFFFFu;  // no expert's weights are loaded yet

        const uint32_t block = rows_ * pitch_;
        const uint32_t nUp = Up(n_, vec::GRAIN);
        const uint32_t rUp = Up(rows_, vec::GRAIN);

        const uint32_t wBytes=nUp+vec::GRAIN,xBytes=rUp+vec::GRAIN;
        const uint32_t workBase=wBytes+xBytes;
        rawW_=LocalTensor<int8_t>(AscendC::TPosition::VECCALC,0,wBytes);
        rawX_=LocalTensor<int8_t>(AscendC::TPosition::VECCALC,wBytes,xBytes);
        for(uint16_t i=0;i<Outputs;++i) AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
        uint32_t at = 0;
        outF_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,block);   at += block * sizeof(float);
        outH_ = LocalTensor<OutT>(AscendC::TPosition::VECCALC,workBase+at,block);    at += block * sizeof(OutT) * Outputs;
        wf_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,nUp);       at += nUp * sizeof(float);
        wsf_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,nUp);      at += nUp * sizeof(float);
        wsRaw_ = wsf_.template ReinterpretCast<WScaleT>();  at += nUp * sizeof(float);
        xf_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,rUp);       at += rUp * sizeof(float);
        xsf_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,rUp);      at += rUp * sizeof(float);
        xh_ = LocalTensor<half>(AscendC::TPosition::VECCALC,workBase+at,rUp); at += rUp * sizeof(half);
        xbr_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,rUp * 8); at += rUp * 8 * sizeof(float);
        xsbr_ = LocalTensor<float>(AscendC::TPosition::VECCALC,workBase+at,rUp * 8);
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
        for(uint16_t i=0;i<Outputs;++i) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
    }

private:
    // Mode0 preserves the reference order; Modes1/2 cache f32(w*WS).
    __aicore__ inline void LoadExpert(uint32_t expert) {
        if (expert == held_) return;
        held_ = expert;
        const uint32_t nUp = Up(n_, vec::GRAIN);

        LocalTensor<int8_t> wr = rawW_;
        AscendC::DataCopyPad(wr, wGm_[uint64_t(expert) * n_],
                             AscendC::DataCopyExtParams{1, n_, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int8_t>{true, 0, 0, 0});
        AscendC::DataCopyPad(wsRaw_, wsGm_[uint64_t(expert) * n_],
                             AscendC::DataCopyExtParams{
                                 1, uint32_t(n_ * sizeof(WScaleT)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<WScaleT>{true, 0, 0, 0});
    }
    __aicore__ inline void CastExpert() {
        const uint32_t nUp=Up(n_,vec::GRAIN);
        LocalTensor<int8_t> wr=rawW_;
        // int8 widens in two hops: the hardware has s82f16 and f162f32, not
        // s82f32. wsRaw_ rides in on the same MTE2 run the queue waits for.
        AscendC::Cast(whBuf(), wr, AscendC::RoundMode::CAST_NONE, nUp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(wf_, whBuf(), AscendC::RoundMode::CAST_NONE, nUp);
        CastScale(wsf_, nUp);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr(Mode!=0){ AscendC::Mul(wf_,wf_,wsf_,nUp); AscendC::PipeBarrier<PIPE_V>(); }
    }

    /// The half staging for w borrows outF_, which is not written until the
    /// row loop below.
    __aicore__ inline LocalTensor<half> whBuf() {
        return outF_.template ReinterpretCast<half>();
    }

    /// FLOAT32 weightScale is already usable; BFLOAT16 needs one widen.
    __aicore__ inline void CastScale(const LocalTensor<float>& dst, uint32_t count) {
        if constexpr (AscendC::IsSameType<WScaleT, float>::value) {
            (void)dst; (void)count; // FP32 WS was DMA-loaded directly into original wsf_.
        } else {
            AscendC::Cast(dst, wsRaw_, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void CopyRows(LocalTensor<int8_t> xr,uint32_t row0,uint32_t have) {
        AscendC::DataCopyPad(xr, xGm_[row0], AscendC::DataCopyExtParams{1, have, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int8_t>{true, 0, 0, 0});
        AscendC::DataCopyPad(xsf_, xsGm_[row0],
                             AscendC::DataCopyExtParams{1, have * 4u, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{true, 0, 0, 0});
    }

    __aicore__ inline void Block(uint32_t expert, uint32_t row0, uint32_t have) {
        // TQue frees xr after its Cast, but xsf_/wsRaw_ can still be read
        // by later vector operations. Order those reads before any new input
        // DMA, using an event not reserved by either input queue.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        const uint32_t rUp = Up(have, vec::GRAIN);
        LocalTensor<int8_t> xr=rawX_;
        const bool reload=expert!=held_;
        if constexpr(EarlyInput)CopyRows(xr,row0,have);
        LoadExpert(expert);
        if constexpr(!EarlyInput)CopyRows(xr,row0,have);
        // One static fence covers early X/XS and optional W/WS, including cache hits.
        Barrier<AscendC::HardEvent::MTE2_V>();
        if(reload)CastExpert();
        AscendC::Cast(xh_, xr, AscendC::RoundMode::CAST_NONE, rUp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(xf_, xh_, AscendC::RoundMode::CAST_NONE, rUp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BrcbRepeatParams br;
        br.dstBlkStride = 1;
        br.dstRepStride = 8;
        if constexpr(Mode==1){AscendC::Mul(xf_,xf_,xsf_,rUp); AscendC::PipeBarrier<PIPE_V>();}
        AscendC::Brcb(xbr_, xf_, uint8_t((have+7)/8), br);
        if constexpr(Mode!=1) AscendC::Brcb(xsbr_, xsf_, uint8_t((have+7)/8), br);
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
        if constexpr(Mode==0) {
            rp.src0RepStride=uint16_t(pitch_/8);rp.src1BlkStride=1;rp.src1RepStride=0;
            for(uint32_t c=0;c<n_;c+=64) AscendC::Mul(outF_[c],outF_[c],wsf_[c],uint64_t(MinU(64,n_-c)),uint8_t(have),rp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr(Mode!=1) {
            rp.src0RepStride=uint16_t(pitch_/8);rp.src1BlkStride=0;rp.src1RepStride=1;
            for(uint32_t c=0;c<n_;c+=64) AscendC::Mul(outF_[c],outF_[c],xsbr_,uint64_t(MinU(64,n_-c)),uint8_t(have),rp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const uint16_t slot=outputIndex_++%Outputs;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        auto target=outH_[slot*rows_*pitch_];
        AscendC::Cast(target, outF_, AscendC::RoundMode::CAST_RINT, have * pitch_);
        Barrier<AscendC::HardEvent::V_MTE3>();
        // One store for the whole block. srcStride 0 walks the UB rows at
        // Up(blockLen, 32) = pitch_ * sizeof(OutT); the destination is packed.
        AscendC::DataCopyPad(yGm_[uint64_t(row0) * n_], target,
                             AscendC::DataCopyExtParams{
                                 uint16_t(have), uint32_t(n_ * sizeof(OutT)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
    }

    GlobalTensor<int8_t> xGm_, wGm_;
    GlobalTensor<float> xsGm_;
    GlobalTensor<WScaleT> wsGm_;
    GlobalTensor<int64_t> glGm_;
    GlobalTensor<OutT> yGm_;
    LocalTensor<int8_t> rawW_,rawX_;
    LocalTensor<float> outF_, wf_, wsf_, xf_, xsf_;
    LocalTensor<float> xbr_, xsbr_;
    LocalTensor<WScaleT> wsRaw_;
    LocalTensor<half> xh_;
    LocalTensor<OutT> outH_;
    const VTiling* t_;
    uint32_t outputIndex_=0;
    uint32_t core_, blocks_, n_, rows_, pitch_, held_;
};


__global__ __aicore__ void SearchP10FixedUb7300000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,VTiling t){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 AscendC::InitSocState();
 FixedUbOuter<float,half,1,32,true,1> op;
 op.Init(x,w,xs,ws,gl,y,&t);op.Process();
 AscendC::PipeBarrier<PIPE_ALL>();
}
}

