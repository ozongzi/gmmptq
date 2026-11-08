struct EmptyResource {};
static_assert(std::is_trivially_default_constructible<EmptyResource>::value && std::is_trivially_destructible<EmptyResource>::value,"static resources must have no queue constructor/destructor");
template<bool Static>using P13Buf=std::conditional_t<Static,EmptyResource,AscendC::TBuf<TPosition::VECCALC>>;
template<bool Static>using P13Que=std::conditional_t<Static,EmptyResource,AscendC::TQue<AscendC::QuePosition::VECIN,1>>;
template<bool Static>using P13Pipe=std::conditional_t<Static,EmptyResource,AscendC::TPipe>;
namespace balanced_short_group {
constexpr uint32_t GRAIN=32;
constexpr uint32_t LANES=8;
constexpr uint32_t SCRATCH_BYTES=8192;
}
// Generic short-group matmul: E<=128, M<=256, K<=512; full-N row ownership.
// Host supplies maximum KC<=128 and matrix capacity; each piece adapts KC.
// K<=512 guarantees every possible int8 accumulation is exact in float32.
template <typename WScaleT, typename OutT, bool Static=false>
class BalancedOverlapShortGroupMatmul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR wScale,
                                GM_ADDR groupList, GM_ADDR y, const VTiling* t,
                                P13Pipe<Static>* pipe) {
        t_ = t;
        if constexpr(Static) {
            xsAll_=LocalTensor<float>(TPosition::VECIN,0,256);
            groupList_=LocalTensor<int64_t>(TPosition::VECIN,1024,128);
            rawIssued_=false;
        } else {
            pipe->InitBuffer(xsAllBuf_, 256 * sizeof(float));
            xsAll_ = xsAllBuf_.template Get<float>();
            pipe->InitBuffer(groupBuf_, 128 * 8);
            groupList_ = groupBuf_.template Get<int64_t>();
        }
        xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
        wGm_.SetGlobalBuffer((__gm__ int8_t*)weight);
        xsGm_.SetGlobalBuffer((__gm__ float*)xScale);
        wsGm_.SetGlobalBuffer((__gm__ WScaleT*)wScale);
        glGm_.SetGlobalBuffer((__gm__ int64_t*)groupList);
        yGm_.SetGlobalBuffer((__gm__ OutT*)y);

        core_ = AscendC::GetBlockIdx();
        blocks_ = t_->coreNum;
        kc_ = t_->kc;
        nb_ = t_->nb;
        nBlocks_ = Ceil(t_->n, nb_);

        if constexpr(Static) out_=LocalTensor<OutT>(TPosition::VECOUT,2048,Up(t_->n,32));
        else pipe->InitBuffer(outBuf_, Up(t_->n,32)*sizeof(OutT));
        outputIssued_ = false;
        const uint32_t block = kc_ * nb_;
        capacity_ = block;
        const uint32_t nbAligned = Up(t_->n, balanced_short_group::GRAIN);

        // The INT8 staging goes through queues, not a TBuf. A TQue is the
        // documented destination for an MTE2 copy and it carries the MTE2->V
        // handover itself; a bare TBuf needs that sync written by hand, and
        // after the EVENT_ID0 lesson (see design.md) hand-written event ids
        // are not worth the risk.
        if constexpr(Static) {
            const uint32_t xb=2048+Up(t_->n,32)*sizeof(OutT);
            const uint32_t wb=xb+Up(kc_,BLK)+BLK;
            xraw_=LocalTensor<int8_t>(TPosition::VECIN,xb,Up(kc_,BLK)+BLK);
            wraw_=LocalTensor<int8_t>(TPosition::VECIN,wb,Up(block,BLK)+BLK);
            workBase_=wb+Up(block,BLK)+BLK;
        } else {
            pipe->InitBuffer(xq_, 1, Up(kc_, BLK) + BLK);
            pipe->InitBuffer(wq_, 1, Up(block, BLK) + BLK);
        }

        // Rounded up a block, because the cast counts are rounded up too.
        const uint32_t blockUp = Up(block, balanced_short_group::GRAIN);
        const uint32_t fprod = blockUp * sizeof(float);
        const uint32_t fx = Up(kc_, balanced_short_group::GRAIN) * sizeof(float);
        const uint32_t fn = nbAligned * sizeof(float);
        const uint32_t fscratch = balanced_short_group::SCRATCH_BYTES; // xbr <=128*8 floats; no index table.
        // Dedicated half staging for the two widening hops. An earlier version
        // reinterpreted prod_ as half for this; giving each its own buffer
        // removes the aliasing question entirely and costs block*2 bytes.
        const uint32_t hw = blockUp * sizeof(half);
        const uint32_t hx = Up(kc_, balanced_short_group::GRAIN) * sizeof(half);
        if constexpr(!Static) pipe->InitBuffer(work_, fprod + fx + 2 * fn + balanced_short_group::LANES * 4 + hw + hx +
                                     fscratch);
        uint32_t at = 0;
        // Fused strided Mul overwrites weights with products. The fold
        // likewise operates in place; no second f32 matrix is needed.
        wf_ = WorkView<float>(blockUp, at);               at += fprod;
        prod_ = wf_;
        xf_ = WorkView<float>(Up(kc_, balanced_short_group::GRAIN), at);   at += fx;
        acc_ = WorkView<float>(nbAligned, at);            at += fn;
        wsf_ = WorkView<float>(nbAligned, at);            at += fn;
        xsUb_ = WorkView<float>(balanced_short_group::LANES, at);          at += balanced_short_group::LANES * 4;
        wh_ = WorkView<half>(blockUp, at);                at += hw;
        xh_ = WorkView<half>(Up(kc_, balanced_short_group::GRAIN), at);    at += hx;
        scratch_ = WorkView<uint8_t>(fscratch, at);
        // scratch_ holds K*8 float broadcasts during the K loop, then raw
        // weight scales after reduction; their lifetimes do not overlap.

    }

    __aicore__ inline void Process() {
        AscendC::DataCopyPad(xsAll_,xsGm_,
            AscendC::DataCopyExtParams{1,t_->m*4,0,0,0},
            AscendC::DataCopyPadExtParams<float>{false,0,0,0});
        // Immutable group and per-token scale tables are loaded once.
        // Each vector core owns complete output rows in increasing order.
        AscendC::DataCopyPad(groupList_, glGm_,
            AscendC::DataCopyExtParams{1, t_->e * 8, 0, 0, 0},
            AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        Barrier<AscendC::HardEvent::MTE2_S>();
        uint32_t expert = 0;
        uint32_t end = uint32_t(groupList_.GetValue(0));
        const uint32_t nBlocks = Ceil(t_->n,32);
        const uint32_t total = t_->m*nBlocks;
        uint32_t first = uint32_t(uint64_t(total)*core_/blocks_);
        const uint32_t last = uint32_t(uint64_t(total)*(core_+1)/blocks_);
        // Search from the endpoint nearest this core's first owned row.
        // Reverse traversal uses M as the last cumulative group endpoint.
        const uint32_t firstRow=first/nBlocks;
        const uint32_t lastBegin=t_->e>1u
            ? (t_->listType==0 ? uint32_t(groupList_.GetValue(t_->e-2u))
                              : t_->m-uint32_t(groupList_.GetValue(t_->e-1u))) : 0u;
        if(first<last && firstRow>=lastBegin) {
            expert=t_->e-1u;
            end=t_->m;
        } else if(first<last && firstRow>=end && firstRow>=t_->m/2u) {
            expert=t_->e-1u;
            end=t_->m;
            while(expert>0u) {
                const uint32_t begin=t_->listType==0
                    ? uint32_t(groupList_.GetValue(expert-1u))
                    : end-uint32_t(groupList_.GetValue(expert));
                if(firstRow>=begin)break;
                end=begin;
                --expert;
            }
        }
        while (first < last) {
            const uint32_t row = first/nBlocks;
            const uint32_t col = (first%nBlocks)*32;
            const uint32_t take = MinU(last-first,nBlocks-first%nBlocks);
            while (row >= end && expert + 1 < t_->e) {
                ++expert;
                const uint32_t value=uint32_t(groupList_.GetValue(expert));
                end=t_->listType==0?value:end+value;
            }
            Row(expert,row,col,MinU(t_->n-col,take*32));
            first+=take;
        }
        // Scalar-side drain before returning from the final async store.
        if (outputIssued_) Sync<AscendC::HardEvent::MTE3_S>();
        if constexpr(Static) if(rawIssued_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        }
    }

private:
    template <AscendC::HardEvent E>
    __aicore__ inline void Sync() {
        // Queue FreeTensor may own V_MTE2 event0/1. Fetch an unused ID.
        if constexpr(Static) {
            constexpr uint8_t id=(E==AscendC::HardEvent::MTE2_V || E==AscendC::HardEvent::V_MTE2)?2u:0u;
            AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);
        } else {
            const auto id = GetTPipePtr()->FetchEventID(E);
            AscendC::SetFlag<E>(id);
            AscendC::WaitFlag<E>(id);
        }
    }

    template<typename T> __aicore__ inline LocalTensor<T> WorkView(uint32_t count,uint32_t offset) {
        if constexpr(Static) return LocalTensor<T>(TPosition::VECCALC,workBase_+offset,count);
        else return work_.template GetWithOffset<T>(count,offset);
    }
    __aicore__ inline LocalTensor<OutT> Output() {
        if constexpr(Static) return out_;
        else return outBuf_.template Get<OutT>();
    }
    /// `pitch` is the row width the working blocks actually use. DataCopyPad
    /// addresses UB in whole 32 B blocks, so a strided 2D copy of `cols` bytes
    /// per row lands them `Up(cols, 32)` apart, not `cols` apart. When the
    /// block spans the whole of N the source is contiguous and one flat copy
    /// keeps the rows packed, which is what makes N=2 cheap; otherwise the
    /// padded pitch is carried through every downstream op. Columns past
    /// `cols` end up in acc_ too and are simply not written out.
    __aicore__ inline void Row(uint32_t expert, uint32_t row, uint32_t col, uint32_t cols) {
        const bool flat = false; // Host nb is Up(N,32), matching DMA padding.
        const uint32_t pitch = flat ? cols : Up(cols, BLK);
        kc_ = t_->kc;
        while (kc_*pitch>capacity_) kc_/=2;
        // 16 elements is one block for the 2-byte output type.
        const uint32_t accLen = Up(pitch, balanced_short_group::GRAIN);

        AscendC::Duplicate(acc_, 0.0f, accLen);
        AscendC::PipeBarrier<PIPE_V>();
        // Once per row, not once per k chunk: pitch is fixed for the row and
        // the table only depends on pitch.
        // No index table: Brcb and strided Mul broadcast x directly below.

        IssueInputs(expert,row,col,cols,flat,0,MinU(kc_,t_->k));
        for (uint32_t k0 = 0; k0 < t_->k; k0 += kc_) {
            const uint32_t klen = MinU(kc_, t_->k - k0);
            LoadChunk(expert, row, col, cols, pitch, flat, k0, klen);
        }

        LoadWeightScale(expert, col, cols, accLen);
        AscendC::Mul(acc_, acc_, wsf_, accLen);
        AscendC::PipeBarrier<PIPE_V>();
        // Immutable scale table was DMA-loaded before the group-list
        // MTE2_S handoff. No per-row scale DMA or buffer-reuse dependency.
        const float xs = xsAll_.GetValue(row);
        Sync<AscendC::HardEvent::S_V>();
        AscendC::Muls(acc_, acc_, xs, accLen);
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<OutT> out = Output();
        // Only the next output conversion needs to wait for the old store;
        // all intervening input DMA, casts, multiplies and folds may overlap.
        if (outputIssued_) Sync<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(out, acc_, AscendC::RoundMode::CAST_RINT, accLen);
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams p{1, cols * uint32_t(sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[uint64_t(row) * t_->n + col], out, p);
        outputIssued_ = true;
    }

    __aicore__ inline void IssueInputs(uint32_t expert,uint32_t row,uint32_t col,
                                     uint32_t cols,bool flat,uint32_t k0,uint32_t klen) {
        LocalTensor<int8_t> xi8;
        if constexpr(Static) {
            if(rawIssued_) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            xi8=xraw_;
        } else xi8=xq_.template AllocTensor<int8_t>();
        {
            AscendC::DataCopyPadExtParams<int8_t> pad{true, 0, 0, 0};
            AscendC::DataCopyExtParams p{1, klen, 0, 0, 0};
            AscendC::DataCopyPad(xi8, xGm_[uint64_t(row) * t_->k + k0], p, pad);
        }
        if constexpr(Static) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        else xq_.EnQue(xi8);

        LocalTensor<int8_t> wi8;
        if constexpr(Static) {
            if(rawIssued_) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            wi8=wraw_;
        } else wi8=wq_.template AllocTensor<int8_t>();
        {
            AscendC::DataCopyPadExtParams<int8_t> pad{true, 0, 0, 0};
            AscendC::DataCopyExtParams p = flat
                ? AscendC::DataCopyExtParams{1, klen * cols, 0, 0, 0}
                : AscendC::DataCopyExtParams{static_cast<uint16_t>(klen), cols,
                                             t_->n - cols, 0, 0};
            AscendC::DataCopyPad(wi8,
                                 wGm_[(uint64_t(expert) * t_->k + k0) * t_->n + col],
                                 p, pad);
        }
        if constexpr(Static) {AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);rawIssued_=true;}
        else wq_.EnQue(wi8);

    }

    __aicore__ inline void LoadChunk(uint32_t expert, uint32_t row, uint32_t col,
                                     uint32_t cols, uint32_t pitch, bool flat,
                                     uint32_t k0, uint32_t klen) {
        // X has a scalar K tail, so clear its small aligned buffer.
        // Weight rows have an aligned pitch: only the absent rows need zeroing.
        // The valid rows are overwritten by the widening casts below.
        if (klen < kc_) {
            AscendC::Duplicate(xf_, 0.0f, Up(kc_, balanced_short_group::GRAIN));
            AscendC::Duplicate(wf_[klen*pitch], 0.0f, (kc_-klen)*pitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
        LocalTensor<int8_t> xr,wr;
        if constexpr(Static) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);xr=xraw_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);wr=wraw_;
        } else {xr=xq_.template DeQue<int8_t>();wr=wq_.template DeQue<int8_t>();}

        // INT8 widens to FLOAT32 in two hops: the hardware has s8->f16 and
        // f16->f32 but no s8->f32 (Intrinsic_vconv in the SoC's
        // platform_config lists s82f16, not an int8 source for f32). The half
        // staging areas borrow buffers that are not live yet -- prod_ is not
        // written until the BroadCast below, scratch_ is only BroadCast's own
        // temporary.
        // Every cast count is rounded up to a whole 32 B block. A vector op
        // whose count is under one block does nothing -- which is what made
        // tp3, tp5, tp6 and tp10 fail: their K is 1 or 2, so klen was 1 or 2
        // and neither widening hop ran. The elements past klen read the zero
        // padding DataCopyPad left behind, so widening them changes nothing.
        const uint32_t welems = Up(klen * pitch, balanced_short_group::GRAIN);
        const uint32_t xelems = Up(klen, balanced_short_group::GRAIN);
        AscendC::Cast(xh_, xr, AscendC::RoundMode::CAST_NONE, xelems);
        AscendC::Cast(wh_, wr, AscendC::RoundMode::CAST_NONE, welems);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr(Static) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        } else {xq_.FreeTensor(xr);wq_.FreeTensor(wr);}
        if(k0+kc_<t_->k)
            IssueInputs(expert,row,col,cols,flat,k0+kc_,MinU(kc_,t_->k-k0-kc_));

        AscendC::Cast(xf_, xh_, AscendC::RoundMode::CAST_NONE, xelems);
        AscendC::Cast(wf_, wh_, AscendC::RoundMode::CAST_NONE, welems);
        AscendC::PipeBarrier<PIPE_V>();
        // The final widening ends wh_'s input lifetime; scale DMA can overlap
        // Brcb, products and reduction without touching the live xbr scratch.
        if (k0 + klen == t_->k) IssueWeightScale(expert, col, cols);
        // Eight x values become eight rows of eight copies each. Mul's
        // src1 block stride0 repeats that row's x across64 output columns;
        // src1 repeat stride1 advances eight floats for the next K row.
        LocalTensor<float> xbr = scratch_.template ReinterpretCast<float>();
        AscendC::Brcb(xbr, xf_, uint8_t(kc_ / 8),
                      AscendC::BrcbRepeatParams{1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams repeat;
        repeat.dstBlkStride = 1;
        repeat.src0BlkStride = 1;
        repeat.src1BlkStride = 0;
        repeat.dstRepStride = uint8_t(pitch / 8);
        repeat.src0RepStride = uint8_t(pitch / 8);
        repeat.src1RepStride = 1;
        // The hardware repeat field is uint8: K256 must be two128-row
        // batches rather than a truncated zero-repeat instruction.
        for (uint32_t r0 = 0; r0 < kc_; r0 += 128) {
            const uint8_t repeats = uint8_t(MinU(128, kc_ - r0));
            for (uint32_t c = 0; c < pitch; c += 64) {
                const uint64_t mask = MinU(64, pitch - c);
                AscendC::Mul(prod_[r0*pitch+c], wf_[r0*pitch+c], xbr[r0*8],
                              mask, repeats, repeat);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Fold k down into acc_ here, in the same function that produced
        // prod_. An earlier version had this in a separate Fold() and read
        // zeros from prod_ there even though a probe right after the Mul read
        // the correct value -- not worth another round of investigation when
        // keeping it local is also simpler.
        //
        // Halving keeps both operands contiguous and 32 B aligned while
        // (rows/2)*pitch is a multiple of eight floats. For pitch >= 8 that
        // runs down to one row and finishes with a single vector add. Below
        // that it stops early and the remaining handful of rows are summed
        // with scalar reads: a vector op whose count is under one 32 B block
        // does nothing at all, so the tail cannot be vectorised.
        uint32_t rows = kc_;
        while (rows > 1 && (((rows / 2) * pitch) % balanced_short_group::LANES) == 0) {
            const uint32_t half = rows / 2;
            AscendC::Add(prod_, prod_, prod_[half * pitch], half * pitch);
            AscendC::PipeBarrier<PIPE_V>();
            rows = half;
        }
        if (rows == 1 && pitch >= balanced_short_group::LANES) {
            AscendC::Add(acc_, acc_, prod_, pitch);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < pitch; ++c) {
            float sum = acc_.GetValue(c);
            for (uint32_t r = 0; r < rows; ++r) {
                sum += prod_.GetValue(r * pitch + c);
            }
            acc_.SetValue(c, sum);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void IssueWeightScale(uint32_t expert,uint32_t col,uint32_t cols) {
        Sync<AscendC::HardEvent::V_MTE2>();
        const uint64_t base=uint64_t(expert)*t_->n+col;
        AscendC::DataCopyPadExtParams<WScaleT> pad{true,0,0,0};
        AscendC::DataCopyExtParams p{1,cols*uint32_t(sizeof(WScaleT)),0,0,0};
        if constexpr(sizeof(WScaleT)==4) {
            AscendC::DataCopyPad(wsf_,wsGm_[base],p,pad);
        } else {
            auto raw=wh_.template ReinterpretCast<WScaleT>();
            AscendC::DataCopyPad(raw,wsGm_[base],p,pad);
        }
    }
    __aicore__ inline void LoadWeightScale(uint32_t,uint32_t,uint32_t,uint32_t nbAligned) {
        Sync<AscendC::HardEvent::MTE2_V>();
        if constexpr(sizeof(WScaleT)!=4) {
            auto raw=wh_.template ReinterpretCast<WScaleT>();
            AscendC::Cast(wsf_,raw,AscendC::RoundMode::CAST_NONE,nbAligned);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    P13Buf<Static> xsAllBuf_;
    LocalTensor<float> xsAll_;
    P13Buf<Static> outBuf_;
    bool outputIssued_;
    P13Buf<Static> groupBuf_;
    LocalTensor<int64_t> groupList_;
    const VTiling* t_;
    GlobalTensor<int8_t> xGm_, wGm_;
    GlobalTensor<float> xsGm_;
    GlobalTensor<WScaleT> wsGm_;
    GlobalTensor<int64_t> glGm_;
    GlobalTensor<OutT> yGm_;
    P13Que<Static> xq_, wq_;
    P13Buf<Static> work_;
    LocalTensor<float> prod_, wf_, xf_, acc_, wsf_, xsUb_;
    LocalTensor<half> wh_, xh_;
    LocalTensor<uint8_t> scratch_;
    uint32_t core_, blocks_, kc_, nb_, nBlocks_, capacity_;
    uint32_t workBase_;bool rawIssued_;
    LocalTensor<int8_t> xraw_,wraw_;
    LocalTensor<OutT> out_;
};

