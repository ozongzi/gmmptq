// One output row, one or two channels, arbitrary runtime K<=8192 and expert list.
// Fixed GatherMask patterns derive from the two-channel ND layout, not a case.

template <typename ScaleT,typename OutT>
__global__ __aicore__ void IntegerProductLongNarrowDot(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
                                        GM_ADDR gl,GM_ADDR y,GeneralVectorDims p) {
    using namespace AscendC;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;
    GlobalTensor<ScaleT> wsg;GlobalTensor<OutT> yg;GlobalTensor<int64_t> glg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);wsg.SetGlobalBuffer((__gm__ ScaleT*)ws);
    yg.SetGlobalBuffer((__gm__ OutT*)y);glg.SetGlobalBuffer((__gm__ int64_t*)gl);
    const uint32_t k32=GvUp(p.k,32),k64=GvUp(p.k,64);
    const uint32_t w32=GvUp(p.k*p.n,32),whCount=GvUp(p.k*p.n,128);
    const uint32_t partialCount=k64/64,partialPad=GvUp(partialCount,16);
    const uint32_t groups=partialPad/16;
    // Independent integer operands; only exact products widen to float.
    TPipe pipe;TBuf<TPosition::VECCALC> ub;pipe.InitBuffer(ub,108320);
    auto raw=ub.Get<uint8_t>();
    auto xi=raw[0].ReinterpretCast<int8_t>();
    auto wi=raw[8192].ReinterpretCast<int8_t>();
    auto xh=raw[24576].ReinterpretCast<half>();
    auto x16=raw[40960].ReinterpretCast<int16_t>();
    auto w16=raw[57344].ReinterpretCast<int16_t>();
    auto wf=raw[73728].ReinterpretCast<float>();
    auto red1=raw[106496].ReinterpretCast<float>();
    auto red2=raw[107008].ReinterpretCast<float>();
    auto xsl=raw[107040].ReinterpretCast<float>();
    auto wsl=raw[107072].ReinterpretCast<ScaleT>();
    auto wsf=raw[107136].ReinterpretCast<float>();
    auto result=raw[107200].ReinterpretCast<float>();
    auto out=raw[107264].ReinterpretCast<OutT>();
    auto gll=raw[107296].ReinterpretCast<int64_t>();
    const uint32_t c=GetBlockIdx();
    if(c>=p.n) return;
    uint32_t expert=0;
    if (p.e>1) {
        DataCopyPad(gll,glg,DataCopyExtParams{1,p.e*8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
        GvBarrier<HardEvent::MTE2_S>();
        // With M=1 the first positive count or cumulative endpoint identifies
        // the sole row's expert, for both valid groupList representations.
        while (expert+1<p.e && gll.GetValue(expert)==0) ++expert;
    }
    // Each invocation invalidates the two scalar input cache lines before
    // requesting them; their fills can overlap the independent vector dot.
    auto selectedScale=wsg[expert*p.n+c];
    DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(xsg);
    DataCacheCleanAndInvalid<ScaleT,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(selectedScale);
    GlobalTensor<uint64_t> scalePreload;
    scalePreload.SetGlobalBuffer((__gm__ uint64_t*)xs);
    DataCachePreload(scalePreload,int16_t(0));
    scalePreload.SetGlobalBuffer((__gm__ uint64_t*)(ws+uint64_t(expert*p.n+c)*sizeof(ScaleT)));
    DataCachePreload(scalePreload,int16_t(0));
    DataCopyPad(xi,xg,DataCopyExtParams{1,p.k,0,0,0},
                DataCopyPadExtParams<int8_t>{true,0,uint8_t(k32-p.k),0});
    DataCopyPad(wi,wg[uint64_t(expert)*p.k*p.n],DataCopyExtParams{1,p.k*p.n,0,0,0},
                DataCopyPadExtParams<int8_t>{true,0,uint8_t(w32-p.k*p.n),0});
    GvBarrier<HardEvent::MTE2_V>();
    if(k64>k32) Duplicate(x16,int16_t(0),k64);
    const uint32_t weightWords=p.n==2?w32/2:k32;
    if(k64>weightWords) Duplicate(w16,int16_t(0),k64);
    Cast(xh,xi,RoundMode::CAST_NONE,k32);
    PipeBarrier<PIPE_V>();
    Cast(x16,xh,RoundMode::CAST_RINT,k32);PipeBarrier<PIPE_V>();
    if(p.n==2) {
        auto packed=wi.ReinterpretCast<int16_t>();
        if(c==0) {
            ShiftLeft(w16,packed,int16_t(8),int32_t(weightWords));PipeBarrier<PIPE_V>();
            ShiftRight(w16,w16,int16_t(8),int32_t(weightWords));
        } else ShiftRight(w16,packed,int16_t(8),int32_t(weightWords));
    } else {
        // xh is dead after its signed16 conversion; reuse only after that V op.
        Cast(xh,wi,RoundMode::CAST_NONE,k32);PipeBarrier<PIPE_V>();
        Cast(w16,xh,RoundMode::CAST_RINT,k32);
    }
    PipeBarrier<PIPE_V>();
    Mul(w16,w16,x16,k64);PipeBarrier<PIPE_V>();
    Cast(wf,w16,RoundMode::CAST_NONE,k64);PipeBarrier<PIPE_V>();
    {
        if (partialPad>partialCount) Duplicate(red1,0.0f,partialPad);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(red1,wf,64,partialCount,1,1,8);PipeBarrier<PIPE_V>();
        // Each second-stage sum covers <=1024 products and remains integer-exact.
        WholeReduceSum(red2,red1,16,groups,1,1,2);GvBarrier<HardEvent::V_S>();
        int32_t sum=0;
        for (uint32_t i=0;i<groups;++i) sum+=int32_t(red2.GetValue(i));
        float scale;
        if constexpr (sizeof(ScaleT)==4) scale=wsg.GetValue(expert*p.n+c);
        else {
            GlobalTensor<uint16_t> bits;bits.SetGlobalBuffer((__gm__ uint16_t*)ws);
            scale=GvFloat(uint32_t(bits.GetValue(expert*p.n+c))<<16);
        }
        float weighted=float(sum)*scale;
        result.SetValue(0,weighted*xsg.GetValue(0));GvBarrier<HardEvent::S_V>();
    }
    Cast(out,result,RoundMode::CAST_RINT,16);GvBarrier<HardEvent::V_MTE3>();
    DataCopyPad(yg[c],out,DataCopyExtParams{1,2,0,0,0});
}

