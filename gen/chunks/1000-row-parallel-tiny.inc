// Structure: <=64 outputs, <=8 reduction terms, <=16 experts. One AIV.
// All real dimensions, list representation, scales and values remain runtime inputs.
// Derived from the provided v011 teammate implementation, SHA8941c16e6a.

template<typename ScaleT,typename OutT>
__global__ __aicore__ void RowParallelTiny(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
    GM_ADDR groups,GM_ADDR y,uint32_t packed){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GeneralVectorDims s{(packed&3u)+1u,((packed>>7)&7u)+1u,
        ((packed>>2)&31u)+1u,((packed>>10)&127u)+1u,(packed>>17)&1u};
    using namespace AscendC;
    // Advisory reads of the first valid cache line of each input tensor.
    // No state or data from previous invocations is retained.
    GlobalTensor<uint64_t> preloadTensor;
    preloadTensor.SetGlobalBuffer((__gm__ uint64_t*)x);
    DataCachePreload(preloadTensor,int16_t(0));
    preloadTensor.SetGlobalBuffer((__gm__ uint64_t*)w);
    DataCachePreload(preloadTensor,int16_t(0));
    preloadTensor.SetGlobalBuffer((__gm__ uint64_t*)xs);
    DataCachePreload(preloadTensor,int16_t(0));
    preloadTensor.SetGlobalBuffer((__gm__ uint64_t*)ws);
    DataCachePreload(preloadTensor,int16_t(0));
    preloadTensor.SetGlobalBuffer((__gm__ uint64_t*)groups);
    DataCachePreload(preloadTensor,int16_t(0));
    uint32_t m=GetBlockIdx(),owner=0,start=0;
    GlobalTensor<int64_t> gg;gg.SetGlobalBuffer((__gm__ int64_t*)groups);
    for(uint32_t e=0;e<s.e;++e){
        uint32_t v=uint32_t(gg.GetValue(e)),end=s.listType==0?v:start+v;
        if(m<end){owner=e;break;}start=end;
    }
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;
    GlobalTensor<ScaleT> wsg;GlobalTensor<OutT> yg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);wsg.SetGlobalBuffer((__gm__ ScaleT*)ws);
    yg.SetGlobalBuffer((__gm__ OutT*)y);
    InitSocState(); // Preserve original TPipe constructor hardware-state init.
    LocalTensor<float> f(TPosition::VECCALC,0,32);
    LocalTensor<OutT> out(TPosition::VECCALC,128,32);
    float row=xsg.GetValue(m);
    for(uint32_t n=0;n<s.n;++n){
        int32_t sum=0;
        for(uint32_t k=0;k<s.k;++k)sum+=int32_t(xg.GetValue(uint64_t(m)*s.k+k))*int32_t(wg.GetValue((uint64_t(owner)*s.k+k)*s.n+n));
        float scale; if constexpr(sizeof(ScaleT)==4) scale=*(__gm__ float*)(ws+(uint64_t(owner)*s.n+n)*4); else {union {uint32_t u;float f;} cv; cv.u=uint32_t(*(__gm__ uint16_t*)(ws+(uint64_t(owner)*s.n+n)*2))<<16;scale=cv.f;}
        f.SetValue(n,float(sum)*scale*row);
    }
    GvBarrier<HardEvent::S_V>();Cast(out,f,RoundMode::CAST_RINT,s.n);GvBarrier<HardEvent::V_MTE3>();
    DataCopyPad(yg[uint64_t(m)*s.n],out,DataCopyExtParams{1,s.n*sizeof(OutT),0,0,0});
    PipeBarrier<PIPE_ALL>(); // Preserve original TPipe::Destroy final drain.
}

