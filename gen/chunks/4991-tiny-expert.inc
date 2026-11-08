namespace gmmptq {
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchTinyExpert3500000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR groups,GM_ADDR y,uint32_t packed){
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
    const uint32_t owner=GetBlockIdx();
    GlobalTensor<int64_t> gg;gg.SetGlobalBuffer((__gm__ int64_t*)groups);
    const uint32_t begin=owner?uint32_t(gg.GetValue(owner-1)):0u;
    // Valid GT0 covers exactlyM rows: last cumulative endpoint is implied byM.
    const uint32_t end=owner+1<s.e?uint32_t(gg.GetValue(owner)):s.m;
    if(begin==end)return;
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;
    GlobalTensor<ScaleT> wsg;GlobalTensor<OutT> yg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);wsg.SetGlobalBuffer((__gm__ ScaleT*)ws);
    yg.SetGlobalBuffer((__gm__ OutT*)y);
    InitSocState(); // Preserve original TPipe constructor hardware-state init.
    LocalTensor<float> f(TPosition::VECCALC,0,32);
    LocalTensor<OutT> out(TPosition::VECCALC,128,32);
    for(uint32_t m=begin;m<end;++m){
    float row=xsg.GetValue(m);
    for(uint32_t n=0;n<s.n;++n){
        int32_t sum=0;
        for(uint32_t k=0;k<s.k;++k)sum+=int32_t(xg.GetValue(uint64_t(m)*s.k+k))*int32_t(wg.GetValue((uint64_t(owner)*s.k+k)*s.n+n));
        float scale; if constexpr(sizeof(ScaleT)==4) scale=*(__gm__ float*)(ws+(uint64_t(owner)*s.n+n)*4); else {union {uint32_t u;float f;} cv; cv.u=uint32_t(*(__gm__ uint16_t*)(ws+(uint64_t(owner)*s.n+n)*2))<<16;scale=cv.f;}
        f.SetValue((m-begin)*s.n+n,float(sum)*scale*row);
    }
    }
    const uint32_t count=(end-begin)*s.n;
    GvBarrier<HardEvent::S_V>();Cast(out,f,RoundMode::CAST_RINT,count);GvBarrier<HardEvent::V_MTE3>();
    DataCopyPad(yg[uint64_t(begin)*s.n],out,DataCopyExtParams{1,count*sizeof(OutT),0,0,0});
    PipeBarrier<PIPE_ALL>(); // Preserve original TPipe::Destroy final drain.
}
}

