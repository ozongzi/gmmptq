namespace tiny_row_stdev {using namespace gmmptq;
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchTinyRowSTDev3400000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR groups,GM_ADDR y,uint32_t packed){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GeneralVectorDims s{(packed&3u)+1u,((packed>>7)&7u)+1u,
        ((packed>>2)&31u)+1u,((packed>>10)&127u)+1u,(packed>>17)&1u};
    s.n=1u;
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
    GlobalTensor<float> earlyXs;earlyXs.SetGlobalBuffer((__gm__ float*)xs);
    float row=earlyXs.GetValue(m);
    GlobalTensor<int64_t> gg;gg.SetGlobalBuffer((__gm__ int64_t*)groups);
    if(m*2u>=s.m){
        uint32_t end=s.m;
        owner=0u;
        for(uint32_t e=s.e-1u;e>0u;--e){
            const uint32_t begin=s.listType==0?uint32_t(gg.GetValue(e-1u)):end-uint32_t(gg.GetValue(e));
            if(m>=begin){owner=e;break;}end=begin;
        }
    }else{
        owner=s.e-1u;
        for(uint32_t e=0;e+1u<s.e;++e){
            uint32_t v=uint32_t(gg.GetValue(e)),end=s.listType==0?v:start+v;
            if(m<end){owner=e;break;}start=end;
        }
    }
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;
    GlobalTensor<ScaleT> wsg;GlobalTensor<OutT> yg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);wsg.SetGlobalBuffer((__gm__ ScaleT*)ws);
    yg.SetGlobalBuffer((__gm__ OutT*)y);
    uint32_t encodedRow=0;
    for(uint32_t n=0;n<s.n;++n){
        float scale; if constexpr(sizeof(ScaleT)==4) scale=*(__gm__ float*)(ws+(uint64_t(owner)*s.n+n)*4); else {union {uint32_t u;float f;} cv; cv.u=uint32_t(*(__gm__ uint16_t*)(ws+(uint64_t(owner)*s.n+n)*2))<<16;scale=cv.f;}
        int32_t sum=0;
        for(uint32_t k=0;k<s.k;++k)sum+=int32_t(xg.GetValue(uint64_t(m)*s.k+k))*int32_t(wg.GetValue((uint64_t(owner)*s.k+k)*s.n+n));
        encodedRow|=uint32_t(GvEncode<(sizeof(ScaleT)==2)>(float(sum)*scale*row))<<(16*n);
    }
    if(s.n==1)WriteGmByPassDCache(reinterpret_cast<__gm__ uint16_t*>(y)+m,uint16_t(encodedRow));
    else WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t*>(y)+m,encodedRow);
    DataSyncBarrier<MemDsbT::DDR>();
}
}

