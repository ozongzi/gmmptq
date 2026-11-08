namespace expert_output {
using namespace gmmptq;
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchTinyExpertOutput6400000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR groups,GM_ADDR y,uint32_t packed){
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
    uint64_t word0=0,word1=0;
    const uint64_t wb=uint64_t(owner)*4u;
    const int32_t w00=int32_t(wg.GetValue(wb)),w01=int32_t(wg.GetValue(wb+1u));
    const int32_t w10=int32_t(wg.GetValue(wb+2u)),w11=int32_t(wg.GetValue(wb+3u));
    float scale0,scale1;
    if constexpr(sizeof(ScaleT)==4){
        scale0=*(__gm__ float*)(ws+uint64_t(owner)*8u);
        scale1=*(__gm__ float*)(ws+uint64_t(owner)*8u+4u);
    }else{
        union {uint32_t u;float f;} cv;
        cv.u=uint32_t(*(__gm__ uint16_t*)(ws+uint64_t(owner)*4u))<<16;scale0=cv.f;
        cv.u=uint32_t(*(__gm__ uint16_t*)(ws+uint64_t(owner)*4u+2u))<<16;scale1=cv.f;
    }
    // Per-invocation bounded row-input snapshot. No UB array or persistent state.
    const uint32_t snapshotRows=end-begin;
    float sx0=0.0f,sx1=0.0f,sx2=0.0f,sx3=0.0f;
    int32_t ax00=0,ax01=0,ax10=0,ax11=0,ax20=0,ax21=0,ax30=0,ax31=0;
    sx0=xsg.GetValue(begin);
    ax00=int32_t(xg.GetValue(uint64_t(begin)*2u));
    ax01=int32_t(xg.GetValue(uint64_t(begin)*2u+1u));
    if(snapshotRows>1u){
        sx1=xsg.GetValue(begin+1u);
        ax10=int32_t(xg.GetValue(uint64_t(begin+1u)*2u));
        ax11=int32_t(xg.GetValue(uint64_t(begin+1u)*2u+1u));
    }
    if(snapshotRows>2u){
        sx2=xsg.GetValue(begin+2u);
        ax20=int32_t(xg.GetValue(uint64_t(begin+2u)*2u));
        ax21=int32_t(xg.GetValue(uint64_t(begin+2u)*2u+1u));
    }
    if(snapshotRows>3u){
        sx3=xsg.GetValue(begin+3u);
        ax30=int32_t(xg.GetValue(uint64_t(begin+3u)*2u));
        ax31=int32_t(xg.GetValue(uint64_t(begin+3u)*2u+1u));
    }
    for(uint32_t m=begin;m<end;++m){
        const uint32_t snapshotIndex=m-begin;
        const float row=snapshotIndex==0u?sx0:(snapshotIndex==1u?sx1:(snapshotIndex==2u?sx2:sx3));
        const int32_t x0=snapshotIndex==0u?ax00:(snapshotIndex==1u?ax10:(snapshotIndex==2u?ax20:ax30));
        const int32_t x1=snapshotIndex==0u?ax01:(snapshotIndex==1u?ax11:(snapshotIndex==2u?ax21:ax31));
        const int32_t sum0=x0*w00+x1*w10,sum1=x0*w01+x1*w11;
        const uint32_t encoded=uint32_t(GvEncode<(sizeof(ScaleT)==2)>(float(sum0)*scale0*row))|
            (uint32_t(GvEncode<(sizeof(ScaleT)==2)>(float(sum1)*scale1*row))<<16u);
        const uint32_t index=m-begin;
        if(index<2u)word0|=uint64_t(encoded)<<(index*32u);
        else word1|=uint64_t(encoded)<<((index-2u)*32u);
    }
    const uint32_t rows=end-begin;
    auto* dst=reinterpret_cast<__gm__ uint32_t*>(y)+begin;
    if(begin&1u){
        WriteGmByPassDCache(dst,uint32_t(word0));
        if(rows==3u)WriteGmByPassDCache(reinterpret_cast<__gm__ uint64_t*>(dst+1u),(word0>>32u)|(word1<<32u));
        else if(rows==2u)WriteGmByPassDCache(dst+1u,uint32_t(word0>>32u));
    }else{
        if(rows>=2u)WriteGmByPassDCache(reinterpret_cast<__gm__ uint64_t*>(dst),word0);
        else WriteGmByPassDCache(dst,uint32_t(word0));
        if(rows==4u)WriteGmByPassDCache(reinterpret_cast<__gm__ uint64_t*>(dst+2u),word1);
        else if(rows==3u)WriteGmByPassDCache(dst+2u,uint32_t(word1));
    }
    DataSyncBarrier<MemDsbT::DDR>();
}
}

