namespace gmmptq {
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchCubeCompact4700002(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
    GM_ADDR gl,GM_ADDR y,GM_ADDR staging,uint32_t dims) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=(((dims>>0u)&511u)+1u);
 p.n=(((dims>>9u)&511u)+1u);
 p.e=(((dims>>18u)&15u)+1u);
 p.cores=(((dims>>22u)&31u)+1u);
 p.listType=(((dims>>27u)&1u)+0u);
 p.kl=(p.k+31u)&~31u;p.wstride=p.n;

    CubeFullKUnitBody<ScaleT,OutT,64,128,384,32,false,true,true,true>(x,w,xs,ws,gl,y,staging,p);
}
inline void LaunchNarrowSingleKSelected(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
    GM_ADDR gl,GM_ADDR y,const Plan& p,aclrtStream stream) {
 const uint32_t dims=((p.k-1u))|((p.n-1u)<<9u)|((p.e-1u)<<18u)|((p.cores-1u)<<22u)|((p.listType)<<27u);

    Scratch scratch(CubeFullKStagingBytes(p));
    try {
        SearchCubeCompact4700002<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
        Check(aclrtSynchronizeStream(stream));
    } catch(...) {aclrtSynchronizeStream(stream);throw;}
}
}

