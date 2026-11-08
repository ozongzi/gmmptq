namespace gmmptq {
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchCubeCompact4700003(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
    GM_ADDR gl,GM_ADDR y,GM_ADDR staging,uint32_t dims) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=((((dims>>0u)&3u)+1u)<<7u);
 p.n=((((dims>>2u)&7u)+1u)<<7u);
 p.e=(((dims>>5u)&15u)+1u);
 p.cores=(((dims>>9u)&31u)+1u);
 p.listType=(((dims>>14u)&1u)+0u);
 p.kl=p.k;p.wstride=p.n;

    CubeFullKBody<ScaleT,OutT,64,128,128,64,false,true,false,false,2u,true>(x,w,xs,ws,gl,y,staging,p);
}
inline void LaunchAlignedMedium64(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,
    GM_ADDR y,const Plan& p,bool bf,aclrtStream stream) {
 const uint32_t dims=((p.k/128u-1u))|((p.n/128u-1u)<<2u)|((p.e-1u)<<5u)|((p.cores-1u)<<9u)|((p.listType)<<14u);

    Scratch stage(CubeFullKStagingBytes(p));
    try {
        if(bf)SearchCubeCompact4700003<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)stage.p,dims);
        else SearchCubeCompact4700003<float,half><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)stage.p,dims);
        Check(aclrtSynchronizeStream(stream));
    }catch(...){aclrtSynchronizeStream(stream);throw;}
}
}

