namespace gmmptq {
template<typename ScaleT,typename OutT>
__global__ __aicore__ __attribute__((aligned(128))) void SearchCubeCompact4700001(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,GM_ADDR staging,uint32_t dims) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=(((dims>>0u)&511u)+1u);
 p.n=(((dims>>9u)&1023u)+1u);
 p.e=(((dims>>19u)&15u)+1u);
 p.cores=(((dims>>23u)&31u)+1u);
 p.listType=(((dims>>28u)&1u)+0u);
 p.kl=(p.k+31u)&~31u;p.wstride=p.n;

 CubeFullKBodyScalarHalf<ScaleT,OutT,192,128,96,128,false,true,false,true,true,true,true,true,true,true,false>(x,w,xs,ws,gl,y,staging,p);
}
inline void LaunchMediumScalarHalf(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
 GM_ADDR gl,GM_ADDR y,const Plan& p,aclrtStream stream) {
 const uint32_t dims=((p.k-1u))|((p.n-1u)<<9u)|((p.e-1u)<<19u)|((p.cores-1u)<<23u)|((p.listType)<<28u);

 Scratch scratch(CubeFullKStagingBytes(p)/2u);
 try {
  SearchCubeCompact4700001<float,half><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
  Check(aclrtSynchronizeStream(stream));
 }catch(...){aclrtSynchronizeStream(stream);throw;}
}
}

