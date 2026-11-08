namespace mix11 {
using namespace gmmptq;
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchP2Mix115300000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,GM_ADDR st,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
 Plan p{};
 p.k=(dims&511u)+1u;p.n=((dims>>9u)&1023u)+1u;
 p.e=((dims>>19u)&127u)+1u;p.cores=((dims>>26u)&31u)+1u;
 p.listType=dims>>31u;p.kl=(p.k+31u)&~31u;p.wstride=p.n;
 CubeFullKBody<ScaleT,OutT,96,128,256,96,false,false,true,true,1u,true,true,true>(x,w,xs,ws,gl,y,st,p);
}
inline uint64_t Upper(const Plan& p){return ((uint64_t(p.m)+95u)/96u+p.e-1u)*((uint64_t(p.n)+127u)/128u);}
inline void Launch(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,const Plan& incoming,bool bf,aclrtStream stream){
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
 if(!platform)throw std::runtime_error("Cannot obtain physical AIC count");
 Plan p=incoming;p.cores=std::min(p.cores,platform->GetCoreNumAic());
 if(!p.cores)throw std::runtime_error("No physical AIC cores");
 if(p.cores>32u || Upper(p)>4ull*p.cores){packed32::LaunchPacked32(x,w,xs,ws,gl,y,incoming,bf,stream);return;}
 p.vcores=p.cores;
 Scratch scratch(CubeFullKStagingBytes(p));
 try{
  const uint32_t dims=(p.k-1u)|((p.n-1u)<<9u)|((p.e-1u)<<19u)|((p.cores-1u)<<26u)|(p.listType<<31u);
  if(bf)SearchP2Mix115300000<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
  else SearchP2Mix115300000<float,half><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
  Check(aclrtSynchronizeStream(stream));
 }catch(...){aclrtSynchronizeStream(stream);throw;}
}
}

