namespace packed32 {
using namespace gmmptq;
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchP2Packed324300000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,GM_ADDR st,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=(dims&511u)+1u;
 p.n=((dims>>9u)&1023u)+1u;
 p.e=((dims>>19u)&127u)+1u;
 p.cores=((dims>>26u)&31u)+1u;
 p.listType=dims>>31u;
 p.kl=(p.k+31u)&~31u;p.wstride=p.n;
 CubeFullKBody<ScaleT,OutT,96,128,512,48,false,false,true,true>(x,w,xs,ws,gl,y,st,p);
}
inline void LaunchPacked32(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,
                             GM_ADDR y,const Plan& incoming,bool bf,aclrtStream stream) {
    auto* shortPlatform=platform_ascendc::PlatformAscendCManager::GetInstance();
    if(!shortPlatform)throw std::runtime_error("Cannot obtain physical AIC count");
    const uint32_t physicalCores=shortPlatform->GetCoreNumAic();
    Plan p=incoming;
    p.cores=std::min(incoming.cores,physicalCores);
    p.vcores=p.cores*2u;
    if(!p.cores)throw std::runtime_error("No physical AIC cores");
    Scratch scratch(CubeFullKStagingBytes(p));
    try {
        if(p.cores<=32u){
            const uint32_t dims=(p.k-1u)|((p.n-1u)<<9u)|((p.e-1u)<<19u)|((p.cores-1u)<<26u)|(p.listType<<31u);
            if(bf)SearchP2Packed324300000<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
            else SearchP2Packed324300000<float,half><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,dims);
        }else{
        if(bf) SearchP2Fixed964100000<bfloat16_t,bfloat16_t>
            <<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,p);
        else SearchP2Fixed964100000<float,half>
            <<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,p);
        }
        Check(aclrtSynchronizeStream(stream));
    } catch(...) {aclrtSynchronizeStream(stream);throw;}
}
}

