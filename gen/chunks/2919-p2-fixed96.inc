namespace gmmptq {
template<typename ScaleT,typename OutT>
__global__ __aicore__ void SearchP2Fixed964100000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,GM_ADDR st,Plan p){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 CubeFullKBody<ScaleT,OutT,96,128,512,48,false,false,true,true>(x,w,xs,ws,gl,y,st,p);
}
inline void LaunchPublishedFixed96(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,
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
        if(bf) SearchP2Fixed964100000<bfloat16_t,bfloat16_t>
            <<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,p);
        else SearchP2Fixed964100000<float,half>
            <<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)scratch.p,p);
        Check(aclrtSynchronizeStream(stream));
    } catch(...) {aclrtSynchronizeStream(stream);throw;}
}
}

