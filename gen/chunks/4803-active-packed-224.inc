namespace gmmptq {
__aicore__ inline uint32_t ActivePackCount(const AscendC::GlobalTensor<int64_t>& groups,
 uint32_t e,uint32_t listType){
 const uint32_t v=uint32_t(groups.GetValue(e));
 return listType==1u?v:v-(e?uint32_t(groups.GetValue(e-1u)):0u);
}
// Same original row format and DMA descriptors, balanced over activeExpertCount*K virtual rows.
template<bool StaticNoPipe = false,bool Double64 = false>
__aicore__ inline void PackActiveWeightRowsInline(GM_ADDR source,GM_ADDR dest,
 uint32_t rows,uint32_t k,uint32_t n,uint32_t pitch,uint32_t cores,GM_ADDR gl,uint32_t listType){
 using namespace AscendC;
 GlobalTensor<int64_t> groups;groups.SetGlobalBuffer((__gm__ int64_t*)gl);
 const uint32_t experts=rows/k;uint32_t active=0;
 for(uint32_t e=0;e<experts;++e)if(ActivePackCount(groups,e,listType)>0u)++active;
 const uint32_t core=GetBlockIdx(),virtualRows=active*k;
 const uint32_t begin=uint32_t(uint64_t(virtualRows)*core/cores);
 const uint32_t end=uint32_t(uint64_t(virtualRows)*(core+1u)/cores);
 typename FullKPipeType<StaticNoPipe>::Type pipe;typename FullKBufferType<StaticNoPipe,TPosition::VECCALC>::Type buf;
 if constexpr(StaticNoPipe)buf.bytes=131072;else pipe.InitBuffer(buf,131072);
 auto data=buf.template Get<int8_t>();GlobalTensor<int8_t> src,dst;
 src.SetGlobalBuffer((__gm__ int8_t*)source);dst.SetGlobalBuffer((__gm__ int8_t*)dest);
 const uint32_t batch=(Double64?65536u:131072u)/pitch;
 uint32_t issued=0;
 uint32_t e=0,ordinal=0;
 if(begin<end){
  const uint32_t target=begin/k;
  for(;e<experts;++e)if(ActivePackCount(groups,e,listType)>0u){if(ordinal==target)break;++ordinal;}
 }
 if constexpr(Double64) {
 for(uint32_t row=begin;row<end;){
  const uint32_t have=Min(Min(batch,end-row),k-row%k);
  const uint32_t realRow=e*k+row%k;
  const uint32_t slot=issued&1u;
  if(issued>=2u)WaitFlag<HardEvent::MTE3_MTE2>(slot+1u);
  LocalTensor<int8_t> chunk=data[slot*65536u];
  DataCopyPad(chunk,src[uint64_t(realRow)*n],
   DataCopyExtParams{uint16_t(have),n,0,(pitch-Up(n,32u))/32u,0},
   DataCopyPadExtParams<int8_t>{true,0,0,0});
  SetFlag<HardEvent::MTE2_MTE3>(slot+1u);
  WaitFlag<HardEvent::MTE2_MTE3>(slot+1u);
  DataCopyPad(dst[uint64_t(realRow)*pitch],chunk,
   DataCopyExtParams{1,have*pitch,0,0,0});
  SetFlag<HardEvent::MTE3_MTE2>(slot+1u);
  ++issued;
  row+=have;
  if(row<end&&row%k==0u){
   do{++e;}while(e<experts&&ActivePackCount(groups,e,listType)==0u);
  }
 }
 if(issued>0u)WaitFlag<HardEvent::MTE3_MTE2>(1u);
 if(issued>1u)WaitFlag<HardEvent::MTE3_MTE2>(2u);
 }else{
 for(uint32_t row=begin;row<end;){
  const uint32_t have=Min(Min(batch,end-row),k-row%k);
  const uint32_t realRow=e*k+row%k;
  DataCopyPad(data,src[uint64_t(realRow)*n],
   DataCopyExtParams{uint16_t(have),n,0,(pitch-Up(n,32u))/32u,0},
   DataCopyPadExtParams<int8_t>{true,0,0,0});
  Barrier<HardEvent::MTE2_MTE3>();
  DataCopyPad(dst[uint64_t(realRow)*pitch],data,
   DataCopyExtParams{1,have*pitch,0,0,0});
  Barrier<HardEvent::MTE3_MTE2>();
  row+=have;
  if(row<end&&row%k==0u){
   do{++e;}while(e<experts&&ActivePackCount(groups,e,listType)==0u);
  }
 }
 }
 Barrier<HardEvent::MTE3_S>();
 if constexpr(StaticNoPipe)pipe.Destroy(); // Preserve the original destructor PIPE_ALL.
}

template<bool BF>
 __global__ __aicore__ void SearchCubeCompact4700004(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
 GM_ADDR gl,GM_ADDR y,GM_ADDR st,GM_ADDR packed,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=(((dims>>0u)&1023u)+1u);
 p.n=(((dims>>10u)&2047u)+1u);
 p.e=(((dims>>21u)&15u)+1u);
 p.cores=(((dims>>25u)&31u)+1u);
 p.listType=(((dims>>30u)&1u)+0u);
 p.kl=(p.k+31u)&~31u;p.wstride=(p.n+127u)&~127u;
 p.vcores=2u*p.cores;

 if ASCEND_IS_AIV {PackActiveWeightRowsInline<true,true>(w,packed,p.e*p.k,p.k,p.n,p.wstride,p.vcores,gl,p.listType);}
 AscendC::SyncAll<false>();
 if constexpr(BF)CubeFullKUnitBody<bfloat16_t,bfloat16_t,224,128,128,64,false,true,false,true,true>(x,packed,xs,ws,gl,y,st,p);
 else CubeFullKBodyScalarHalf<float,half,224,128,128,64,false,true,false,true>(x,packed,xs,ws,gl,y,st,p);
 }
template<bool BF>
 __global__ __aicore__ void SearchCubeCompact4700005(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
 GM_ADDR gl,GM_ADDR y,GM_ADDR st,GM_ADDR packed,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Plan p{};
 p.k=(((dims>>0u)&1023u)+1u);
 p.n=(((dims>>10u)&2047u)+1u);
 p.e=(((dims>>21u)&15u)+1u);
 p.cores=(((dims>>25u)&31u)+1u);
 p.listType=(((dims>>30u)&1u)+0u);
 p.kl=(p.k+31u)&~31u;p.wstride=(p.n+127u)&~127u;
 p.vcores=2u*p.cores;

 if ASCEND_IS_AIV {PackActiveWeightRowsInline<true>(w,packed,p.e*p.k,p.k,p.n,p.wstride,p.vcores,gl,p.listType);}
 AscendC::SyncAll<false>();
 if constexpr(BF)CubeFullKUnitBody<bfloat16_t,bfloat16_t,224,128,128,64,false,true,false,true>(x,packed,xs,ws,gl,y,st,p);
 else CubeFullKBodyScalarHalf<float,half,224,128,128,64,false,true,false,true,false,false,false,false,true,false,true>(x,packed,xs,ws,gl,y,st,p);
 }
template<bool BF>
inline void LaunchActivePacked224(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,
 GM_ADDR y,GM_ADDR packed,const Plan& p,aclrtStream stream){
 const uint32_t dims=((p.k-1u))|((p.n-1u)<<10u)|((p.e-1u)<<21u)|((p.cores-1u)<<25u)|((p.listType)<<30u);

 Scratch staging(CubeFullKStagingBytes(p)/(BF?1u:2u));
 try {
  if constexpr(BF) SearchCubeCompact4700004<true><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)staging.p,packed,dims);
  else SearchCubeCompact4700005<false><<<p.cores,nullptr,stream>>>(x,w,xs,ws,gl,y,(GM_ADDR)staging.p,packed,dims);
  Check(aclrtSynchronizeStream(stream));
 }catch(...){aclrtSynchronizeStream(stream);throw;}
}
}

