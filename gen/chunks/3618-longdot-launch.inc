template<uint32_t Capacity,bool BF,bool ScalarOutput>
__global__ __aicore__ void SearchLongDotCapacity11840033(
 GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR y,uint32_t k){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 longdot_capacity::Dot<Capacity,false,BF?3u:1u,!BF,false,BF,ScalarOutput> op;
 op.Run(x,w,xs,ws,y,k);
}
// FP odd scale addresses use public byte-DMA scales, preserving WFirst.
// This is a new path; original 132 correctness runs did not cover it.
template<uint32_t Capacity,bool ScalarOutput>
__global__ __aicore__ void SearchLongDotCapacityDMAScales(
 GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR y,uint32_t k){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 longdot_capacity::Dot<Capacity,false,3u,true,false,false,ScalarOutput> op;
 op.Run(x,w,xs,ws,y,k);
}
namespace longdot_capacity {
template<uint32_t Capacity,bool BF> inline void Launch(
 GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR y,uint32_t k,bool scalar,aclrtStream stream){
 if constexpr(!BF){
  if(((reinterpret_cast<uintptr_t>(xs)|reinterpret_cast<uintptr_t>(ws))&3u)!=0u){
   if(scalar)SearchLongDotCapacityDMAScales<Capacity,true><<<2,nullptr,stream>>>(x,w,xs,ws,y,k);
   else SearchLongDotCapacityDMAScales<Capacity,false><<<2,nullptr,stream>>>(x,w,xs,ws,y,k);
   return;
  }
 }
 if(scalar)SearchLongDotCapacity11840033<Capacity,BF,true><<<2,nullptr,stream>>>(x,w,xs,ws,y,k);
 else SearchLongDotCapacity11840033<Capacity,BF,false><<<2,nullptr,stream>>>(x,w,xs,ws,y,k);
}
}

