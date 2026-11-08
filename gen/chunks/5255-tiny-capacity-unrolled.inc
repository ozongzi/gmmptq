namespace tiny_capacity_unrolled {using namespace gmmptq;using namespace AscendC;
template<uint32_t Cap> __aicore__ inline uint64_t Load(GM_ADDR p,uint32_t bytes){
 if(bytes==0u)return 0u;
 if constexpr(Cap==1u)return uint64_t(*(__gm__ uint8_t*)p);
 else {
  if(bytes==Cap && (reinterpret_cast<uintptr_t>(p)&(Cap-1u))==0u){
   if constexpr(Cap==8u)return *(__gm__ uint64_t*)p;
   else if constexpr(Cap==4u)return *(__gm__ uint32_t*)p;
   else return *(__gm__ uint16_t*)p;
  }
  constexpr uint32_t Half=Cap/2u;
  uint64_t lo=Load<Half>(p,Min(bytes,Half));
  if(bytes>Half)lo|=Load<Half>(p+Half,bytes-Half)<<(Half*8u);
  return lo;
 }
}
template<uint32_t Cap> __aicore__ inline void Store(GM_ADDR p,uint64_t bits,uint32_t bytes){
 if(bytes==0u)return;
 if constexpr(Cap==2u)WriteGmByPassDCache((__gm__ uint16_t*)p,uint16_t(bits));
 else{
  if(bytes==Cap && (reinterpret_cast<uintptr_t>(p)&(Cap-1u))==0u){
   if constexpr(Cap==8u)WriteGmByPassDCache((__gm__ uint64_t*)p,bits);
   else WriteGmByPassDCache((__gm__ uint32_t*)p,uint32_t(bits));
  }else{
   constexpr uint32_t Half=Cap/2u;Store<Half>(p,bits,Min(bytes,Half));
   if(bytes>Half)Store<Half>(p+Half,bits>>(Half*8u),bytes-Half);
  }
 }
}
__aicore__ inline float F(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}
__aicore__ inline int32_t S(uint64_t u,uint32_t i){return int32_t(int8_t((u>>(i*8u))&255u));}
template<bool BF,uint32_t KC,uint32_t NC,bool Full,bool Single>
__aicore__ inline void Rows(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR y,uint32_t m,uint32_t actualK,uint32_t actualN,uint32_t owners){
 constexpr uint32_t MC=(4u<8u/KC?4u:8u/KC)<8u/NC?(4u<8u/KC?4u:8u/KC):8u/NC;
 constexpr uint32_t XCap=MC*KC, WCap=KC*NC, SCap=NC*(BF?2u:4u);
 static_assert(XCap<=8u && WCap<=8u && SCap<=8u && MC*NC<=8u,"capacity family");
 const uint32_t k=Full?KC:actualK,n=Full?NC:actualN;
 const uint64_t xx=Load<XCap>(x,m*k),sx0=Load<8>(xs,Min(8u,m*4u));const uint64_t sx1=m>2u?Load<8>(xs+8u,m*4u-8u):0u;
 uint64_t ww=0,sw=0,out0=0,out1=0;uint32_t held=0xffffffffu;
 if constexpr(Single){ww=Load<WCap>(w,k*n);sw=Load<SCap>(ws,n*(BF?2u:4u));}
 #pragma unroll
 for(uint32_t row=0;row<MC;++row)if(row<m){
  if constexpr(!Single){const uint32_t owner=(owners>>(8u*row))&255u;if(owner!=held){ww=Load<WCap>(w+uint64_t(owner)*k*n,k*n);sw=Load<SCap>(ws+uint64_t(owner)*n*(BF?2u:4u),n*(BF?2u:4u));held=owner;}}
  const float scale=F(uint32_t((row<2u?sx0:sx1)>>(32u*(row&1u))));
  #pragma unroll
  for(uint32_t col=0;col<NC;++col)if(Full || col<n){
   int32_t dot=0;
   #pragma unroll
   for(uint32_t t=0;t<KC;++t)if(Full || t<k)dot+=S(xx,row*k+t)*S(ww,t*n+col);
   const uint32_t sb=BF?uint32_t(uint16_t(sw>>(16u*col)))<<16u:uint32_t(sw>>(32u*col));
   const float weighted=float(dot)*F(sb);const uint16_t v=GvEncode<BF>(weighted*scale);const uint32_t q=row*n+col;
   if(q<4u)out0|=uint64_t(v)<<(q*16u);else out1|=uint64_t(v)<<((q-4u)*16u);
  }
 }
 const uint32_t bytes=m*n*2u;Store<8>(y,out0,Min(bytes,8u));if(bytes>8u)Store<8>(y+8u,out1,bytes-8u);
}
template<bool BF,uint32_t KC,uint32_t NC> __global__ __aicore__ void SearchTinyCapacityUnrolled11850012(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 const uint32_t m=(dims&3u)+1u,k=((dims>>2u)&7u)+1u,n=((dims>>5u)&3u)+1u,e=((dims>>7u)&127u)+1u,gt=(dims>>14u)&1u;
 GlobalTensor<uint64_t> preload;
 if(e>1u){preload.SetGlobalBuffer((__gm__ uint64_t*)gl);DataCachePreload(preload,int16_t(0));}
 preload.SetGlobalBuffer((__gm__ uint64_t*)x);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)xs);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)w);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)ws);DataCachePreload(preload,int16_t(0));
 uint32_t owners=0;
 if(e>1u){uint32_t done=0;uint64_t end=0;
  for(uint32_t expert=0;expert+1u<e && done<m;++expert){const uint64_t v=Load<8>(gl+uint64_t(expert)*8u,8u);end=gt?end+v:v;while(done<m && uint64_t(done)<end){owners|=expert<<(8u*done);++done;}}
  while(done<m){owners|=(e-1u)<<(8u*done);++done;}
 }
 if(e==1u){
  if(k==KC && n==NC)Rows<BF,KC,NC,true,true>(x,w,xs,ws,y,m,k,n,0u);
  else Rows<BF,KC,NC,false,true>(x,w,xs,ws,y,m,k,n,0u);
 }else{
  if(k==KC && n==NC)Rows<BF,KC,NC,true,false>(x,w,xs,ws,y,m,k,n,owners);
  else Rows<BF,KC,NC,false,false>(x,w,xs,ws,y,m,k,n,owners);
 }
 DataSyncBarrier<MemDsbT::DDR>();
}
}

namespace tiny_capacity_unrolled {
template<bool BF> inline void Select(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,uint32_t dims,aclrtStream stream){
 const uint32_t k=((dims>>2u)&7u)+1u,n=((dims>>5u)&3u)+1u;
 if(k<=1u){
  if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,1u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,1u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,1u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,1u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  if constexpr(BF){
  if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,1u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,1u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  }
 }
 if(k<=2u){
  if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,2u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,2u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,2u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,2u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  if constexpr(BF){
  if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,2u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,2u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  }
 }
 if(k<=4u){
  if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,4u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,4u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
  if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,4u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,4u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 if(k<=8u){
  if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityUnrolled11850012<BF,8u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityUnrolled11850012<BF,8u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 throw std::invalid_argument("outside registered power-of-two capacity family");
}
}

