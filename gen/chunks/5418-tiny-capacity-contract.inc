namespace tiny_capacity_contract {using namespace gmmptq;using namespace AscendC;
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
// Row capacity controls the physically live scalar words, not an exact-M guard.
template<uint32_t RC> struct RowScales {
 uint64_t low;
 __aicore__ inline RowScales(GM_ADDR p,uint32_t rows):low(Load<RC*4u>(p,rows*4u)){}
 __aicore__ inline float Get(uint32_t row) const{return F(uint32_t(low>>(32u*row)));}
};
template<> struct RowScales<4u> {
 uint64_t low,high;
 __aicore__ inline RowScales(GM_ADDR p,uint32_t rows):low(Load<8>(p,8u)),high(Load<8>(p+8u,rows*4u-8u)){}
 __aicore__ inline float Get(uint32_t row) const{return F(uint32_t((row<2u?low:high)>>(32u*(row&1u))));}
};
template<uint32_t Cells> struct Output {
 uint64_t low=0;
 __aicore__ inline void Put(uint32_t q,uint16_t v){low|=uint64_t(v)<<(16u*q);}
 __aicore__ inline void Write(GM_ADDR p,uint32_t bytes){Store<Cells*2u>(p,low,bytes);}
};
template<> struct Output<8u> {
 uint64_t low=0,high=0;
 __aicore__ inline void Put(uint32_t q,uint16_t v){if(q<4u)low|=uint64_t(v)<<(16u*q);else high|=uint64_t(v)<<(16u*(q-4u));}
 __aicore__ inline void Write(GM_ADDR p,uint32_t bytes){Store<8>(p,low,8u);if(bytes>8u)Store<8>(p+8u,high,bytes-8u);}
};
template<bool BF,uint32_t RC,uint32_t KC,uint32_t NC,bool Full>
__aicore__ inline void Rows(GM_ADDR w,GM_ADDR ws,GM_ADDR y,uint32_t m,uint32_t actualK,uint32_t actualN,uint32_t owners,uint64_t xx,const RowScales<RC>& rowScales){
 constexpr uint32_t MC=RC;
 constexpr uint32_t XCap=MC*KC, WCap=KC*NC, SCap=NC*(BF?2u:4u);
 static_assert(XCap<=8u && WCap<=8u && SCap<=8u && MC*NC<=8u,"capacity family");
 const uint32_t k=Full?KC:actualK,n=Full?NC:actualN;
 Output<RC*NC> output;
 uint64_t ww=0,sw=0;uint32_t held=0xffffffffu;
 #pragma unroll
 for(uint32_t row=0;row<MC;++row)if(row<m){
  const uint32_t owner=(owners>>(8u*row))&255u;if(owner!=held){ww=Load<WCap>(w+uint64_t(owner)*k*n,k*n);sw=Load<SCap>(ws+uint64_t(owner)*n*(BF?2u:4u),n*(BF?2u:4u));held=owner;}
  const float scale=rowScales.Get(row);
  #pragma unroll
  for(uint32_t col=0;col<NC;++col)if(Full || col<n){
   int32_t dot=0;
   #pragma unroll
   for(uint32_t t=0;t<KC;++t)if(Full || t<k)dot+=S(xx,row*k+t)*S(ww,t*n+col);
   const uint32_t sb=BF?uint32_t(uint16_t(sw>>(16u*col)))<<16u:uint32_t(sw>>(32u*col));
   const float weighted=float(dot)*F(sb);const uint16_t v=GvEncode<BF>(weighted*scale);const uint32_t q=row*n+col;
   output.Put(q,v);
  }
 }
 output.Write(y,m*n*2u);
}
// Independent addresses; no loaded GL value controls issuance of the next GM read.
template<uint32_t RC> __aicore__ inline uint32_t BatchOwners(GM_ADDR gl,uint32_t e,uint32_t m,uint32_t gt){
 uint32_t v0=0u,v1=0u,v2=0u,v3=0u,v4=0u,v5=0u,v6=0u;
 if(e>1u)v0=uint32_t(Load<4>(gl+0u,4u));
 if(e>2u)v1=uint32_t(Load<4>(gl+8u,4u));
 if(e>3u)v2=uint32_t(Load<4>(gl+16u,4u));
 if(e>4u)v3=uint32_t(Load<4>(gl+24u,4u));
 if(e>5u)v4=uint32_t(Load<4>(gl+32u,4u));
 if(e>6u)v5=uint32_t(Load<4>(gl+40u,4u));
 if(e>7u)v6=uint32_t(Load<4>(gl+48u,4u));
 if(gt){v1+=v0;v2+=v1;v3+=v2;v4+=v3;v5+=v4;v6+=v5;}
 uint32_t owners=0u;
 #pragma unroll
 for(uint32_t row=0u;row<RC;++row)if(row<m){
  uint32_t owner=0u;
  owner+=uint32_t(e>1u && v0<=row);
  owner+=uint32_t(e>2u && v1<=row);
  owner+=uint32_t(e>3u && v2<=row);
  owner+=uint32_t(e>4u && v3<=row);
  owner+=uint32_t(e>5u && v4<=row);
  owner+=uint32_t(e>6u && v5<=row);
  owner+=uint32_t(e>7u && v6<=row);
  owners|=owner<<(8u*row);
 }
 return owners;
}
// Host binds each dimension to its least power-of-two capacity bucket.
template<uint32_t C> __aicore__ inline uint32_t DecodeCapacity(uint32_t encodedMinus1){
 static_assert(C==1u||C==2u||C==4u||C==8u,"power-of-two dimension capacity");
 if constexpr(C<=2u)return C;
 else return C/2u+1u+(encodedMinus1&(C/2u-1u));
}
template<bool BF,uint32_t RC,uint32_t KC,uint32_t NC> __global__ __aicore__ void SearchTinyCapacityContract11850017(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,uint32_t dims){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 const uint32_t m=DecodeCapacity<RC>(dims&3u),k=DecodeCapacity<KC>((dims>>2u)&7u),n=DecodeCapacity<NC>((dims>>5u)&3u),e=((dims>>7u)&127u)+1u,gt=(dims>>14u)&1u;
 GlobalTensor<uint64_t> preload;
 preload.SetGlobalBuffer((__gm__ uint64_t*)gl);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)x);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)xs);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)w);DataCachePreload(preload,int16_t(0));
 preload.SetGlobalBuffer((__gm__ uint64_t*)ws);DataCachePreload(preload,int16_t(0));
 // Input scalar state is loaded before either owner-resolution path.
 const uint64_t xx=Load<RC*KC>(x,m*k);const RowScales<RC> rowScales(xs,m);
 // Host admits only E2..8 with naturally aligned GL fully contained in one64B line.
 const uint32_t owners=BatchOwners<RC>(gl,e,m,gt);
 if(k==KC && n==NC)Rows<BF,RC,KC,NC,true>(w,ws,y,m,k,n,owners,xx,rowScales);
 else Rows<BF,RC,KC,NC,false>(w,ws,y,m,k,n,owners,xx,rowScales);
 DataSyncBarrier<MemDsbT::DDR>();
}
}

namespace tiny_capacity_contract {
template<bool BF> inline void Select(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR gl,GM_ADDR y,uint32_t dims,aclrtStream stream){
 const uint32_t m=(dims&3u)+1u,k=((dims>>2u)&7u)+1u,n=((dims>>5u)&3u)+1u;
 if(m<=1u){
 if(k<=1u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,1u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,1u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,1u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,1u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if constexpr(BF){
 if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,1u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,1u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(k<=2u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,2u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,2u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,2u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,2u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if constexpr(BF){
 if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,2u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,2u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(k<=4u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,4u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,4u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,4u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,4u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 if(k<=8u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,1u,8u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,1u,8u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(m<=2u){
 if(k<=1u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,1u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,1u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,1u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,1u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if constexpr(BF){
 if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,1u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,1u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(k<=2u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,2u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,2u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,2u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,2u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if constexpr(BF){
 if(n<=4u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,2u,4u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,2u,4u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(k<=4u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,4u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,4u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,2u,4u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,2u,4u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 if(m<=4u){
 if(k<=1u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,4u,1u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,4u,1u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,4u,1u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,4u,1u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 if(k<=2u){
 if(n<=1u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,4u,2u,1u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,4u,2u,1u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 if(n<=2u){
#ifdef TINY_CAPACITY_CPU
 SearchTinyCapacityContract11850017<BF,4u,2u,2u>(x,w,xs,ws,gl,y,dims);
#else
 SearchTinyCapacityContract11850017<BF,4u,2u,2u><<<1,nullptr,stream>>>(x,w,xs,ws,gl,y,dims);
#endif
 return;}
 }
 }
 throw std::invalid_argument("outside registered M/K/N capacity family");
}
}

