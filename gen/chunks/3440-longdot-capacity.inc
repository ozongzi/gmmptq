#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "acl/acl.h"
#include "kernel_operator.h"
#include <type_traits>
// KCapacity/N2/E1 fixed family, shared two-output and original channel ownership.
namespace longdot_capacity {
using namespace AscendC;
struct Pointers {uint64_t x,w,xs,ws,y;};
static_assert(sizeof(Pointers)==40,"five pointer launch ABI");
template<HardEvent E>__aicore__ inline void Fence(){SetFlag<E>(0);WaitFlag<E>(0);}
__aicore__ inline float FloatBits(uint32_t bits){union{uint32_t u;float f;}v;v.u=bits;return v.f;}

template<uint32_t Capacity,bool Merged,uint32_t ScalePath,bool WFirst,bool Balanced,bool BF,bool ScalarOutput=false>
class Dot {
    static_assert(!Merged && !Balanced,"one channel per core; exact original integer reduction");
    static_assert(Capacity==1024u || Capacity==2048u || Capacity==4096u || Capacity==8192u,"power-of-two reduction capacity");
    static constexpr uint32_t CH=Merged?2u:1u;
    static constexpr uint32_t BYTES=13u*Capacity+(CH-1u)*6u*Capacity+CH*544u+160u;
    static_assert(BYTES<=188160u,"UB overflow");
    using Scale=typename std::conditional<BF,bfloat16_t,float>::type;
    using Out=typename std::conditional<BF,bfloat16_t,half>::type;
public:
    __aicore__ inline void Run(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,GM_ADDR y,uint32_t k){
        k_=k;
        InitSocState();
        LocalTensor<uint8_t> raw(TPosition::VECCALC,0,BYTES);
        uint32_t at=0;
        xi_=raw[at].template ReinterpretCast<int8_t>();at+=Capacity;
        wi_=raw[at].template ReinterpretCast<int8_t>();at+=(2u*Capacity);
        xh_=raw[at].template ReinterpretCast<half>();at+=(2u*Capacity);
        x16_=raw[at].template ReinterpretCast<int16_t>();at+=(2u*Capacity);
        w16_=raw[at].template ReinterpretCast<int16_t>();at+=CH*(2u*Capacity);
        wf_=raw[at].template ReinterpretCast<float>();at+=CH*(4u*Capacity);
        red1_=raw[at].template ReinterpretCast<float>();at+=CH*512;
        red2_=raw[at].template ReinterpretCast<float>();at+=CH*32;
        xsUb_=raw[at].template ReinterpretCast<float>();at+=32;
        wsUb_=raw[at].template ReinterpretCast<Scale>();at+=32;
        result_=raw[at].template ReinterpretCast<float>();at+=64;
        out_=raw[at].template ReinterpretCast<Out>();
        xg_.SetGlobalBuffer((__gm__ int8_t*)x);wg_.SetGlobalBuffer((__gm__ int8_t*)w);
        xsg_.SetGlobalBuffer((__gm__ float*)xs);wsg_.SetGlobalBuffer((__gm__ Scale*)ws);
        wsBits_.SetGlobalBuffer((__gm__ uint16_t*)ws);yg_.SetGlobalBuffer((__gm__ Out*)y);
        channel_=Merged?0u:GetBlockIdx();
        if(channel_>=2u){PipeBarrier<PIPE_ALL>();return;}
        if constexpr(ScalePath==0u || ScalePath==2u)PrepareScales(xs,ws);
        // Runtime tails are zeroed on device, before DMA takes ownership.
        if(k_ < Capacity){
            Duplicate(xi_.template ReinterpretCast<int16_t>(),int16_t(0),Capacity/2u);
            Duplicate(wi_.template ReinterpretCast<int16_t>(),int16_t(0),Capacity);
            Fence<HardEvent::V_MTE2>();
        }
        if constexpr(WFirst){LoadW();LoadX();}else{LoadX();LoadW();}
        if constexpr(ScalePath==1u || ScalePath==3u)PrepareScales(xs,ws);
        Fence<HardEvent::MTE2_V>();
        Cast(xh_,xi_,RoundMode::CAST_NONE,Capacity);PipeBarrier<PIPE_V>();
        Cast(x16_,xh_,RoundMode::CAST_RINT,Capacity);PipeBarrier<PIPE_V>();
        auto packed=wi_.template ReinterpretCast<int16_t>();
        if constexpr(Merged){
            ShiftLeft(w16_,packed,int16_t(8),Capacity);
            ShiftRight(w16_[Capacity],packed,int16_t(8),Capacity);
            PipeBarrier<PIPE_V>();
            ShiftRight(w16_,w16_,int16_t(8),Capacity);
        }else{
            if(channel_==0u){
                ShiftLeft(w16_,packed,int16_t(8),Capacity);PipeBarrier<PIPE_V>();
                ShiftRight(w16_,w16_,int16_t(8),Capacity);
            }else ShiftRight(w16_,packed,int16_t(8),Capacity);
        }
        PipeBarrier<PIPE_V>();
        Mul(w16_,w16_,x16_,Capacity);
        if constexpr(Merged)Mul(w16_[Capacity],w16_[Capacity],x16_,Capacity);
        PipeBarrier<PIPE_V>();
        // Pair exact INT16 products as two signed INT32 lanes, then reduce.
        // Reuse the original Capacity-float scratch; no UB or DMA change.
        auto packedProducts=w16_.template ReinterpretCast<int32_t>();
        auto integerSums=wf_.template ReinterpretCast<int32_t>();
        auto highProducts=integerSums[Capacity/2u];
        ShiftLeft(integerSums,packedProducts,int32_t(16),Capacity/2u);
        PipeBarrier<PIPE_V>();
        ShiftRight(integerSums,integerSums,int32_t(16),Capacity/2u);
        ShiftRight(highProducts,packedProducts,int32_t(16),Capacity/2u);
        PipeBarrier<PIPE_V>();
        Add(integerSums,integerSums,highProducts,Capacity/2u);
        PipeBarrier<PIPE_V>();
        // Each repeat consumes the next independent64-I32 row and the
        // previous repeat's accumulator as src1. Public Add overlap exception:
        // dst==src1, dstRepStride=src1RepStride=0. No I32 sum intrinsic.
        Add(integerSums,integerSums[64u],integerSums,uint64_t(64),
            uint8_t(Capacity/128u-1u),BinaryRepeatParams{1,1,1,0,8,0});
        PipeBarrier<PIPE_V>();
        Add(integerSums,integerSums,integerSums[32u],32u);
        PipeBarrier<PIPE_V>();
        Add(integerSums,integerSums,integerSums[16u],16u);
        PipeBarrier<PIPE_V>();
        Add(integerSums,integerSums,integerSums[8u],8u);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        if constexpr(ScalePath>=2u)Fence<HardEvent::MTE2_S>();
        const float xsValue=ScalePath>=2u?xsUb_.GetValue(0):xsg_.GetValue(0);
        const int32_t sum0=Sum<0>();
        const float ws0=WeightScale(channel_);
        const float weighted0=float(sum0)*ws0;
        if constexpr(ScalarOutput){
        const uint16_t encoded0=gmmptq::GvEncode<BF>(weighted0*xsValue);
        WriteGmByPassDCache(reinterpret_cast<__gm__ uint16_t*>(y)+channel_,encoded0);
        if constexpr(Merged){
            const int32_t sum1=Sum<8>();
            const float weighted1=float(sum1)*WeightScale(1);
            const uint16_t encoded1=gmmptq::GvEncode<BF>(weighted1*xsValue);
            WriteGmByPassDCache(reinterpret_cast<__gm__ uint16_t*>(y)+1u,encoded1);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        }else{
        result_.SetValue(0,weighted0*xsValue);
        if constexpr(Merged){
            const int32_t sum1=Sum<8>();
            const float weighted1=float(sum1)*WeightScale(1);
            result_.SetValue(1,weighted1*xsValue);
        }
        Fence<HardEvent::S_V>();
        Cast(out_,result_,RoundMode::CAST_RINT,16);Fence<HardEvent::V_MTE3>();
        DataCopyPad(yg_[channel_],out_,DataCopyExtParams{1,CH*2u,0,0,0});
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void LoadX(){
        DataCopyPad(xi_,xg_,DataCopyExtParams{1,k_,0,0,0},DataCopyPadExtParams<int8_t>{true,0,uint8_t((32u-k_%32u)%32u),0});
    }
    __aicore__ inline void LoadW(){
        DataCopyPad(wi_,wg_,DataCopyExtParams{1,2u*k_,0,0,0},DataCopyPadExtParams<int8_t>{true,0,uint8_t((32u-(2u*k_)%32u)%32u),0});
    }
    __aicore__ inline void PrepareScales(GM_ADDR xs,GM_ADDR ws){
        if constexpr(ScalePath<2u){
            DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(xsg_);
            auto selected=wsg_[channel_];
            DataCacheCleanAndInvalid<Scale,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(selected);
            const bool crosses=(uint64_t(ws)&63u)+2u*sizeof(Scale)>64u;
            if constexpr(Merged)if(crosses){
                auto tail=wsg_[1];
                DataCacheCleanAndInvalid<Scale,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(tail);
            }
            GlobalTensor<uint64_t> prefetch;
            prefetch.SetGlobalBuffer((__gm__ uint64_t*)xs);DataCachePreload(prefetch,int16_t(0));
            prefetch.SetGlobalBuffer((__gm__ uint64_t*)(ws+channel_*sizeof(Scale)));DataCachePreload(prefetch,int16_t(0));
            if constexpr(Merged)if(crosses){
                prefetch.SetGlobalBuffer((__gm__ uint64_t*)(ws+sizeof(Scale)));DataCachePreload(prefetch,int16_t(0));
            }
        }else{
            DataCopyPad(xsUb_,xsg_,DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            DataCopyPad(wsUb_,wsg_,DataCopyExtParams{1,2u*sizeof(Scale),0,0,0},DataCopyPadExtParams<Scale>{false,0,0,0});
        }
    }
    __aicore__ inline float WeightScale(uint32_t channel){
        if constexpr(BF){
            const uint32_t bits=ScalePath>=2u?wsUb_.template ReinterpretCast<uint16_t>().GetValue(channel):wsBits_.GetValue(channel);
            return FloatBits(bits<<16);
        }else return ScalePath>=2u?wsUb_.GetValue(channel):wsg_.GetValue(channel);
    }
    template<uint32_t Base>__aicore__ inline int32_t Sum(){
        static_assert(Base==0u,"one channel per core");
        auto values=wf_.template ReinterpretCast<int32_t>();
        int32_t total=0;
        for(uint32_t i=0;i<8u;++i)total+=values.GetValue(i);
        return total;
    }
    uint32_t channel_,k_;
    GlobalTensor<int8_t>xg_,wg_;GlobalTensor<float>xsg_;GlobalTensor<Scale>wsg_;
    GlobalTensor<uint16_t>wsBits_;GlobalTensor<Out>yg_;
    LocalTensor<int8_t>xi_,wi_;LocalTensor<half>xh_;LocalTensor<int16_t>x16_,w16_;
    LocalTensor<float>wf_,red1_,red2_,xsUb_,result_;LocalTensor<Scale>wsUb_;LocalTensor<Out>out_;
};
} // namespace longdot_capacity


