// Packed signed-byte split, exact int16 products, float pair sums.
// One dynamic N1 resource family; original contiguous row ownership and scale order.
template<typename ScaleT,typename OutT,uint32_t R=32,uint32_t C=800>
__global__ __aicore__ void SearchN1FixedLayoutSafe7100000(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
    GM_ADDR groups,GM_ADDR y,GeneralVectorDims s,uint32_t cores){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    static_assert(C%32u==0 && R==32u,"aligned pair rows");
    constexpr uint32_t ROW4=(R*4u+31u)/32u*32u,ROW2=(R*2u+31u)/32u*32u;
    constexpr uint32_t UB_BYTES=R*C+C+2u*R*C+4u*R*C+2u*C+ROW4+ROW4+ROW2+R*32u+ROW4;
    static_assert(UB_BYTES==183072u && UB_BYTES<=188160u,"pair UB bound");
    TPipe pipe;
    uint32_t at=0;
    LocalTensor<int8_t> xl(TPosition::VECCALC,at,(R*C)/1u);at+=(R*C);
    LocalTensor<int8_t> wl(TPosition::VECCALC,at,(C)/1u);at+=(C);
    LocalTensor<int16_t> even(TPosition::VECCALC,at,(R*C)/2u);at+=(R*C);
    LocalTensor<int16_t> odd(TPosition::VECCALC,at,(R*C)/2u);at+=(R*C);
    LocalTensor<float> f(TPosition::VECCALC,at,(R*C*2u)/4u);at+=(R*C*2u);
    LocalTensor<float> other(TPosition::VECCALC,at,(R*C*2u)/4u);at+=(R*C*2u);
    LocalTensor<int16_t> weven(TPosition::VECCALC,at,(C)/2u);at+=(C);
    LocalTensor<int16_t> wodd(TPosition::VECCALC,at,(C)/2u);at+=(C);
    LocalTensor<float> sums(TPosition::VECCALC,at,(ROW4)/4u);at+=(ROW4);
    LocalTensor<float> scales(TPosition::VECCALC,at,(ROW4)/4u);at+=(ROW4);
    LocalTensor<OutT> out(TPosition::VECCALC,at,(ROW2)/2u);at+=(ROW2);
    LocalTensor<float> spaced(TPosition::VECCALC,at,(R*32u)/4u);at+=(R*32u);
    LocalTensor<int32_t> index(TPosition::VECCALC,at,(ROW4)/4u);at+=(ROW4);
    // Fixed layout, 183072 bytes: the original TBuf sequence less the 1600-byte
    // `scratch` the two-level reduce needed. Every later offset shifts down by
    // 1600 = 50*32, so all of them stay 32-byte aligned (see :590).
    CreateVecIndex(index,int32_t(0),R);PipeBarrier<PIPE_V>();
    Muls(index,index,int32_t(32),R);PipeBarrier<PIPE_V>();
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;GlobalTensor<int64_t> gg;GlobalTensor<OutT> yg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);gg.SetGlobalBuffer((__gm__ int64_t*)groups);yg.SetGlobalBuffer((__gm__ OutT*)y);
    uint32_t begin=uint64_t(s.m)*GetBlockIdx()/cores,finish=uint64_t(s.m)*(GetBlockIdx()+1)/cores;
    uint32_t width=(s.k+31)/32*32,start=0;
    const uint32_t pairs=width/2u;
    for(uint32_t e=0;e<s.e;++e){
        uint32_t entry=gg.GetValue(e),end=s.listType==0?entry:start+entry;
        uint32_t first=begin>start?begin:start,last=finish<end?finish:end;
        if(first<last){
            // Explicit right-padding DMA writes every split-input byte, including odd partner.
                        DataCopyPad(wl,wg[uint64_t(e)*s.k],DataCopyExtParams{1,s.k,0,0,0},DataCopyPadExtParams<int8_t>{true,0,uint8_t(width-s.k),0});
            GvBarrier<HardEvent::MTE2_V>();
            auto wp=wl.ReinterpretCast<int16_t>();
            ShiftLeft(weven,wp,int16_t(8),pairs);
            ShiftRight(wodd,wp,int16_t(8),pairs);PipeBarrier<PIPE_V>();
            ShiftRight(weven,weven,int16_t(8),pairs);GvBarrier<HardEvent::V_MTE2>();
            float weightScale;
            if constexpr(sizeof(ScaleT)==4)weightScale=*(__gm__ float*)(ws+uint64_t(e)*4);
            else {union{uint32_t u;float f;}v;v.u=uint32_t(*(__gm__ uint16_t*)(ws+uint64_t(e)*2))<<16;weightScale=v.f;}
            for(uint32_t m=first;m<last;m+=R){
                uint32_t nr=last-m<R?last-m:R;
                                DataCopyPad(xl,xg[uint64_t(m)*s.k],DataCopyExtParams{uint16_t(nr),s.k,0,0,0},DataCopyPadExtParams<int8_t>{true,0,uint8_t(width-s.k),0});
                DataCopyPad(scales,xsg[m],DataCopyExtParams{1,nr*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                GvBarrier<HardEvent::MTE2_V>();
                auto xp=xl.ReinterpretCast<int16_t>();
                ShiftLeft(even,xp,int16_t(8),nr*pairs);
                ShiftRight(odd,xp,int16_t(8),nr*pairs);PipeBarrier<PIPE_V>();
                ShiftRight(even,even,int16_t(8),nr*pairs);PipeBarrier<PIPE_V>();
                for(uint32_t col=0;col<pairs;col+=128u){
                    const uint64_t mask=pairs-col<128u?pairs-col:128u;
                    const BinaryRepeatParams row{1,1,1,uint8_t(width/32u),uint8_t(width/32u),0};
                    Mul(even[col],even[col],weven[col],mask,uint8_t(nr),row);
                    Mul(odd[col],odd[col],wodd[col],mask,uint8_t(nr),row);
                }
                PipeBarrier<PIPE_V>();
                Cast(f,even,RoundMode::CAST_NONE,nr*pairs);
                Cast(other,odd,RoundMode::CAST_NONE,nr*pairs);PipeBarrier<PIPE_V>();
                // Never add inint16: two16384 products would overflow32767.
                Add(f,f,other,nr*pairs);PipeBarrier<PIPE_V>();
                // Halving fold with 32-byte-aligned split points, then one
                // reduce over the last <=64.  f[(k+1)/2 .. pairs) is zero:
                // xl and wl are both DataCopyPad'ed with paddingValue 0 over
                // [k, width), so even/odd are 0 there, Mul keeps them 0 and
                // Cast gives 0.0f -- every original index lands in exactly one
                // final slot.  Four levels and not nine because each split
                // point is rounded UP to 8 floats: f[50] would be byte 200,
                // which is not 32-byte aligned, and an unaligned vector
                // operand start faults the unit (ACL 507035, see :590).
                {
                    const BinaryRepeatParams tr{1,1,1,uint8_t(pairs/8u),uint8_t(pairs/8u),uint8_t(pairs/8u)};
                    uint32_t act=pairs;
                    while(act>64u){
                        const uint32_t half=((act+1u)/2u+7u)&~7u, tail=act-half;
                        for(uint32_t c=0;c<tail;c+=64u){
                            const uint64_t mask=tail-c<64u?tail-c:64u;
                            Add(f[c],f[c],f[half+c],mask,uint8_t(nr),tr);
                        }
                        PipeBarrier<PIPE_V>();
                        act=half;
                    }
                    WholeReduceSum(spaced,f,uint64_t(act),uint8_t(nr),8,1,uint8_t(pairs/8u));
                    PipeBarrier<PIPE_V>();
                }
                Gather(sums,spaced,index.ReinterpretCast<uint32_t>(),0,nr);PipeBarrier<PIPE_V>();
                Muls(sums,sums,weightScale,nr);PipeBarrier<PIPE_V>();
                Mul(sums,sums,scales,nr);PipeBarrier<PIPE_V>();
                Cast(out,sums,RoundMode::CAST_RINT,nr);GvBarrier<HardEvent::V_MTE3>();
                DataCopyPad(yg[m],out,DataCopyExtParams{1,nr*2,0,0,0});
                GvBarrier<HardEvent::MTE3_V>();GvBarrier<HardEvent::V_MTE2>();
            }
        }
        start=end;
    }
}

