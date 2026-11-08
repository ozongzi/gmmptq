// N=1 matrix-vector products, arbitrary runtime M/E/K and either group encoding.
// Contiguous output ownership; full expert weights cached inside one invocation.
// K chunks <=1024 keep every float partial integer-exact, then accumulate int32.

template <uint32_t Rows,typename ScaleT,typename OutT>
__global__ __aicore__ void RowDot(GM_ADDR x,GM_ADDR w,GM_ADDR xs,GM_ADDR ws,
                                 GM_ADDR gl,GM_ADDR y,GeneralVectorDims p) {
    using namespace AscendC;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    static_assert(Rows>=1 && Rows<=16,"row batch must fit vector UB at maximum K");
    GlobalTensor<int8_t> xg,wg;GlobalTensor<float> xsg;
    GlobalTensor<ScaleT> wsg;GlobalTensor<OutT> yg;GlobalTensor<int64_t> glg;
    xg.SetGlobalBuffer((__gm__ int8_t*)x);wg.SetGlobalBuffer((__gm__ int8_t*)w);
    xsg.SetGlobalBuffer((__gm__ float*)xs);wsg.SetGlobalBuffer((__gm__ ScaleT*)ws);
    yg.SetGlobalBuffer((__gm__ OutT*)y);glg.SetGlobalBuffer((__gm__ int64_t*)gl);
    const uint32_t block=GetBlockIdx(),cores=GetBlockNum();
    const uint32_t lo=p.m*block/cores,hi=p.m*(block+1)/cores;
    if (lo>=hi) return;
    const uint32_t cap=GvMin(p.k,1024),cap32=GvUp(cap,32),cap64=GvUp(cap,64);
    const uint32_t wk32=GvUp(p.k,32),wk64=GvUp(p.k,64),rr=GvUp(Rows,16);
    uint32_t bytes=Rows*(cap32*3+cap64*4)+wk32*3+wk64*4+Rows*64+rr*18+128+GvUp(p.e*8,32);
    TPipe pipe;TBuf<TPosition::VECCALC> ub;pipe.InitBuffer(ub,GvUp(bytes,32));
    auto raw=ub.Get<uint8_t>();uint32_t at=0;
    auto xi=raw[at].ReinterpretCast<int8_t>();at+=Rows*cap32;
    auto xh=raw[at].ReinterpretCast<half>();at+=Rows*cap32*2;
    auto xf=raw[at].ReinterpretCast<float>();at+=Rows*cap64*4;
    auto wi=raw[at].ReinterpretCast<int8_t>();at+=wk32;
    auto wh=raw[at].ReinterpretCast<half>();at+=wk32*2;
    auto wf=raw[at].ReinterpretCast<float>();at+=wk64*4;
    auto red1=raw[at].ReinterpretCast<float>();at+=Rows*64;
    auto red2=raw[at].ReinterpretCast<float>();at+=rr*4;
    auto acc=raw[at].ReinterpretCast<int32_t>();at+=rr*4;
    auto partial=raw[at].ReinterpretCast<int32_t>();at+=rr*4;
    auto xsl=raw[at].ReinterpretCast<float>();at+=rr*4;
    auto out=raw[at].ReinterpretCast<OutT>();at+=rr*2;
    auto wsl=raw[at].ReinterpretCast<ScaleT>();at+=64;
    auto wsf=raw[at].ReinterpretCast<float>();at+=64;
    auto gll=raw[at].ReinterpretCast<int64_t>();
    DataCopyPad(gll,glg,DataCopyExtParams{1,p.e*8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
    GvBarrier<HardEvent::MTE2_S>();
    uint32_t start=0;
    for (uint32_t e=0;e<p.e;++e) {
        const uint32_t value=uint32_t(gll.GetValue(e));
        const uint32_t end=p.listType==0?value:start+value;
        uint32_t row=start>lo?start:lo,stop=end<hi?end:hi;start=end;
        if (row>=stop) continue;
        GvBarrier<HardEvent::V_MTE2>();
        DataCopyPad(wi,wg[uint64_t(e)*p.k],DataCopyExtParams{1,p.k,0,0,0},
                    DataCopyPadExtParams<int8_t>{true,0,uint8_t(wk32-p.k),0});
        DataCopyPad(wsl,wsg[e],DataCopyExtParams{1,uint32_t(sizeof(ScaleT)),0,0,0},
                    DataCopyPadExtParams<ScaleT>{false,0,0,0});
        GvBarrier<HardEvent::MTE2_V>();
        Cast(wh,wi,RoundMode::CAST_NONE,wk32);
        if constexpr (sizeof(ScaleT)==2) Cast(wsf,wsl,RoundMode::CAST_NONE,16);
        Duplicate(wf,0.0f,wk64);PipeBarrier<PIPE_V>();
        Cast(wf,wh,RoundMode::CAST_NONE,wk32);GvBarrier<HardEvent::V_S>();
        float scale;
        if constexpr (sizeof(ScaleT)==4) scale=wsl.GetValue(0);
        else scale=wsf.GetValue(0);
        GvBarrier<HardEvent::S_V>();
        for (;row<stop;row+=Rows) {
            const uint32_t have=GvMin(Rows,stop-row);
            GvBarrier<HardEvent::V_MTE2>();
            DataCopyPad(xsl,xsg[row],DataCopyExtParams{1,have*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            if (p.k>1024) Duplicate(acc,int32_t(0),GvUp(have,8));
            for (uint32_t k0=0;k0<p.k;k0+=cap) {
                uint32_t len=GvMin(cap,p.k-k0),kp=GvUp(len,32),fp=GvUp(len,64);
                // Mandatory input-buffer reuse fence: MTE3_V alone cannot stop
                // next-batch MTE2 from overwriting scales/input before V reads it.
                GvBarrier<HardEvent::V_MTE2>();
                DataCopyPad(xi,xg[uint64_t(row)*p.k+k0],
                            DataCopyExtParams{uint16_t(have),len,p.k-len,0,0},
                            DataCopyPadExtParams<int8_t>{true,0,uint8_t(kp-len),0});
                GvBarrier<HardEvent::MTE2_V>();
                Cast(xh,xi,RoundMode::CAST_NONE,have*kp);
                Duplicate(xf,0.0f,have*fp);Duplicate(red1,0.0f,have*16);PipeBarrier<PIPE_V>();
                for (uint32_t r=0;r<have;++r) Cast(xf[r*fp],xh[r*kp],RoundMode::CAST_NONE,kp);
                PipeBarrier<PIPE_V>();GvMulRows(xf,wf[k0],have,fp);PipeBarrier<PIPE_V>();
                for (uint32_t r=0;r<have;++r) WholeReduceSum(red1[r*16],xf[r*fp],64,fp/64,1,1,8);
                PipeBarrier<PIPE_V>();WholeReduceSum(red2,red1,16,have,1,1,2);PipeBarrier<PIPE_V>();
                if (p.k>1024) {
                    Cast(partial,red2,RoundMode::CAST_RINT,GvUp(have,8));PipeBarrier<PIPE_V>();
                    Add(acc,acc,partial,GvUp(have,8));PipeBarrier<PIPE_V>();
                }
            }
            // A complete <=1024-term reduction is already exact in red2.
            if (p.k>1024) { Cast(red2,acc,RoundMode::CAST_RINT,GvUp(have,8));PipeBarrier<PIPE_V>(); }
            Muls(red2,red2,scale,GvUp(have,8));PipeBarrier<PIPE_V>();
            Mul(red2,red2,xsl,GvUp(have,8));PipeBarrier<PIPE_V>();
            Cast(out,red2,RoundMode::CAST_RINT,GvUp(have,16));GvBarrier<HardEvent::V_MTE3>();
            DataCopyPad(yg[row],out,DataCopyExtParams{1,have*2,0,0,0});GvBarrier<HardEvent::MTE3_V>();
        }
    }
}

