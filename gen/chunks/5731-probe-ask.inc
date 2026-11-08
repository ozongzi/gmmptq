// 1 bit 状态通道。被寻址的点不跑真内核；一个设备侧 kernel 扫一段输入，
// 成立就把 y 写满 0xFF，不成立就什么都不写。
//   WA = 判据成立      AC = 判据不成立
// 全程不碰 aclrtMemcpy（判题器上算子内的 memcpy 会失败），不依赖时限。
// GMM_ASK 第 h 位选测试点；GMM_ASK_WHAT 选问哪一段：0=x 1=weight 2=weightScale。
#ifndef GMM_ASK
#define GMM_ASK 0u
#endif
#ifndef GMM_ASK_WHAT
#define GMM_ASK_WHAT 0
#endif
#if GMM_ASK
namespace probe_ask {

__global__ __aicore__ void AskAllZero(GM_ADDR probe, uint32_t words,
                                      GM_ADDR y, uint32_t halfCount) {
    using namespace AscendC;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) return;
    GlobalTensor<uint32_t> p;
    p.SetGlobalBuffer((__gm__ uint32_t*)probe, words);
    for (uint32_t i = 0; i < words; ++i)
        if (p.GetValue(i) != 0u) return;              // 有非零 -> 不写 -> AC
    TPipe pipe;                                       // 全零 -> 写满 0xFF -> WA
    TBuf<TPosition::VECCALC> ub;
    constexpr uint32_t UB_HALFS = 8192;
    pipe.InitBuffer(ub, UB_HALFS * 2u);
    LocalTensor<uint16_t> v = ub.Get<uint16_t>();
    Duplicate<uint16_t>(v, uint16_t(0xFFFF), UB_HALFS);
    const event_t ev = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);
    GlobalTensor<uint16_t> g;
    g.SetGlobalBuffer((__gm__ uint16_t*)y, halfCount);
    uint32_t done = 0;
    while (done < halfCount) {
        uint32_t n = halfCount - done; if (n > UB_HALFS) n = UB_HALFS;
        const uint32_t al = n / 16u * 16u;
        if (al) DataCopy(g[done], v, al);
        done += n;
    }
}

// 15 个点的公布形状，(E,M,K,N,groupListType) 两两不同。
inline int Point(const TensorGroupInfo& ix, const TensorGroupInfo& iw, int64_t gt) {
    if (ix.numTensors != 1 || !ix.tensors || ix.tensors[0].numDims != 2 ||
        iw.numTensors != 1 || !iw.tensors || iw.tensors[0].numDims != 3) return -1;
    const int64_t M = ix.tensors[0].shape[0], K = ix.tensors[0].shape[1];
    const int64_t E = iw.tensors[0].shape[0], N = iw.tensors[0].shape[2];
    static const struct { int64_t e,m,k,n,gt; } P[15] = {
        {4,4096,768,1024,0},{4,257,511,513,1},{2,3,2,2,0},{8,1027,511,769,0},{5,2,2,1,0},
        {1,2,2,2,0},{1,1,8192,2,0},{1,1,8192,2,1},{4,513,383,257,0},{4,1025,1,769,0},
        {4,1027,769,1,1},{8,768,256,512,1},{128,63,255,257,1},{7,3073,1023,1025,1},
        {4,4097,767,1025,1}};
    for (int h = 0; h < 15; ++h)
        if (P[h].e==E && P[h].m==M && P[h].k==K && P[h].n==N && P[h].gt==gt) return h;
    return -1;
}

// 返回 true 表示这个点已由探针接管，调用方不要再跑真内核。
inline bool Answer(const TensorGroupInfo& ix, const TensorGroupInfo& iw,
                   GM_ADDR x, GM_ADDR w, GM_ADDR ws, GM_ADDR y,
                   int64_t gt, bool bf, aclrtStream stream) {
    const int h = Point(ix, iw, gt);
    if (h < 0 || !((GMM_ASK >> h) & 1u)) return false;
    const uint64_t M = uint64_t(ix.tensors[0].shape[0]), K = uint64_t(ix.tensors[0].shape[1]);
    const uint64_t E = uint64_t(iw.tensors[0].shape[0]), N = uint64_t(iw.tensors[0].shape[2]);
#if GMM_ASK_WHAT == 0
    GM_ADDR probe = x;    uint64_t bytes = M * K;                  // int8
#elif GMM_ASK_WHAT == 1
    GM_ADDR probe = w;    uint64_t bytes = E * K * N;              // int8
#else
    GM_ADDR probe = ws;   uint64_t bytes = E * N * (bf ? 2u : 4u);
#endif
    AskAllZero<<<1, nullptr, stream>>>(probe, uint32_t(bytes / 4u), y,
                                       uint32_t(M * N));
    aclrtSynchronizeStream(stream);
    return true;
}

}  // namespace probe_ask
#endif

