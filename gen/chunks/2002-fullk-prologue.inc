// Geometry-specialized full-K Cube engine. Logical M/K/N/E and expert counts
// always come from Plan and groupList. Host must satisfy resources.json guards.
// The only compile-time choices are tiling, traversal, and buffer strategy.

using AscendC::TPipe; using AscendC::TBuf; using AscendC::TPosition;
using AscendC::LocalTensor; using AscendC::InitSocState; using AscendC::PipeBarrier;
// Default aliases are the original public SDK types. Only bounded P2 selects static lifecycle.
template<bool Static> struct FullKPipeType { using Type = TPipe; };
template<> struct FullKPipeType<true> {
    struct Type {
        __aicore__ inline Type() { InitSocState(); }
        __aicore__ inline void Destroy() { PipeBarrier<PIPE_ALL>(); }
    };
};
template<bool Static, TPosition Pos> struct FullKBufferType { using Type = TBuf<Pos>; };
template<TPosition Pos> struct FullKBufferType<true, Pos> {
    struct Type {
        uint32_t bytes;
        template<typename T> __aicore__ inline LocalTensor<T> Get() {
            return LocalTensor<T>(Pos, 0, bytes / sizeof(T));
        }
    };
};
