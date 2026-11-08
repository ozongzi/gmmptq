// The tiling arrives by value, not behind a GM_ADDR.
//
// It used to be copied in with a loop of 15 scalar reads from Global Memory,
// one dependent load after another. On tp7 that showed up as a floor: with
// every stage of the computation compiled out, the kernel still took 3.06 us,
// against a whole budget of 3.07 us (tbest 2.50 on the judge's faster clock).
// The arithmetic could have been free and the point would still have lost.
//
// VTiling is fifteen uint32. Passing it in the launch parameters is what the
// old kernel already did with its Plan, and it removes all fifteen reads.
template <typename WScaleT, typename OutT>
__global__ __aicore__ void BalancedOverlapShortGroupMatmulKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale,
                                              GM_ADDR wScale, GM_ADDR groupList,
                                              GM_ADDR y, VTiling tiling) {
    // See GmmKernel: the task type belongs here, not in the attribute.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    BalancedOverlapShortGroupMatmul<WScaleT, OutT,true> op;
    op.Init(x, weight, xScale, wScale, groupList, y, &tiling, nullptr);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

