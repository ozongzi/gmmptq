extern "C" void run_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR weight, const TensorGroupInfo& info_weight,
    GM_ADDR xScale, const TensorGroupInfo& info_xScale,
    GM_ADDR weightScale, const TensorGroupInfo& info_weightScale,
    GM_ADDR groupList, const TensorGroupInfo& info_groupList,
    GM_ADDR y, const TensorGroupInfo& info_y,
    int64_t availableCoreNum, aclrtStream stream,
    int64_t groupListType, const char* outputDtype) {
    using namespace gmmptq;
    if (!outputDtype ||
        (std::strcmp(outputDtype, "float16") && std::strcmp(outputDtype, "bfloat16")) ||
        (groupListType != 0 && groupListType != 1) || availableCoreNum <= 0)
        throw std::invalid_argument("GroupedMatmulPerTokenQuant: invalid attributes/core count");
    bool bf = std::strcmp(outputDtype, "bfloat16") == 0;
#if GMM_ASK
    if (probe_ask::Answer(info_x, info_weight, x, weight, weightScale, y,
                          groupListType, bf, stream)) return;
#endif
    const auto& a = Tensor(info_x, 2, 3);
    const auto& b = Tensor(info_weight, 3, 3);
    const auto& xs = Tensor(info_xScale, 1, 0);
    const auto& ws = Tensor(info_weightScale, 2, bf ? 2 : 0);
    const auto& gl = Tensor(info_groupList, 1, 6);
    const auto& out = Tensor(info_y, 2, bf ? 2 : 1);
    int64_t M = a.shape[0], K = a.shape[1], E = b.shape[0], N = b.shape[2];
    if (M < 1 || M > 32768 || K < 1 || K > 8192 || E < 1 || E > 128 || N < 1 || N > 8192 ||
        b.shape[1] != K || xs.shape[0] != M || ws.shape[0] != E || ws.shape[1] != N ||
        gl.shape[0] != E || out.shape[0] != M || out.shape[1] != N)
        throw std::invalid_argument("GroupedMatmulPerTokenQuant: shape mismatch");
    // Non-natural typed input views use existing bounded all-DMA staging.
    if(K<=8 && E<=16 && uint64_t(M)*uint64_t(N)<=64u &&
       (((reinterpret_cast<uintptr_t>(xScale)&3u)!=0u) ||
        ((reinterpret_cast<uintptr_t>(weightScale)&(bf?1u:3u))!=0u) ||
        ((reinterpret_cast<uintptr_t>(groupList)&7u)!=0u))) {
        const uint32_t dims=uint32_t((M-1)|((N-1)<<6)|((K-1)<<12)|((E-1)<<15)|(groupListType<<19));
        if(bf)TinyShortReduction<true><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,dims);
        else TinyShortReduction<false><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,dims);
        return;
    }
    const bool scalarOutputWindow=((reinterpret_cast<uintptr_t>(y)&127u)+uint64_t(M)*uint64_t(N)*2u)<=32u;
    // Register-capacity algorithms consume runtime tensors and group metadata.
    // Fixed 2x2 register microtile; up to four output rows fit two uint64 words
    // per expert. Complete original GT0 expert-parallel resource domain.
    if(!bf && groupListType==0 && E>=2 && E<=M && M<=4 &&
       uint64_t(E)<=2*uint64_t(availableCoreNum) && K==2 && N==2 &&
       scalarOutputWindow &&
       ((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(weight)|
         reinterpret_cast<uintptr_t>(xScale)|reinterpret_cast<uintptr_t>(weightScale)|
         reinterpret_cast<uintptr_t>(groupList)|reinterpret_cast<uintptr_t>(y))&63u)==0u){
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|
                                      ((E-1)<<10)|(groupListType<<17));
        expert_output::SearchTinyExpertOutput6400000<float,half><<<uint32_t(E),nullptr,stream>>>(
            x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    // Single register-capacity algorithm, every dimension and grouping stays runtime.
    if(uint64_t(M)*K<=8u && uint64_t(K)*N<=8u && uint64_t(M)*4u<=16u &&
       uint64_t(N)*(bf?2u:4u)<=8u && uint64_t(M)*N*2u<=16u &&
       scalarOutputWindow && (reinterpret_cast<uintptr_t>(y)&1u)==0u){
        const uint32_t dims=uint32_t((M-1)|((K-1)<<2)|((N-1)<<5)|((E-1)<<7)|(groupListType<<14));
        // Complete resource family; only naturally aligned single-line GL batches use independent owner reads.
        if(E>=2u && E<=8u && (reinterpret_cast<uintptr_t>(groupList)&7u)==0u &&
           (reinterpret_cast<uintptr_t>(groupList)&63u)+uint64_t(E)*8u<=64u){
            if(bf)tiny_capacity_contract::Select<true>(x,weight,xScale,weightScale,groupList,y,dims,stream);
            else tiny_capacity_contract::Select<false>(x,weight,xScale,weightScale,groupList,y,dims,stream);
            return;
        }
        if(bf)tiny_capacity_unrolled::Select<true>(x,weight,xScale,weightScale,groupList,y,dims,stream);
        else tiny_capacity_unrolled::Select<false>(x,weight,xScale,weightScale,groupList,y,dims,stream);
        return;
    }
    // A single row with two interleaved channels; all true K lengths in the
    // capacity domain, both dtypes/GT, runtime inputs and scales remain dynamic.
    if(E==1 && M==1 && N==2 && K>=1024 && K<=8192){
        const bool scalar=(uint64_t(y)&31u)==0u && scalarOutputWindow;
        if(K<=1024){
            if(bf)longdot_capacity::Launch<1024,true>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            else longdot_capacity::Launch<1024,false>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            return;
        }
        if(K<=2048){
            if(bf)longdot_capacity::Launch<2048,true>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            else longdot_capacity::Launch<2048,false>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            return;
        }
        if(K<=4096){
            if(bf)longdot_capacity::Launch<4096,true>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            else longdot_capacity::Launch<4096,false>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            return;
        }
        if(K<=8192){
            if(bf)longdot_capacity::Launch<8192,true>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            else longdot_capacity::Launch<8192,false>(x,weight,xScale,weightScale,y,uint32_t(K),scalar,stream);
            return;
        }
    }
    GeneralVectorDims vp{uint32_t(M),uint32_t(K),uint32_t(N),uint32_t(E),uint32_t(groupListType)};
    // Every participating tensor fits one aligned scalar-cache line. The
    // predicate uses only metadata and addresses, never host tensor contents.
    const bool tinyCacheAligned=((reinterpret_cast<uintptr_t>(x)|
        reinterpret_cast<uintptr_t>(weight)|reinterpret_cast<uintptr_t>(xScale)|
        reinterpret_cast<uintptr_t>(weightScale)|reinterpret_cast<uintptr_t>(groupList)|
        reinterpret_cast<uintptr_t>(y))&63u)==0;
    // E1 validgroupmetadata implies owner0; retain the boundedcompleteinputfamily.
    if(scalarOutputWindow && E==1 && M<=4 && uint64_t(M)<=2*uint64_t(availableCoreNum) && N<=32 && K<=8 &&
       uint64_t(M)*uint64_t(N)<=32u && uint64_t(M)*uint64_t(K)<=64u &&
       uint64_t(K)*uint64_t(N)<=64u && uint64_t(M)*4u<=64u &&
       uint64_t(N)*(bf?2u:4u)<=64u && N<=2 && (reinterpret_cast<uintptr_t>(y)&uint64_t(N*2-1))==0) {
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|(groupListType<<17));
        if(bf)SearchTinyE1STDev3400000<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        else SearchTinyE1STDev3400000<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    if(E==1 && M<=4 && uint64_t(M)<=2*uint64_t(availableCoreNum) && N<=32 && K<=8 &&
       uint64_t(M)*uint64_t(N)<=32u && uint64_t(M)*uint64_t(K)<=64u &&
       uint64_t(K)*uint64_t(N)<=64u && uint64_t(M)*4u<=64u &&
       uint64_t(N)*(bf?2u:4u)<=64u) {
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|(groupListType<<17));
        if(bf)E1FixedUbRowTiny<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        else E1FixedUbRowTiny<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    if (E==1 && K<=8 && uint64_t(M)*uint64_t(N)<=32 &&
        uint64_t(M)*uint64_t(K)<=64 && uint64_t(E)*uint64_t(K)*uint64_t(N)<=64 &&
        uint64_t(M)*4<=64 && uint64_t(E)*uint64_t(N)*(bf?2u:4u)<=64 && tinyCacheAligned) {
        const uint32_t tinyCacheDims=uint32_t((M-1)|((N-1)<<4)|((K-1)<<9)|
                                             ((E-1)<<12)|(groupListType<<15));
        if (bf) TinyCacheReduction<true><<<1,nullptr,stream>>>(
            x,weight,xScale,weightScale,groupList,y,tinyCacheDims);
        else TinyCacheReduction<false><<<1,nullptr,stream>>>(
            x,weight,xScale,weightScale,groupList,y,tinyCacheDims);
        return;
    }
    // GT0 expert intervals; last cumulative endpoint follows from completeMcoverage.
    if(groupListType==0 && E>=2 && E<=M && M<=4 && uint64_t(E)<=2*uint64_t(availableCoreNum) && K<=8 && N<=32 && uint64_t(M)*uint64_t(N)<=32) {
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));
        if(K==2 && N==2){
            if(!bf && tinyCacheAligned && scalarOutputWindow){
                expert_output::SearchTinyExpertOutput6400000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
                return;
            }
            if(bf)expert_reuse::SearchTinyExpertReuse6300000<bfloat16_t,bfloat16_t><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
            else expert_reuse::SearchTinyExpertReuse6300000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
            return;
        }
        if(bf)SearchTinyExpert3500000<bfloat16_t,bfloat16_t><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        else SearchTinyExpert3500000<float,half><<<uint32_t(E),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    // One vector core per short output row; all dimensions remain inputs.
    if (M<=4 && uint64_t(M)<=2*uint64_t(availableCoreNum) && N<=32 && K<=8) {
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));
        if(bf && N==1 && (reinterpret_cast<uintptr_t>(y)&1u)==0 && scalarOutputWindow){
            tiny_row_stdev::SearchTinyRowSTDev3400000<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
            return;
        }
        if (bf) RowParallelTiny<bfloat16_t,bfloat16_t><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        else RowParallelTiny<float,half><<<uint32_t(M),nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    // Supplement the same short-row resource family when the core budget is lower.
    if (M<=4 && uint64_t(M)>2*uint64_t(availableCoreNum) && N<=32 && K<=8) {
        const uint32_t blocks=uint32_t(std::min<uint64_t>(uint64_t(M),2*uint64_t(availableCoreNum)));
        const uint32_t rowDims=uint32_t((M-1)|((N-1)<<2)|((K-1)<<7)|((E-1)<<10)|(groupListType<<17));
        if (bf) BudgetStridedRowTiny<bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        else BudgetStridedRowTiny<float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,rowDims);
        return;
    }
    if (uint64_t(M)*uint64_t(N)<=64 && K<=8 && E<=16) {
        if (bf) TinyShortReduction<true><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,((vp.m-1u)|((vp.n-1u)<<6)|((vp.k-1u)<<12)|((vp.e-1u)<<15)|(vp.listType<<19)));
        else TinyShortReduction<false><<<1,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,((vp.m-1u)|((vp.n-1u)<<6)|((vp.k-1u)<<12)|((vp.e-1u)<<15)|(vp.listType<<19)));
        return;
    }
    if (M==1 && N<=2) {
        const uint32_t blocks=uint32_t(std::min<int64_t>(2*availableCoreNum,N));
        if (bf) IntegerProductLongNarrowDot<bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);
        else IntegerProductLongNarrowDot<float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);
        return;
    }
    if (N==1) {
        const uint32_t blocks=uint32_t(std::min<int64_t>(2*availableCoreNum,M));
        // Compact 800-column staging permits a 32-row batch within UB.
        if (K<=800) {
            if (bf) SearchN1FixedLayoutSafe7100000<bfloat16_t,bfloat16_t,32,800><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp,blocks);
            else SearchN1FixedLayoutSafe7100000<float,half,32,800><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp,blocks);
            return;
        }
        if (bf) RowDot<16,bfloat16_t,bfloat16_t><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);
        else RowDot<16,float,half><<<blocks,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vp);
        return;
    }
    Plan p{};
    p.m=uint32_t(M);p.k=uint32_t(K);p.n=uint32_t(N);p.e=uint32_t(E);
    p.listType=uint32_t(groupListType);p.kp=UpH(p.k,32);p.wstride=p.n;
    p.cores=uint32_t(availableCoreNum);p.vcores=p.cores*2;
    if (p.k==1) {
        // RR32 single output: runtime UB and stride guard, unchanged Mode1 arithmetic.
        const uint32_t selectedPitch=UpH(p.n,16u),selectedNUp=UpH(p.n,32u);
        const uint64_t selectedUb=uint64_t(32u)*selectedPitch*6u+
            12u*selectedNUp+80u*32u+selectedNUp+32u+32u+32u;
        if(!bf && p.m>=256u && p.n>=512u && p.n<=1024u &&
           selectedPitch/8u<=255u && selectedUb<=188160u) {
            VTiling selected{p.m,p.k,p.n,p.e,p.listType,
                std::min(p.vcores,(p.m+31u)/32u+p.e),32u,selectedPitch};
            selected.coreNum=std::min(selected.coreNum,16u);
        p10_fixed_ub::SearchP10FixedUb7300000<<<selected.coreNum,nullptr,stream>>>(
                x,weight,xScale,weightScale,groupList,y,selected);
            return;
        }
        // FP16 K1 resource family: full-column UB <=179840 bytes.
        if(!bf && p.m>=256u && p.n>=512u && p.n<=1024u) {
            VTiling selected{p.m,p.k,p.n,p.e,p.listType,
                std::min(p.vcores,(p.m+19u)/20u+p.e),20u,UpH(p.n,16u)};
            PublishedOuter<<<selected.coreNum,nullptr,stream>>>(
                x,weight,xScale,weightScale,groupList,y,selected);
            return;
        }
        VTiling vt{};BuildOuterTiling(p,vt);
        if (bf) GmmOuterKernel<bfloat16_t,bfloat16_t><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);
        else GmmOuterKernel<float,half><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);
        return;
    }
    // Many short experts: do not pay Cube padding and cross-core handshakes.
    if (p.e>=32 && p.m<=256 && p.m<=2*p.e && p.k<=512 && p.n<=512) {
        VTiling vt=BuildBalancedOverlapShortGroup(p);
        if (bf) BalancedOverlapShortGroupMatmulKernel<bfloat16_t,bfloat16_t><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);
        else BalancedOverlapShortGroupMatmulKernel<float,half><<<vt.coreNum,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,vt);
        return;
    }
    // Medium short experts: original FullK body, measured on recovered groups.
    // Group boundaries and all numerical tensors remain runtime inputs.
    if(!bf && p.cores<=20u && p.e>1u && p.e<=16u && p.m/p.e>=96u &&
       p.m/p.e<256u && p.k>256u && p.k<=512u && p.n>640u && p.n<=1024u &&
       (p.k%128u || p.n%128u)) {
        Plan selected=p;
        if(ConfigureCubeFullK<192,128,96,128,true,false>(selected)) {
            LaunchMediumScalarHalf(x,weight,xScale,weightScale,groupList,y,selected,stream);return;
        }
    }
    // Confirmed aligned-medium int32 backend; actual group boundaries stay on device.
    if(p.cores<=20u && p.e>1u && p.e<=16u && p.m/p.e>=64u && p.m/p.e<256u &&
       p.n>=384u && p.n<=640u && p.n%128u==0u &&
       p.k>=128u && p.k<=512u && p.k%128u==0u) {
        Plan selected=p;
        if(ConfigureCubeFullK<64,128,128,64,true,false>(selected)) {
            LaunchAlignedMedium64(x,weight,xScale,weightScale,groupList,y,selected,bf,stream);return;
        }
    }
    // Preserve original packed-family metadata predicates; allocate only when
    // one of the retained active pack routes will actually launch.
    const bool packWeights=p.m/p.e>=256u && p.n>=512u && p.n%32u!=0u && p.k>=128u &&
        uint64_t(p.m/p.e>=512u?464u:480u)*p.kp<=L1_BYTES;
    if(packWeights) {
        Plan packed=p;
        packed.wstride=UpH(p.n,128u);
        if(bf && p.cores<=20u && p.e<=16u && p.m/p.e<512u &&
           p.k>512u && p.k<=1024u && p.n<=1280u &&
           uint64_t(480u)*p.kp<=524032u &&
           ConfigureCubeFullK<224,128,128,64,true,false>(packed)) {
            Scratch packedWeight(uint64_t(p.e)*p.k*packed.wstride);
            LaunchActivePacked224<true>(x,weight,xScale,weightScale,groupList,y,
                (GM_ADDR)packedWeight.p,packed,stream);return;
        }
        if(!bf && ConfigureCubeFullK<224,128,128,64,true,false>(packed) &&
           p.e<=16u && p.cores<=20u && p.k<=1024u && p.n<=1280u &&
           uint64_t(480u)*p.kp<=524032u) {
            Scratch packedWeight(uint64_t(p.e)*p.k*packed.wstride);
            LaunchActivePacked224<false>(x,weight,xScale,weightScale,groupList,y,
                (GM_ADDR)packedWeight.p,packed,stream);return;
        }
    }
    const uint32_t rowsPerExpert=p.m/p.e;
    // Narrow BF16 resource family: one K chunk, runtime expert boundaries.
    if(bf && p.cores<=20u && p.e>1u && p.e<=16u && p.m/p.e>=96u &&
       p.m/p.e<256u && p.n>=192u && p.n<=320u && p.k>256u && p.k<=384u) {
        Plan selected=p;
        if(ConfigureCubeFullK<64,128,384,32,true,true>(selected)) {
            LaunchNarrowSingleKSelected(x,weight,xScale,weightScale,groupList,y,selected,stream);return;
        }
    }
    // Resident expert/N panels, one full-K Mmad and bulk B loader when aligned.
    if (rowsPerExpert<96 && p.n<=640 && p.kp<=512 &&
        ConfigureCubeFullK<96,128,512,48,false,true>(p)) {
        mix11::Launch(x,weight,xScale,weightScale,groupList,y,p,bf,stream);
        return;
    }
    // Aligned FP16 full-K resource family: exact scalar 2^-12 half transport.
    // Host uses metadata only; all expert boundaries and scales remain dynamic.
    if(!bf && availableCoreNum<=20 && rowsPerExpert>=256u && p.n>=512u &&
       p.n%128u==0u && p.k%128u==0u && p.wstride==p.n &&
       uint64_t(256u+2u*128u)*p.kp<=524032u) {
        Plan selected=p;
        if(ConfigureCubeFullK<256,128,128,64,true,false>(selected)) {
            p1_half_pair::Launch(x,weight,xScale,weightScale,groupList,y,selected,stream);return;
        }
    }
    // General legal-metadata fallback: original chunked L1 implementation.
    p.wstride=p.n;
    LeanPlanCube(p);
    const size_t maxTiles=(size_t((p.m+p.m0-1)/p.m0)+p.e)*p.nTiles;
    Scratch scratch(maxTiles*p.m0*p.n0*sizeof(int32_t));
    try {
        if (bf) CubeMix<bfloat16_t,bfloat16_t><<<p.cores,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,(GM_ADDR)scratch.p,p);
        else CubeMix<float,half><<<p.cores,nullptr,stream>>>(x,weight,xScale,weightScale,groupList,y,(GM_ADDR)scratch.p,p);
        Check(aclrtSynchronizeStream(stream));
    } catch (...) { aclrtSynchronizeStream(stream);throw; }
}

