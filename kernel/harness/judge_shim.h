// The two structs the judge's harness hands run_kernel, reconstructed from how
// our submission consumes them. The judge does not ship these to us, so a
// local build has to supply an equivalent.
//
// The dtype codes are read back out of the submission's own Tensor() checks:
// x and weight are 3, xScale is 0, weightScale is 0 or 2, groupList is 6, and
// y is 1 or 2. So 0=float32, 1=float16, 2=bfloat16, 3=int8, 6=int64. These
// only have to agree with the submission, since the submission is the only
// thing that validates them.
#pragma once
#include <cstdint>

enum JudgeDtype {
    JUDGE_FLOAT32 = 0,
    JUDGE_FLOAT16 = 1,
    JUDGE_BFLOAT16 = 2,
    JUDGE_INT8 = 3,
    JUDGE_INT64 = 6,
};

struct TensorInfo {
    int32_t numDims;
    const int64_t* shape;
    int32_t dtype;
};

struct TensorGroupInfo {
    int32_t numTensors;
    const TensorInfo* tensors;
};
