#pragma once
#include "common.cuh"
#include "conv2d-implicit.cuh"

// Tensor-core implicit GEMM with f32 accumulation. X_H: NHWC f16 input, K_H: [K][RS][C] f16 filter. Optional epilogue:
// Y = (conv + bias[k]) + residual (residual may alias Y).
void ggml_cuda_conv2d_implicit_acc32(ggml_backend_cuda_context & ctx, const half * X_H, const half * K_H, float * Y_D,
                                     const param_t & P, cudaStream_t st, const float * bias = nullptr,
                                     const float * residual = nullptr);
