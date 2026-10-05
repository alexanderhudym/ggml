#pragma once
#include "common.cuh"

#define CUDA_CONV2D_BLOCK_SIZE 256
void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// fuses CONV_2D -> [RESHAPE] -> ADD(bias) [-> ADD(residual)] starting at node i; returns the number of extra nodes
// consumed (0 = not fused).
int ggml_cuda_try_fuse_conv2d_bias(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
                                   const half * x_nhwc = nullptr);

// conv would stage its input as NHWC f16 (tensor-core implicit GEMM path)
bool ggml_cuda_conv2d_accepts_nhwc(const ggml_backend_cuda_context & ctx, const ggml_tensor * conv);
// runs CONV_2D node i (+ the bias/residual fusion) on an input the producer already staged; returns extra nodes used
int ggml_cuda_conv2d_run_nhwc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const half * x_nhwc);
// UPSCALE(nearest 2x) / PAD(zeros) -> CONV_2D starting at node i; returns the extra nodes consumed (0 = not fused)
int ggml_cuda_try_fuse_stage_conv2d(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
