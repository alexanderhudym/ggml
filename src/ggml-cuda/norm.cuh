#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GROUP_NORM(gn) [* w] [+ b] [-> silu] written to dst; returns false (nothing launched) if unsupported
bool ggml_cuda_op_group_norm_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * gn, const ggml_tensor * w,
                                   const ggml_tensor * b, bool silu, ggml_tensor * dst);

// the same, but written as the NHWC f16 input of a following tensor-core conv (out: ne0*ne1*ne2*ne3 halves) instead of
// dst (only its address/shape are used, to pick the identical kernels); false = unsupported, nothing launched
bool ggml_cuda_op_group_norm_fused_nhwc(ggml_backend_cuda_context & ctx, const ggml_tensor * gn, const ggml_tensor * w,
                                        const ggml_tensor * b, bool silu, const ggml_tensor * dst, half * out);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
