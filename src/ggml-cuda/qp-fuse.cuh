#include "common.cuh"

// Fusions for DiT blocks, both bit-identical to the unfused kernels:
//   NORM -> MUL(row)           layer norm without affine followed by a broadcast row scale (adaLN modulation)
//   MUL(row) -> ADD(residual)  gated residual x + h * g
void ggml_cuda_op_norm_mul_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * row, ggml_tensor * dst);
void ggml_cuda_op_mul_add_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * h, const ggml_tensor * row,
                                const ggml_tensor * residual, ggml_tensor * dst);
