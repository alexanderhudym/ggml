#include "common.cuh"

void ggml_cuda_op_rms_norm_rope_pe(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_rms_norm_rope_pe_supported(const ggml_tensor * op);
