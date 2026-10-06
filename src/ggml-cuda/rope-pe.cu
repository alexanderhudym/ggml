#include "rope-pe.cuh"

// GGML_OP_RMS_NORM_ROPE_PE: rms_norm(x) * w, rotation by a precomputed [[cos, -sin], [sin, cos]] table on adjacent pairs,
// heads moved in front of the tokens. One warp per row of D = 32*G values (lane l holds columns 32g + l).
//
// Bit-identical to the unfused graph (rms_norm_f32<256, do_multiply> + sd.cpp's Rope::apply_rope built from
// cont/repeat/mul/add): the sum of squares follows rms_norm_f32's block_reduce with 256 threads (each warp's xor
// butterfly over its 32 columns, then a butterfly over the 8 warp sums with the missing warps as 0.0f), and the
// rotation is (x0 * pe0) + (x1 * pe1) without FMA contraction, as the separate mul and add kernels compute it.
template <int G>
static __global__ void rms_norm_rope_pe_f32(const float * __restrict__ x, const float * __restrict__ w,
                                            const float * __restrict__ pe, float * __restrict__ dst,
                                            const int64_t nrows, const int heads, const int tokens, const float eps) {
    constexpr int D = 32 * G;
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    // input row = h + heads*(t + tokens*n), output row = t + tokens*(h + heads*n)
    const int64_t h  = row % heads;
    const int64_t tn = row / heads;
    const int64_t t  = tn % tokens;
    const int64_t n  = tn / tokens;

    x += row * D;
    float v[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        v[g] = x[g * WARP_SIZE + lane];
    }

    float s[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        s[g] = 0.0f;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        s[g] = warp_reduce_sum(__fmul_rn(v[g], v[g]));
    }
    // butterfly over lanes 0..7 of the second block_reduce stage (xor 16 and 8 only add zeros)
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(s[0], s[4]), __fadd_rn(s[2], s[6])),
                                __fadd_rn(__fadd_rn(s[1], s[5]), __fadd_rn(s[3], s[7])));
    const float mean  = sum / D;
    const float scale = rsqrtf(mean + eps);

    const float * pe_t = pe + t * (2 * D);
    float * out = dst + (t + (int64_t) tokens * (h + (int64_t) heads * n)) * D;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int col = g * WARP_SIZE + lane;
        float y = __fmul_rn(scale, v[g]);
        if (w != nullptr) {
            y = __fmul_rn(y, w[col]);
        }
        const float other = __shfl_xor_sync(0xffffffff, y, 1);
        const float y0    = (col & 1) ? other : y;
        const float y1    = (col & 1) ? y : other;
        // pe index for (in k, out o, pair i, token t) = k + 2*o + 4*i + 2*D*t, and 2*o + 4*i = 2*col
        const float2 p = *(const float2 *) (pe_t + 2 * col);
        out[col] = __fadd_rn(__fmul_rn(y0, p.x), __fmul_rn(y1, p.y));
    }
}

bool ggml_cuda_rms_norm_rope_pe_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const int64_t D = x->ne[0];
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && D % 32 == 0 && D / 32 >= 1 && D / 32 <= 8 &&
           ggml_is_contiguous(x) && ggml_is_contiguous(op->src[1]) &&
           (op->src[2] == nullptr || ggml_is_contiguous(op->src[2]));
}

void ggml_cuda_op_rms_norm_rope_pe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * pe = dst->src[1];
    const ggml_tensor * w  = dst->src[2];
    const float eps        = ggml_get_op_params_f32(dst, 0);
    const int64_t D        = x->ne[0];
    const int heads        = (int) x->ne[1];
    const int tokens       = (int) x->ne[2];
    const int64_t nrows    = ggml_nrows(x);
    GGML_ASSERT(ggml_cuda_rms_norm_rope_pe_supported(dst));

    constexpr int rows_per_block = 4;
    const dim3 grid((unsigned) ((nrows + rows_per_block - 1) / rows_per_block));
    const dim3 block(rows_per_block * WARP_SIZE);
    cudaStream_t stream = ctx.stream();
    const float * xd = (const float *) x->data;
    const float * wd = w ? (const float *) w->data : nullptr;
    const float * pd = (const float *) pe->data;
    float * dd       = (float *) dst->data;
    switch (D / 32) {
        case 1: rms_norm_rope_pe_f32<1><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 2: rms_norm_rope_pe_f32<2><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 3: rms_norm_rope_pe_f32<3><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 4: rms_norm_rope_pe_f32<4><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 5: rms_norm_rope_pe_f32<5><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 6: rms_norm_rope_pe_f32<6><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 7: rms_norm_rope_pe_f32<7><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        case 8: rms_norm_rope_pe_f32<8><<<grid, block, 0, stream>>>(xd, wd, pd, dd, nrows, heads, tokens, eps); break;
        default: GGML_ABORT("unsupported head dim");
    }
    CUDA_CHECK(cudaGetLastError());
}
