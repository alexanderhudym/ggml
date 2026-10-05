#pragma once
#include "common.cuh"
#include "cp-async.cuh"

constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
constexpr unsigned int SWIZZLE_BITS_1 = 4;
constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
constexpr unsigned int SWIZZLE_BITS_2 = 2;

typedef struct{
    unsigned int      n;                              //batch size
    unsigned int      c;                              //number if channels
    unsigned int      h;                              //height
    unsigned int      w;                              //width
    unsigned int      k;                              //number of filters
    unsigned int      r;                              //filter height
    unsigned int      s;                              //filter width
    unsigned int      u;                              //stride height
    unsigned int      v;                              //stride width
    unsigned int      p;                              //padding height
    unsigned int      q;                              //padding width
    unsigned int      d_h;                            //dilation height
    unsigned int      d_w;                            //dilation width
    unsigned int      Oh;                             //output height
    unsigned int      Ow;                             //output width
    uint3 OW_fastdiv;
    uint3 RS_fastdiv;
    uint3 OHOW_fastdiv;
    int64_t inc_next[3];
    unsigned int inChannelOffset;
    unsigned int weightKOffset;
    unsigned int PQ;
    unsigned int KPQ;
    unsigned int NKPQ;
    unsigned int CHW;
} param_t;


template<const unsigned int K_STRID>
__device__ void clear_mask(unsigned int masks_[][2], bool clear = true) {

#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
        masks_[s][0] = clear ? 0 : masks_[s][0];
        masks_[s][1] = clear ? 0 : masks_[s][1];
    }
}

template<const unsigned int K_STRID>
__device__ void add_byte_offset(int64_t element_offset[], const int64_t offset) {
#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
       element_offset[s] += offset;
    }
}

template<const unsigned int TILE_ROWS,
         const unsigned int TILE_COLS,
         const unsigned int A_K_STRID,
         const unsigned int ROW_STEP>
__device__ void prepareIteratorA(unsigned int thread_row,
                                 unsigned int masks[][2],
                                 int64_t element_offset[],
                                 const param_t param) {
    int offset_n[A_K_STRID];
    int offset_p[A_K_STRID];
    int offset_q[A_K_STRID];

#pragma unroll
    for (int s = 0; s < A_K_STRID; ++s) {

        const unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        offset_n[s]  = fastdiv(gemm_i, param.OHOW_fastdiv);
        unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
        offset_p[s] = fastdiv(npq_res, param.OW_fastdiv); //* param.u - param.p;
        offset_q[s] = fastmodulo(npq_res, param.OW_fastdiv); // * param.v - param.q;
        const int h = offset_p[s] * (int)param.u - (int) param.p;
        const int w = offset_q[s] * (int)param.v - (int) param.q;

        element_offset[s] =  offset_n[s] * (int64_t)param.CHW + h * (int64_t)(param.inChannelOffset) + w * (int64_t)param.c;

        thread_row += ROW_STEP;
    }

    clear_mask<A_K_STRID>(masks);

    for (int r = 0; r < param.r; ++r) {
#pragma unroll
      for (int s_idx = 0; s_idx < A_K_STRID; ++s_idx) {
        const int h = offset_p[s_idx] * param.u - param.p + r * param.d_h;

        bool pred = (offset_n[s_idx] < param.n && h >= 0 && h < param.h);
        masks[s_idx][0] |= (pred << r);
      }
    }

    for (int s = 0; s < param.s; ++s) {
#pragma unroll
      for (int s_idx = 0; s_idx < A_K_STRID; ++s_idx) {
        const int w = offset_q[s_idx] * param.v - param.q + s * param.d_w;
        bool pred = (w >= 0 && w < param.w);
        masks[s_idx][1] |= (pred << s);
      }
    }
}

// 16-byte global -> shared copy that writes zeros when pred_guard is false
__device__ __forceinline__ void cp_async_zfill(void * ptr, const void * global_ptr, bool pred_guard) {
#ifdef CP_ASYNC_AVAILABLE
    unsigned int smem_ptr = ggml_cuda_cvta_generic_to_shared(ptr);
    int src_in_bytes = pred_guard ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_ptr), "l"(global_ptr), "r"(src_in_bytes));
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(global_ptr);
    GGML_UNUSED(pred_guard);
    NO_DEVICE_CODE;
#endif
}

// swizzled shared-memory index of the 16-byte vector iter_idx of a 32-column f16 tile (no bank conflicts in ldmatrix)
__device__ __forceinline__ unsigned int swizzle_vec_index(unsigned int idx) {
    idx = idx ^ ((idx & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
    return idx ^ ((idx & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
}

// First filter tile (r = s = 0, channel block of start_k): TILE_ROWS output channels x 32 input channels.
template<unsigned int TILE_ROWS, unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleB(
    const half * __restrict__ src,
    half * __restrict__ dst,
    const unsigned int curC,
    const int64_t ki,
    const unsigned int end_k,
    unsigned int thread_row,
    const unsigned int thread_col,
    const param_t & param
) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int TILE_COLS_VECTORIZED = 32 / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    float4 * dst_float4 = reinterpret_cast<float4 *>(dst);

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        const unsigned int src_index = thread_row * param.weightKOffset + ki;
        const unsigned int dst_index = swizzle_vec_index(thread_row * TILE_COLS_VECTORIZED + thread_col);
        cp_async_zfill(&dst_float4[dst_index], &src[src_index], thread_row + blockIdx.x * TILE_ROWS < param.k && curC < end_k);
        thread_row += ROW_STEP;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(curC);
    GGML_UNUSED(ki);
    GGML_UNUSED(end_k);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}

// First input tile (r = s = 0): TILE_ROWS output positions x 32 input channels. Returns the channel the thread loads.
template<unsigned int TILE_ROWS, unsigned int NUM_THREADS>
__device__ __forceinline__ unsigned int tileMemcpySwizzleA(
    const half * __restrict__ src,
    half * __restrict__ dst,
    unsigned int masks[][2],
    const int64_t element_offset[],
    unsigned int thread_row,
    const unsigned int thread_col,
    const unsigned int start_k,
    const unsigned int end_k
) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int TILE_COLS_VECTORIZED = 32 / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    float4 * dst_float4 = reinterpret_cast<float4 *>(dst);

    const unsigned int curC = start_k + thread_col * 8;
    clear_mask<NUM_ITERS>(masks, curC >= end_k);

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        const bool valid = (masks[i][0] & 1u) && (masks[i][1] & 1u);
        const unsigned int dst_index = swizzle_vec_index(thread_row * TILE_COLS_VECTORIZED + thread_col);
        cp_async_zfill(&dst_float4[dst_index], &src[element_offset[i] + curC], valid);
        thread_row += ROW_STEP;
    }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(masks);
    GGML_UNUSED(element_offset);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    NO_DEVICE_CODE;
    return 0;
#endif
}

// Input tile of filter position (curR, curS) and channel block block_k.
template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS>
__device__ __forceinline__ unsigned int tileMemcpyAsyncLoadA(
    const half * __restrict__ src,
    half * __restrict__ dst,
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    const unsigned int thread_col,
    unsigned int iter_idx,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int oldC
) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    constexpr unsigned int ITER_STEPS = ROW_STEP * TILE_COLS_VECTORIZED;

    float4 * dst_float4 = reinterpret_cast<float4 *>(dst);

    const unsigned int curC = start_k + block_k + thread_col * 8;
    if (curC > oldC) {
        clear_mask<NUM_ITERS>(masks, curC >= end_k);
    }

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        const bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        cp_async_zfill(&dst_float4[swizzle_vec_index(iter_idx)], &src[element_offset[i] + curC], valid);
        iter_idx += ITER_STEPS;
    }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(masks);
    GGML_UNUSED(element_offset);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(iter_idx);
    GGML_UNUSED(block_k);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(oldC);
    NO_DEVICE_CODE;
    return 0;
#endif
}

// Filter tile of filter position (r, s): src_idx0 is the thread's first source element, krow_idx its output channel.
template<unsigned int TILE_ROWS, unsigned int TILE_COLS, unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyAsyncLoadB(
    const half * src,
    half * dst,
    const unsigned int curC,
    const int64_t ki,
    const unsigned int end_k,
    unsigned int iter_src_idx,
    unsigned int iter_dst_idx,
    unsigned int krow_idx,
    const int ITER_SRC_STEPS,
    const unsigned int k_total
) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    constexpr unsigned int ITER_DST_STEPS = ROW_STEP * TILE_COLS_VECTORIZED;

    float4 * dst_float4 = reinterpret_cast<float4 *>(dst);

    iter_src_idx += ki;

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        cp_async_zfill(&dst_float4[swizzle_vec_index(iter_dst_idx)], &src[iter_src_idx], krow_idx < k_total && curC < end_k);
        iter_src_idx += ITER_SRC_STEPS;
        krow_idx += ROW_STEP;
        iter_dst_idx += ITER_DST_STEPS;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(curC);
    GGML_UNUSED(ki);
    GGML_UNUSED(end_k);
    GGML_UNUSED(iter_src_idx);
    GGML_UNUSED(iter_dst_idx);
    GGML_UNUSED(krow_idx);
    GGML_UNUSED(ITER_SRC_STEPS);
    GGML_UNUSED(k_total);
    NO_DEVICE_CODE;
#endif
}

// out = (conv(dst->src[1], dst->src[0]) + bias[k]) + residual, written to out (default dst->data); bias ([OC] f32)
// and residual (same shape as dst, may alias out) are optional. Takes a f16 kernel of any size up to 32x32 and an input
// with a multiple of 8 channels, on a device and a build with sm_80 tensor cores.
// x_nhwc: optional NHWC f16 input already staged by the producer (then dst->src[1] is not read).
void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * bias,
                                  const float * residual, float * out, const half * x_nhwc);

// f32 NCHW [N, C, Hs, Ws] -> f16 NHWC [N, H, W, C]; mode (nhwc-stage.cuh): 0 plain (Ws = W, Hs = H), 1 nearest 2x
// upscale (W = 2 Ws, H = 2 Hs), 2 zero pad by lp0 left / lp1 top (and whatever is left right/bottom); C % 8 == 0
void ggml_cuda_conv2d_nhwc_stage(const float * src, half * dst, int C, int W, int H, int N, int mode, int Ws, int Hs,
                                 int lp0, int lp1, cudaStream_t st);
