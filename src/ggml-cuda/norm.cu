#include "norm.cuh"
#include "unary.cuh"
#include <algorithm>
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// ---------------------------------------------------------------------------------------------------------------------
// Parallel group norm.
// The stock kernel above runs one block per group (grid = 32 for a VAE), i.e. ~1 block per SM, and reads x three
// times. Here: (1) a stats kernel splits every group into up to GN_MAX_CHUNKS chunks and writes per-chunk shifted
// sums (x - K, (x - K)^2 with K = first element of the group, so E[x^2] - E[x]^2 does not cancel), (2) an apply kernel
// over (channel plane chunk, channel, sample) merges the chunk sums of its group in double in a fixed order, then
// writes ((x - mean) * rstd) [* w[c]] [+ b[c]] [-> silu] with float4 loads/stores. ggml_cuda_try_fuse uses (2) to
// fold the GROUP_NORM -> MUL(gamma) -> ADD(beta) [-> SILU] chain that ggml_ext_group_norm + ggml_silu_inplace produce.
// Groups of <= GN_SMALL_MAX elements (UNet, small latents) instead use one block per group that holds the group in
// registers (one read, exact two-pass mean/variance, one write).
// All kernels are elementwise-safe when x aliases dst (stats only reads; apply reads x[i] and writes dst[i]; K is
// saved by the stats kernel, not re-read by apply).
// ---------------------------------------------------------------------------------------------------------------------

#define GN_BLOCK      256
#define GN_MAX_CHUNKS 64
#define GN_MIN_CHUNK  2048

template <bool vec>
static __global__ void group_norm_stats_f32(const float * __restrict__ x, float2 * __restrict__ partials,
                                            const int64_t group_size, const int64_t chunk, const int nchunks) {
    const int     gi    = blockIdx.y;
    const int     ci    = blockIdx.x;
    const float * xg    = x + (int64_t) gi * group_size;
    const int64_t start = (int64_t) ci * chunk;
    const int64_t end   = min(start + chunk, group_size);

    ggml_cuda_pdl_sync();
    const float K = xg[0];

    float s = 0.0f;
    float q = 0.0f;
    if constexpr (vec) {
        const float4 * x4 = (const float4 *) (xg + start);
        const int      n4 = (int) ((end - start) / 4);
#pragma unroll 4
        for (int i = threadIdx.x; i < n4; i += GN_BLOCK) {
            const float4 v  = x4[i];
            const float  d0 = v.x - K, d1 = v.y - K, d2 = v.z - K, d3 = v.w - K;
            s += (d0 + d1) + (d2 + d3);
            q += (d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3);
        }
    } else {
        for (int64_t i = start + threadIdx.x; i < end; i += GN_BLOCK) {
            const float d = xg[i] - K;
            s += d;
            q += d * d;
        }
    }

    __shared__ float2 s_red[GN_BLOCK / WARP_SIZE];
    const float2 r = block_reduce<block_reduce_method::SUM, GN_BLOCK>(make_float2(s, q), s_red);
    if (threadIdx.x == 0) {
        partials[(int64_t) gi * (nchunks + 1) + ci] = r;
        if (ci == 0) {
            partials[(int64_t) gi * (nchunks + 1) + nchunks] = make_float2(K, 0.0f);
        }
    }
}

// (mean, rstd) of a group from its chunk partials, merged in double in a fixed order; called by one full warp,
// every lane gets the result
static __device__ float2 group_norm_merge(const float2 * __restrict__ p, const int nchunks, const int64_t group_size,
                                          const float eps) {
    double S = 0.0, Q = 0.0;
    for (int k = threadIdx.x; k < nchunks; k += WARP_SIZE) {
        const float2 v = p[k];
        S += v.x;
        Q += v.y;
    }
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        S += __shfl_xor_sync(0xffffffff, S, off, WARP_SIZE);
        Q += __shfl_xor_sync(0xffffffff, Q, off, WARP_SIZE);
    }
    const double m   = S / (double) group_size;
    const double var = fmax(Q / (double) group_size - m * m, 0.0);
    return make_float2((float) ((double) p[nchunks].x + m), rsqrtf((float) var + eps));
}

// chunks of a group for the statistics kernel: elements per chunk (a multiple of 4) and their count
static void group_norm_chunking(const int64_t group_size, int64_t & chunk, int & nchunks) {
    chunk   = std::max<int64_t>((group_size + GN_MAX_CHUNKS - 1) / GN_MAX_CHUNKS, GN_MIN_CHUNK);
    chunk   = (chunk + 3) / 4 * 4;
    nchunks = (int) ((group_size + chunk - 1) / chunk);
}

template <bool vec, bool has_w, bool has_b, bool silu>
static __global__ void group_norm_apply_f32(const float * x, float * dst, const float2 * __restrict__ partials,
                                            const float * __restrict__ w, const float * __restrict__ b,
                                            const int64_t hw, const int cpg, const int ngroups, const int64_t group_size,
                                            const int nchunks, const int64_t per_block, const float eps) {
    const int c      = blockIdx.y;
    const int sample = blockIdx.z;
    const int gi     = sample * ngroups + c / cpg;

    __shared__ float s_mean, s_rstd;

    ggml_cuda_pdl_sync();
    if (threadIdx.x < WARP_SIZE) {
        const float2 ms = group_norm_merge(partials + (int64_t) gi * (nchunks + 1), nchunks, group_size, eps);
        if (threadIdx.x == 0) {
            s_mean = ms.x;
            s_rstd = ms.y;
        }
    }
    __syncthreads();

    const float   mean = s_mean;
    const float   rstd = s_rstd;
    const float   wc   = has_w ? w[c] : 1.0f;
    const float   bc   = has_b ? b[c] : 0.0f;
    const int64_t base = ((int64_t) sample * gridDim.y + c) * hw;
    const int64_t i0   = (int64_t) blockIdx.x * per_block;
    const int64_t i1   = min(i0 + per_block, hw);

    auto f = [&](float v) {
        float t = (v - mean) * rstd;
        if constexpr (has_w) {
            t = t * wc;
        }
        if constexpr (has_b) {
            t = t + bc;
        }
        if constexpr (silu) {
            t = ggml_cuda_op_silu_single(t);
        }
        return t;
    };

    if constexpr (vec) {
        const float4 * x4 = (const float4 *) (x + base + i0);
        float4 *       d4 = (float4 *) (dst + base + i0);
        const int      n4 = (int) ((i1 - i0) / 4);
#pragma unroll 4
        for (int i = threadIdx.x; i < n4; i += GN_BLOCK) {
            float4 v = x4[i];
            v.x      = f(v.x);
            v.y      = f(v.y);
            v.z      = f(v.z);
            v.w      = f(v.w);
            d4[i]    = v;
        }
    } else {
        for (int64_t i = i0 + threadIdx.x; i < i1; i += GN_BLOCK) {
            dst[base + i] = f(x[base + i]);
        }
    }
}

#define GN_SMALL_BLOCK 1024
#define GN_SMALL_NV    8  // float4 per thread held in registers
#define GN_SMALL_MAX   (4 * GN_SMALL_NV * GN_SMALL_BLOCK)  // largest group of the one-block path, 32768 elements

// one block per group for small groups (UNet / VAE bottleneck): the group is read once into registers, mean and
// centred variance are exact two-pass reductions, then the normalised (+gamma, +beta, +silu) values are written.
template <bool has_w, bool has_b, bool silu>
static __global__ void __launch_bounds__(GN_SMALL_BLOCK)
group_norm_small_f32(const float * x, float * dst, const float * __restrict__ w, const float * __restrict__ b,
                     const int hw, const int cpg, const int ngroups, const int group_size, const float eps) {
    const int      gi = blockIdx.x;
    const int      c0 = (gi % ngroups) * cpg;
    const float4 * x4 = (const float4 *) (x + (int64_t) gi * group_size);
    float4 *       d4 = (float4 *) (dst + (int64_t) gi * group_size);
    const int      n4 = group_size / 4;

    __shared__ float s_a[GN_SMALL_BLOCK / WARP_SIZE];
    __shared__ float s_b[GN_SMALL_BLOCK / WARP_SIZE];

    ggml_cuda_pdl_sync();
    float4 v[GN_SMALL_NV];
    float  s = 0.0f;
#pragma unroll
    for (int k = 0; k < GN_SMALL_NV; ++k) {
        const int i = threadIdx.x + k * GN_SMALL_BLOCK;
        v[k]        = i < n4 ? x4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
    }
    s                = block_reduce<block_reduce_method::SUM, GN_SMALL_BLOCK>(s, s_a);
    const float mean = s / group_size;

    float q = 0.0f;
#pragma unroll
    for (int k = 0; k < GN_SMALL_NV; ++k) {
        const int i = threadIdx.x + k * GN_SMALL_BLOCK;
        if (i < n4) {
            const float d0 = v[k].x - mean, d1 = v[k].y - mean, d2 = v[k].z - mean, d3 = v[k].w - mean;
            q += (d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3);
        }
    }
    q                = block_reduce<block_reduce_method::SUM, GN_SMALL_BLOCK>(q, s_b);
    const float rstd = rsqrtf(q / group_size + eps);

#pragma unroll
    for (int k = 0; k < GN_SMALL_NV; ++k) {
        const int i = threadIdx.x + k * GN_SMALL_BLOCK;
        if (i < n4) {
            const int   c  = c0 + (i * 4) / hw;
            const float wc = has_w ? w[c] : 1.0f;
            const float bc = has_b ? b[c] : 0.0f;
            auto        f  = [&](float t) {
                t = (t - mean) * rstd;
                if constexpr (has_w) {
                    t = t * wc;
                }
                if constexpr (has_b) {
                    t = t + bc;
                }
                if constexpr (silu) {
                    t = ggml_cuda_op_silu_single(t);
                }
                return t;
            };
            float4 r;
            r.x   = f(v[k].x);
            r.y   = f(v[k].y);
            r.z   = f(v[k].z);
            r.w   = f(v[k].w);
            d4[i] = r;
        }
    }
}

template <bool has_w, bool has_b>
static void group_norm_small_launch(bool silu, int ngi, cudaStream_t stream, const float * x, float * dst,
                                    const float * w, const float * b, int hw, int cpg, int ngroups, int group_size,
                                    float eps) {
    if (silu) {
        group_norm_small_f32<has_w, has_b, true>
            <<<ngi, GN_SMALL_BLOCK, 0, stream>>>(x, dst, w, b, hw, cpg, ngroups, group_size, eps);
    } else {
        group_norm_small_f32<has_w, has_b, false>
            <<<ngi, GN_SMALL_BLOCK, 0, stream>>>(x, dst, w, b, hw, cpg, ngroups, group_size, eps);
    }
}

template <bool vec, bool has_w, bool has_b>
static void group_norm_apply_launch(bool silu, dim3 grid, cudaStream_t stream, const float * x, float * dst,
                                    const float2 * partials, const float * w, const float * b, int64_t hw, int cpg,
                                    int ngroups, int64_t group_size, int nchunks, int64_t per_block, float eps) {
    if (silu) {
        group_norm_apply_f32<vec, has_w, has_b, true><<<grid, GN_BLOCK, 0, stream>>>(
            x, dst, partials, w, b, hw, cpg, ngroups, group_size, nchunks, per_block, eps);
    } else {
        group_norm_apply_f32<vec, has_w, has_b, false><<<grid, GN_BLOCK, 0, stream>>>(
            x, dst, partials, w, b, hw, cpg, ngroups, group_size, nchunks, per_block, eps);
    }
}

// x, dst: contiguous [ne0, ne1, ne2, ne3] f32, ne2 % num_groups == 0; w, b: nullptr or ne2 contiguous f32
static void group_norm_par_f32_cuda(ggml_backend_cuda_context & ctx, const float * x, float * dst, const float * w,
                                    const float * b, bool silu, const int64_t * ne, int num_groups, float eps) {
    cudaStream_t  stream     = ctx.stream();
    const int64_t hw         = ne[0] * ne[1];
    const int     cpg        = (int) (ne[2] / num_groups);
    const int64_t group_size = hw * cpg;
    const int     ngi        = (int) (num_groups * ne[3]);
    const bool    vec        = hw % 4 == 0 && ((uintptr_t) x % 16) == 0 && ((uintptr_t) dst % 16) == 0;

    if (vec && group_size <= GN_SMALL_MAX) {
        if (w && b) {
            group_norm_small_launch<true, true>(silu, ngi, stream, x, dst, w, b, (int) hw, cpg, num_groups,
                                                (int) group_size, eps);
        } else if (w) {
            group_norm_small_launch<true, false>(silu, ngi, stream, x, dst, w, b, (int) hw, cpg, num_groups,
                                                 (int) group_size, eps);
        } else {
            GGML_ASSERT(!b);
            group_norm_small_launch<false, false>(silu, ngi, stream, x, dst, w, b, (int) hw, cpg, num_groups,
                                                  (int) group_size, eps);
        }
        return;
    }

    int64_t chunk;
    int     nchunks;
    group_norm_chunking(group_size, chunk, nchunks);

    ggml_cuda_pool_alloc<float2> partials(ctx.pool(), (size_t) ngi * (nchunks + 1));

    const dim3 grid_s(nchunks, ngi, 1);
    if (vec) {
        group_norm_stats_f32<true><<<grid_s, GN_BLOCK, 0, stream>>>(x, partials.get(), group_size, chunk, nchunks);
    } else {
        group_norm_stats_f32<false><<<grid_s, GN_BLOCK, 0, stream>>>(x, partials.get(), group_size, chunk, nchunks);
    }

    const int64_t per_block = 16 * GN_BLOCK;  // elements of one channel plane per block (4 float4 per thread)
    const dim3    grid_a((unsigned) ((hw + per_block - 1) / per_block), (unsigned) ne[2], (unsigned) ne[3]);

#define GN_APPLY(V)                                                                                                  \
    if (w && b) {                                                                                                    \
        group_norm_apply_launch<V, true, true>(silu, grid_a, stream, x, dst, partials.get(), w, b, hw, cpg,          \
                                               num_groups, group_size, nchunks, per_block, eps);                     \
    } else if (w) {                                                                                                  \
        group_norm_apply_launch<V, true, false>(silu, grid_a, stream, x, dst, partials.get(), w, b, hw, cpg,         \
                                                num_groups, group_size, nchunks, per_block, eps);                    \
    } else {                                                                                                         \
        GGML_ASSERT(!b);                                                                                             \
        group_norm_apply_launch<V, false, false>(silu, grid_a, stream, x, dst, partials.get(), w, b, hw, cpg,        \
                                                 num_groups, group_size, nchunks, per_block, eps);                   \
    }
    if (vec) {
        GN_APPLY(true)
    } else {
        GN_APPLY(false)
    }
#undef GN_APPLY
}

bool ggml_cuda_op_group_norm_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * gn, const ggml_tensor * w,
                                   const ggml_tensor * b, bool silu, ggml_tensor * dst) {
    const ggml_tensor * src0       = gn->src[0];
    const int           num_groups = gn->op_params[0];
    float               eps;
    memcpy(&eps, gn->op_params + 1, sizeof(float));

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(dst) || !ggml_are_same_shape(src0, dst) || num_groups <= 0 ||
        src0->ne[2] % num_groups != 0 || src0->ne[2] > 65535 || src0->ne[3] > 65535 ||
        (int64_t) num_groups * src0->ne[3] > 65535) {
        return false;
    }
    // elementwise-safe only for exact aliasing (x == dst) or disjoint buffers
    const char * x0 = (const char *) src0->data;
    const char * d0 = (const char *) dst->data;
    if (x0 != d0 && x0 < d0 + ggml_nbytes(dst) && d0 < x0 + ggml_nbytes(src0)) {
        return false;
    }
    group_norm_par_f32_cuda(ctx, (const float *) src0->data, (float *) dst->data,
                            w ? (const float *) w->data : nullptr, b ? (const float *) b->data : nullptr, silu,
                            src0->ne, num_groups, eps);
    return true;
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    if (ggml_cuda_op_group_norm_fused(ctx, dst, nullptr, nullptr, false, dst)) {
        return;
    }

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
