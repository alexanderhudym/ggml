#include "qp-fuse.cuh"

// norm_f32 (norm.cu) with the multiply by a broadcast row folded into the store: ((x - mean) * inv_std) * w
template <int block_size>
static __global__ void norm_mul_f32(const float * x, float * dst, const int ncols, const int64_t stride_row,
                                    const int64_t stride_channel, const int64_t stride_sample, const float eps,
                                    const float * w, const int w_ne1, const int w_ne2, const int w_ne3,
                                    const int64_t w_s1, const int64_t w_s2, const int64_t w_s3) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x   += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;
    w   += (sample % w_ne3) * w_s3 + (channel % w_ne2) * w_s2 + (row % w_ne1) * w_s1;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean    = mean_var.x / ncols;
    const float var     = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float y = (x[col] - mean) * inv_std;
        dst[col]      = __fmul_rn(y, w[col]);
    }
}

void ggml_cuda_op_norm_mul_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * row, ggml_tensor * dst) {
    const ggml_tensor * src0 = norm->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && row->type == GGML_TYPE_F32);
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const size_t ts = sizeof(float);
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const dim3 blocks_num(ne01, ne02, ne03);
    const float * x = (const float *) src0->data;
    float * d       = (float *) dst->data;
    const float * w = (const float *) row->data;
    cudaStream_t stream = ctx.stream();
    if (ne00 < 1024) {
        norm_mul_f32<WARP_SIZE><<<blocks_num, WARP_SIZE, 0, stream>>>(
            x, d, ne00, src0->nb[1] / ts, src0->nb[2] / ts, src0->nb[3] / ts, eps,
            w, row->ne[1], row->ne[2], row->ne[3], row->nb[1] / ts, row->nb[2] / ts, row->nb[3] / ts);
    } else {
        norm_mul_f32<1024><<<blocks_num, 1024, 32 * sizeof(float2), stream>>>(
            x, d, ne00, src0->nb[1] / ts, src0->nb[2] / ts, src0->nb[3] / ts, eps,
            w, row->ne[1], row->ne[2], row->ne[3], row->nb[1] / ts, row->nb[2] / ts, row->nb[3] / ts);
    }
    CUDA_CHECK(cudaGetLastError());
}

// dst = r + h * w[row]; all contiguous except w (contiguous rows, broadcast over rows/channels/samples)
static __global__ void mul_add_f32(const float * __restrict__ h, const float * __restrict__ r, float * dst,
                                   const float * __restrict__ w, const int ne0, const int ne1, const int ne2,
                                   const int w_ne1, const int w_ne2, const int w_ne3,
                                   const int64_t w_s1, const int64_t w_s2, const int64_t w_s3) {
    const int64_t row = blockIdx.y;  // flattened i1 + ne1*(i2 + ne2*i3)
    const int i1 = row % ne1;
    const int i2 = (row / ne1) % ne2;
    const int i3 = row / ((int64_t) ne1 * ne2);
    const float * wr = w + (i3 % w_ne3) * w_s3 + (i2 % w_ne2) * w_s2 + (i1 % w_ne1) * w_s1;
    const int64_t base = row * ne0;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < ne0; i0 += gridDim.x * blockDim.x) {
        dst[base + i0] = __fadd_rn(r[base + i0], __fmul_rn(h[base + i0], wr[i0]));
    }
}

void ggml_cuda_op_mul_add_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * h, const ggml_tensor * row,
                                const ggml_tensor * residual, ggml_tensor * dst) {
    const int ne0 = dst->ne[0];
    const int64_t nrows = ggml_nrows(dst);
    const int block = 256;
    const dim3 grid((unsigned) std::min<int64_t>((ne0 + block - 1) / block, 64), (unsigned) nrows);
    const size_t ts = sizeof(float);
    GGML_ASSERT(nrows <= 65535);
    mul_add_f32<<<grid, block, 0, ctx.stream()>>>((const float *) h->data, (const float *) residual->data, (float *) dst->data,
                                                  (const float *) row->data, ne0, dst->ne[1], dst->ne[2],
                                                  row->ne[1], row->ne[2], row->ne[3], row->nb[1] / ts, row->nb[2] / ts, row->nb[3] / ts);
    CUDA_CHECK(cudaGetLastError());
}
