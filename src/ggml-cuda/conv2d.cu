#include "conv2d.cuh"
#include "convert.cuh"
#include "mma.cuh"
#include "conv2d-implicit.cuh"

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int64_t out_coord,
                                                     int64_t kern_coord,
                                                     int64_t stride,
                                                     int64_t dilation,
                                                     int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ static int64_t kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ static void unpack_indices(int64_t             global_idx,
                                          const conv_params & P,
                                          int64_t &           n,
                                          int64_t &           c,
                                          int64_t &           out_y,
                                          int64_t &           out_x) {
        out_x = global_idx % P.OW;
        out_y = (global_idx / P.OW) % P.OH;
        c     = (global_idx / (P.OW * P.OH)) % P.OC;
        n     = global_idx / (P.OW * P.OH * P.OC);
    }
};

template <typename T, typename Layout>
static __global__ void conv2d_kernel(const float * __restrict__ input,
                                     const T * __restrict__ kernel,
                                     float * __restrict__ output,
                                     const conv_params P) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    Layout::unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val = input[Layout::input_index(n, c_in, in_y, in_x, P)];
                const T kernel_val = kernel[Layout::kernel_index(c_out, c_in, ky, kx, P)];
                acc += (input_val * ggml_cuda_cast<float>(kernel_val));
            }
        }
    }

    // [N, OC, OH, OW]
    output[Layout::output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    const int blocks = (P.TOTAL + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
    conv2d_kernel<T, whcn_layout><<<blocks, CUDA_CONV2D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, P);
}

static __global__ void
conv2d_pad_f16(const float * input, half * output, int iw, int ih, int pw, int ph, int px, int py, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int x = i % pw - px, y = i / pw % ph - py, nc = i / (pw * ph);
    output[i] = __float2half(
        (unsigned) x < (unsigned) iw && (unsigned) y < (unsigned) ih ? input[(nc * ih + y) * iw + x] : 0.0f);
}

template <int KW, int KH, bool use_mma>
static __global__ void conv2d_implicit_gemm_f16(const half * __restrict__ input,
                                                const half * __restrict__ weight,
                                                float * __restrict__ output,
                                                const conv_params P,
                                                const int         split_k) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = 4 * warp_size;
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int AS = BK / 2 + 4;
    constexpr int BS = BN / 2 + 4;
    __shared__ __align__(16) half2 a_s[BM][AS];
    __shared__ __align__(16) half2 b_s[BK][BS];

    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int kw = KW ? KW : int(P.KW), kh = KH ? KH : int(P.KH);
    const int ic = int(P.IC), oc = int(P.OC);
    const int sx = int(P.ST_X), sy = int(P.ST_Y);
    const int dx = int(P.DL_X), dy = int(P.DL_Y);
    const int n = blockIdx.z / split_k, split = blockIdx.z % split_k;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    const int k_total   = ic * kw * kh;
    const int load_lane = warp_size == 32 ? threadIdx.x : threadIdx.x % (BN / 2);
    const int load_row  = threadIdx.y * (warp_size / (BN / 2)) + (warp_size == 32 ? 0 : threadIdx.x / (BN / 2));
    const int spatial   = n0 + 2 * load_lane;
    const int spatial0 = min(spatial, ow * oh - 1), spatial1 = min(spatial + 1, ow * oh - 1);
    const int y0 = spatial0 / ow, x0 = spatial0 % ow;
    const int y1 = spatial1 / ow, x1 = spatial1 % ow;
    const int pos0 = y0 * sy * iw + x0 * sx, pos1 = y1 * sy * iw + x1 * sx;

    [[maybe_unused]] const int wm = threadIdx.y / 2 * 32, wn = threadIdx.y % 2 * 32;
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    using tile_ab = tile<16, 8, half2, get_input_data_layout()>;
#    if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    // AMD accumulator fragments transpose the input fragment's row/column mapping.
    using tile_c = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
#    else
    using tile_c = tile<16, 16, float>;
#    endif
    [[maybe_unused]] tile_c c[2][2];
#else
    if constexpr (use_mma) {
        NO_DEVICE_CODE;
        return;
    }
#endif
    constexpr int              RM = 4, RN = BM * BN / (nthreads * RM);
    [[maybe_unused]] const int simt_m = tid / (BN / RN) * RM, simt_n = tid % (BN / RN) * RN;
    [[maybe_unused]] float     c_simt[RM][RN] = {};
    const int                  tiles          = (k_total + BK - 1) / BK;
    const int                  begin          = int(int64_t(tiles) * split / split_k) * BK;
    const int                  end            = int(int64_t(tiles) * (split + 1) / split_k) * BK;
    for (int k0 = begin; k0 < end; k0 += BK) {
        if (k_total % 8 == 0 && uintptr_t(weight) % 16 == 0) {
#pragma unroll
            for (int i = tid; i < BM * BK / 8; i += nthreads) {
                const int  row = i / (BK / 8), col = 8 * (i % (BK / 8));
                const int4 v                 = m0 + row < oc && k0 + col < k_total ?
                                                   ((const int4 *) weight)[((m0 + row) * k_total + k0 + col) / 8] :
                                                   make_int4(0, 0, 0, 0);
                *(int4 *) &a_s[row][col / 2] = v;
            }
        } else {
#pragma unroll
            for (int i = tid; i < BM * BK / 2; i += nthreads) {
                const int row = i / (BK / 2), col = 2 * (i % (BK / 2));
                half      lo = __float2half(0.0f), hi = lo;
                if (m0 + row < oc && k0 + col < k_total) {
                    lo = weight[(m0 + row) * k_total + k0 + col];
                    if (k0 + col + 1 < k_total) {
                        hi = weight[(m0 + row) * k_total + k0 + col + 1];
                    }
                }
                a_s[row][col / 2] = __halves2half2(lo, hi);
            }
        }
#pragma unroll
        for (int k = load_row; k < BK; k += nthreads / (BN / 2)) {
            const int ki = k0 + k;
            const int ci = ki / (kw * kh), ky = ki / kw % kh, kx = ki % kw;
            const int offset = ki < k_total ? (n * ic + ci) * ih * iw + ky * dy * iw + kx * dx : 0;
            half      lo = __float2half(0.0f), hi = lo;
            if (ki < k_total && spatial < ow * oh) {
                lo = input[offset + pos0];
            }
            if (ki < k_total && spatial + 1 < ow * oh) {
                hi = input[offset + pos1];
            }
            b_s[k][load_lane] = __halves2half2(lo, hi);
        }
        __syncthreads();
        if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
            for (int k = 0; k < BK; k += 16) {
                tile_ab a[2], b[2];
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_ldmatrix(a[i], &a_s[wm + 16 * i][k / 2], AS);
                    load_ldmatrix_trans(b[i], &b_s[k][(wn + 16 * i) / 2], BS);
                }
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
#    pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        mma(c[i][j], a[i], b[j]);
                    }
                }
            }
#endif
        } else {
#pragma unroll 4
            for (int k = 0; k < BK; ++k) {
                float a[RM], b[RN];
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    a[i] = __half2float(((const half *) a_s[simt_m + i])[k]);
                }
#pragma unroll
                for (int j = 0; j < RN; ++j) {
                    b[j] = __half2float(((const half *) b_s[k])[simt_n + j]);
                }
#pragma unroll
                for (int i = 0; i < RM; ++i) {
#pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        c_simt[i][j] += a[i] * b[j];
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
        for (int i = 0; i < 2; ++i) {
#    pragma unroll
            for (int j = 0; j < 2; ++j) {
#    pragma unroll
                for (int l = 0; l < c[i][j].ne; ++l) {
                    const int co  = m0 + wm + 16 * i + c[i][j].get_i(l);
                    const int pos = n0 + wn + 16 * j + c[i][j].get_j(l);
                    if (co < oc && pos < ow * oh) {
                        output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c[i][j].x[l];
                    }
                }
            }
        }
#endif
    } else {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int co = m0 + simt_m + i, pos = n0 + simt_n + j;
                if (co < oc && pos < ow * oh) {
                    output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c_simt[i][j];
                }
            }
        }
    }
}

static __global__ void conv2d_reduce_split_k(const float * __restrict__ partial,
                                             float * __restrict__ output,
                                             const int total,
                                             const int per_batch,
                                             const int split_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int     n   = i / per_batch;
    const float * src = partial + int64_t(n) * (split_k - 1) * per_batch + i;
    float         sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += src[int64_t(k) * per_batch];
    }
    output[i] = sum;
}

template <bool use_mma>
static void conv2d_launch_implicit_gemm(const half *        input,
                                        const half *        weight,
                                        float *             output,
                                        const conv_params & params,
                                        int                 split_k,
                                        dim3                grid,
                                        dim3                block,
                                        cudaStream_t        stream) {
    if (params.KW == 3 && params.KH == 3) {
        conv2d_implicit_gemm_f16<3, 3, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else if (params.KW == 1 && params.KH == 1) {
        conv2d_implicit_gemm_f16<1, 1, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else {
        conv2d_implicit_gemm_f16<0, 0, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    }
}

static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
}

// true if the conv runs on the tensor-core implicit GEMM (conv2d-implicit.cu), which reads its input as NHWC f16:
// NVIDIA build and device with sm_80 tensor cores, f16 filter up to 32x32, a multiple of 8 input channels, positive
// stride and dilation, non-negative padding, 32-bit tensor sizes (param_t), batch within the staging grid
bool ggml_cuda_conv2d_accepts_nhwc(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const auto &        device = ggml_cuda_info().devices[ctx.device];
    const int32_t *     p      = (const int32_t *) dst->op_params;
    return dst->op == GGML_OP_CONV_2D && kernel->type == GGML_TYPE_F16 && input->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && ggml_is_contiguous(input) && ggml_is_contiguous(kernel) &&
           input->ne[2] == kernel->ne[2] && p[6] == 0 && p[0] > 0 && p[1] > 0 && p[2] >= 0 && p[3] >= 0 && p[4] > 0 &&
           p[5] > 0 && GGML_CUDA_CC_IS_NVIDIA(device.cc) && ampere_mma_available(device.cc) &&
           input->ne[2] % 8 == 0 && kernel->ne[0] <= 32 && kernel->ne[1] <= 32 && input->ne[3] <= 65535 &&
           ggml_nelements(input) <= INT_MAX && ggml_nelements(dst) <= INT_MAX && ggml_nelements(kernel) <= INT_MAX;
}

// Runs the conv. With bias/residual/out set it may apply the fused epilogue out = (conv + bias) + residual inside
// the conv kernel (tensor-core path); returns true if it did, false if the plain conv result is in dst->data.
static bool conv2d_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * bias = nullptr,
                       const float * residual = nullptr, float * out = nullptr, const half * x_nhwc = nullptr) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int64_t IW = input->ne[0];   // input_w
    const int64_t IH = input->ne[1];   // input_h
    const int64_t OW = dst->ne[0];     // output_w
    const int64_t OH = dst->ne[1];     // output_h
    const int64_t KW = kernel->ne[0];  // kernel_w
    const int64_t KH = kernel->ne[1];  // kernel_h
    const int64_t IC = input->ne[2];   // input_channels
    const int64_t OC = kernel->ne[3];  // ouptut_chanles
    const int64_t B  = input->ne[3];   // n_batches

    const int64_t total  = B * OC * OH * OW;
    conv_params   params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (ggml_cuda_conv2d_accepts_nhwc(ctx, dst)) {
        ggml_cuda_op_conv2d_implicit(ctx, dst, bias, residual, out, x_nhwc);
        return out != nullptr;
    }
    GGML_ASSERT(!x_nhwc);
    const bool   use_mma =
        turing_mma_available(device.cc) || amd_wmma_available(device.cc) || amd_mfma_available(device.cc);
    // MUSA can share the tiling without a native fragment implementation in mma.cuh.
    const bool use_simt   = GGML_CUDA_CC_IS_MTHREADS(device.cc);
    const bool pointwise  = KW == 1 && KH == 1 && ST_X == 1 && ST_Y == 1 && PD_X == 0 && PD_Y == 0;
    const bool use_blas   = pointwise && fast_fp16_hardware_available(device.cc);
    // Short reductions on small maps do not amortize conversion and launch costs.
    const bool small_conv = IC * KW * KH < 64 && OW * OH < 512;

    const int64_t limit    = INT_MAX - 256;
    const int64_t padded_w = IW + 2 * int64_t(PD_X), padded_h = IH + 2 * int64_t(PD_Y);
    const bool    padded_fits = padded_w > 0 && padded_w <= limit && padded_h > 0 && padded_h <= limit &&
                             padded_w * padded_h <= limit && IC * B <= limit / (padded_w * padded_h);
    if (kernel->type == GGML_TYPE_F16 && (use_mma || use_blas || use_simt) && (use_blas || !small_conv) &&
        ggml_nelements(input) <= limit && ggml_nelements(kernel) <= limit && total <= limit && padded_fits &&
        PD_X >= 0 && PD_Y >= 0 && ST_X > 0 && ST_Y > 0 && DL_X > 0 && DL_Y > 0 &&
        (OW - 1) * ST_X + (KW - 1) * DL_X < padded_w && (OH - 1) * ST_Y + (KH - 1) * DL_Y < padded_h &&
        (OC + 63) / 64 <= 65535 && B <= 65535) {
        const int pw = int(padded_w), ph = int(padded_h);
        const int padded_total = int(padded_w * padded_h * IC * B);

        ggml_cuda_pool_alloc<half> x_half(ctx.pool(), padded_total);
        // Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32.
        if (PD_X == 0 && PD_Y == 0) {
            ggml_get_to_fp16_cuda(input->type)(X_D, x_half.get(), padded_total, st);
        } else {
            conv2d_pad_f16<<<(padded_total + 255) / 256, 256, 0, st>>>(X_D, x_half.get(), int(IW), int(IH), pw, ph,
                                                                       PD_X, PD_Y, padded_total);
        }
        const conv_params padded_params = { pw, ph, OW, OH, KW, KH, ST_X, ST_Y, 0, 0, DL_X, DL_Y, IC, OC, B, total };
        if (use_blas) {
            const float    alpha = 1.0f, beta = 0.0f;
            const int      positions = int(OW * OH);
            cublasHandle_t cublas_h  = ctx.cublas_handle();
            for (int n = 0; n < B; ++n) {
                CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha,
                                          x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, K_D,
                                          CUDA_R_16F, int(IC), &beta, Y_D + int64_t(n) * OC * positions, CUDA_R_32F,
                                          positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            return false;
        }
        const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
        const int     target  = 8 * ggml_cuda_info().devices[ctx.device].nsm;
        // Split long reductions so small spatial maps still occupy the GPU.
        const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                               std::max(int64_t(1), (target + blocks - 1) / blocks) }));

        ggml_cuda_pool_alloc<float> partial(ctx.pool());
        float *                     result = split_k == 1 ? Y_D : partial.alloc(total * split_k);
        const dim3                  block(device.warp_size, 4);
        const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        if (use_mma) {
            conv2d_launch_implicit_gemm<true>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                              block, st);
        } else {
            conv2d_launch_implicit_gemm<false>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                               block, st);
        }
        if (split_k > 1) {
            conv2d_reduce_split_k<<<(total + 255) / 256, 256, 0, st>>>(result, Y_D, int(total), int(OC * OW * OH),
                                                                       split_k);
        }
        return false;
    }

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
    return false;
}

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    conv2d_run(ctx, dst);
}

// ---- CONV_2D -> [RESHAPE] -> ADD(bias) [-> ADD(residual)] fusion -------------------------------------------------
// sd.cpp builds every biased conv as ggml_add(conv, reshape(bias, [1,1,OC,1])), and ResNet blocks follow it with
// ggml_add(h, x). The fused path does the same fp32 adds in the same order (bit-identical), either in the
// tensor-core conv epilogue or, for the other conv paths, in one elementwise pass instead of two.

static __global__ void conv2d_bias_residual(const float * x, const float * __restrict__ bias, const float * residual,
                                            float * dst, const int64_t total, const int64_t PQ, const int64_t K) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    float v = x[i];
    if (bias) {
        v += bias[(i / PQ) % K];
    }
    if (residual) {
        v += residual[i];
    }
    dst[i] = v;
}

static bool conv2d_fuse_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

int ggml_cuda_try_fuse_conv2d_bias(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
                                   const half * x_nhwc) {
    ggml_tensor * conv = cgraph->nodes[i];
    if (conv->op != GGML_OP_CONV_2D || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv) ||
        (conv->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i) != 1) {
        return 0;
    }
    // the bias reshape (a no-op view) usually sits between the conv and the add
    int j = i + 1;
    while (j < cgraph->n_nodes && j <= i + 2 && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * add = cgraph->nodes[j];
    if (add->op != GGML_OP_ADD || !(add->flags & GGML_TENSOR_FLAG_COMPUTE) || add->type != GGML_TYPE_F32 ||
        add->src[0] != conv || !ggml_are_same_shape(add, conv) || !ggml_is_contiguous(add)) {
        return 0;
    }
    const ggml_tensor * bias = add->src[1];
    const int64_t       OC   = conv->ne[2];
    if (bias->type != GGML_TYPE_F32 || bias->ne[0] != 1 || bias->ne[1] != 1 || bias->ne[2] != OC || bias->ne[3] != 1 ||
        bias->nb[2] != sizeof(float) || bias->data == nullptr) {
        return 0;
    }
    const ggml_tensor * input  = conv->src[1];
    const ggml_tensor * weight = conv->src[0];

    // optional residual add right after: ADD(add, r) or ADD(r, add)
    ggml_tensor *       dst      = add;
    const ggml_tensor * residual = nullptr;
    const int           k        = j + 1;
    if (k < cgraph->n_nodes && !(add->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(cgraph, j) == 1) {
        ggml_tensor *       add2 = cgraph->nodes[k];
        const ggml_tensor * r    = nullptr;
        if (add2->op == GGML_OP_ADD && (add2->flags & GGML_TENSOR_FLAG_COMPUTE) && add2->type == GGML_TYPE_F32) {
            r = add2->src[0] == add ? add2->src[1] : add2->src[1] == add ? add2->src[0] : nullptr;
        }
        // the residual is read and dst written at the same index by the same thread, so it may alias dst exactly
        if (r && r != add && r->type == GGML_TYPE_F32 && ggml_are_same_shape(r, add) && ggml_are_same_shape(add2, add) &&
            ggml_is_contiguous(r) && ggml_is_contiguous(add2) && r->data != nullptr &&
            (r->data == add2->data || !conv2d_fuse_overlap(r, add2)) && !conv2d_fuse_overlap(r, conv)) {
            dst      = add2;
            residual = r;
        }
    }
    // the conv reads its input/weights while the epilogue writes dst: they must not share memory
    auto dst_ok = [&](const ggml_tensor * d) {
        return !conv2d_fuse_overlap(d, input) && !conv2d_fuse_overlap(d, weight) && !conv2d_fuse_overlap(d, bias);
    };
    if (!dst_ok(dst)) {
        dst      = add;
        residual = nullptr;
        if (!dst_ok(dst)) {
            return 0;
        }
    }

    const float * b_d = (const float *) bias->data;
    const float * r_d = residual ? (const float *) residual->data : nullptr;
    float *       out = (float *) dst->data;
    if (!conv2d_run(ctx, conv, b_d, r_d, out, x_nhwc)) {
        const int64_t total = ggml_nelements(conv);
        conv2d_bias_residual<<<(total + 255) / 256, 256, 0, ctx.stream()>>>(
            (const float *) conv->data, b_d, r_d, out, total, conv->ne[0] * conv->ne[1], OC);
    }
    return (dst == add ? j : k) - i;
}

// ---- producer -> CONV_2D with the NHWC f16 input written by the producer ------------------------------------------
// The tensor-core conv stages its f32 NCHW input as f16 NHWC in a separate pass (read 4 B + write 2 B per element).
// When the conv is the only consumer of its input, the producer can write that layout directly:
//  - GROUP_NORM -> MUL -> ADD [-> SILU] (ggml_cuda_try_fuse_group_norm, norm.cu's NHWC apply kernel), and
//  - UPSCALE (nearest, exactly 2x) and PAD (zeros, W/H only) folded into the staging read (below).
// Values are the same f32 results rounded to f16 the same way, so the conv output is bit-identical.

int ggml_cuda_conv2d_run_nhwc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const half * x_nhwc) {
    const int n_fused = ggml_cuda_try_fuse_conv2d_bias(ctx, cgraph, i, x_nhwc);
    if (n_fused > 0) {
        return n_fused;
    }
    conv2d_run(ctx, cgraph->nodes[i], nullptr, nullptr, nullptr, x_nhwc);
    return 0;
}

int ggml_cuda_try_fuse_stage_conv2d(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * prod = cgraph->nodes[i];
    const ggml_tensor * src  = prod->src[0];
    if (src == nullptr || prod->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
        !ggml_is_contiguous(prod) || prod->ne[2] != src->ne[2] || prod->ne[3] != src->ne[3] ||
        (prod->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i) != 1) {
        return 0;
    }
    int mode = -1, lp0 = 0, lp1 = 0;
    if (prod->op == GGML_OP_UPSCALE) {
        // NEAREST without flags, exactly 2x in W and H: upscale_f32 reads src(x / 2.0f, y / 2.0f) = (x / 2, y / 2)
        if (ggml_get_op_params_i32(prod, 0) == GGML_SCALE_MODE_NEAREST && prod->ne[0] == 2 * src->ne[0] &&
            prod->ne[1] == 2 * src->ne[1]) {
            mode = 1;
        }
    } else if (prod->op == GGML_OP_PAD) {
        const int32_t * pp = (const int32_t *) prod->op_params;  // lp0 rp0 lp1 rp1 lp2 rp2 lp3 rp3 circular
        if (pp[0] >= 0 && pp[1] >= 0 && pp[2] >= 0 && pp[3] >= 0 && pp[4] == 0 && pp[5] == 0 && pp[6] == 0 &&
            pp[7] == 0 && pp[8] == 0 && prod->ne[0] == src->ne[0] + pp[0] + pp[1] &&
            prod->ne[1] == src->ne[1] + pp[2] + pp[3]) {
            mode = 2;
            lp0  = pp[0];
            lp1  = pp[2];
        }
    }
    if (mode < 0 || prod->ne[0] > INT_MAX / 4 || prod->ne[1] > INT_MAX / 4 || prod->ne[3] > 65535) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j]) && j <= i + 2) {
        ++j;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * conv = cgraph->nodes[j];
    if (conv->op != GGML_OP_CONV_2D || conv->src[1] != prod || !(conv->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        !ggml_cuda_conv2d_accepts_nhwc(ctx, conv)) {
        return 0;
    }
    ggml_cuda_pool_alloc<half> x_nhwc(ctx.pool(), ggml_nelements(prod));
    ggml_cuda_conv2d_nhwc_stage((const float *) src->data, x_nhwc.get(), int(prod->ne[2]), int(prod->ne[0]),
                                int(prod->ne[1]), int(prod->ne[3]), mode, int(src->ne[0]), int(src->ne[1]), lp0, lp1,
                                ctx.stream());
    return j - i + ggml_cuda_conv2d_run_nhwc(ctx, cgraph, j, x_nhwc.get());
}
