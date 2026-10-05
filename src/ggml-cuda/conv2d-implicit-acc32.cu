#include "conv2d-implicit-acc32.cuh"

// Tensor-core implicit-GEMM convolution after the multistage design of bssrdf's llama.cpp PR #15805 (MIT): NHWC f16
// input, filter [K][RS][C] f16, 256x128x32 block tiles, a two-stage cp.async pipeline, swizzled shared memory and x4
// ldmatrix fragments. The mma is m16n8k16 with f32 accumulators and the epilogue is f32, so results match the
// f32-accumulating kernels to rounding. f32 accumulators are register-heavy, so the output-channel tile is 128, the
// warp tile 128x32 (212 registers, no spills).

// x4 ldmatrix fragments for a 128x32 (rows x k) A warp tile: reg[m][k] = rows 16m..16m+15, k block k.
__device__ __forceinline__ void acc32_ldmatrix_a(const half * src, uint32_t (&reg)[8][4][2]) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int smem_stride     = 32;
    const unsigned int     logical_offset  = (threadIdx.x % 32) * smem_stride;
    unsigned int           swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
    swizzled_offset                        = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
    uint32_t           src_addr            = ggml_cuda_cvta_generic_to_shared(src + swizzled_offset);
    constexpr uint32_t k_xor[4]            = { 0, 0b10000, 0b110000, 0b10000 };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        src_addr ^= k_xor[k];
#pragma unroll
        for (int m = 0; m < 8; m += 2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(reg[m][k][0]), "=r"(reg[m][k][1]), "=r"(reg[m + 1][k][0]), "=r"(reg[m + 1][k][1])
                         : "r"(src_addr + uint32_t(16 * m * smem_stride * sizeof(half))));
        }
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

// x4 ldmatrix fragments for a 32x32 (output channels x k) B warp tile: reg[k][n] = channels 8n..8n+7, k block k.
__device__ __forceinline__ void acc32_ldmatrix_b(const half * src, uint32_t (&reg)[4][4]) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int smem_stride     = 32;
    const unsigned int     logical_offset  = (threadIdx.x % 32) * smem_stride;
    unsigned int           swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
    swizzled_offset                        = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
    uint32_t           src_addr            = ggml_cuda_cvta_generic_to_shared(src + swizzled_offset);
    constexpr uint32_t k_xor[4]            = { 0, 0b10000, 0b110000, 0b10000 };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        src_addr ^= k_xor[k];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(reg[k][0]), "=r"(reg[k][1]), "=r"(reg[k][2]), "=r"(reg[k][3])
                     : "r"(src_addr));
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

// Two k8 fragments (k blocks 2kk, 2kk+1) concatenate into one m16n8k16 fragment for both A and B.
template <unsigned int MT, unsigned int NT>
__device__ __forceinline__ void acc32_mma(float (&acc)[MT][NT][4], const uint32_t (&A)[MT][4][2],
                                          const uint32_t (&B)[4][NT]) {
#ifdef CP_ASYNC_AVAILABLE
#pragma unroll
    for (unsigned int kk = 0; kk < 2; kk++) {
#pragma unroll
        for (unsigned int n = 0; n < NT; n++) {
#pragma unroll
            for (unsigned int m = 0; m < MT; m++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[m][n][0]), "+f"(acc[m][n][1]), "+f"(acc[m][n][2]), "+f"(acc[m][n][3])
                    : "r"(A[m][2 * kk][0]), "r"(A[m][2 * kk][1]), "r"(A[m][2 * kk + 1][0]), "r"(A[m][2 * kk + 1][1]),
                      "r"(B[2 * kk][n]), "r"(B[2 * kk + 1][n]));
            }
        }
    }
#else
    GGML_UNUSED(acc);
    GGML_UNUSED(A);
    GGML_UNUSED(B);
    NO_DEVICE_CODE;
#endif
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int NUM_THREADS>
static __global__ void __launch_bounds__(NUM_THREADS, 1)
    conv2d_implicit_kernel_acc32(const half * __restrict__ input,
                                 const half * __restrict__ kernel,
                                 float * output,
                                 const param_t param,
                                 const unsigned int ksplit,
                                 const float * __restrict__ bias,
                                 const float * residual) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr unsigned int MMA_M                = 16;
    constexpr unsigned int MMA_N                = 8;
    constexpr unsigned int mma_tiles_per_warp_k = 4;
    constexpr unsigned int mma_tiles_per_warp_m = WM / MMA_M;
    constexpr unsigned int mma_tiles_per_warp_n = WN / MMA_N;
    static_assert(BM == 256 && BN == 128 && BK == 32 && WM == 128 && WN == 32 && NUM_THREADS == 256);

    const unsigned int z                   = blockIdx.z;
    const unsigned int ks                  = (param.c + ksplit - 1) / ksplit;
    const unsigned int start_k             = z * ks;
    const unsigned int end_k               = min(start_k + ks, param.c);
    const unsigned int num_block_tiles_k   = (ks + (BK - 1)) / BK;
    const unsigned int num_block_tiles_krs = num_block_tiles_k * param.r * param.s;

    constexpr unsigned int TILE_COLS_VECTORIZED = BK / 8;
    constexpr unsigned int ROW_STEP             = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int A_K_STRID            = BM / ROW_STEP;

    unsigned int masks_a[A_K_STRID][2];
    int64_t      element_offset_a[A_K_STRID];
    int64_t      element_offset_b;

    const unsigned int block_m    = blockIdx.y;
    const unsigned int block_n    = blockIdx.x;
    const unsigned int warp_m     = threadIdx.y;
    const unsigned int warp_n     = threadIdx.x / WARP_SIZE;
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int       thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    extern __shared__ half shmem[];
    constexpr int          BUFFER_SIZE = BM * BK + BK * BN;
    half *                 SA1         = shmem;
    half *                 SB1         = &shmem[BM * BK];
    half *                 SA2         = &shmem[BUFFER_SIZE];
    half *                 SB2         = SA2 + BM * BK;

    float    acc[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

#pragma unroll
    for (unsigned int m = 0; m < mma_tiles_per_warp_m; m++) {
#pragma unroll
        for (unsigned int n = 0; n < mma_tiles_per_warp_n; n++) {
#pragma unroll
            for (unsigned int l = 0; l < 4; l++) {
                acc[m][n][l] = 0.0f;
            }
        }
    }

    const unsigned int A_warp_tile_offset = warp_m * WM * BK;
    const unsigned int B_warp_tile_offset = warp_n * WN * BK;

    prepareIteratorA<BM, BK, A_K_STRID, ROW_STEP>(thread_row, masks_a, element_offset_a, param);

    const unsigned int iter_src_idx   = thread_row * param.weightKOffset;
    const unsigned int iter_dst_idx   = thread_row * TILE_COLS_VECTORIZED + thread_col;
    const unsigned int krow_idx       = thread_row + blockIdx.x * BN;
    const int          ITER_SRC_STEPS = ROW_STEP * param.weightKOffset;

    const half * A_block_gmem = input;
    const half * B_block_gmem = kernel + block_n * BN * param.weightKOffset;

    unsigned int curC = tileMemcpySwizzleA<BM, NUM_THREADS>(A_block_gmem, SA1, masks_a, element_offset_a, thread_row,
                                                            thread_col, start_k, end_k);
    element_offset_b  = curC;
    tileMemcpySwizzleB<BN, NUM_THREADS>(B_block_gmem, SB1, curC, element_offset_b, end_k, thread_row, thread_col,
                                        param);
    asm volatile("cp.async.commit_group;\n" ::);

    unsigned int block_k   = 0;
    unsigned int block_krs = 1;
    int          s         = 0;
    int          r         = 0;

    while (block_krs < num_block_tiles_krs) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();

        int next_idx = 0;
        ++s;
        if (s == param.s) {
            s = 0;
            ++r;
            if (r < param.r) {
                next_idx = 1;
            } else {
                r        = 0;
                next_idx = 2;
            }
        }
        add_byte_offset<A_K_STRID>(element_offset_a, param.inc_next[next_idx]);
        if (next_idx == 2) {
            ++block_k;
        }

        curC = tileMemcpyAsyncLoadA<BM, BK, NUM_THREADS>(A_block_gmem, SA2, r, s, masks_a, element_offset_a,
                                                         thread_col, iter_dst_idx, block_k * BK, start_k, end_k, curC);
        element_offset_b = (r * param.s + s) * param.c + curC;
        tileMemcpyAsyncLoadB<BN, BK, NUM_THREADS>(B_block_gmem, SB2, curC, element_offset_b, end_k, iter_src_idx,
                                                  iter_dst_idx, krow_idx, ITER_SRC_STEPS, param.k);
        asm volatile("cp.async.commit_group;\n" ::);

        acc32_ldmatrix_a(SA1 + A_warp_tile_offset, A_register);
        acc32_ldmatrix_b(SB1 + B_warp_tile_offset, B_register);
        acc32_mma(acc, A_register, B_register);

        half * tmp = SA1;
        SA1        = SA2;
        SA2        = tmp;
        tmp        = SB1;
        SB1        = SB2;
        SB2        = tmp;
        block_krs++;
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    acc32_ldmatrix_a(SA1 + A_warp_tile_offset, A_register);
    acc32_ldmatrix_b(SB1 + B_warp_tile_offset, B_register);
    acc32_mma(acc, A_register, B_register);

    // f32 accumulator fragment: c0,c1 at row g, cols 2t,2t+1; c2,c3 at row g+8. Rows are output positions (contiguous
    // in NCHW), so every store instruction writes whole 32-byte sectors.
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int g       = lane / 4;
    const unsigned int t       = lane % 4;
    const int64_t      z_off   = int64_t(z) * param.NKPQ;
    const unsigned int m_base  = block_m * BM + warp_m * WM + g;
    const unsigned int ch_base = block_n * BN + warp_n * WN + 2 * t;
#pragma unroll
    for (unsigned int m = 0; m < mma_tiles_per_warp_m; m++) {
        // load this m-tile's residual before any of its stores: residual may alias output, so the compiler would
        // otherwise serialize every load behind the previous store
        float res_v[2][mma_tiles_per_warp_n][2];
        if (residual) {
#pragma unroll
            for (unsigned int h = 0; h < 2; h++) {
                const unsigned int gemm_i = m_base + m * MMA_M + 8 * h;
                const unsigned int n      = fastdiv(gemm_i, param.OHOW_fastdiv);
                const unsigned int col    = fastmodulo(gemm_i, param.OHOW_fastdiv);
                const int64_t      base   = z_off + int64_t(n) * param.KPQ + col;
#pragma unroll
                for (unsigned int nn = 0; nn < mma_tiles_per_warp_n; nn++) {
#pragma unroll
                    for (unsigned int e = 0; e < 2; e++) {
                        const unsigned int ch = ch_base + nn * MMA_N + e;
                        res_v[h][nn][e] = n < param.n && ch < param.k ? residual[base + int64_t(ch) * param.PQ] : 0.0f;
                    }
                }
            }
        }
#pragma unroll
        for (unsigned int h = 0; h < 2; h++) {
            const unsigned int gemm_i = m_base + m * MMA_M + 8 * h;
            const unsigned int n      = fastdiv(gemm_i, param.OHOW_fastdiv);
            const unsigned int col    = fastmodulo(gemm_i, param.OHOW_fastdiv);
            if (n >= param.n) {
                continue;
            }
            const int64_t base = z_off + int64_t(n) * param.KPQ + col;
#pragma unroll
            for (unsigned int nn = 0; nn < mma_tiles_per_warp_n; nn++) {
#pragma unroll
                for (unsigned int e = 0; e < 2; e++) {
                    const unsigned int ch = ch_base + nn * MMA_N + e;
                    if (ch < param.k) {
                        const int64_t idx = base + int64_t(ch) * param.PQ;
                        // fused CONV_2D -> ADD(bias) -> ADD(residual): same f32 adds in the same order
                        float         v   = acc[m][nn][2 * h + e];
                        if (bias) {
                            v += bias[ch];
                        }
                        if (residual) {
                            v += res_v[h][nn][e];
                        }
                        output[idx] = v;
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(input);
    GGML_UNUSED(kernel);
    GGML_UNUSED(output);
    GGML_UNUSED(param);
    GGML_UNUSED(ksplit);
    GGML_UNUSED(bias);
    GGML_UNUSED(residual);
    NO_DEVICE_CODE;
#endif
}

static __global__ void conv2d_acc32_reduce_split_k(const float * __restrict__ partial, float * dst,
                                                   const unsigned int ne, const unsigned int ksplit,
                                                   const float * __restrict__ bias, const float * residual,
                                                   const unsigned int PQ, const unsigned int K) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int z = 0; z < ksplit; ++z) {
        sum += partial[int64_t(z) * ne + i];
    }
    if (bias) {
        sum += bias[(i / PQ) % K];
    }
    if (residual) {
        sum += residual[i];
    }
    dst[i] = sum;
}

void ggml_cuda_conv2d_implicit_acc32(ggml_backend_cuda_context & ctx, const half * X_H, const half * K_H, float * Y_D,
                                     const param_t & P, cudaStream_t st, const float * bias, const float * residual) {
    constexpr unsigned int BM = 256, BN = 128, BK = 32, WM = 128, WN = 32, NumThreads = 256;
    constexpr unsigned int shmem_bytes = (BM * BK + BK * BN) * 2 * sizeof(half);
    const unsigned int     BlocksM     = (P.n * P.Oh * P.Ow + BM - 1) / BM;
    const unsigned int     BlocksN     = (P.k + BN - 1) / BN;
    const unsigned int     nsm         = (unsigned int) ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    // split the channel loop over the grid when the output tiles alone do not fill the SMs; the f32 partial sums are
    // added in split order by one thread per output element
    unsigned int ksplit = 1;
    if (BlocksM * BlocksN < 2 * nsm) {
        int max_remaining_waves = -1, candidate = -1;
        int ks                  = min(20, int(nsm / (BlocksM * BlocksN)));
        if (ks < 2 && (BlocksM * BlocksN) % nsm < nsm * 4 / 5) {
            ks = 20;
        }
        for (int j = 2; j <= ks; j++) {
            const int remainder = (BlocksM * BlocksN * j) % nsm;
            if (P.c % (8 * j) == 0) {
                if (remainder == 0) {
                    candidate = j;
                    break;
                } else if (remainder > max_remaining_waves) {
                    max_remaining_waves = remainder;
                    candidate           = j;
                }
            }
        }
        if (candidate != -1) {
            ksplit = candidate;
        }
    }

    const auto kern = conv2d_implicit_kernel_acc32<BM, BN, BK, WM, WN, NumThreads>;
    CUDA_SET_SHARED_MEMORY_LIMIT(kern, shmem_bytes);
    const dim3 grid(BlocksN, BlocksM, ksplit);
    const dim3 block(WARP_SIZE * (BN / WN), BM / WM);
    if (ksplit == 1) {
        kern<<<grid, block, shmem_bytes, st>>>(X_H, K_H, Y_D, P, 1, bias, residual);
        return;
    }
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), size_t(ksplit) * P.NKPQ);
    kern<<<grid, block, shmem_bytes, st>>>(X_H, K_H, partial.get(), P, ksplit, nullptr, nullptr);
    conv2d_acc32_reduce_split_k<<<(P.NKPQ + 255) / 256, 256, 0, st>>>(partial.get(), Y_D, P.NKPQ, ksplit, bias,
                                                                      residual, P.PQ, P.k);
}
