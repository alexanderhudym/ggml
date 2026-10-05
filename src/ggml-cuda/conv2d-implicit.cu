#include "convert.cuh"
#include "conv2d-implicit.cuh"
#include "conv2d-implicit-acc32.cuh"
#include "nhwc-stage.cuh"

#define CUDA_NCHW_2_NHWC_TILE_DIM 32
#define CUDA_NCHW_2_NHWC_BLOCK_NM 8
#define CUDA_NCHW_2_NHWC_BLOCK_ROWS 8
#define CUDA_NCHW_2_NHWC_BLOCK_C 64

constexpr uint32_t filter_swizzle_mask(uint32_t n, uint32_t m) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    int count = 0;
    while ((m >>= 1) != 0)
        ++count;
    return n << count;
}

template <typename src_T, typename dst_T>
static __global__ void NCHW2NHWC(const src_T *src, dst_T * dst, const int ne, const int ne00, const int ne01){

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    int x  = blockIdx.x * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.x;
    int y  = blockIdx.y * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.y;
    int tx = blockIdx.y * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.x;  // transpose block offset
    int ty = blockIdx.x * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.y;

    __shared__ src_T tile[CUDA_NCHW_2_NHWC_TILE_DIM][CUDA_NCHW_2_NHWC_TILE_DIM];
#pragma unroll
    for(int i = 0; i < CUDA_NCHW_2_NHWC_BLOCK_NM; ++i){

        const unsigned int imat = blockIdx.z * CUDA_NCHW_2_NHWC_BLOCK_NM + i;
        if(imat >= nmat)
            break;
#pragma unroll
        for (int j = 0; j < CUDA_NCHW_2_NHWC_TILE_DIM; j += CUDA_NCHW_2_NHWC_BLOCK_ROWS){
            if(x < ne01 && y + j < ne00){
                const int row = threadIdx.y+j;
                const int col = threadIdx.x ^ row;
                tile[row][col] = src[imat*n + (y+j)*ne01 + x];
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < CUDA_NCHW_2_NHWC_TILE_DIM; j += CUDA_NCHW_2_NHWC_BLOCK_ROWS){
            if(ty + j < ne01 && tx < ne00){
                const int col = (threadIdx.y+j) ^ threadIdx.x;
                dst[imat*n + (ty+j)*ne00 + tx] = ggml_cuda_cast<dst_T>(tile[threadIdx.x][col]);
            }
        }
        __syncthreads();  // the next matrix overwrites tile entries other warps may still read
    }
}

template <typename src_T, typename dst_T, const unsigned int mask, const int rs, const unsigned int blk_c>
static __global__ void NCHW2NHWC(const src_T *src, dst_T * dst, const int ne, const int ne00, const int ne01, param_t P){

    const int64_t n = ne00 * ne01;

    const unsigned int tx   = threadIdx.x;
    const unsigned int bx   = blockIdx.x;
    const unsigned int by   = blockIdx.y;

    const unsigned int blk = (bx+1) * blk_c <= ne00 ? blk_c : ne00 - bx * blk_c;

    __shared__ src_T tile[rs*blk_c];


#pragma unroll
    for (unsigned int j = 0; j < rs; j++){
        const int i = j * blk + tx;
        const unsigned int row = fastmodulo(i, P.RS_fastdiv);
        const unsigned int col = fastdiv(i, P.RS_fastdiv);
        const unsigned int src_index = by*n + bx * blk_c * rs + j * blk + tx;
        unsigned int idx = row * blk_c + col;
        idx =  idx ^ ((idx & mask) >> 4);
        if (src_index < ne && tx < blk) {
          tile[idx] = src[src_index];
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned int j = 0; j < rs; j++){
        const unsigned int dst_index = by*n + j*ne00 + bx*blk_c + tx;
        if(dst_index < ne && tx < blk){
          unsigned int idx = j*blk_c + tx;
          idx =  idx ^ ((idx & mask) >> 4);
          dst[dst_index] = ggml_cuda_cast<dst_T>(tile[idx]);
        }
    }
}



// staging into NHWC f16 straight from the producer's input (nearest 2x upscale / zero pad), see nhwc-stage.cuh
template <int MODE, bool VEC>
static __global__ void __launch_bounds__(NHWC_THREADS)
    nhwc_stage_f32_f16(const float * __restrict__ src, half * __restrict__ dst, const int C, const int W,
                       const int64_t HW, const nhwc_src_map m) {
    const int64_t n = blockIdx.z;
    nhwc_stage_tile<MODE, VEC>(src + n * C * m.HWs, dst + n * C * HW, C, W, HW, blockIdx.y * NHWC_TC,
                               (int64_t) blockIdx.x * NHWC_TP, nhwc_identity(), m);
}

void ggml_cuda_conv2d_nhwc_stage(const float * src, half * dst, int C, int W, int H, int N, int mode, int Ws, int Hs,
                                 int lp0, int lp1, cudaStream_t st) {
    GGML_ASSERT(C % 8 == 0 && (uintptr_t) dst % 16 == 0);
    const int64_t      HW = int64_t(W) * H;
    const nhwc_src_map m  = { Ws, Hs, lp0, lp1, int64_t(Ws) * Hs };
    const dim3         grid((unsigned) ((HW + NHWC_TP - 1) / NHWC_TP), (unsigned) ((C + NHWC_TC - 1) / NHWC_TC), N);
    if (mode == NHWC_SRC_PLAIN) {
        GGML_ASSERT(Ws == W && Hs == H);
        if (HW % 4 == 0 && (uintptr_t) src % 16 == 0) {
            nhwc_stage_f32_f16<NHWC_SRC_PLAIN, true><<<grid, NHWC_THREADS, 0, st>>>(src, dst, C, W, HW, m);
        } else {
            nhwc_stage_f32_f16<NHWC_SRC_PLAIN, false><<<grid, NHWC_THREADS, 0, st>>>(src, dst, C, W, HW, m);
        }
    } else if (mode == NHWC_SRC_UP2) {
        GGML_ASSERT(W == 2 * Ws && H == 2 * Hs);
        if (W % 4 == 0 && (uintptr_t) src % 8 == 0) {
            nhwc_stage_f32_f16<NHWC_SRC_UP2, true><<<grid, NHWC_THREADS, 0, st>>>(src, dst, C, W, HW, m);
        } else {
            nhwc_stage_f32_f16<NHWC_SRC_UP2, false><<<grid, NHWC_THREADS, 0, st>>>(src, dst, C, W, HW, m);
        }
    } else {
        GGML_ASSERT(mode == NHWC_SRC_PAD && lp0 >= 0 && lp1 >= 0 && lp0 + Ws <= W && lp1 + Hs <= H);
        nhwc_stage_f32_f16<NHWC_SRC_PAD, false><<<grid, NHWC_THREADS, 0, st>>>(src, dst, C, W, HW, m);
    }
}

// Filter [K][C][RS] f16 -> [K][RS][C] f16 (the layout the implicit GEMM reads), a 3x3 filter with the swizzled tile.
static void filter_to_krsc(const half * K_D, half * K_H, const param_t & P, cudaStream_t st) {
    const int64_t ne   = int64_t(P.c) * P.r * P.s * P.k;
    const int64_t ne00 = P.c;
    const int64_t ne01 = int64_t(P.r) * P.s;
    if (ne01 == 9) {
        constexpr unsigned int mask = filter_swizzle_mask(9, CUDA_NCHW_2_NHWC_BLOCK_C);
        const dim3 grid((ne00 + CUDA_NCHW_2_NHWC_BLOCK_C - 1) / CUDA_NCHW_2_NHWC_BLOCK_C, ne / (ne00 * ne01), 1);
        NCHW2NHWC<half, half, mask, 9, CUDA_NCHW_2_NHWC_BLOCK_C><<<grid, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(
            K_D, K_H, ne, ne00, ne01, P);
    } else {
        const dim3 grid((ne01 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                        (ne00 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                        (ne / (ne00 * ne01) + CUDA_NCHW_2_NHWC_BLOCK_NM - 1) / CUDA_NCHW_2_NHWC_BLOCK_NM);
        NCHW2NHWC<half, half><<<grid, dim3(CUDA_NCHW_2_NHWC_TILE_DIM, CUDA_NCHW_2_NHWC_BLOCK_ROWS), 0, st>>>(
            K_D, K_H, ne, ne00, ne01);
    }
}

void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * bias,
                                  const float * residual, float * out, const half * x_nhwc) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const half *        K_D    = (const half *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = out ? out : (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel) && kernel->type == GGML_TYPE_F16);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const unsigned int ST_X = p[0];  // stride_x
    const unsigned int ST_Y = p[1];  // stride_y
    const unsigned int PD_X = p[2];  // padding_x
    const unsigned int PD_Y = p[3];  // padding_y
    const unsigned int DL_X = p[4];  // dilation_x
    const unsigned int DL_Y = p[5];  // dilation_y

    GGML_ASSERT(p[6] == false);

    const unsigned int IW = input->ne[0];   // input_w
    const unsigned int IH = input->ne[1];   // input_h
    const unsigned int OW = dst->ne[0];     // output_w
    const unsigned int OH = dst->ne[1];     // output_h
    const unsigned int KW = kernel->ne[0];  // kernel_w
    const unsigned int KH = kernel->ne[1];  // kernel_h
    const unsigned int IC = input->ne[2];   // input_channels
    const unsigned int OC = kernel->ne[3];  // output_channels
    const unsigned int B  = input->ne[3];   // n_batches

    // input offsets of the next filter column, row and channel block
    const int64_t inc[3] = {
        int64_t(IC) * DL_X,
        int64_t(IW) * IC * DL_Y - int64_t(KW - 1) * IC * DL_X,
        -int64_t(KH - 1) * IW * IC * DL_Y - int64_t(KW - 1) * IC * DL_X,
    };

    const param_t P = { B, IC, IH, IW, OC, KH, KW, ST_Y, ST_X, PD_Y, PD_X, DL_Y, DL_X, OH, OW,
                        init_fastdiv_values(OW),
                        init_fastdiv_values(KW*KH),
                        init_fastdiv_values(OW*OH),
                        { inc[0], inc[1], inc[2] },
                        IC*IW,
                        IC*KW*KH,
                        OW*OH,
                        OC*OW*OH,
                        B*OC*OW*OH,
                        IC*IW*IH };

    // the tensor-core kernel reads the input as NHWC f16: from the producer, or staged here
    ggml_cuda_pool_alloc<half> input_f16(ctx.pool());
    if (!x_nhwc) {
        input_f16.alloc(ggml_nelements(input));
        ggml_cuda_conv2d_nhwc_stage(X_D, input_f16.get(), IC, IW, IH, B, NHWC_SRC_PLAIN, IW, IH, 0, 0, st);
    }
    ggml_cuda_pool_alloc<half> kernel_f16(ctx.pool());
    const half * K_H = K_D;
    if (KW * KH > 1) {
        kernel_f16.alloc(ggml_nelements(kernel));
        filter_to_krsc(K_D, kernel_f16.get(), P, st);
        K_H = kernel_f16.get();
    }

    ggml_cuda_conv2d_implicit_acc32(ctx, x_nhwc ? x_nhwc : input_f16.get(), K_H, Y_D, P, st, bias, residual);
}
