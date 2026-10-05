#pragma once
// f32 NCHW -> f16 NHWC staging tile for the tensor-core implicit-GEMM conv (conv2d-implicit.cu), shared by
//  - the plain / nearest-upscaled / padded staging pass (conv2d-implicit.cu, ggml_cuda_conv2d_nhwc_stage), and
//  - the GroupNorm(+gamma,beta,SiLU) apply kernel that writes the conv input directly (norm.cu).
// Every value is rounded f32 -> f16 to nearest, so a conv fed from a producer sees the same input as one that stages
// the producer's f32 output itself.
#include "common.cuh"

#define NHWC_TC      64   // channels per tile
#define NHWC_TP      64   // positions per tile
#define NHWC_THREADS 256

struct nhwc_identity {
    __device__ __forceinline__ float operator()(int /*row*/, float v) const { return v; }
};

// Source of the staged tensor: the conv input itself (NHWC_SRC_PLAIN), or the input of the op that produced it, folded
// into the staging read: NHWC_SRC_UP2 = ggml_upscale NEAREST by exactly 2 (output (y, x) reads src (y/2, x/2)),
// NHWC_SRC_PAD = ggml_pad with zeros (output (y, x) reads src (y - lp1, x - lp0), 0 outside).
enum { NHWC_SRC_PLAIN = 0, NHWC_SRC_UP2 = 1, NHWC_SRC_PAD = 2 };
struct nhwc_src_map {
    int     Ws, Hs;    // source width/height
    int     lp0, lp1;  // left/top zero padding (PAD)
    int64_t HWs;       // source channel plane = Ws * Hs
};

// One NHWC_TC x NHWC_TP tile of one sample: dst[p][c] = f(c - c0, src[c][q(p)]) for c in [c0, c0 + TC), p in
// [p0, p0 + TP). src/dst point at the sample. Output is W x H (HW = W * H positions), q(p) as described above.
// VEC (float4 / float2 loads): PLAIN needs HW % 4 == 0 and 16-byte aligned src; UP2 needs W % 4 == 0 and 8-byte
// aligned src; PAD has no vector path. C % 8 == 0 and 16-byte aligned dst are always required (16-byte stores).
template <int MODE, bool VEC, typename F>
static __device__ __forceinline__ void nhwc_stage_tile(const float * __restrict__ src, half * __restrict__ dst,
                                                       const int C, const int W, const int64_t HW, const int c0,
                                                       const int64_t p0, const F & f, const nhwc_src_map m) {
    static_assert(MODE == NHWC_SRC_PLAIN || MODE == NHWC_SRC_UP2 || MODE == NHWC_SRC_PAD, "MODE");
    static_assert(!(VEC && MODE == NHWC_SRC_PAD), "PAD is scalar");
    static_assert(NHWC_THREADS == 256 && NHWC_TC == 64 && NHWC_TP == 64, "tile");
    __shared__ __align__(16) float tile[NHWC_TC][NHWC_TP + 4];

    const int     tid  = threadIdx.x;
    // load: 16 float4 per row, 16 rows per pass
    const int     vr   = tid / 16;
    const int     vc   = tid % 16;
    const int64_t p    = p0 + 4 * vc;
    const bool    full = p0 + NHWC_TP <= HW;

    // source offsets of positions p .. p+3 (-1: zero padding / outside)
    int64_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t pj = p + j;
        if constexpr (MODE == NHWC_SRC_PLAIN) {
            q[j] = pj;
        } else {
            const int64_t y = pj / W;
            const int     x = (int) (pj - y * W);
            if constexpr (MODE == NHWC_SRC_UP2) {
                q[j] = (y / 2) * m.Ws + x / 2;
            } else {
                const int64_t ys = y - m.lp1;
                const int     xs = x - m.lp0;
                q[j]             = ys >= 0 && ys < m.Hs && xs >= 0 && xs < m.Ws ? ys * m.Ws + xs : -1;
            }
        }
        if (pj >= HW) {
            q[j] = -1;
        }
    }

#pragma unroll
    for (int k = 0; k < NHWC_TC / 16; ++k) {
        const int r = vr + 16 * k;
        const int c = c0 + r;
        float4    v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c < C) {
            const float * row = src + (int64_t) c * m.HWs;
            if (VEC && full) {
                if constexpr (MODE == NHWC_SRC_PLAIN) {
                    v = *(const float4 *) (row + q[0]);
                } else if constexpr (MODE == NHWC_SRC_UP2) {
                    const float2 u = *(const float2 *) (row + q[0]);
                    v              = make_float4(u.x, u.x, u.y, u.y);
                }
            } else {
                v.x = q[0] >= 0 ? row[q[0]] : 0.0f;
                v.y = q[1] >= 0 ? row[q[1]] : 0.0f;
                v.z = q[2] >= 0 ? row[q[2]] : 0.0f;
                v.w = q[3] >= 0 ? row[q[3]] : 0.0f;
            }
            v.x = f(r, v.x);
            v.y = f(r, v.y);
            v.z = f(r, v.z);
            v.w = f(r, v.w);
        }
        *(float4 *) &tile[r][4 * vc] = v;
    }
    __syncthreads();

    // store: 8 channels (16 bytes) of one position per thread, 8 threads per position, 32 positions per pass
    const int cq = tid % 8;
    const int pp = tid / 8;
    const int cc = c0 + cq * 8;
#pragma unroll
    for (int k = 0; k < NHWC_TP / 32; ++k) {
        const int     pl = pp + 32 * k;
        const int64_t gp = p0 + pl;
        if (gp < HW && cc < C) {
            half2 h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h[j] = __floats2half2_rn(tile[cq * 8 + 2 * j][pl], tile[cq * 8 + 2 * j + 1][pl]);
            }
            *(uint4 *) (dst + gp * C + cc) = *(const uint4 *) h;
        }
    }
}
