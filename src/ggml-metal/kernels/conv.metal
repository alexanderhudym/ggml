#include "common.h"

typedef void (im2col_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
//    const int64_t IC = tgpg[0];
    const int64_t OH = tgpg[1];
    const int64_t OW = tgpg[2];

    const int64_t KH = ntg[1];
    const int64_t KW = ntg[2];

          int64_t in  = tpitg[0];
    const int64_t ikh = tpitg[1];
    const int64_t ikw = tpitg[2];

    const int64_t iic = tgpig[0];
    const int64_t ioh = tgpig[1];
    const int64_t iow = tgpig[2];

    const int64_t iiw = iow*args.s0 + ikw*args.d0 - args.p0;
    const int64_t iih = ioh*args.s1 + ikh*args.d1 - args.p1;

    int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*args.CHW + (iic*(KH*KW) + ikh*KW + ikw);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        while (in < args.N) {
            pdst[offset_dst] = 0.0f;
            offset_dst += ntg[0]*args.CHW*OH*OW;

            in += ntg[0];
        }
    } else {
        int64_t offset_src = in*args.ofs0 + iic*args.ofs1 + iih*args.IW + iiw;

        while (in < args.N) {
            pdst[offset_dst] = x[offset_src];

            offset_dst += ntg[0]*args.CHW*OH*OW;
            offset_src += ntg[0]*args.ofs0;

            in += ntg[0];
        }
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

template <typename T>
kernel void kernel_im2col_win(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const uint IC  = args.CHW / args.KHW;
    const uint OW  = args.OW;
    const uint OHW = args.OH*OW;
    const uint gid = (tgpig.y*tgpg.x + tgpig.x)*ntg.x + tpitg.x;
    if (gid >= args.N*OHW*IC) {
        return;
    }

    const uint iic = gid % IC;
    const uint pix = gid / IC;
    const uint in  = pix / OHW;
    const uint rp  = pix - in*OHW;
    const uint ioh = rp / OW;
    const uint iow = rp - ioh*OW;

    device const float * xc = x + in*args.ofs0 + iic*args.ofs1;
    device T * pdst = (device T *) dst + (uint64_t) pix*args.CHW + iic*args.KHW;

    for (int ikh = 0; ikh < args.KH; ++ikh) {
        const int iih = ((int) ioh + args.oh0)*args.s1 + ikh*args.d1 - args.p1;
        const bool row_ok = iih >= 0 && iih < args.IH;
        for (int ikw = 0; ikw < args.KW; ++ikw) {
            const int iiw = (int) iow*args.s0 + ikw*args.d0 - args.p0;
            *pdst++ = (row_ok && iiw >= 0 && iiw < args.IW) ? T(xc[iih*args.IW + iiw]) : T(0.0f);
        }
    }
}

template [[host_name("kernel_im2col_win_f32")]] kernel im2col_t kernel_im2col_win<float>;
template [[host_name("kernel_im2col_win_f16")]] kernel im2col_t kernel_im2col_win<half>;

kernel void kernel_im2col_tile_f16(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        threadgroup  half  * patch [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr int NP = 32;
    constexpr int NC = 32;

    const int KHW = args.KHW;
    const int IC  = args.CHW / KHW;
    const int PWD = (NP - 1)*args.s0 + (args.KW - 1)*args.d0 + 1;

    const int ow0 = tgpig.x*NP;
    const int in  = tgpig.y / args.OH;
    const int ioh = tgpig.y - in*args.OH;
    const int ic0 = tgpig.z*NC;

    const int iw0 = ow0*args.s0 - args.p0;

    for (int rr = tiitg/32; rr < NC*args.KH; rr += 8) {
        const int c   = rr / args.KH;
        const int kh  = rr - c*args.KH;
        const int ih  = (ioh + args.oh0)*args.s1 + kh*args.d1 - args.p1;
        const int ic  = ic0 + c;
        threadgroup half * prow = patch + rr*PWD;
        const bool ok = ic < IC && ih >= 0 && ih < args.IH;
        device const float * xrow = x + in*args.ofs0 + (uint64_t) ic*args.ofs1 + (uint) ih*args.IW;
        for (int col = tiitg%32; col < PWD; col += 32) {
            const int iw = iw0 + col;
            prow[col] = (ok && iw >= 0 && iw < args.IW) ? (half) xrow[iw] : (half) 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nc  = min(NC, IC - ic0);
    const int run = nc*KHW;
    const int np  = min(NP, args.OW - ow0);

    device half * out = (device half *) dst + ((uint64_t)(in*args.OH + ioh)*args.OW + ow0)*args.CHW + ic0*KHW;

    for (int kk = tiitg; kk < run; kk += 256) {
        const int c  = kk / KHW;
        const int r  = kk - c*KHW;
        const int kh = r / args.KW;
        const int kw = r - kh*args.KW;
        threadgroup const half * src = patch + (c*args.KH + kh)*PWD + kw*args.d0;
        for (int j = 0; j < np; ++j) {
            out[(uint64_t) j*args.CHW + kk] = src[j*args.s0];
        }
    }
}

// TODO: optimize
typedef void (im2col_ext_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int64_t KHW = (int64_t)args.KHW;

    const int64_t d   = tgpig[0] / args.CHW;
    const int64_t chw = tgpig[0] % args.CHW;
    const int64_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int64_t HW = tgpig[0] % KHW;

    const int64_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= args.N) {
        return;
    }

    const int64_t tpitg_1 = HW / args.KW;
    const int64_t tpitg_2 = HW % args.KW;

    const int64_t iiw = tgpig[2] * args.s0 + tpitg_2 * args.d0 - args.p0;
    const int64_t iih = tgpig[1] * args.s1 + tpitg_1 * args.d1 - args.p1;

    const int64_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * args.CHW +
        (tgpig_0 * KHW + tpitg_1 * args.KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = tpitg_0 * args.ofs0 + tgpig_0 * args.ofs1;
        pdst[offset_dst] = x[offset_src + iih * args.IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

template <typename T>
kernel void kernel_col2im_1d(
        constant ggml_metal_kargs_col2im_1d & args,
        device const T * col,
        device       T * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T_out * args.OC) {
        return;
    }

    const int t_out = idx % args.T_out;
    const int oc    = idx / args.T_out;
    const int t_abs = t_out + args.p0;  // absolute position in uncropped signal

    int t_in_min = (t_abs - args.K + args.s0) / args.s0;  // ceil((t_abs - K + 1) / s0)
    if (t_in_min < 0) {
        t_in_min = 0;
    }
    int t_in_max = t_abs / args.s0;
    if (t_in_max >= args.T_in) {
        t_in_max = args.T_in - 1;
    }

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        const int k = t_abs - t_in * args.s0;
        sum += float(col[(oc * args.K + k) + t_in * args.K_OC]);
    }

    dst[t_out + oc * args.T_out] = T(sum);
}

template [[host_name("kernel_col2im_1d_f32")]]  kernel void kernel_col2im_1d<float>(constant ggml_metal_kargs_col2im_1d &, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_col2im_1d_f16")]]  kernel void kernel_col2im_1d<half>(constant ggml_metal_kargs_col2im_1d &, device const half *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_col2im_1d_bf16")]] kernel void kernel_col2im_1d<bfloat>(constant ggml_metal_kargs_col2im_1d &, device const bfloat *, device bfloat *, uint, uint, uint);
#endif

template <typename TK>
kernel void kernel_conv_2d(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const uint threads_per_tg = ntg.x * ntg.y * ntg.z;
    const uint tg_index = (tgpig.z * tgpg.y + tgpig.y) * tgpg.x + tgpig.x;
    const uint local_thread = tpitg.z * (ntg.x * ntg.y) + tpitg.y * ntg.x + tpitg.x;
    const uint thread_index = tg_index * threads_per_tg + local_thread;
    const uint64_t total_threads = (uint64_t) threads_per_tg * tgpg.x * tgpg.y * tgpg.z;
    const uint64_t total_outputs = (uint64_t) args.N * args.OC * args.OH * args.OW;

    for (uint64_t index = thread_index; index < total_outputs; index += total_threads) {
        uint64_t tmp = index;

        const int32_t ow = tmp % args.OW; tmp /= args.OW;
        const int32_t oh = tmp % args.OH; tmp /= args.OH;
        const int32_t oc = tmp % args.OC; tmp /= args.OC;
        const int32_t  n = tmp;

        float acc = 0.0f;

        const int32_t base_x = ow*args.s0 - args.p0;
        const int32_t base_y = oh*args.s1 - args.p1;

        int32_t ky_start = 0;
        if (base_y < 0) {
            ky_start = (-base_y + args.d1 - 1)/args.d1;
        }
        int32_t ky_end = args.KH;
        const int32_t y_max = args.IH - 1 - base_y;
        if (y_max < 0) {
            ky_end = ky_start;
        } else if (base_y + (args.KH - 1)*args.d1 >= args.IH) {
            ky_end = min(ky_end, y_max/args.d1 + 1);
        }

        int32_t kx_start = 0;
        if (base_x < 0) {
            kx_start = (-base_x + args.d0 - 1)/args.d0;
        }
        int32_t kx_end = args.KW;
        const int32_t x_max = args.IW - 1 - base_x;
        if (x_max < 0) {
            kx_end = kx_start;
        } else if (base_x + (args.KW - 1)*args.d0 >= args.IW) {
            kx_end = min(kx_end, x_max/args.d0 + 1);
        }

        if (ky_start < ky_end && kx_start < kx_end) {
            const uint64_t src_base_n = (uint64_t) n  * args.nb13;
            const uint64_t w_base_oc  = (uint64_t) oc * args.nb03;

            for (int32_t ic = 0; ic < args.IC; ++ic) {
                const uint64_t src_base_nc = src_base_n + (uint64_t) ic * args.nb12;
                const uint64_t w_base_ocic = w_base_oc  + (uint64_t) ic * args.nb02;

                for (int32_t ky = ky_start; ky < ky_end; ++ky) {
                    const int32_t iy = base_y + ky*args.d1;
                    const uint64_t src_base_row = src_base_nc + (uint64_t) iy * args.nb11;
                    const uint64_t w_base_row   = w_base_ocic + (uint64_t) ky * args.nb01;

                    for (int32_t kx = kx_start; kx < kx_end; ++kx) {
                        const int32_t ix = base_x + kx*args.d0;
                        const uint64_t src_offs = src_base_row + (uint64_t) ix * args.nb10;
                        const uint64_t w_offs   = w_base_row   + (uint64_t) kx * args.nb00;

                        const float x = *(device const float *)(src + src_offs);
                        const float w = (float) (*(device const TK *)(weights + w_offs));

                        acc += x * w;
                    }
                }
            }
        }

        const uint64_t dst_offs =
            (uint64_t) n  * args.nb3 +
            (uint64_t) oc * args.nb2 +
            (uint64_t) oh * args.nb1 +
            (uint64_t) ow * args.nb0;

        *(device float *)(dst + dst_offs) = acc;
    }
}

template [[host_name("kernel_conv_2d_f32_f32")]]
kernel void kernel_conv_2d<float>(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_2d_f16_f32")]]
kernel void kernel_conv_2d<half>(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

kernel void kernel_conv_2d_mm_f16_f32(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int P   = args.OH*args.OW;
    const int KHW = args.KH*args.KW;

    const int in = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (P - r0 < NR0) ? (P - r0) : NR0;
    const short nr1 = (args.OC - r1 < NR1) ? (args.OC - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = tiitg % NL0;
    const short iy  = 8*(tiitg % NL1);

    const int p  = r0 + lr0;
    const int oh = p / args.OW;
    const int ow = p - oh*args.OW;

    const uint sw = args.nb10/4;
    const uint sh = args.nb11/4;
    const uint sc = args.nb12/4;

    device const float * x = (device const float *)(src + (uint64_t) in*args.nb13);
    const short lane = tiitg % 32;
    device const half * wsg[8];
    FOR_UNROLL (short j = 0; j < 8; j++) {
        const int row = r1 + 8*sgitg + j;
        wsg[j] = (device const half *)(weights + (uint64_t)(row < args.OC ? row : args.OC - 1)*args.nb03);
    }

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const int nchunk = (args.IC + NK - 1)/NK;
    const int nsteps = KHW*nchunk;

    half va[16];
    half vb[8];

    int  khw = 0;
    int  ic0 = 0;
    bool ok  = false;
    device const float * xp = x;

    auto tap = [&](int t_khw) {
        const int kh = t_khw / args.KW;
        const int kw = t_khw - kh*args.KW;
        const int ih = oh*args.s1 + kh*args.d1 - args.p1;
        const int iw = ow*args.s0 + kw*args.d0 - args.p0;
        ok = ih >= 0 && ih < args.IH && iw >= 0 && iw < args.IW;
        xp = x + (ok ? (uint) ih*sh + (uint) iw*sw : 0);
    };

    auto fetch = [&]() {
        const int ica = ic0 + 16*il0;
        FOR_UNROLL (short i = 0; i < 16; i++) {
            va[i] = (ok && ica + i < args.IC) ? (half) xp[(uint)(ica + i)*sc] : (half) 0.0f;
        }
        const int icb = ic0 + lane;
        FOR_UNROLL (short j = 0; j < 8; j++) {
            vb[j] = icb < args.IC ? wsg[j][icb*KHW + khw] : (half) 0.0f;
        }
    };

    tap(0);
    fetch();

    for (int t = 0; t < nsteps; ++t) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            FOR_UNROLL (short i = 0; i < 16; i++) {
                *(sa + 64*(8*(2*il0 + i/8) + sy) + 8*(i%8) + lx) = va[i];
            }
        }
        {
            const short ib = 4*(lane/8) + sgitg;
            FOR_UNROLL (short j = 0; j < 8; ++j) {
                *(sb + 64*ib + 8*j + lane%8) = vb[j];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t + 1 < nsteps) {
            ic0 += NK;
            if (ic0 >= args.IC) {
                ic0 = 0;
                ++khw;
                tap(khw);
            }
            fetch();
        }

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    device float * out = (device float *) dst + (uint64_t) in*args.OC*P;

    if (nr0 == NR0 && nr1 == NR1) {
        device float * C = out + (r0 + 32*(sgitg & 1)) + (uint64_t)(r1 + 16*(sgitg >> 1))*P;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*P*(i/4), P, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = out + r0 + (uint64_t)(r1 + j)*P;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;

                for (int i = 0; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}


typedef void (conv_transpose_1d_t)(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const float * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_1d(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const     T * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    // For output position j on the time axis, only input positions
    //   i such that i*s0 <= j < i*s0 + K
    // contribute -- i.e. i in [ceil((j - K + 1)/s0), floor(j/s0)]
    // intersected with [0, IL-1]. That's at most ceil(K/s0) values
    // (typically 2 for stride==K/2 transposed convs).
    const int32_t j  = tgpig[0];
    const int32_t s0 = args.s0;
    const int32_t K  = args.K;
    const int32_t IL = args.IL;

    int32_t i_min;
    {
        int32_t a = j - K + 1;
        i_min = a <= 0 ? 0 : (a + s0 - 1) / s0; // ceil(a/s0) for a>0
    }
    int32_t i_max = j / s0;
    if (i_max > IL - 1) i_max = IL - 1;

    float v = 0.0f;
    if (i_min <= i_max) {
        for (int64_t c = 0; c < args.IC; c++) {
            const int32_t kernel_offset = c * tgpg[1] * K + K * tgpig[1];
            const int32_t input_offset  = c * IL;

            for (int32_t i = i_min; i <= i_max; i++) {
                v += float(src0[kernel_offset + j - i * s0]) * src1[input_offset + i];
            }
        }
    }

    device float * dst_ptr = (device float *) (dst + tgpig[0] * args.nb0 + tgpig[1] * args.nb1);

    dst_ptr[0] = v;
}

template [[host_name("kernel_conv_transpose_1d_f32_f32")]]
kernel void kernel_conv_transpose_1d<float>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const float * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

template [[host_name("kernel_conv_transpose_1d_f16_f32")]]
kernel void kernel_conv_transpose_1d<half>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);


typedef void (conv_transpose_2d_t)(
        constant ggml_metal_kargs_conv_transpose_2d & args,
        device const float * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_2d(
        constant ggml_metal_kargs_conv_transpose_2d & args,
        device const T * src0,
        device const float * src1,
        device        char * dst,
        threadgroup float * shared_sum [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int64_t out_x = tgpig[0];
    const int64_t out_y = tgpig[1];
    const int64_t batch = tgpig[2] / args.OC;
    const int64_t out_c = tgpig[2] % args.OC;

    const int64_t kw = tpitg[0];
    const int64_t kh = tpitg[1];

    float v = 0.0f;

    for (int64_t in_c = 0; in_c < args.IC; in_c++) {
        int64_t in_y = out_y - kh;

        if (in_y < 0 || in_y % args.s0) continue;

        in_y /= args.s0;

        if (in_y >= args.IH) continue;

        int64_t in_x = out_x - kw;

        if (in_x < 0 || in_x % args.s0) continue;

        in_x /= args.s0;

        if (in_x >= args.IW) continue;

        const int64_t input_idx = (args.IW * args.IH) * (args.IC * batch + in_c) + (args.IW) * in_y + in_x;
        const int64_t kernel_idx = (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;

        v += (float)src0[kernel_idx] * src1[input_idx];
    }

    const uint tid = tpitg.y * ntg.x + tpitg.x;
    shared_sum[tid] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        const uint num_threads = ntg.x * ntg.y;
        for (uint i = 0; i < num_threads; i++) {
            total += shared_sum[i];
        }

        device float * dst_ptr = (device float *) (dst + batch*args.nb3 + out_c*args.nb2 + out_y * args.nb1 + out_x*args.nb0);
        dst_ptr[0] = total;
    }
}

template [[host_name("kernel_conv_transpose_2d_f32_f32")]]
kernel void kernel_conv_transpose_2d<float>(
    constant ggml_metal_kargs_conv_transpose_2d & args,
    device const float * src0,
    device const float * src1,
    device        char * dst,
    threadgroup float * shared_sum [[threadgroup(0)]],
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3   tpitg[[thread_position_in_threadgroup]],
    uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_transpose_2d_f16_f32")]]
kernel void kernel_conv_transpose_2d<half>(
    constant ggml_metal_kargs_conv_transpose_2d & args,
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    threadgroup float * shared_sum [[threadgroup(0)]],
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3   tpitg[[thread_position_in_threadgroup]],
    uint3     ntg[[threads_per_threadgroup]]);

// grid: x = C tile, y = OH, z = OW * N (for channel-contiguous layouts)
template <typename TK>
kernel void kernel_conv_2d_dw_tiled(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int32_t c = (int32_t)(tgpig.x * ntg.x + tpitg.x);
    if (c >= args.C) {
        return;
    }

    const int32_t oh = tgpig.y;
    const int32_t own = tgpig.z;
    const int32_t ow = own % args.OW;
    const int32_t n  = own / args.OW;

    const int32_t base_y = oh*args.s1 - args.p1;

    int32_t ky_start = 0;
    if (base_y < 0) {
        ky_start = (-base_y + args.d1 - 1)/args.d1;
    }
    int32_t ky_end = args.KH;
    const int32_t y_max = args.IH - 1 - base_y;
    if (y_max < 0) {
        ky_end = ky_start;
    } else if (base_y + (args.KH - 1)*args.d1 >= args.IH) {
        ky_end = min(ky_end, y_max/args.d1 + 1);
    }

    const int32_t base_x = ow*args.s0 - args.p0;

    int32_t kx_start = 0;
    if (base_x < 0) {
        kx_start = (-base_x + args.d0 - 1)/args.d0;
    }
    int32_t kx_end = args.KW;
    const int32_t x_max = args.IW - 1 - base_x;
    if (x_max < 0) {
        kx_end = kx_start;
    } else if (base_x + (args.KW - 1)*args.d0 >= args.IW) {
        kx_end = min(kx_end, x_max/args.d0 + 1);
    }

    float acc = 0.0f;

    if (ky_start < ky_end && kx_start < kx_end) {
        const uint64_t w_base   = (uint64_t) c * args.nb02;
        const uint64_t src_base = (uint64_t) n * args.nb13 + (uint64_t) c * args.nb12;

        for (int32_t ky = ky_start; ky < ky_end; ++ky) {
            const int32_t iy = base_y + ky*args.d1;
            const uint64_t src_row = src_base + (uint64_t) iy * args.nb11;
            const uint64_t w_row = w_base + (uint64_t) ky * args.nb01;

            for (int32_t kx = kx_start; kx < kx_end; ++kx) {
                const int32_t ix = base_x + kx*args.d0;
                const float x = *(device const float *)(src + src_row + (uint64_t) ix * args.nb10);
                const float w = (float)(*(device const TK *)(weights + w_row + (uint64_t) kx * args.nb00));
                acc += x * w;
            }
        }
    }

    const uint64_t dst_offs =
        (uint64_t) n  * args.nb3 +
        (uint64_t) c  * args.nb2 +
        (uint64_t) oh * args.nb1 +
        (uint64_t) ow * args.nb0;

    *(device float *)(dst + dst_offs) = acc;
}

// grid: x = OW tile, y = OH, z = C * N (for spatially-contiguous layouts)
template <typename TK>
kernel void kernel_conv_2d_dw(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int32_t oh = tgpig.y;
    const int32_t cn = tgpig.z;
    const int32_t c  = cn % args.C;
    const int32_t n  = cn / args.C;

    const int32_t base_y = oh*args.s1 - args.p1;

    int32_t ky_start = 0;
    if (base_y < 0) {
        ky_start = (-base_y + args.d1 - 1)/args.d1;
    }
    int32_t ky_end = args.KH;
    const int32_t y_max = args.IH - 1 - base_y;
    if (y_max < 0) {
        ky_end = ky_start;
    } else if (base_y + (args.KH - 1)*args.d1 >= args.IH) {
        ky_end = min(ky_end, y_max/args.d1 + 1);
    }

    const uint64_t w_base   = (uint64_t) c * args.nb02;
    const uint64_t src_base = (uint64_t) n * args.nb13 + (uint64_t) c * args.nb12;

    const int32_t ow = (int32_t)(tgpig.x * ntg.x + tpitg.x);
    if (ow >= args.OW) {
        return;
    }

    float acc = 0.0f;

    const int32_t base_x = ow*args.s0 - args.p0;

    int32_t kx_start = 0;
    if (base_x < 0) {
        kx_start = (-base_x + args.d0 - 1)/args.d0;
    }
    int32_t kx_end = args.KW;
    const int32_t x_max = args.IW - 1 - base_x;
    if (x_max < 0) {
        kx_end = kx_start;
    } else if (base_x + (args.KW - 1)*args.d0 >= args.IW) {
        kx_end = min(kx_end, x_max/args.d0 + 1);
    }

    if (ky_start < ky_end && kx_start < kx_end) {
        for (int32_t ky = ky_start; ky < ky_end; ++ky) {
            const int32_t iy = base_y + ky*args.d1;
            const uint64_t src_row = src_base + (uint64_t) iy * args.nb11;
            const uint64_t w_row = w_base + (uint64_t) ky * args.nb01;

            for (int32_t kx = kx_start; kx < kx_end; ++kx) {
                const int32_t ix = base_x + kx*args.d0;
                const float x = *(device const float *)(src + src_row + (uint64_t) ix * args.nb10);
                const float w = (float)(*(device const TK *)(weights + w_row + (uint64_t) kx * args.nb00));
                acc += x * w;
            }
        }
    }

    const uint64_t dst_offs =
        (uint64_t) n  * args.nb3 +
        (uint64_t) c  * args.nb2 +
        (uint64_t) oh * args.nb1 +
        (uint64_t) ow * args.nb0;

    *(device float *)(dst + dst_offs) = acc;
}

template [[host_name("kernel_conv_2d_dw_f32_f32")]]
kernel void kernel_conv_2d_dw<float>(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_2d_dw_f16_f32")]]
kernel void kernel_conv_2d_dw<half>(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_2d_dw_tiled_f32_f32")]]
kernel void kernel_conv_2d_dw_tiled<float>(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_2d_dw_tiled_f16_f32")]]
kernel void kernel_conv_2d_dw_tiled<half>(
        constant ggml_metal_kargs_conv_2d_dw & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_conv_3d(
        constant ggml_metal_kargs_conv_3d & args,
        device const  char * src0, // Weights [IC * OC, KD, KH, KW]
        device const  char * src1, // Inputs  [IC * N,  ID, IH, IW]
        device       char  * dst,  // Outputs [OC * N,  OD, OH, OW]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {

    // 1. Un-flatten the spatial dimension from Grid X
    int64_t spatial_idx = tgpig.x * 32 + tpitg.x;

    if (spatial_idx >= args.OW * args.OH * args.OD) {
        return; // Thread falls outside the spatial volume
    }

    int64_t od = spatial_idx / (args.OW * args.OH);
    int64_t oh = (spatial_idx / args.OW) % args.OH;
    int64_t ow = spatial_idx % args.OW;

    // 2. Map Y to Channels, Z to Batch
    int64_t oc = tgpig.y;
    int64_t batch_idx = tgpig.z;

    // 3. Calculate anchor coordinates in the Input volume
    int64_t i_w_base = ow * args.s0 - args.p0;
    int64_t i_h_base = oh * args.s1 - args.p1;
    int64_t i_d_base = od * args.s2 - args.p2;

    float sum = 0.0f;

    // 4. Gather Loop (Iterate over Input Channels -> Depth -> Height -> Width)
    for (int64_t ic = 0; ic < args.IC; ++ic) {

        // ggml packs batch and channel together in the 4th dimension
        int64_t src_cn_idx = batch_idx * args.IC + ic;
        int64_t w_cn_idx   = oc * args.IC + ic;

        for (int64_t kz = 0; kz < args.KD; ++kz) {
            int64_t id = i_d_base + kz * args.d2;
            if (id < 0 || id >= args.ID) continue; // Boundary check (Padding)

            for (int64_t ky = 0; ky < args.KH; ++ky) {
                int64_t ih = i_h_base + ky * args.d1;
                if (ih < 0 || ih >= args.IH) continue;

                for (int64_t kx = 0; kx < args.KW; ++kx) {
                    int64_t iw = i_w_base + kx * args.d0;
                    if (iw < 0 || iw >= args.IW) continue;

                    // Convert multi-dimensional coordinates to flat byte offsets
                    int64_t w_idx = kx*args.nb00 + ky*args.nb01 + kz*args.nb02 + w_cn_idx*args.nb03;
                    int64_t i_idx = iw*args.nb10 + ih*args.nb11 + id*args.nb12 + src_cn_idx*args.nb13;

                    // Dereference memory and cast weights to f32 if they were f16
                    float w_val = (float)*(device const T*)((device const char*)src0 + w_idx);
                    float i_val = *(device const float*)((device const char*)src1 + i_idx);

                    sum += w_val * i_val;
                }
            }
        }
    }

    // 5. Write the accumulated value out to RAM
    int64_t dst_cn_idx = batch_idx * args.OC + oc;
    int64_t d_idx = ow*args.nb0 + oh*args.nb1 + od*args.nb2 + dst_cn_idx*args.nb3;

    *(device float*)(dst + d_idx) = sum;
}

// Explicit instantiations so the JIT compiler can find them by name
template [[host_name("kernel_conv_3d_f32_f32")]]
kernel void kernel_conv_3d<float>(
    constant ggml_metal_kargs_conv_3d & args,
    device const char * src0,
    device const char * src1,
    device       char  * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);

// Explicit instantiation for f16 weights
template [[host_name("kernel_conv_3d_f16_f32")]]
kernel void kernel_conv_3d<half>(
    constant ggml_metal_kargs_conv_3d & args,
    device const char  * src0,
    device const char * src1,
    device       char  * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);


kernel void kernel_conv_2d_mmp_f16_f32(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;

    threadgroup half * sb    = (threadgroup half *)(shmem);
    threadgroup half * patch = (threadgroup half *)(shmem + 2048);

    const int KH  = args.KH;
    const int KW  = args.KW;
    const int KHW = KH*KW;
    const int PW  = NR0 + KW - 1;
    const int PR  = KH*PW;

    const int P  = args.OH*args.OW;
    const int in = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr1 = (args.OC - r1 < NR1) ? (args.OC - r1) : NR1;

    const int oh  = r0 / args.OW;
    const int ow0 = r0 - oh*args.OW;

    const uint sw = args.nb10/4;
    const uint sh = args.nb11/4;
    const uint sc = args.nb12/4;

    device const float * x = (device const float *)(src + (uint64_t) in*args.nb13);

    const short lane = tiitg % 32;
    device const half * wsg[8];
    FOR_UNROLL (short j = 0; j < 8; j++) {
        const int row = r1 + 8*sgitg + j;
        wsg[j] = (device const half *)(weights + (uint64_t)(row < args.OC ? row : args.OC - 1)*args.nb03);
    }

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const int npatch = NK*PR;

    for (int ic0 = 0; ic0 < args.IC; ic0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int rr = sgitg; rr < NK*KH; rr += 4) {
            const int c  = rr / KH;
            const int r  = rr - c*KH;
            const int ih = oh + r - args.p1;
            const int ic = ic0 + c;
            threadgroup half * prow = patch + rr*PW;
            if (ic < args.IC && ih >= 0 && ih < args.IH) {
                device const float * xrow = x + (uint) ic*sc + (uint) ih*sh;
                for (int col = lane; col < PW; col += 32) {
                    const int iw = ow0 + col - args.p0;
                    prow[col] = (iw >= 0 && iw < args.IW) ? (half) xrow[(uint) iw*sw] : (half) 0.0f;
                }
            } else {
                for (int col = lane; col < PW; col += 32) {
                    prow[col] = (half) 0.0f;
                }
            }
        }

        for (int khw = 0; khw < KHW; ++khw) {
            const int kh = khw / KW;
            const int kw = khw - kh*KW;

            half vb[8];
            const int icb = ic0 + lane;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                vb[j] = icb < args.IC ? wsg[j][icb*KHW + khw] : (half) 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                const short ib = 4*(lane/8) + sgitg;
                FOR_UNROLL (short j = 0; j < 8; ++j) {
                    *(sb + 64*ib + 8*j + lane%8) = vb[j];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half * lsma = patch + kh*PW + kw + 32*(sgitg%2);
            threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 8*i, PR, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*PR;
                lsmb += 4*64;
            }
        }
    }

    device float * out = (device float *) dst + (uint64_t) in*args.OC*P;

    if (nr1 == NR1) {
        device float * C = out + (r0 + 32*(sgitg & 1)) + (uint64_t)(r1 + 16*(sgitg >> 1))*P;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*P*(i/4), P, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = out + r0 + (uint64_t)(r1 + j)*P;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;

                for (int i = 0; i < NR0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}


kernel void kernel_conv_2d_wino_in(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * src,
        device       half * V,
        constant ggml_metal_kargs_conv_2d_chunk & chunk,
        threadgroup  half * patch [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr int NT  = 16;
    constexpr int NC  = 32;
    constexpr int PWD = 2*NT + 2;

    const int TW = (args.OW + 1)/2;
    const int TH = (args.OH + 1)/2;
    const int Tc = chunk.rows*TW;

    const int tx0 = tgpig.x*NT;
    const int grow = chunk.r0 + tgpig.y;
    const int in  = grow / TH;
    const int ty  = grow - in*TH;
    const int ic0 = tgpig.z*NC;

    const int iy0 = 2*ty  - args.p1;
    const int ix0 = 2*tx0 - args.p0;

    device const char * xn = src + (uint64_t) in*args.nb13;

    for (int rr = tiitg/32; rr < NC*4; rr += 8) {
        const int c  = rr / 4;
        const int r  = rr - 4*c;
        const int ih = iy0 + r;
        const int ic = ic0 + c;
        threadgroup half * prow = patch + rr*PWD;
        const bool ok = ic < args.IC && ih >= 0 && ih < args.IH;
        device const char * xrow = xn + (uint64_t) ic*args.nb12 + (uint64_t) ih*args.nb11;
        for (int col = tiitg%32; col < PWD; col += 32) {
            const int iw = ix0 + col;
            prow[col] = (ok && iw >= 0 && iw < args.IW) ? (half) *(device const float *)(xrow + (uint64_t) iw*args.nb10) : (half) 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = tiitg; e < NT*NC; e += 256) {
        const int j  = e / NC;
        const int c  = e - j*NC;
        const int tx = tx0 + j;
        const int ic = ic0 + c;
        if (tx >= TW || ic >= args.IC) {
            continue;
        }

        threadgroup const half * pd = patch + (c*4)*PWD + 2*j;
        float d[4][4];
        for (short r = 0; r < 4; ++r) {
            for (short q = 0; q < 4; ++q) {
                d[r][q] = pd[r*PWD + q];
            }
        }
        float t[4][4];
        for (short q = 0; q < 4; ++q) {
            t[0][q] = d[0][q] - d[2][q];
            t[1][q] = d[1][q] + d[2][q];
            t[2][q] = d[2][q] - d[1][q];
            t[3][q] = d[1][q] - d[3][q];
        }
        const int tt = tgpig.y*TW + tx;
        device half * v = V + (uint64_t) tt*args.IC + ic;
        const uint64_t sxi = (uint64_t) Tc*args.IC;
        for (short r = 0; r < 4; ++r) {
            v[(4*r + 0)*sxi] = (half)(t[r][0] - t[r][2]);
            v[(4*r + 1)*sxi] = (half)(t[r][1] + t[r][2]);
            v[(4*r + 2)*sxi] = (half)(t[r][2] - t[r][1]);
            v[(4*r + 3)*sxi] = (half)(t[r][1] - t[r][3]);
        }
    }
}

kernel void kernel_conv_2d_wino_w(
        constant ggml_metal_kargs_conv_2d & args,
        device const half * w,
        device       half * U,
        uint gid[[thread_position_in_grid]]) {
    if (gid >= (uint) args.OC*args.IC) {
        return;
    }
    const int ic = gid % args.IC;
    const int oc = gid / args.IC;

    device const half * g = w + ((uint64_t) oc*args.IC + ic)*9;
    float t[4][3];
    for (short q = 0; q < 3; ++q) {
        const float g0 = g[0*3 + q];
        const float g1 = g[1*3 + q];
        const float g2 = g[2*3 + q];
        t[0][q] = g0;
        t[1][q] = 0.5f*(g0 + g1 + g2);
        t[2][q] = 0.5f*(g0 - g1 + g2);
        t[3][q] = g2;
    }
    device half * u = U + (uint64_t) oc*args.IC + ic;
    const uint64_t sxi = (uint64_t) args.OC*args.IC;
    for (short r = 0; r < 4; ++r) {
        u[(4*r + 0)*sxi] = (half) t[r][0];
        u[(4*r + 1)*sxi] = (half)(0.5f*(t[r][0] + t[r][1] + t[r][2]));
        u[(4*r + 2)*sxi] = (half)(0.5f*(t[r][0] - t[r][1] + t[r][2]));
        u[(4*r + 3)*sxi] = (half) t[r][2];
    }
}

kernel void kernel_conv_2d_wino_out(
        constant ggml_metal_kargs_conv_2d & args,
        device const float * M,
        device       char  * dst,
        device const float * bias,
        constant     int   & has_bias,
        constant ggml_metal_kargs_conv_2d_chunk & chunk,
        uint gid[[thread_position_in_grid]]) {
    const int TW = (args.OW + 1)/2;
    const int TH = (args.OH + 1)/2;
    const int Tc = chunk.rows*TW;
    if (gid >= (uint) Tc*args.OC) {
        return;
    }
    const int tl = gid % Tc;
    const int oc = gid / Tc;

    device const float * m = M + (uint64_t) oc*Tc + tl;
    const uint64_t sxi = (uint64_t) args.OC*Tc;
    float a[4][4];
    for (short r = 0; r < 4; ++r) {
        for (short q = 0; q < 4; ++q) {
            a[r][q] = m[(4*r + q)*sxi];
        }
    }
    float b[2][4];
    for (short q = 0; q < 4; ++q) {
        b[0][q] = a[0][q] + a[1][q] + a[2][q];
        b[1][q] = a[1][q] - a[2][q] - a[3][q];
    }

    const int t  = chunk.r0*TW + tl;
    const int in = t / (TH*TW);
    const int rt = t - in*TH*TW;
    const int ty = rt / TW;
    const int tx = rt - ty*TW;
    const int oy = 2*ty;
    const int ox = 2*tx;

    const float bo = has_bias ? bias[oc] : 0.0f;

    device char * dn = dst + (uint64_t) in*args.nb3 + (uint64_t) oc*args.nb2;
    for (short r = 0; r < 2; ++r) {
        if (oy + r >= args.OH) {
            break;
        }
        device float * drow = (device float *)(dn + (uint64_t)(oy + r)*args.nb1);
        drow[ox] = b[r][0] + b[r][1] + b[r][2] + bo;
        if (ox + 1 < args.OW) {
            drow[ox + 1] = b[r][1] - b[r][2] - b[r][3] + bo;
        }
    }
}
