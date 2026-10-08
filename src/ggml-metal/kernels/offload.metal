#include "common.h"

#if __METAL_VERSION__ >= 320

#pragma METAL internals : enable
namespace metal {
constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(2);
}

kernel void kernel_offload_mean(
        constant ggml_metal_kargs_offload & args,
        device const char  * src1,
        device       float * mean,
        uint2 gid[[thread_position_in_grid]]) {
    const int k = gid.x;
    const int s = gid.y;

    if (k >= args.K || s >= args.n_seg) {
        return;
    }

    const int t0 = s == 0 ? 0 : args.seg_end[s - 1];
    const int t1 = args.seg_end[s];

    float sum = 0.0f;
    for (int t = t0; t < t1; ++t) {
        sum += ((device const float *) (src1 + (uint64_t) t*args.nb11))[k];
    }

    mean[(uint64_t) s*args.K + k] = sum/float(t1 - t0);
}

kernel void kernel_offload_cvt(
        constant ggml_metal_kargs_offload & args,
        device const char  * src1,
        device const float * mean,
        volatile coherent(system) device uint * dst,
        volatile coherent(system) device uint * mean_in,
        uint2 gid[[thread_position_in_grid]]) {
    const int c = gid.x;
    const int t = gid.y;

    if (4*c < args.K && t < args.M + args.n_seg) {
        if (t < args.M) {
            float4 v = ((device const float4 *) (src1 + (uint64_t) t*args.nb11))[c];

            if (args.n_seg > 0) {
                int s = 0;
                while (t >= args.seg_end[s]) {
                    ++s;
                }

                const half4 m = half4(((device const float4 *) (mean + (uint64_t) s*args.K))[c]*args.scale);

                v -= float4(m)*args.inv_scale;
            }

            const half4 h = half4(v*args.scale);

            volatile coherent(system) device uint * row = dst + (uint64_t) t*(args.in_stride/4) + 2*(uint64_t) c;

            row[0] = as_type<uint>(h.xy);
            row[1] = as_type<uint>(h.zw);
        } else {
            const int s = t - args.M;

            const half4 m = half4(((device const float4 *) (mean + (uint64_t) s*args.K))[c]*args.scale);

            volatile coherent(system) device uint * row = mean_in + (uint64_t) s*(args.mean_stride/4) + 2*(uint64_t) c;

            row[0] = as_type<uint>(m.xy);
            row[1] = as_type<uint>(m.zw);
        }
    }

    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}

kernel void kernel_offload_fence(
        constant ggml_metal_kargs_offload & args,
        volatile coherent(system) device uint * fence) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    fence[0] = args.seq;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}

kernel void kernel_offload_merge(
        constant ggml_metal_kargs_offload & args,
        device const char  * out,
        device const float * center,
        device       float * dst,
        device atomic_uint * flag,
        uint2 gid[[thread_position_in_grid]]) {
    const int NG = args.N - args.G;
    const int c  = gid.x;
    const int t  = gid.y;

    if (4*c >= NG || t >= args.M) {
        return;
    }

    const ushort4 b = ((device const ushort4 *) (out + (uint64_t) t*args.out_stride))[c];

    if (any((b & ushort4(0x7fff)) >= ushort4(0x7800))) {
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
    }

    float4 o = float4(as_type<half4>(b))*args.inv_scale;

    if (args.n_seg > 0) {
        int s = 0;
        while (t >= args.seg_end[s]) {
            ++s;
        }

        const float4 m = ((device const float4 *) (center + (uint64_t) s*NG))[c];

        if (any((as_type<uint4>(m) & uint4(0x7fffffff)) >= uint4(0x7f800000))) {
            atomic_store_explicit(flag, 1u, memory_order_relaxed);
        }

        o += m;
    }

    ((device float4 *) (dst + (uint64_t) t*args.N + args.G))[c] = o;
}

kernel void kernel_offload_indirect(
        constant ggml_metal_kargs_offload & args,
        device uint * slot) {
    const bool redo = ((volatile device uint *) slot)[0] == 2u || slot[1] != 0u;

    slot[2] = redo ? uint(args.tgx) : 0u;
    slot[3] = redo ? uint(args.tgy) : 0u;
    slot[4] = redo ? 1u : 0u;
    slot[5] = redo ? uint(args.tgd) : 0u;
    slot[6] = redo ? 1u : 0u;
    slot[7] = redo ? 1u : 0u;
}

#endif
