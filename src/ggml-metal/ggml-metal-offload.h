#pragma once

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_OFFLOAD_MAX_SEGMENTS 4

struct ggml_metal_offload_plan {
    int64_t gpu_rows;
    float   in_scale;
    int32_t n_seg;
    int64_t seg[GGML_METAL_OFFLOAD_MAX_SEGMENTS];

    void *  in;
    size_t  in_stride;

    void *  out;
    size_t  out_stride;

    void *  mean_in;
    size_t  mean_in_offset;
    size_t  mean_in_stride;

    void *  center;

    void *  call;
};

typedef bool (*ggml_metal_offload_match_t)(void * user, const char * name, int64_t K, int64_t N, int64_t M, struct ggml_metal_offload_plan * plan);
typedef bool (*ggml_metal_offload_run_t)(void * user, void * call);

GGML_BACKEND_API void ggml_metal_offload_set(void * user, int timeout_ms, ggml_metal_offload_match_t match, ggml_metal_offload_run_t run);

GGML_BACKEND_API void ggml_metal_offload_remove(void);

GGML_BACKEND_API void ggml_metal_offload_stats(int64_t * served, int64_t * recomputed, bool * off);

#ifdef __cplusplus
}
#endif
