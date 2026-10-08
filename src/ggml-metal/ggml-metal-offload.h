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

    void *  call;
};

typedef bool (*ggml_metal_offload_match_t)(void * user, const char * name, int64_t K, int64_t N, int64_t M, struct ggml_metal_offload_plan * plan);
typedef bool (*ggml_metal_offload_run_t)(void * user, void * call);

GGML_BACKEND_API void ggml_metal_offload_set(void * user, int timeout_ms, ggml_metal_offload_match_t match, ggml_metal_offload_run_t run);

GGML_BACKEND_API void ggml_metal_offload_remove(void);

GGML_BACKEND_API void ggml_metal_offload_stats(int64_t * served, int64_t * recomputed, bool * off);

struct ggml_metal_split_weight {
    const char * name;
    int32_t      type;
    int64_t      ne0;
    int64_t      ne1;
    int64_t      gpu_rows;
};

typedef bool (*ggml_metal_split_read_t)(void * user, int32_t index, size_t offset, void * dst, size_t size);

GGML_BACKEND_API bool ggml_metal_split_set(void * user, const struct ggml_metal_split_weight * w, int32_t n, ggml_metal_split_read_t read);

GGML_BACKEND_API void ggml_metal_split_remove(void);

GGML_BACKEND_API void ggml_metal_split_stats(int32_t * tensors, size_t * dropped, int64_t * reads, bool * failed);

#ifdef __cplusplus
}
#endif
