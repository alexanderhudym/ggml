#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

struct ggml_metal_split_call {
    uint64_t fseq;
    int32_t  index;
    uint64_t gpu_bytes;
    uint64_t tail_bytes;

    struct ggml_metal_buffer_id slot;
    struct ggml_metal_buffer_id fence;

    void * event;
};

bool   ggml_metal_split_active(void);

size_t ggml_metal_split_alloc_size(const struct ggml_tensor * t);
bool   ggml_metal_split_matches(const struct ggml_tensor * t);

void   ggml_metal_split_record(void * ctx, const struct ggml_tensor * t);
void   ggml_metal_split_drop(void * ctx);

bool   ggml_metal_split_lookup(const void * ctx, const struct ggml_tensor * t, size_t * gpu_bytes, int32_t * index, size_t * view_offs);
bool   ggml_metal_split_is_alloc(const struct ggml_tensor * t);
size_t ggml_metal_split_extent(const void * ctx, const struct ggml_tensor * t);

bool   ggml_metal_split_read_tail(int32_t index, size_t offset, void * dst, size_t size);

void   ggml_metal_split_prepare(ggml_metal_device_t dev, struct ggml_cgraph * gf);
void   ggml_metal_split_finish(ggml_metal_cmd_buf_t cmd_buf_last);

const struct ggml_metal_split_call * ggml_metal_split_find(const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
