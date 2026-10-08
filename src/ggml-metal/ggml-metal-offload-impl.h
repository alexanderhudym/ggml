#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

struct ggml_metal_offload_call {
    int64_t  gpu_rows;
    float    scale;
    int32_t  n_seg;
    int32_t  seg_end[4];
    uint64_t seq;
    uint64_t in_stride;
    uint64_t out_stride;

    struct ggml_metal_buffer_id in;
    struct ggml_metal_buffer_id out;
    struct ggml_metal_buffer_id scratch;
    struct ggml_metal_buffer_id fence;
    struct ggml_metal_buffer_id slot;

    void * event;
};

void ggml_metal_offload_prepare(ggml_metal_device_t dev, struct ggml_cgraph * gf);

void ggml_metal_offload_finish(ggml_metal_cmd_buf_t cmd_buf_last);

const struct ggml_metal_offload_call * ggml_metal_offload_find(const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
