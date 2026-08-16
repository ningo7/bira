#ifndef BIRA_OPS_H
#define BIRA_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bira_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

enum bira_tensor_format {
    BIRA_TENSOR_FULL_HWC = 0,
    BIRA_TENSOR_BINARY_HWC16 = 1,
};

typedef struct {
    void *data;
    uint16_t height;
    uint16_t width;
    uint16_t channels;
    uint32_t row_stride_bytes;
    uint8_t format;
    bool is_signed;
} bira_tensor_t;

typedef struct {
    const void *data;
    uint16_t rows;
    uint8_t bytes_per_row;
    uint16_t stride_bytes;
} bira_native_blob_t;

typedef struct {
    uint8_t kernel_height;
    uint8_t kernel_width;
    uint8_t padding_height;
    uint8_t padding_width;
    uint16_t groups;
    bool depthwise;
    uint8_t weight_precision;
    uint8_t post_mode;
    bool write_binary;
    bira_native_blob_t weight_low;
    bira_native_blob_t weight_high;
    bira_native_blob_t parameters;
    bira_tensor_t *binary_output;
} bira_conv2d_params_t;

typedef struct {
    uint8_t kernel_height;
    uint8_t kernel_width;
    uint8_t padding_height;
    uint8_t padding_width;
    uint8_t post_mode;
    bool write_binary;
    bira_native_blob_t weight;
    bira_native_blob_t parameters;
    /*
     * One signed int32 -N value per output pixel in row-major order.
     * correction_row_stride_bytes is the byte stride between image rows.
     * The operator packs these scalars into native Parameter Buffer rows.
     */
    const void *correction;
    uint32_t correction_row_stride_bytes;
    const bira_tensor_t *residual;
    bira_tensor_t *binary_output;
} bira_binary_conv2d_params_t;

typedef struct {
    uint16_t tile_height;
    uint16_t tile_width;
    uint16_t tile_count;
} bira_tile_plan_t;

/*
 * ISA generic operators support stride=1 and dilation=1. Tensors use HWC
 * ordering; channels must be <=16 or a multiple of 16 so every external
 * channel block maps directly to one native SPAD row.
 */
int bira_plan_conv2d(
    const bira_runtime_t *runtime,
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    const bira_conv2d_params_t *params,
    bira_tile_plan_t *plan);

int bira_conv2d(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_conv2d_params_t *params);

int bira_binary_conv2d(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_binary_conv2d_params_t *params);

int bira_linear(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_conv2d_params_t *params);

#ifdef __cplusplus
}
#endif

#endif
