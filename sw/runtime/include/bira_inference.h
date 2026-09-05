#ifndef BIRA_INFERENCE_H
#define BIRA_INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bira_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BIRA_TENSOR_NONE UINT16_MAX

/*
 * Tensor layouts describe the accelerator-visible representation. They are
 * deliberately independent from a framework such as PyTorch or ONNX.
 */
enum bira_tensor_layout {
    BIRA_LAYOUT_FULL_DENSE16 = 0,
    BIRA_LAYOUT_FULL_HWC16 = 1,
    BIRA_LAYOUT_BINARY_HWC16 = 2,
    BIRA_LAYOUT_FULL_PACKED_PIXEL_PAIR = 3,
};

enum bira_layer_type {
    BIRA_OP_CONV2D = 0,
    BIRA_OP_DEPTHWISE_CONV2D = 1,
    BIRA_OP_BINARY_CONV2D = 2,
    BIRA_OP_LINEAR = 3,
    BIRA_OP_COLUMN_REDUCE_CONV2D = 4,
};

typedef struct {
    const char *name;
    uint16_t height;
    uint16_t width;
    uint16_t channels;
    uint8_t layout;
    bool is_signed;
} bira_tensor_desc_t;

/*
 * One descriptor is one programmer-visible layer operator. Tensor IDs express
 * dataflow; the application never assigns SPAD rows, contexts or DMA commands.
 * Weight and parameter blobs are generated data, already in native BIRA
 * layout. A binary convolution uses residual_tensor as its full-precision
 * residual input. Operators may produce both a full and a binary tensor.
 */
typedef struct {
    const char *name;
    uint8_t type;
    uint16_t input_tensor;
    uint16_t residual_tensor;
    uint16_t output_tensor;
    uint16_t binary_output_tensor;

    uint8_t kernel_height;
    uint8_t kernel_width;
    uint8_t padding_height;
    uint8_t padding_width;
    /*
     * Zero means derive from output_tensor. Fused layout-changing operators
     * such as pixel shuffle set the accelerator-side output shape explicitly.
     */
    uint16_t execution_output_height;
    uint16_t execution_output_width;
    uint16_t execution_output_channels;
    uint8_t weight_precision;
    uint8_t post_mode;
    bool input_signed;
    bool shuffle_pack2;

    bira_native_blob_t weight_low;
    bira_native_blob_t weight_high;
    bira_native_blob_t parameters;
    /* Packed per-pixel -N biases used to initialize binary Accumulator rows. */
    bira_native_blob_t correction;
} bira_layer_desc_t;

/*
 * One inference description is one API submission. The programmer passes one
 * layer for standalone execution or several consecutive layers for inter-layer
 * optimization. Only tensors crossing this submitted range need bindings.
 */
typedef struct {
    const char *name;
    const bira_tensor_desc_t *tensors;
    size_t tensor_count;
    const bira_layer_desc_t *layers;
    size_t layer_count;
} bira_inference_desc_t;

#define BIRA_INFERENCE_MAX_LAYERS 64u
#define BIRA_INFERENCE_MAX_TENSORS 128u
#define BIRA_INFERENCE_MAX_LOADS_PER_LAYER 8u
#define BIRA_INFERENCE_MAX_STORES_PER_LAYER 2u

enum bira_onchip_space {
    BIRA_SPACE_NONE = 0,
    BIRA_SPACE_FULL = 1,
    BIRA_SPACE_BINARY = 2,
};

typedef struct {
    uint16_t tensor;
    void *data;
} bira_tensor_binding_t;

typedef struct {
    bool allocated;
    bool external_input;
    bool external_output;
    uint8_t space;
    uint16_t first_layer;
    uint16_t last_layer;
    uint16_t base_row;
    uint16_t rows;
} bira_tensor_plan_t;

typedef struct {
    uint16_t tensor;
    const void *static_address;
    uint8_t role;
    uint16_t local_row_offset;
    uint16_t rows;
    uint8_t bytes_per_row;
    uint16_t dram_stride_bytes;
    uint8_t local_stride_rows;
} bira_planned_load_t;

typedef struct {
    uint16_t tensor;
    uint8_t role;
    uint16_t local_row_offset;
    uint16_t rows;
    uint8_t bytes_per_row;
    uint16_t dram_stride_bytes;
    uint8_t local_stride_rows;
} bira_planned_store_t;

typedef struct {
    const bira_layer_desc_t *layer_desc;
    bira_shape_t shape;
    bira_mode_t mode;
    uint16_t address_mask;
    uint16_t base_rows[BIRA_ADDR_ROLE_COUNT];
    bira_planned_load_t
        loads[BIRA_INFERENCE_MAX_LOADS_PER_LAYER];
    size_t load_count;
    bira_planned_store_t
        stores[BIRA_INFERENCE_MAX_STORES_PER_LAYER];
    size_t store_count;
} bira_planned_layer_t;

typedef struct {
    const bira_inference_desc_t *inference;
    const char *name;
    uint16_t layer_count;
    bira_tensor_plan_t tensors[BIRA_INFERENCE_MAX_TENSORS];
    bira_planned_layer_t layers[BIRA_INFERENCE_MAX_LAYERS];
    uint16_t full_rows_used;
    uint16_t binary_rows_used;
    uint16_t eliminated_tensor_transfers;
    bool prepared;
} bira_inference_workspace_t;

struct bira_inference_context;

typedef void (*bira_inference_layer_hook_t)(
    struct bira_inference_context *context,
    size_t layer_index,
    bool before,
    void *user);

typedef struct bira_inference_context {
    const bira_inference_workspace_t *workspace;
    bira_runtime_t *runtime;
    const bira_tensor_binding_t *bindings;
    size_t binding_count;
    bira_inference_layer_hook_t hook;
    void *hook_user;
} bira_inference_context_t;

uint32_t bira_tensor_rows(const bira_tensor_desc_t *tensor);

uint8_t bira_tensor_bytes_per_row(
    const bira_tensor_desc_t *tensor);

size_t bira_tensor_bytes(const bira_tensor_desc_t *tensor);

const char *bira_inference_layer_name(
    const bira_inference_desc_t *inference,
    size_t layer_index);

int bira_inference(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    const bira_inference_desc_t *inference,
    const bira_tensor_binding_t *bindings,
    size_t binding_count,
    bira_inference_layer_hook_t hook,
    void *hook_user);

#ifdef __cplusplus
}
#endif

#endif
