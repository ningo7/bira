#ifndef BFSRCNN_INFERENCE_H
#define BFSRCNN_INFERENCE_H

#include "bira_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BFSRCNN_TENSOR_INPUT = 0,
    BFSRCNN_TENSOR_HEAD,
    BFSRCNN_TENSOR_SHRINK1,
    BFSRCNN_TENSOR_SHRINK2,
    BFSRCNN_TENSOR_STATE0_FULL,
    BFSRCNN_TENSOR_STATE0_BINARY,
    BFSRCNN_TENSOR_STATE1_FULL,
    BFSRCNN_TENSOR_STATE1_BINARY,
    BFSRCNN_TENSOR_STATE2_FULL,
    BFSRCNN_TENSOR_STATE2_BINARY,
    BFSRCNN_TENSOR_STATE3_FULL,
    BFSRCNN_TENSOR_STATE3_BINARY,
    BFSRCNN_TENSOR_STATE4_FULL,
    BFSRCNN_TENSOR_STATE4_BINARY,
    BFSRCNN_TENSOR_STATE5_FULL,
    BFSRCNN_TENSOR_STATE5_BINARY,
    BFSRCNN_TENSOR_STATE6_FULL,
    BFSRCNN_TENSOR_STATE6_BINARY,
    BFSRCNN_TENSOR_STATE7_FULL,
    BFSRCNN_TENSOR_STATE7_BINARY,
    BFSRCNN_TENSOR_STATE8_FULL,
    BFSRCNN_TENSOR_SHUFFLED,
    BFSRCNN_TENSOR_OUTPUT,
    BFSRCNN_TENSOR_COUNT,
} bfsrcnn_tensor_id_t;

typedef enum {
    BFSRCNN_LAYER_HEAD = 0,
    BFSRCNN_LAYER_SHRINK1,
    BFSRCNN_LAYER_SHRINK2,
    BFSRCNN_LAYER_SHRINK3,
    BFSRCNN_LAYER_MAPPING0,
    BFSRCNN_LAYER_MAPPING1,
    BFSRCNN_LAYER_MAPPING2,
    BFSRCNN_LAYER_MAPPING3,
    BFSRCNN_LAYER_MAPPING4,
    BFSRCNN_LAYER_MAPPING5,
    BFSRCNN_LAYER_MAPPING6,
    BFSRCNN_LAYER_MAPPING7,
    BFSRCNN_LAYER_EXPAND,
    BFSRCNN_LAYER_FINAL,
    BFSRCNN_LAYER_COUNT,
} bfsrcnn_layer_id_t;

/*
 * Programmer-written whole-network inference. Passing all fourteen layer
 * descriptions in one call lets the generic engine optimize across layers.
 */
int bfsrcnn_infer(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    const void *input,
    void *output,
    bira_inference_layer_hook_t hook,
    void *hook_user);

/*
 * The same engine executes one layer when the submitted description contains
 * one layer. The caller binds all external inputs and outputs of that layer.
 */
int bfsrcnn_infer_layer(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    bfsrcnn_layer_id_t layer,
    const bira_tensor_binding_t *bindings,
    size_t binding_count,
    bira_inference_layer_hook_t hook,
    void *hook_user);

/*
 * Read-only metadata access is provided for tests, tracing and diagnostics.
 * The only definition remains in the handwritten inference C source.
 */
const bira_inference_desc_t *bfsrcnn_inference_description(void);

#ifdef __cplusplus
}
#endif

#endif
