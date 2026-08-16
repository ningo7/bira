/*
 * Programmer-written BFSRCNN inference.
 *
 * The parameter exporter emits bfsrcnn_bira_data.{h,c}. The programmer writes
 * the layer and tensor descriptions below, then submits one or more consecutive
 * layers to the generic inference engine. No network C code, SPAD address or
 * ISA command is generated from the model.
 */
#include "bfsrcnn_inference.h"
#include "bfsrcnn_bira_data.h"

#include <stddef.h>

#define NATIVE_BLOB(symbol, row_count, row_bytes) \
    { \
        .data = (symbol), \
        .rows = (row_count), \
        .bytes_per_row = (row_bytes), \
        .stride_bytes = (row_bytes), \
    }

#define EMPTY_BLOB \
    { \
        .data = NULL, \
        .rows = 0u, \
        .bytes_per_row = 0u, \
        .stride_bytes = 0u, \
    }

#define FULL_TENSOR(tensor_name, tensor_height, tensor_width, tensor_channels) \
    { \
        .name = (tensor_name), \
        .height = (tensor_height), \
        .width = (tensor_width), \
        .channels = (tensor_channels), \
        .layout = BIRA_LAYOUT_FULL_HWC16, \
        .is_signed = true, \
    }

#define BINARY_TENSOR(tensor_name) \
    { \
        .name = (tensor_name), \
        .height = BFSRCNN_BIRA_INPUT_HEIGHT, \
        .width = BFSRCNN_BIRA_INPUT_WIDTH, \
        .channels = 16u, \
        .layout = BIRA_LAYOUT_BINARY_HWC16, \
        .is_signed = false, \
    }

static const bira_tensor_desc_t bfsrcnn_tensors[] = {
    [BFSRCNN_TENSOR_INPUT] = {
        .name = "input",
        .height = BFSRCNN_BIRA_INPUT_HEIGHT,
        .width = BFSRCNN_BIRA_INPUT_WIDTH,
        .channels = 1u,
        .layout = BIRA_LAYOUT_FULL_DENSE16,
        .is_signed = false,
    },
    [BFSRCNN_TENSOR_HEAD] =
        FULL_TENSOR(
            "head",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            48u),
    [BFSRCNN_TENSOR_SHRINK1] =
        FULL_TENSOR(
            "shrink1",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            32u),
    [BFSRCNN_TENSOR_SHRINK2] =
        FULL_TENSOR(
            "shrink2",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            32u),
    [BFSRCNN_TENSOR_STATE0_FULL] =
        FULL_TENSOR(
            "state0.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE0_BINARY] =
        BINARY_TENSOR("state0.binary"),
    [BFSRCNN_TENSOR_STATE1_FULL] =
        FULL_TENSOR(
            "state1.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE1_BINARY] =
        BINARY_TENSOR("state1.binary"),
    [BFSRCNN_TENSOR_STATE2_FULL] =
        FULL_TENSOR(
            "state2.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE2_BINARY] =
        BINARY_TENSOR("state2.binary"),
    [BFSRCNN_TENSOR_STATE3_FULL] =
        FULL_TENSOR(
            "state3.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE3_BINARY] =
        BINARY_TENSOR("state3.binary"),
    [BFSRCNN_TENSOR_STATE4_FULL] =
        FULL_TENSOR(
            "state4.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE4_BINARY] =
        BINARY_TENSOR("state4.binary"),
    [BFSRCNN_TENSOR_STATE5_FULL] =
        FULL_TENSOR(
            "state5.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE5_BINARY] =
        BINARY_TENSOR("state5.binary"),
    [BFSRCNN_TENSOR_STATE6_FULL] =
        FULL_TENSOR(
            "state6.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE6_BINARY] =
        BINARY_TENSOR("state6.binary"),
    [BFSRCNN_TENSOR_STATE7_FULL] =
        FULL_TENSOR(
            "state7.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_STATE7_BINARY] =
        BINARY_TENSOR("state7.binary"),
    [BFSRCNN_TENSOR_STATE8_FULL] =
        FULL_TENSOR(
            "state8.full",
            BFSRCNN_BIRA_INPUT_HEIGHT,
            BFSRCNN_BIRA_INPUT_WIDTH,
            16u),
    [BFSRCNN_TENSOR_SHUFFLED] = {
        .name = "shuffled",
        .height = BFSRCNN_BIRA_OUTPUT_HEIGHT,
        .width = BFSRCNN_BIRA_OUTPUT_WIDTH,
        .channels = 8u,
        .layout = BIRA_LAYOUT_FULL_PACKED_PIXEL_PAIR,
        .is_signed = true,
    },
    [BFSRCNN_TENSOR_OUTPUT] = {
        .name = "output",
        .height = BFSRCNN_BIRA_OUTPUT_HEIGHT,
        .width = BFSRCNN_BIRA_OUTPUT_WIDTH,
        .channels = 1u,
        .layout = BIRA_LAYOUT_FULL_DENSE16,
        .is_signed = false,
    },
};

#define LAYER_EDGES( \
    operator_name, operator_type, input_id, residual_id, \
    output_id, binary_output_id) \
    .name = (operator_name), \
    .type = (operator_type), \
    .input_tensor = (input_id), \
    .residual_tensor = (residual_id), \
    .output_tensor = (output_id), \
    .binary_output_tensor = (binary_output_id)

#define MAPPING_LAYER( \
    operator_name, input_full, input_binary, output_full, output_binary, \
    weight_symbol, parameter_symbol) \
    { \
        LAYER_EDGES( \
            (operator_name), BIRA_OP_BINARY_CONV2D, \
            (input_binary), (input_full), (output_full), \
            (output_binary)), \
        .kernel_height = 3u, \
        .kernel_width = 3u, \
        .padding_height = 1u, \
        .padding_width = 1u, \
        .weight_precision = BIRA_WEIGHT_W2, \
        .post_mode = BIRA_POST_BINARY_FUSED, \
        .input_signed = false, \
        .weight_low = NATIVE_BLOB( \
            (weight_symbol), BFSRCNN_BIRA_MAPPING_WEIGHT_ROWS, 2u), \
        .weight_high = EMPTY_BLOB, \
        .parameters = NATIVE_BLOB( \
            (parameter_symbol), \
            BFSRCNN_BIRA_MAPPING_0_PARAMETERS_SIZE / 64u, 64u), \
        .correction = NATIVE_BLOB( \
            bfsrcnn_bira_mapping_correction, \
            BFSRCNN_BIRA_MAPPING_CORRECTION_SIZE / 64u, 64u), \
    }

static const bira_layer_desc_t bfsrcnn_layers[] = {
    [BFSRCNN_LAYER_HEAD] = {
        LAYER_EDGES(
            "head", BIRA_OP_CONV2D,
            BFSRCNN_TENSOR_INPUT, BIRA_TENSOR_NONE,
            BFSRCNN_TENSOR_HEAD, BIRA_TENSOR_NONE),
        .kernel_height = 3u,
        .kernel_width = 3u,
        .padding_height = 1u,
        .padding_width = 1u,
        .weight_precision = BIRA_WEIGHT_W16,
        .post_mode = BIRA_POST_INT_PRELU,
        .input_signed = false,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_head_weight,
            BFSRCNN_BIRA_HEAD_WEIGHT_LOW_ROWS, 16u),
        .weight_high = NATIVE_BLOB(
            bfsrcnn_bira_head_weight
                + BFSRCNN_BIRA_HEAD_WEIGHT_PLANE_BYTES,
            BFSRCNN_BIRA_HEAD_WEIGHT_LOW_ROWS, 16u),
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_head_parameters,
            BFSRCNN_BIRA_HEAD_PARAMETERS_SIZE / 64u, 64u),
    },
    [BFSRCNN_LAYER_SHRINK1] = {
        LAYER_EDGES(
            "shrink1", BIRA_OP_CONV2D,
            BFSRCNN_TENSOR_HEAD, BIRA_TENSOR_NONE,
            BFSRCNN_TENSOR_SHRINK1, BIRA_TENSOR_NONE),
        .kernel_height = 1u,
        .kernel_width = 1u,
        .padding_height = 0u,
        .padding_width = 0u,
        .weight_precision = BIRA_WEIGHT_W4,
        .post_mode = BIRA_POST_INT_RELU,
        .input_signed = true,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_shrink1_weight,
            BFSRCNN_BIRA_SHRINK1_WEIGHT_ROWS, 16u),
        .weight_high = EMPTY_BLOB,
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_shrink1_parameters,
            BFSRCNN_BIRA_SHRINK1_PARAMETERS_SIZE / 64u, 64u),
    },
    [BFSRCNN_LAYER_SHRINK2] = {
        LAYER_EDGES(
            "shrink2", BIRA_OP_DEPTHWISE_CONV2D,
            BFSRCNN_TENSOR_SHRINK1, BIRA_TENSOR_NONE,
            BFSRCNN_TENSOR_SHRINK2, BIRA_TENSOR_NONE),
        .kernel_height = 3u,
        .kernel_width = 3u,
        .padding_height = 1u,
        .padding_width = 1u,
        .weight_precision = BIRA_WEIGHT_W16,
        .post_mode = BIRA_POST_INT_RELU,
        .input_signed = false,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_shrink2_weight,
            BFSRCNN_BIRA_SHRINK2_WEIGHT_LOW_ROWS, 16u),
        .weight_high = NATIVE_BLOB(
            bfsrcnn_bira_shrink2_weight
                + BFSRCNN_BIRA_SHRINK2_WEIGHT_PLANE_BYTES,
            BFSRCNN_BIRA_SHRINK2_WEIGHT_LOW_ROWS, 16u),
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_shrink2_parameters,
            BFSRCNN_BIRA_SHRINK2_PARAMETERS_SIZE / 64u, 64u),
    },
    [BFSRCNN_LAYER_SHRINK3] = {
        LAYER_EDGES(
            "shrink3", BIRA_OP_CONV2D,
            BFSRCNN_TENSOR_SHRINK2, BIRA_TENSOR_NONE,
            BFSRCNN_TENSOR_STATE0_FULL,
            BFSRCNN_TENSOR_STATE0_BINARY),
        .kernel_height = 1u,
        .kernel_width = 1u,
        .padding_height = 0u,
        .padding_width = 0u,
        .weight_precision = BIRA_WEIGHT_W8,
        .post_mode = BIRA_POST_INT_SIGNED_SIGN,
        .input_signed = false,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_shrink3_weight,
            BFSRCNN_BIRA_SHRINK3_WEIGHT_ROWS, 16u),
        .weight_high = EMPTY_BLOB,
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_shrink3_parameters,
            BFSRCNN_BIRA_SHRINK3_PARAMETERS_SIZE / 64u, 64u),
    },
    [BFSRCNN_LAYER_MAPPING0] =
        MAPPING_LAYER(
            "mapping0",
            BFSRCNN_TENSOR_STATE0_FULL,
            BFSRCNN_TENSOR_STATE0_BINARY,
            BFSRCNN_TENSOR_STATE1_FULL,
            BFSRCNN_TENSOR_STATE1_BINARY,
            bfsrcnn_bira_mapping_0_weight,
            bfsrcnn_bira_mapping_0_parameters),
    [BFSRCNN_LAYER_MAPPING1] =
        MAPPING_LAYER(
            "mapping1",
            BFSRCNN_TENSOR_STATE1_FULL,
            BFSRCNN_TENSOR_STATE1_BINARY,
            BFSRCNN_TENSOR_STATE2_FULL,
            BFSRCNN_TENSOR_STATE2_BINARY,
            bfsrcnn_bira_mapping_1_weight,
            bfsrcnn_bira_mapping_1_parameters),
    [BFSRCNN_LAYER_MAPPING2] =
        MAPPING_LAYER(
            "mapping2",
            BFSRCNN_TENSOR_STATE2_FULL,
            BFSRCNN_TENSOR_STATE2_BINARY,
            BFSRCNN_TENSOR_STATE3_FULL,
            BFSRCNN_TENSOR_STATE3_BINARY,
            bfsrcnn_bira_mapping_2_weight,
            bfsrcnn_bira_mapping_2_parameters),
    [BFSRCNN_LAYER_MAPPING3] =
        MAPPING_LAYER(
            "mapping3",
            BFSRCNN_TENSOR_STATE3_FULL,
            BFSRCNN_TENSOR_STATE3_BINARY,
            BFSRCNN_TENSOR_STATE4_FULL,
            BFSRCNN_TENSOR_STATE4_BINARY,
            bfsrcnn_bira_mapping_3_weight,
            bfsrcnn_bira_mapping_3_parameters),
    [BFSRCNN_LAYER_MAPPING4] =
        MAPPING_LAYER(
            "mapping4",
            BFSRCNN_TENSOR_STATE4_FULL,
            BFSRCNN_TENSOR_STATE4_BINARY,
            BFSRCNN_TENSOR_STATE5_FULL,
            BFSRCNN_TENSOR_STATE5_BINARY,
            bfsrcnn_bira_mapping_4_weight,
            bfsrcnn_bira_mapping_4_parameters),
    [BFSRCNN_LAYER_MAPPING5] =
        MAPPING_LAYER(
            "mapping5",
            BFSRCNN_TENSOR_STATE5_FULL,
            BFSRCNN_TENSOR_STATE5_BINARY,
            BFSRCNN_TENSOR_STATE6_FULL,
            BFSRCNN_TENSOR_STATE6_BINARY,
            bfsrcnn_bira_mapping_5_weight,
            bfsrcnn_bira_mapping_5_parameters),
    [BFSRCNN_LAYER_MAPPING6] =
        MAPPING_LAYER(
            "mapping6",
            BFSRCNN_TENSOR_STATE6_FULL,
            BFSRCNN_TENSOR_STATE6_BINARY,
            BFSRCNN_TENSOR_STATE7_FULL,
            BFSRCNN_TENSOR_STATE7_BINARY,
            bfsrcnn_bira_mapping_6_weight,
            bfsrcnn_bira_mapping_6_parameters),
    [BFSRCNN_LAYER_MAPPING7] =
        MAPPING_LAYER(
            "mapping7",
            BFSRCNN_TENSOR_STATE7_FULL,
            BFSRCNN_TENSOR_STATE7_BINARY,
            BFSRCNN_TENSOR_STATE8_FULL,
            BIRA_TENSOR_NONE,
            bfsrcnn_bira_mapping_7_weight,
            bfsrcnn_bira_mapping_7_parameters),
    [BFSRCNN_LAYER_EXPAND] = {
        LAYER_EDGES(
            "expand", BIRA_OP_CONV2D,
            BFSRCNN_TENSOR_STATE8_FULL, BIRA_TENSOR_NONE,
            BFSRCNN_TENSOR_SHUFFLED, BIRA_TENSOR_NONE),
        .kernel_height = 1u,
        .kernel_width = 1u,
        .padding_height = 0u,
        .padding_width = 0u,
        .execution_output_height = BFSRCNN_BIRA_INPUT_HEIGHT,
        .execution_output_width = BFSRCNN_BIRA_INPUT_WIDTH,
        .execution_output_channels = 128u,
        .weight_precision = BIRA_WEIGHT_W8,
        .post_mode = BIRA_POST_INT_PRELU,
        .input_signed = true,
        .shuffle_pack2 = true,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_expand_weight,
            BFSRCNN_BIRA_EXPAND_WEIGHT_ROWS, 16u),
        .weight_high = EMPTY_BLOB,
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_expand_parameters,
            BFSRCNN_BIRA_EXPAND_PARAMETERS_SIZE / 64u, 64u),
    },
    [BFSRCNN_LAYER_FINAL] = {
        LAYER_EDGES(
            "final", BIRA_OP_COLUMN_REDUCE_CONV2D,
            BFSRCNN_TENSOR_SHUFFLED, BFSRCNN_TENSOR_INPUT,
            BFSRCNN_TENSOR_OUTPUT, BIRA_TENSOR_NONE),
        .kernel_height = 3u,
        .kernel_width = 3u,
        .padding_height = 1u,
        .padding_width = 1u,
        .weight_precision = BIRA_WEIGHT_W16,
        .post_mode = BIRA_POST_FINAL_BILINEAR_RESIDUAL,
        .input_signed = true,
        .weight_low = NATIVE_BLOB(
            bfsrcnn_bira_final_weight,
            BFSRCNN_BIRA_FINAL_WEIGHT_LOW_ROWS, 16u),
        .weight_high = NATIVE_BLOB(
            bfsrcnn_bira_final_weight
                + BFSRCNN_BIRA_FINAL_WEIGHT_PLANE_BYTES,
            BFSRCNN_BIRA_FINAL_WEIGHT_LOW_ROWS, 16u),
        .parameters = NATIVE_BLOB(
            bfsrcnn_bira_final_parameters,
            BFSRCNN_BIRA_FINAL_PARAMETERS_SIZE / 64u, 64u),
    },
};

static const bira_inference_desc_t bfsrcnn_network = {
    .name = "BFSRCNN x4",
    .tensors = bfsrcnn_tensors,
    .tensor_count = BFSRCNN_TENSOR_COUNT,
    .layers = bfsrcnn_layers,
    .layer_count = BFSRCNN_LAYER_COUNT,
};

const bira_inference_desc_t *bfsrcnn_inference_description(void)
{
    return &bfsrcnn_network;
}

int bfsrcnn_infer(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    const void *input,
    void *output,
    bira_inference_layer_hook_t hook,
    void *hook_user)
{
    bira_tensor_binding_t bindings[2];
    if (runtime == NULL || workspace == NULL
        || input == NULL || output == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    bindings[0].tensor = BFSRCNN_TENSOR_INPUT;
    bindings[0].data = (void *)input;
    bindings[1].tensor = BFSRCNN_TENSOR_OUTPUT;
    bindings[1].data = output;
    return bira_inference(
        runtime,
        workspace,
        &bfsrcnn_network,
        bindings,
        2u,
        hook,
        hook_user);
}

int bfsrcnn_infer_layer(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    bfsrcnn_layer_id_t layer,
    const bira_tensor_binding_t *bindings,
    size_t binding_count,
    bira_inference_layer_hook_t hook,
    void *hook_user)
{
    bira_inference_desc_t inference;
    if (runtime == NULL || workspace == NULL
        || bindings == NULL || binding_count == 0u
        || layer >= BFSRCNN_LAYER_COUNT) {
        return BIRA_ERR_ARGUMENT;
    }
    inference.name = bfsrcnn_layers[layer].name;
    inference.tensors = bfsrcnn_tensors;
    inference.tensor_count = BFSRCNN_TENSOR_COUNT;
    inference.layers = &bfsrcnn_layers[layer];
    inference.layer_count = 1u;
    return bira_inference(
        runtime,
        workspace,
        &inference,
        bindings,
        binding_count,
        hook,
        hook_user);
}
