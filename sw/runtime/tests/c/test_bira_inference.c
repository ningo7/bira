#include "bira_inference.h"
#include "bira_trace.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    TENSOR_INPUT = 0,
    TENSOR_HIDDEN,
    TENSOR_OUTPUT,
    TENSOR_COUNT,
};

static const uint8_t weights0[16] = {0};
static const uint8_t weights1[16] = {0};
static const uint8_t parameters[512] = {0};

static const bira_tensor_desc_t tensors[] = {
    {
        .name = "input",
        .height = 1u,
        .width = 16u,
        .channels = 16u,
        .layout = BIRA_LAYOUT_FULL_HWC16,
        .is_signed = false,
    },
    {
        .name = "hidden",
        .height = 1u,
        .width = 16u,
        .channels = 16u,
        .layout = BIRA_LAYOUT_FULL_HWC16,
        .is_signed = false,
    },
    {
        .name = "output",
        .height = 1u,
        .width = 16u,
        .channels = 16u,
        .layout = BIRA_LAYOUT_FULL_HWC16,
        .is_signed = false,
    },
};

#define TEST_BLOB(data_symbol, row_count, row_bytes) \
    { \
        .data = (data_symbol), \
        .rows = (row_count), \
        .bytes_per_row = (row_bytes), \
        .stride_bytes = (row_bytes), \
    }

#define TEST_LAYER(operator_name, input_id, output_id, weight_symbol) \
    { \
        .name = (operator_name), \
        .type = BIRA_OP_CONV2D, \
        .input_tensor = (input_id), \
        .residual_tensor = BIRA_TENSOR_NONE, \
        .output_tensor = (output_id), \
        .binary_output_tensor = BIRA_TENSOR_NONE, \
        .kernel_height = 1u, \
        .kernel_width = 1u, \
        .weight_precision = BIRA_WEIGHT_W4, \
        .post_mode = BIRA_POST_INT_RELU, \
        .weight_low = TEST_BLOB((weight_symbol), 1u, 16u), \
        .parameters = TEST_BLOB(parameters, 8u, 64u), \
    }

static const bira_layer_desc_t layers[] = {
    TEST_LAYER("conv0", TENSOR_INPUT, TENSOR_HIDDEN, weights0),
    TEST_LAYER("conv1", TENSOR_HIDDEN, TENSOR_OUTPUT, weights1),
};

static const bira_inference_desc_t network = {
    .name = "two-conv",
    .tensors = tensors,
    .tensor_count = TENSOR_COUNT,
    .layers = layers,
    .layer_count = 2u,
};

static const bira_inference_desc_t second_layer = {
    .name = "conv1",
    .tensors = tensors,
    .tensor_count = TENSOR_COUNT,
    .layers = &layers[1],
    .layer_count = 1u,
};

static size_t count_command(
    const bira_trace_t *trace,
    uint8_t funct)
{
    size_t index;
    size_t count = 0u;
    for (index = 0u; index < trace->command_count; ++index) {
        if (trace->commands[index].funct == funct) {
            ++count;
        }
    }
    return count;
}

int main(void)
{
    uint8_t input[256] = {0};
    uint8_t hidden[256] = {0};
    uint8_t output[256] = {0};
    bira_trace_t trace;
    bira_runtime_t runtime;
    bira_inference_workspace_t workspace;
    bira_tensor_binding_t network_bindings[] = {
        {.tensor = TENSOR_INPUT, .data = input},
        {.tensor = TENSOR_OUTPUT, .data = output},
    };
    bira_tensor_binding_t layer_bindings[] = {
        {.tensor = TENSOR_HIDDEN, .data = hidden},
        {.tensor = TENSOR_OUTPUT, .data = output},
    };

    assert(bira_trace_init(&trace, 0u) == BIRA_OK);
    bira_runtime_init(&runtime, bira_trace_driver(&trace), NULL);
    assert(bira_inference(
        &runtime,
        &workspace,
        &network,
        network_bindings,
        sizeof(network_bindings) / sizeof(network_bindings[0]),
        NULL,
        NULL) == BIRA_OK);
    assert(workspace.eliminated_tensor_transfers == 1u);
    assert(workspace.layers[0].base_rows[BIRA_ROLE_OUTPUT_FULL]
        == workspace.layers[1].base_rows[BIRA_ROLE_INPUT]);
    assert(count_command(&trace, BIRA_FUNCT_EXEC_CONV) == 2u);
    assert(count_command(&trace, BIRA_FUNCT_STORE_2D) == 1u);
    assert(count_command(&trace, BIRA_FUNCT_LOAD_2D) == 5u);

    bira_trace_destroy(&trace);
    assert(bira_trace_init(&trace, 0u) == BIRA_OK);
    bira_runtime_init(&runtime, bira_trace_driver(&trace), NULL);
    assert(bira_inference(
        &runtime,
        &workspace,
        &second_layer,
        layer_bindings,
        sizeof(layer_bindings) / sizeof(layer_bindings[0]),
        NULL,
        NULL) == BIRA_OK);
    assert(count_command(&trace, BIRA_FUNCT_EXEC_CONV) == 1u);
    assert(count_command(&trace, BIRA_FUNCT_STORE_2D) == 1u);
    assert(count_command(&trace, BIRA_FUNCT_LOAD_2D) == 3u);

    printf(
        "inference engine ran two layers together and one layer alone\n");
    bira_trace_destroy(&trace);
    return 0;
}
