#include "bira_ops.h"
#include "bira_trace.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    enum {
        HEIGHT = 20,
        WIDTH = 20,
        CHANNELS = 16,
        WEIGHT_ROWS = 144,
        PARAMETER_ROWS = 8
    };
    uint8_t *input = malloc(HEIGHT * WIDTH * CHANNELS);
    uint8_t *output = calloc(HEIGHT * WIDTH * CHANNELS, 1u);
    uint8_t *weight_low = calloc(WEIGHT_ROWS * 16u, 1u);
    uint8_t *weight_high = calloc(WEIGHT_ROWS * 16u, 1u);
    uint8_t *parameters = calloc(PARAMETER_ROWS * 64u, 1u);
    bira_tensor_t input_tensor;
    bira_tensor_t output_tensor;
    bira_conv2d_params_t params;
    bira_capabilities_t capabilities = bira_default_capabilities();
    bira_trace_t trace;
    bira_runtime_t runtime;
    bira_tile_plan_t plan;
    size_t index;
    size_t exec_count = 0u;

    assert(input != NULL && output != NULL);
    assert(weight_low != NULL && weight_high != NULL);
    assert(parameters != NULL);
    for (index = 0u; index < HEIGHT * WIDTH * CHANNELS; ++index) {
        input[index] = (uint8_t)index;
    }
    memset(&input_tensor, 0, sizeof(input_tensor));
    input_tensor.data = input;
    input_tensor.height = HEIGHT;
    input_tensor.width = WIDTH;
    input_tensor.channels = CHANNELS;
    input_tensor.format = BIRA_TENSOR_FULL_HWC;
    input_tensor.is_signed = true;

    memset(&output_tensor, 0, sizeof(output_tensor));
    output_tensor.data = output;
    output_tensor.height = HEIGHT;
    output_tensor.width = WIDTH;
    output_tensor.channels = CHANNELS;
    output_tensor.format = BIRA_TENSOR_FULL_HWC;
    output_tensor.is_signed = true;

    memset(&params, 0, sizeof(params));
    params.kernel_height = 3u;
    params.kernel_width = 3u;
    params.padding_height = 1u;
    params.padding_width = 1u;
    params.groups = 1u;
    params.weight_precision = BIRA_WEIGHT_W16;
    params.post_mode = BIRA_POST_INT_PRELU;
    params.weight_low.data = weight_low;
    params.weight_low.rows = WEIGHT_ROWS;
    params.weight_low.bytes_per_row = 16u;
    params.weight_high.data = weight_high;
    params.weight_high.rows = WEIGHT_ROWS;
    params.weight_high.bytes_per_row = 16u;
    params.parameters.data = parameters;
    params.parameters.rows = PARAMETER_ROWS;
    params.parameters.bytes_per_row = 64u;

    /*
     * Small banks force spatial tiling while preserving the ISA organization.
     * The operator must derive local padding for every edge tile.
     */
    capabilities.bank_rows = 64u;
    capabilities.full_banks = 10u;
    capabilities.accumulator_banks = 2u;
    capabilities.max_image_height = 32u;
    capabilities.max_image_width = 32u;
    assert(bira_trace_init(&trace, 0u) == BIRA_OK);
    bira_runtime_init(
        &runtime, bira_trace_driver(&trace), &capabilities);

    assert(bira_plan_conv2d(
        &runtime,
        &input_tensor,
        &output_tensor,
        &params,
        &plan) == BIRA_OK);
    assert(plan.tile_count > 1u);
    assert(plan.tile_height <= HEIGHT);
    assert(plan.tile_width <= WIDTH);
    assert(bira_conv2d(
        &runtime,
        &input_tensor,
        &output_tensor,
        &params) == BIRA_OK);

    for (index = 0u; index < trace.command_count; ++index) {
        if (trace.commands[index].funct == BIRA_FUNCT_EXEC_CONV) {
            ++exec_count;
        }
    }
    assert(exec_count == plan.tile_count);
    assert(runtime.next_context == plan.tile_count % capabilities.contexts);
    printf(
        "%ux%u convolution used %ux%u tiles (%zu EXEC commands)\n",
        WIDTH,
        HEIGHT,
        plan.tile_width,
        plan.tile_height,
        exec_count);

    bira_trace_destroy(&trace);

    {
        uint8_t *binary_input = calloc(HEIGHT * WIDTH * 2u, 1u);
        uint8_t *binary_output = calloc(HEIGHT * WIDTH * 2u, 1u);
        uint8_t *residual = calloc(HEIGHT * WIDTH * CHANNELS, 1u);
        int32_t *correction = calloc(
            HEIGHT * WIDTH, sizeof(*correction));
        uint8_t *binary_weight = calloc(WEIGHT_ROWS * 2u, 1u);
        bira_tensor_t binary_input_tensor;
        bira_tensor_t binary_output_tensor;
        bira_tensor_t residual_tensor;
        bira_binary_conv2d_params_t binary_params;

        assert(binary_input != NULL && binary_output != NULL);
        assert(residual != NULL && correction != NULL);
        assert(binary_weight != NULL);
        memset(&binary_input_tensor, 0, sizeof(binary_input_tensor));
        binary_input_tensor.data = binary_input;
        binary_input_tensor.height = HEIGHT;
        binary_input_tensor.width = WIDTH;
        binary_input_tensor.channels = CHANNELS;
        binary_input_tensor.format = BIRA_TENSOR_BINARY_HWC16;

        memset(&binary_output_tensor, 0, sizeof(binary_output_tensor));
        binary_output_tensor.data = binary_output;
        binary_output_tensor.height = HEIGHT;
        binary_output_tensor.width = WIDTH;
        binary_output_tensor.channels = CHANNELS;
        binary_output_tensor.format = BIRA_TENSOR_BINARY_HWC16;

        memset(&residual_tensor, 0, sizeof(residual_tensor));
        residual_tensor.data = residual;
        residual_tensor.height = HEIGHT;
        residual_tensor.width = WIDTH;
        residual_tensor.channels = CHANNELS;
        residual_tensor.format = BIRA_TENSOR_FULL_HWC;
        residual_tensor.is_signed = true;

        memset(&binary_params, 0, sizeof(binary_params));
        binary_params.kernel_height = 3u;
        binary_params.kernel_width = 3u;
        binary_params.padding_height = 1u;
        binary_params.padding_width = 1u;
        binary_params.post_mode = BIRA_POST_BINARY_FUSED;
        binary_params.write_binary = true;
        binary_params.weight.data = binary_weight;
        binary_params.weight.rows = WEIGHT_ROWS;
        binary_params.weight.bytes_per_row = 2u;
        binary_params.parameters.data = parameters;
        binary_params.parameters.rows = PARAMETER_ROWS;
        binary_params.parameters.bytes_per_row = 64u;
        binary_params.correction = correction;
        binary_params.correction_row_stride_bytes =
            WIDTH * sizeof(*correction);
        binary_params.residual = &residual_tensor;
        binary_params.binary_output = &binary_output_tensor;

        memset(output, 0, HEIGHT * WIDTH * CHANNELS);
        output_tensor.is_signed = true;
        assert(bira_trace_init(&trace, 0u) == BIRA_OK);
        bira_runtime_init(
            &runtime, bira_trace_driver(&trace), &capabilities);
        assert(bira_binary_conv2d(
            &runtime,
            &binary_input_tensor,
            &output_tensor,
            &binary_params) == BIRA_OK);
        exec_count = 0u;
        for (index = 0u; index < trace.command_count; ++index) {
            if (trace.commands[index].funct
                == BIRA_FUNCT_EXEC_CONV) {
                ++exec_count;
            }
        }
        assert(exec_count > 1u);
        printf(
            "binary convolution used %zu tiled EXEC commands\n",
            exec_count);
        bira_trace_destroy(&trace);
        free(binary_weight);
        free(correction);
        free(residual);
        free(binary_output);
        free(binary_input);
    }

    {
        uint8_t scalar_input[16];
        uint8_t scalar_output[16 * 16];
        uint8_t scalar_weight_low[27 * 16];
        uint8_t scalar_weight_high[27 * 16];
        uint8_t scalar_parameters[24 * 64];
        bira_tensor_t scalar_input_tensor;
        bira_tensor_t scalar_output_tensor;
        bira_conv2d_params_t scalar_params;
        const bira_command_t *input_load = NULL;

        for (index = 0u; index < sizeof(scalar_input); ++index) {
            scalar_input[index] = (uint8_t)(index + 3u);
        }
        memset(scalar_output, 0, sizeof(scalar_output));
        memset(scalar_weight_low, 0, sizeof(scalar_weight_low));
        memset(scalar_weight_high, 0, sizeof(scalar_weight_high));
        memset(scalar_parameters, 0, sizeof(scalar_parameters));
        memset(&scalar_input_tensor, 0, sizeof(scalar_input_tensor));
        scalar_input_tensor.data = scalar_input;
        scalar_input_tensor.height = 4u;
        scalar_input_tensor.width = 4u;
        scalar_input_tensor.channels = 1u;
        scalar_input_tensor.format = BIRA_TENSOR_FULL_HWC;
        memset(&scalar_output_tensor, 0, sizeof(scalar_output_tensor));
        scalar_output_tensor.data = scalar_output;
        scalar_output_tensor.height = 4u;
        scalar_output_tensor.width = 4u;
        scalar_output_tensor.channels = 16u;
        scalar_output_tensor.format = BIRA_TENSOR_FULL_HWC;
        scalar_output_tensor.is_signed = true;
        memset(&scalar_params, 0, sizeof(scalar_params));
        scalar_params.kernel_height = 3u;
        scalar_params.kernel_width = 3u;
        scalar_params.padding_height = 1u;
        scalar_params.padding_width = 1u;
        scalar_params.groups = 1u;
        scalar_params.weight_precision = BIRA_WEIGHT_W16;
        scalar_params.post_mode = BIRA_POST_INT_PRELU;
        scalar_params.weight_low.data = scalar_weight_low;
        scalar_params.weight_low.rows = 27u;
        scalar_params.weight_low.bytes_per_row = 16u;
        scalar_params.weight_high.data = scalar_weight_high;
        scalar_params.weight_high.rows = 27u;
        scalar_params.weight_high.bytes_per_row = 16u;
        scalar_params.parameters.data = scalar_parameters;
        scalar_params.parameters.rows = 24u;
        scalar_params.parameters.bytes_per_row = 64u;

        assert(bira_trace_init(&trace, 0u) == BIRA_OK);
        bira_runtime_init(
            &runtime, bira_trace_driver(&trace), &capabilities);
        {
            int scalar_status = bira_conv2d(
                &runtime,
                &scalar_input_tensor,
                &scalar_output_tensor,
                &scalar_params);
            if (scalar_status != BIRA_OK) {
                fprintf(stderr, "scalar conv status=%d\n", scalar_status);
            }
            assert(scalar_status == BIRA_OK);
        }
        for (index = 0u; index < trace.command_count; ++index) {
            uint64_t descriptor = trace.commands[index].rs2;
            if (trace.commands[index].funct == BIRA_FUNCT_LOAD_2D
                && ((descriptor >> 3) & 0xfu) == BIRA_ROLE_INPUT) {
                input_load = &trace.commands[index];
                break;
            }
        }
        assert(input_load != NULL);
        assert(((input_load->rs2 >> 21) & 0x3fffu) == 1u);
        assert(((input_load->rs2 >> 35) & 0x7fu) == 16u);
        for (index = 0u; index < sizeof(scalar_input); ++index) {
            size_t offset = (size_t)(
                input_load->rs1 - trace.memory_base + index);
            assert(trace.memory[offset] == scalar_input[index]);
        }
        bira_trace_destroy(&trace);
    }

    free(parameters);
    free(weight_high);
    free(weight_low);
    free(output);
    free(input);
    return 0;
}
