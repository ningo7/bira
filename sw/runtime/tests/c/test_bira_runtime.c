#include "bira_runtime.h"
#include "bira_trace.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void fill_layer(
    bira_layer_t *layer,
    const bira_load_t *loads,
    size_t load_count,
    bira_store_t *store)
{
    memset(layer, 0, sizeof(*layer));
    layer->name = "test_dense";
    layer->context_id = BIRA_CONTEXT_AUTO;
    layer->shape.input_height = 2u;
    layer->shape.input_width = 2u;
    layer->shape.input_channels = 16u;
    layer->shape.output_height = 2u;
    layer->shape.output_width = 2u;
    layer->shape.output_channels = 16u;
    layer->shape.kernel_height = 1u;
    layer->shape.kernel_width = 1u;
    layer->mode.array_mode = BIRA_ARRAY_DENSE;
    layer->mode.weight_precision = BIRA_WEIGHT_W8;
    layer->mode.input_signed = true;
    layer->mode.post_mode = BIRA_POST_INT_RELU;
    layer->mode.write_full = true;
    layer->address_mask =
        (UINT16_C(1) << BIRA_ROLE_INPUT)
        | (UINT16_C(1) << BIRA_ROLE_WEIGHT_LOW)
        | (UINT16_C(1) << BIRA_ROLE_PARAM)
        | (UINT16_C(1) << BIRA_ROLE_ACCUMULATOR)
        | (UINT16_C(1) << BIRA_ROLE_OUTPUT_FULL);
    layer->base_rows[BIRA_ROLE_INPUT] = 0u;
    layer->base_rows[BIRA_ROLE_WEIGHT_LOW] = 1024u;
    layer->base_rows[BIRA_ROLE_PARAM] = 0u;
    layer->base_rows[BIRA_ROLE_ACCUMULATOR] = 0u;
    layer->base_rows[BIRA_ROLE_OUTPUT_FULL] = 2048u;
    layer->loads = loads;
    layer->load_count = load_count;
    layer->stores = store;
    layer->store_count = store == NULL ? 0u : 1u;
}

int main(int argc, char **argv)
{
    uint8_t input[64];
    uint8_t weight[128];
    uint8_t parameter[512];
    uint8_t output[64];
    bira_load_t loads[3];
    bira_store_t store;
    bira_layer_t layer;
    bira_trace_t trace;
    bira_runtime_t runtime;
    bira_event_t event;
    size_t index;

    assert(argc == 4);
    for (index = 0u; index < sizeof(input); ++index) {
        input[index] = (uint8_t)(index + 1u);
    }
    for (index = 0u; index < sizeof(weight); ++index) {
        weight[index] = (uint8_t)(0xa0u + (index & 0x0fu));
    }
    memset(parameter, 0x5a, sizeof(parameter));
    memset(output, 0, sizeof(output));
    memset(loads, 0, sizeof(loads));
    memset(&store, 0, sizeof(store));

    loads[0].dram_address = input;
    loads[0].role = BIRA_ROLE_INPUT;
    loads[0].rows = 4u;
    loads[0].bytes_per_row = 16u;
    loads[0].dram_stride_bytes = 16u;
    loads[0].local_stride_rows = 1u;

    loads[1].dram_address = weight;
    loads[1].role = BIRA_ROLE_WEIGHT_LOW;
    loads[1].rows = 8u;
    loads[1].bytes_per_row = 16u;
    loads[1].dram_stride_bytes = 16u;
    loads[1].local_stride_rows = 1u;

    loads[2].dram_address = parameter;
    loads[2].role = BIRA_ROLE_PARAM;
    loads[2].rows = 8u;
    loads[2].bytes_per_row = 64u;
    loads[2].dram_stride_bytes = 64u;
    loads[2].local_stride_rows = 1u;

    store.dram_address = output;
    store.role = BIRA_ROLE_OUTPUT_FULL;
    store.rows = 4u;
    store.bytes_per_row = 16u;
    store.dram_stride_bytes = 16u;
    store.local_stride_rows = 1u;

    fill_layer(&layer, loads, 3u, &store);
    assert(bira_trace_init(&trace, UINT64_C(0x20000)) == BIRA_OK);
    bira_runtime_init(&runtime, bira_trace_driver(&trace), NULL);

    assert(bira_validate_layer(&runtime, &layer) == BIRA_OK);
    assert(bira_submit_layer(&runtime, &layer, &event) == BIRA_OK);
    assert(event.context_id == 0u);
    assert(bira_wait_event(&runtime, &event) == BIRA_OK);

    assert(trace.command_count == 14u);
    assert(trace.commands[0].funct == BIRA_FUNCT_CFG_SHAPE);
    assert(trace.commands[8].funct == BIRA_FUNCT_LOAD_2D);
    assert(trace.commands[11].funct == BIRA_FUNCT_EXEC_CONV);
    assert(trace.commands[12].funct == BIRA_FUNCT_STORE_2D);
    assert(trace.commands[13].funct == BIRA_FUNCT_FENCE);
    assert(trace.commands[8].rs1 >= trace.memory_base);
    assert(trace.commands[8].rs1 != (uint64_t)(uintptr_t)input);
    assert(trace.memory[
        (size_t)(trace.commands[8].rs1 - trace.memory_base)] == input[0]);
    assert(trace.mapping_count == 4u);

    assert(bira_trace_write_files(
        &trace, argv[1], argv[2], argv[3]) == BIRA_OK);
    printf(
        "recorded %zu commands and %zu DRAM bytes\n",
        trace.command_count,
        trace.memory_size);
    bira_trace_destroy(&trace);
    return 0;
}
