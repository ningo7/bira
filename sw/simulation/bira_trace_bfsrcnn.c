/* Record commands from the programmer-written BFSRCNN inference code. */
#include "bfsrcnn_bira_data.h"
#include "bfsrcnn_inference.h"
#include "bira_trace.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    bira_runtime_t *runtime;
    uint8_t **stage_outputs;
    int status;
} debug_stage_context_t;

static int read_exact(const char *path, void *data, size_t size)
{
    FILE *file = fopen(path, "rb");
    int status = 0;
    if (file == NULL) {
        return -1;
    }
    if (fread(data, size, 1u, file) != 1u || fgetc(file) != EOF) {
        status = -1;
    }
    if (fclose(file) != 0) {
        status = -1;
    }
    return status;
}

static void store_debug_stage(
    bira_inference_context_t *context,
    size_t operator_index,
    bool before,
    void *user)
{
    debug_stage_context_t *debug =
        (debug_stage_context_t *)user;
    const bira_planned_layer_t *compiled =
        &context->workspace->layers[operator_index];
    const bira_layer_desc_t *op = compiled->layer_desc;
    const bira_tensor_desc_t *tensor =
        &context->workspace->inference->tensors[
            op->output_tensor];
    uint32_t rows;
    uint8_t bytes_per_row;
    uint8_t context_id;
    uint64_t descriptor;
    bira_event_t event;

    if (before || operator_index + 1u >= BFSRCNN_LAYER_COUNT
        || debug->status != BIRA_OK) {
        return;
    }
    rows = bira_tensor_rows(tensor);
    bytes_per_row = bira_tensor_bytes_per_row(tensor);
    debug->stage_outputs[operator_index] =
        calloc(rows, bytes_per_row);
    if (debug->stage_outputs[operator_index] == NULL
        || rows > UINT16_MAX) {
        debug->status = BIRA_ERR_CAPACITY;
        return;
    }
    context_id = (uint8_t)(
        (debug->runtime->next_context
            + debug->runtime->capabilities.contexts - 1u)
        % debug->runtime->capabilities.contexts);
    descriptor = bira_pack_dma_desc(
        context_id,
        BIRA_ROLE_OUTPUT_FULL,
        0u,
        (uint16_t)rows,
        bytes_per_row,
        bytes_per_row,
        1u);
    debug->status = bira_issue(
        debug->runtime,
        BIRA_FUNCT_STORE_2D,
        (uint64_t)(uintptr_t)debug->stage_outputs[operator_index],
        descriptor,
        false,
        NULL);
    event.context_id = context_id;
    if (debug->status == BIRA_OK) {
        debug->status = bira_wait_event(debug->runtime, &event);
    }
}

int main(int argc, char **argv)
{
    const size_t input_size =
        BFSRCNN_BIRA_INPUT_HEIGHT * BFSRCNN_BIRA_INPUT_WIDTH;
    const size_t output_size =
        BFSRCNN_BIRA_OUTPUT_HEIGHT * BFSRCNN_BIRA_OUTPUT_WIDTH;
    uint8_t *input;
    uint8_t *output;
    bira_trace_t trace;
    bira_runtime_t runtime;
    bira_inference_workspace_t workspace;
    uint8_t *stage_outputs[BFSRCNN_LAYER_COUNT - 1u] = {0};
    debug_stage_context_t debug;
    unsigned int layer;
    int status;

    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s INPUT COMMANDS MEMORY MANIFEST\n",
            argv[0]);
        return 2;
    }
    input = malloc(input_size);
    output = calloc(output_size, 1u);
    if (input == NULL || output == NULL
        || read_exact(argv[1], input, input_size) != 0) {
        free(output);
        free(input);
        return 3;
    }
    status = bira_trace_init(&trace, 0u);
    if (status == BIRA_OK) {
        bira_runtime_init(&runtime, bira_trace_driver(&trace), NULL);
    }
    debug.runtime = &runtime;
    debug.stage_outputs = stage_outputs;
    debug.status = BIRA_OK;
    if (status == BIRA_OK) {
        status = bfsrcnn_infer(
            &runtime,
            &workspace,
            input,
            output,
            store_debug_stage,
            &debug);
        if (status != BIRA_OK) {
            fprintf(
                stderr, "BFSRCNN run failed with status %d\n",
                status);
        }
    }
    if (status == BIRA_OK) {
        status = debug.status;
    }
    if (status == BIRA_OK) {
        status = bira_trace_write_files(
            &trace, argv[2], argv[3], argv[4]);
    }
    if (status == BIRA_OK) {
        printf(
            "wrote %zu commands and %zu DRAM bytes; "
            "eliminated %u intermediate transfers\n",
            trace.command_count,
            trace.memory_size,
            workspace.eliminated_tensor_transfers);
    } else {
        fprintf(stderr, "BFSRCNN trace failed with status %d\n", status);
    }
    bira_trace_destroy(&trace);
    for (layer = 0u; layer + 1u < BFSRCNN_LAYER_COUNT; ++layer) {
        free(stage_outputs[layer]);
    }
    free(output);
    free(input);
    return status == BIRA_OK ? 0 : 4;
}
