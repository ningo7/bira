#include "bfsrcnn_inference.h"
#include "bira_trace.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const bira_inference_desc_t *network =
        bfsrcnn_inference_description();
    bira_trace_t trace;
    bira_runtime_t runtime;
    bira_inference_workspace_t workspace;
    unsigned int layer;

    for (layer = 0u; layer < BFSRCNN_LAYER_COUNT; ++layer) {
        bira_tensor_binding_t bindings[BIRA_INFERENCE_MAX_TENSORS];
        void *buffers[BIRA_INFERENCE_MAX_TENSORS] = {0};
        size_t binding_count = 0u;
        size_t tensor;
        int status = BIRA_OK;

        if (bira_trace_init(&trace, 0u) != BIRA_OK) {
            return 1;
        }
        bira_runtime_init(
            &runtime, bira_trace_driver(&trace), NULL);
        for (tensor = 0u;
             tensor < network->tensor_count;
             ++tensor) {
            size_t bytes;
            bytes = bira_tensor_bytes(
                &network->tensors[tensor]);
            buffers[binding_count] = calloc(bytes, 1u);
            if (buffers[binding_count] == NULL) {
                status = BIRA_ERR_CAPACITY;
                break;
            }
            bindings[binding_count].tensor = (uint16_t)tensor;
            bindings[binding_count].data =
                buffers[binding_count];
            ++binding_count;
        }
        if (status == BIRA_OK) {
            status = bfsrcnn_infer_layer(
                &runtime,
                &workspace,
                (bfsrcnn_layer_id_t)layer,
                bindings,
                binding_count,
                NULL,
                NULL);
        }
        if (status != BIRA_OK) {
            fprintf(
                stderr,
                "standalone layer %s failed: %d\n",
                bira_inference_layer_name(network, layer),
                status);
            return 2;
        }
        if (trace.command_count == 0u) {
            return 3;
        }
        for (tensor = 0u; tensor < binding_count; ++tensor) {
            free(buffers[tensor]);
        }
        bira_trace_destroy(&trace);
    }
    printf(
        "prepared and submitted all %u BFSRCNN layers independently\n",
        (unsigned int)BFSRCNN_LAYER_COUNT);
    return 0;
}
