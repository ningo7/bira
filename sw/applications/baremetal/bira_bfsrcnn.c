#include "bfsrcnn_inference.h"
#include "bira.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define BFSRCNN_INPUT_BYTES 1024u
#define BFSRCNN_OUTPUT_BYTES 16384u

extern const uint8_t bira_bfsrcnn_fixture_input[BFSRCNN_INPUT_BYTES];
extern const uint8_t bira_bfsrcnn_fixture_expected[BFSRCNN_OUTPUT_BYTES];

static uint8_t output[BFSRCNN_OUTPUT_BYTES] __attribute__((aligned(64)));
static bira_inference_workspace_t workspace;
static uint64_t layer_start_cycle;
static uint64_t profiled_layer_cycles;

static inline uint64_t read_cycle(void)
{
    uint64_t cycle;

    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static void layer_hook(
    bira_inference_context_t *context,
    size_t layer,
    bool before,
    void *user)
{
    const char *name;

    (void)context;
    (void)user;
    name = bira_inference_layer_name(
        context->workspace->inference, layer);
    if (before) {
        printf("RUN  %s\n", name);
        layer_start_cycle = read_cycle();
    } else {
        uint64_t cycles = read_cycle() - layer_start_cycle;

        profiled_layer_cycles += cycles;
        printf(
            "DONE %s cycles=%lu\n",
            name,
            (unsigned long)cycles);
    }
}

int main(void)
{
    bira_runtime_t runtime;
    uint64_t program_start;
    uint64_t inference_start;
    uint64_t inference_cycles;
    size_t index;
    int status;

    program_start = read_cycle();
    bira_runtime_init(&runtime, bira_rocc_driver(), NULL);
    __asm__ volatile("fence rw, rw" ::: "memory");
    inference_start = read_cycle();
    status = bfsrcnn_infer(
        &runtime,
        &workspace,
        bira_bfsrcnn_fixture_input,
        output,
        layer_hook,
        NULL);
    inference_cycles = read_cycle() - inference_start;
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (status != BIRA_OK) {
        printf("FAIL bfsrcnn run: %d\n", status);
        return 1;
    }

    for (index = 0; index < BFSRCNN_OUTPUT_BYTES; ++index) {
        if (output[index] != bira_bfsrcnn_fixture_expected[index]) {
            printf(
                "FAIL bfsrcnn output[%lu]: got=0x%x expected=0x%x\n",
                (unsigned long)index,
                output[index],
                bira_bfsrcnn_fixture_expected[index]);
            return 1;
        }
    }

    printf(
        "profiled_layer_cycles=%lu\n",
        (unsigned long)profiled_layer_cycles);
    printf(
        "inference_cycles=%lu\n",
        (unsigned long)inference_cycles);
    printf(
        "program_cycles=%lu\n",
        (unsigned long)(read_cycle() - program_start));
    printf("PASS bira_bfsrcnn (%u bytes)\n", BFSRCNN_OUTPUT_BYTES);
    return 0;
}
