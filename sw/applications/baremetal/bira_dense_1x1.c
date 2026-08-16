#include "bira_test_common.h"

#include <stdio.h>

static const int8_t input[16] __attribute__((aligned(64))) = {
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
static int8_t weights[16][16] __attribute__((aligned(64)));
static uint8_t parameters[8][64] __attribute__((aligned(64)));
static int8_t output[16] __attribute__((aligned(64)));

int main(void)
{
    unsigned row;
    unsigned lane;

    for (row = 0; row < 16; ++row) {
        for (lane = 0; lane < 16; ++lane) {
            weights[row][lane] = row == lane ? 1 : 0;
        }
    }
    bira_pack_multibit_identity_parameters(parameters);

    bira_cfg_shape(BIRA_TEST_CTX, 1, 1, 16, 1, 1, 16, 1, 1, 0, 0);
    bira_cfg_addr(BIRA_TEST_CTX, BIRA_ROLE_INPUT, BIRA_FULL_INPUT_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_WEIGHT_LOW, BIRA_FULL_WEIGHT_BASE);
    bira_cfg_addr(BIRA_TEST_CTX, BIRA_ROLE_PARAM, BIRA_PARAMETER_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_ACCUMULATOR, BIRA_ACCUMULATOR_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_OUTPUT_FULL, BIRA_FULL_OUTPUT_BASE);
    bira_cfg_mode(
        BIRA_TEST_CTX,
        BIRA_ARRAY_DENSE,
        BIRA_WEIGHT_W8,
        true,
        BIRA_POST_NONE,
        false,
        true,
        false);
    bira_cfg_commit(BIRA_TEST_CTX);

    bira_cpu_fence();
    bira_load_2d(
        input, bira_descriptor(BIRA_ROLE_INPUT, 0, 1, 16, 16));
    bira_load_2d(
        weights, bira_descriptor(BIRA_ROLE_WEIGHT_LOW, 0, 16, 16, 16));
    bira_load_2d(
        parameters, bira_descriptor(BIRA_ROLE_PARAM, 0, 8, 64, 64));
    bira_exec_conv(BIRA_TEST_CTX);
    bira_store_2d(
        output, bira_descriptor(BIRA_ROLE_OUTPUT_FULL, 0, 1, 16, 16));

    if (bira_expect_fence_ok("dense execution") != 0) {
        return 1;
    }
    bira_cpu_fence();
    if (bira_expect_bytes(
            "dense output",
            (const uint8_t *)output,
            (const uint8_t *)input,
            sizeof(input)) != 0) {
        return 1;
    }

    printf("PASS bira_dense_1x1\n");
    return 0;
}
