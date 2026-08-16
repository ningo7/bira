#include "bira_test_common.h"

#include <stdio.h>

static const uint8_t input[2] __attribute__((aligned(64))) = {0xa5, 0x3c};
static uint8_t weights[16][2] __attribute__((aligned(64)));
static uint8_t residual[16] __attribute__((aligned(64)));
static uint8_t parameters[8][64] __attribute__((aligned(64)));
static int32_t correction[16] __attribute__((aligned(64)));
static int8_t output[16] __attribute__((aligned(64)));
static const int8_t expected[16] __attribute__((aligned(64))) = {
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16};

int main(void)
{
    unsigned row;

    for (row = 0; row < 16; ++row) {
        weights[row][0] = input[0];
        weights[row][1] = input[1];
        correction[row] = -16;
    }
    bira_pack_binary_identity_parameters(parameters);

    bira_cfg_shape(BIRA_TEST_CTX, 1, 1, 16, 1, 1, 16, 1, 1, 0, 0);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_INPUT, BIRA_BINARY_INPUT_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_WEIGHT_LOW, BIRA_BINARY_WEIGHT_BASE);
    bira_cfg_addr(BIRA_TEST_CTX, BIRA_ROLE_PARAM, BIRA_PARAMETER_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_RESIDUAL, BIRA_FULL_RESIDUAL_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_CORRECTION, BIRA_CORRECTION_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_ACCUMULATOR, BIRA_ACCUMULATOR_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_OUTPUT_FULL, BIRA_FULL_OUTPUT_BASE);
    bira_cfg_mode(
        BIRA_TEST_CTX,
        BIRA_ARRAY_BINARY,
        BIRA_WEIGHT_W2,
        false,
        BIRA_POST_BINARY_FUSED,
        false,
        true,
        false);
    bira_cfg_commit(BIRA_TEST_CTX);

    bira_cpu_fence();
    bira_load_2d(
        input, bira_descriptor(BIRA_ROLE_INPUT, 0, 1, 2, 2));
    bira_load_2d(
        weights, bira_descriptor(BIRA_ROLE_WEIGHT_LOW, 0, 16, 2, 2));
    bira_load_2d(
        residual, bira_descriptor(BIRA_ROLE_RESIDUAL, 0, 1, 16, 16));
    bira_load_2d(
        parameters, bira_descriptor(BIRA_ROLE_PARAM, 0, 8, 64, 64));
    bira_load_2d(
        correction,
        bira_descriptor(BIRA_ROLE_CORRECTION, 0, 1, 64, 64));
    bira_exec_conv(BIRA_TEST_CTX);
    bira_store_2d(
        output, bira_descriptor(BIRA_ROLE_OUTPUT_FULL, 0, 1, 16, 16));

    if (bira_expect_fence_ok("binary execution") != 0) {
        return 1;
    }
    bira_cpu_fence();
    if (bira_expect_bytes(
            "binary output",
            (const uint8_t *)output,
            (const uint8_t *)expected,
            sizeof(expected)) != 0) {
        return 1;
    }

    printf("PASS bira_binary_1x1\n");
    return 0;
}
