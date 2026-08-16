#include "bira_test_common.h"

#include <stdio.h>

static const uint8_t input[4][16] __attribute__((aligned(64))) = {
    {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
     0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00},
    {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31},
    {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32},
};
static uint8_t output[4][16] __attribute__((aligned(64)));

int main(void)
{
    const uint64_t descriptor =
        bira_descriptor(BIRA_ROLE_INPUT, 0, 4, 16, 16);

    bira_cfg_shape(BIRA_TEST_CTX, 1, 1, 16, 1, 1, 16, 1, 1, 0, 0);
    bira_cfg_addr(BIRA_TEST_CTX, BIRA_ROLE_INPUT, BIRA_FULL_INPUT_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_WEIGHT_LOW, BIRA_FULL_WEIGHT_BASE);
    bira_cfg_addr(BIRA_TEST_CTX, BIRA_ROLE_PARAM, BIRA_PARAMETER_BASE);
    bira_cfg_addr(
        BIRA_TEST_CTX, BIRA_ROLE_ACCUMULATOR, BIRA_ACCUMULATOR_BASE);
    bira_cfg_mode(
        BIRA_TEST_CTX,
        BIRA_ARRAY_DENSE,
        BIRA_WEIGHT_W8,
        true,
        BIRA_POST_NONE,
        false,
        false,
        false);
    bira_cfg_commit(BIRA_TEST_CTX);

    bira_cpu_fence();
    bira_load_2d(input, descriptor);
    if (bira_expect_fence_ok("DMA load") != 0) {
        return 1;
    }
    bira_store_2d(output, descriptor);
    if (bira_expect_fence_ok("DMA store") != 0) {
        return 1;
    }
    bira_cpu_fence();

    if (bira_expect_bytes(
            "DMA loopback", &output[0][0], &input[0][0], sizeof(input)) != 0) {
        return 1;
    }
    printf("PASS bira_dma_loopback\n");
    return 0;
}
