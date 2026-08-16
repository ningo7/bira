#include "bira_test_common.h"

#include <stdio.h>

int main(void)
{
    uint64_t status;

    status = bira_status_global(true);
    if (status != 0) {
        printf("FAIL initial global status=0x%lx\n", status);
        return 1;
    }
    if (bira_tlb_flush() != 0) {
        printf("FAIL TLB flush\n");
        return 1;
    }
    if (bira_fence_all() != 1) {
        printf("FAIL empty global fence\n");
        return 1;
    }

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

    status = bira_status_context(BIRA_TEST_CTX, false);
    if ((status & 0xf) != 0xd || (status & (1ull << 12)) != 0) {
        printf("FAIL committed context status=0x%lx\n", status);
        return 1;
    }
    if (bira_expect_fence_ok("status") != 0) {
        return 1;
    }

    printf("PASS bira_status\n");
    return 0;
}
