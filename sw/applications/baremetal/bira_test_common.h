#ifndef BIRA_TEST_COMMON_H
#define BIRA_TEST_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "bira_rocc.h"

#define BIRA_TEST_CTX 0u

#define BIRA_FULL_INPUT_BASE 0u
#define BIRA_FULL_WEIGHT_BASE 1024u
#define BIRA_FULL_RESIDUAL_BASE 2048u
#define BIRA_FULL_OUTPUT_BASE 3072u

#define BIRA_BINARY_INPUT_BASE 0u
#define BIRA_BINARY_WEIGHT_BASE 1024u
#define BIRA_BINARY_OUTPUT_BASE 2048u

#define BIRA_PARAMETER_BASE 0u
#define BIRA_CORRECTION_BASE 64u
#define BIRA_ACCUMULATOR_BASE 0u

void bira_cpu_fence(void);
uint64_t bira_descriptor(
    uint32_t role,
    uint32_t local_row_offset,
    uint32_t rows,
    uint32_t bytes_per_row,
    uint32_t dram_stride_bytes);
int bira_expect_fence_ok(const char *phase);
int bira_expect_bytes(
    const char *name,
    const uint8_t *actual,
    const uint8_t *expected,
    size_t size);
void bira_set_bits(
    uint8_t *bytes,
    unsigned bit_offset,
    unsigned bit_width,
    uint64_t value);
void bira_pack_multibit_identity_parameters(uint8_t rows[8][64]);
void bira_pack_binary_identity_parameters(uint8_t rows[8][64]);

#endif
