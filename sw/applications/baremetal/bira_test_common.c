#include "bira_test_common.h"

#include <stdio.h>
#include <string.h>

void bira_cpu_fence(void)
{
    __asm__ volatile("fence rw, rw" ::: "memory");
}

uint64_t bira_descriptor(
    uint32_t role,
    uint32_t local_row_offset,
    uint32_t rows,
    uint32_t bytes_per_row,
    uint32_t dram_stride_bytes)
{
    return bira_pack_dma_desc(
        BIRA_TEST_CTX,
        role,
        local_row_offset,
        rows,
        bytes_per_row,
        dram_stride_bytes,
        1);
}

int bira_expect_fence_ok(const char *phase)
{
    const uint64_t result = bira_fence_context(BIRA_TEST_CTX);
    if (result != 1) {
        printf("FAIL %s: fence=0x%lx\n", phase, result);
        return 1;
    }
    return 0;
}

int bira_expect_bytes(
    const char *name,
    const uint8_t *actual,
    const uint8_t *expected,
    size_t size)
{
    size_t index;
    for (index = 0; index < size; ++index) {
        if (actual[index] != expected[index]) {
            printf(
                "FAIL %s[%lu]: got=0x%x expected=0x%x\n",
                name,
                (unsigned long)index,
                actual[index],
                expected[index]);
            return 1;
        }
    }
    return 0;
}

void bira_set_bits(
    uint8_t *bytes,
    unsigned bit_offset,
    unsigned bit_width,
    uint64_t value)
{
    unsigned bit;
    for (bit = 0; bit < bit_width; ++bit) {
        const unsigned destination = bit_offset + bit;
        const uint8_t mask = (uint8_t)(1u << (destination & 7u));
        if ((value >> bit) & 1u) {
            bytes[destination >> 3] |= mask;
        } else {
            bytes[destination >> 3] &= (uint8_t)~mask;
        }
    }
}

static void pack_multibit_record(uint8_t *record)
{
    bira_set_bits(record, 40, 2, 1);
    bira_set_bits(record, 62, 32, (uint32_t)-128);
    bira_set_bits(record, 94, 32, 127);
}

void bira_pack_multibit_identity_parameters(uint8_t rows[8][64])
{
    unsigned row;
    memset(rows, 0, 8u * 64u);
    for (row = 0; row < 8; ++row) {
        pack_multibit_record(&rows[row][0]);
        pack_multibit_record(&rows[row][32]);
    }
}

static void pack_binary_record(uint8_t *record)
{
    bira_set_bits(record, 138, 32, (uint32_t)-128);
    bira_set_bits(record, 170, 32, 127);
}

void bira_pack_binary_identity_parameters(uint8_t rows[8][64])
{
    unsigned row;
    memset(rows, 0, 8u * 64u);
    for (row = 0; row < 8; ++row) {
        pack_binary_record(&rows[row][0]);
        pack_binary_record(&rows[row][32]);
    }
}
