#ifndef BIRA_ROCC_H
#define BIRA_ROCC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum bira_funct {
    BIRA_FUNCT_CFG_SHAPE  = 0x40,
    BIRA_FUNCT_CFG_ADDR   = 0x41,
    BIRA_FUNCT_CFG_MODE   = 0x42,
    BIRA_FUNCT_CFG_COMMIT = 0x43,
    BIRA_FUNCT_LOAD_2D    = 0x44,
    BIRA_FUNCT_EXEC_CONV  = 0x45,
    BIRA_FUNCT_STORE_2D   = 0x46,
    BIRA_FUNCT_FENCE      = 0x47,
    BIRA_FUNCT_STATUS     = 0x48,
    BIRA_FUNCT_TLB_FLUSH  = 0x49,
};

enum bira_addr_role {
    BIRA_ROLE_INPUT         = 0,
    BIRA_ROLE_WEIGHT_LOW    = 1,
    BIRA_ROLE_WEIGHT_HIGH   = 2,
    BIRA_ROLE_PARAM         = 3,
    BIRA_ROLE_RESIDUAL      = 4,
    /* Packed per-pixel binary Accumulator initialization biases (-N). */
    BIRA_ROLE_CORRECTION    = 5,
    BIRA_ROLE_ACCUMULATOR   = 6,
    BIRA_ROLE_OUTPUT_FULL   = 7,
    BIRA_ROLE_OUTPUT_BINARY = 8,
};

enum bira_array_mode {
    BIRA_ARRAY_DENSE         = 0,
    BIRA_ARRAY_DEPTHWISE     = 1,
    BIRA_ARRAY_BINARY        = 2,
    BIRA_ARRAY_COLUMN_REDUCE = 3,
};

enum bira_weight_precision {
    BIRA_WEIGHT_W2  = 0,
    BIRA_WEIGHT_W4  = 1,
    BIRA_WEIGHT_W8  = 2,
    BIRA_WEIGHT_W16 = 3,
};

enum bira_post_mode {
    BIRA_POST_NONE                    = 0,
    BIRA_POST_INT_PRELU               = 1,
    BIRA_POST_INT_RELU                = 2,
    BIRA_POST_INT_SIGNED_SIGN         = 3,
    BIRA_POST_BINARY_FUSED            = 4,
    BIRA_POST_FINAL_BILINEAR_RESIDUAL = 5,
};

static inline uint64_t bira_pack_shape_a(
    uint32_t ctx_id,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t input_c,
    uint32_t output_h)
{
    return (uint64_t)ctx_id
        | ((uint64_t)input_h << 3)
        | ((uint64_t)input_w << 15)
        | ((uint64_t)input_c << 27)
        | ((uint64_t)output_h << 39);
}

static inline uint64_t bira_pack_shape_b(
    uint32_t output_w,
    uint32_t output_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t pad_h,
    uint32_t pad_w)
{
    return (uint64_t)output_w
        | ((uint64_t)output_c << 12)
        | ((uint64_t)kernel_h << 24)
        | ((uint64_t)kernel_w << 28)
        | ((uint64_t)pad_h << 32)
        | ((uint64_t)pad_w << 36);
}

static inline uint64_t bira_pack_addr(
    uint32_t ctx_id,
    uint32_t role,
    uint32_t base_row)
{
    return (uint64_t)ctx_id
        | ((uint64_t)role << 3)
        | ((uint64_t)base_row << 7);
}

static inline uint64_t bira_pack_mode(
    uint32_t ctx_id,
    uint32_t array_mode,
    uint32_t weight_precision,
    bool input_signed,
    uint32_t post_mode,
    bool shuffle_pack2,
    bool write_full,
    bool write_binary)
{
    return (uint64_t)ctx_id
        | ((uint64_t)array_mode << 3)
        | ((uint64_t)weight_precision << 5)
        | ((uint64_t)input_signed << 7)
        | ((uint64_t)post_mode << 8)
        | ((uint64_t)shuffle_pack2 << 11)
        | ((uint64_t)write_full << 12)
        | ((uint64_t)write_binary << 13);
}

static inline bool bira_dma_desc_is_valid(
    uint32_t ctx_id,
    uint32_t role,
    uint32_t local_row_offset,
    uint32_t rows,
    uint32_t bytes_per_row,
    uint32_t dram_stride_bytes,
    uint32_t local_stride_rows)
{
    return ctx_id < 8 && role < 16
        && local_row_offset < (1u << 14)
        && rows > 0 && rows < (1u << 14)
        && bytes_per_row > 0 && bytes_per_row < (1u << 7)
        && dram_stride_bytes >= bytes_per_row
        && dram_stride_bytes < (1u << 16)
        && local_stride_rows > 0 && local_stride_rows < (1u << 6);
}

static inline uint64_t bira_pack_dma_desc(
    uint32_t ctx_id,
    uint32_t role,
    uint32_t local_row_offset,
    uint32_t rows,
    uint32_t bytes_per_row,
    uint32_t dram_stride_bytes,
    uint32_t local_stride_rows)
{
    return (uint64_t)ctx_id
        | ((uint64_t)role << 3)
        | ((uint64_t)local_row_offset << 7)
        | ((uint64_t)rows << 21)
        | ((uint64_t)bytes_per_row << 35)
        | ((uint64_t)dram_stride_bytes << 42)
        | ((uint64_t)local_stride_rows << 58);
}

#if defined(__riscv) && (__riscv_xlen == 64)
#define BIRA_HAS_ROCC 1

#define BIRA_INSN_NORET(funct, value1, value2)                         \
    __asm__ volatile (                                                \
        ".insn r 0x7b, 3, %2, x0, %0, %1"                            \
        :                                                             \
        : "r"((uint64_t)(value1)), "r"((uint64_t)(value2)),          \
          "i"(funct)                                                  \
        : "memory")

#define BIRA_INSN_RET(funct, value1, value2)                           \
    ({                                                                \
        uint64_t bira_result__;                                        \
        __asm__ volatile (                                            \
            ".insn r 0x7b, 7, %3, %0, %1, %2"                        \
            : "=r"(bira_result__)                                     \
            : "r"((uint64_t)(value1)), "r"((uint64_t)(value2)),      \
              "i"(funct)                                             \
            : "memory");                                              \
        bira_result__;                                                  \
    })

static inline void bira_cfg_shape(
    uint32_t ctx_id,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t input_c,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t output_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t pad_h,
    uint32_t pad_w)
{
    BIRA_INSN_NORET(
        BIRA_FUNCT_CFG_SHAPE,
        bira_pack_shape_a(ctx_id, input_h, input_w, input_c, output_h),
        bira_pack_shape_b(output_w, output_c, kernel_h, kernel_w, pad_h, pad_w));
}

static inline void bira_cfg_addr(
    uint32_t ctx_id,
    uint32_t role,
    uint32_t base_row)
{
    BIRA_INSN_NORET(
        BIRA_FUNCT_CFG_ADDR, bira_pack_addr(ctx_id, role, base_row), 0);
}

static inline void bira_cfg_mode(
    uint32_t ctx_id,
    uint32_t array_mode,
    uint32_t weight_precision,
    bool input_signed,
    uint32_t post_mode,
    bool shuffle_pack2,
    bool write_full,
    bool write_binary)
{
    BIRA_INSN_NORET(
        BIRA_FUNCT_CFG_MODE,
        bira_pack_mode(ctx_id, array_mode, weight_precision, input_signed,
                      post_mode, shuffle_pack2, write_full, write_binary),
        0);
}

static inline void bira_cfg_commit(uint32_t ctx_id)
{
    BIRA_INSN_NORET(BIRA_FUNCT_CFG_COMMIT, ctx_id, 0);
}

static inline void bira_load_2d(const void *dram_va, uint64_t descriptor)
{
    BIRA_INSN_NORET(
        BIRA_FUNCT_LOAD_2D, (uintptr_t)dram_va, descriptor);
}

static inline void bira_exec_conv(uint32_t ctx_id)
{
    BIRA_INSN_NORET(BIRA_FUNCT_EXEC_CONV, ctx_id, 0);
}

static inline void bira_store_2d(void *dram_va, uint64_t descriptor)
{
    BIRA_INSN_NORET(
        BIRA_FUNCT_STORE_2D, (uintptr_t)dram_va, descriptor);
}

static inline uint64_t bira_fence_all(void)
{
    return BIRA_INSN_RET(BIRA_FUNCT_FENCE, 0, 0);
}

static inline uint64_t bira_fence_context(uint32_t ctx_id)
{
    return BIRA_INSN_RET(
        BIRA_FUNCT_FENCE, 1u | ((uint64_t)ctx_id << 1), 0);
}

static inline uint64_t bira_status_global(bool clear_error)
{
    return BIRA_INSN_RET(
        BIRA_FUNCT_STATUS, (uint64_t)clear_error << 4, 0);
}

static inline uint64_t bira_status_context(
    uint32_t ctx_id,
    bool clear_error)
{
    const uint64_t request =
        1u | ((uint64_t)ctx_id << 1) | ((uint64_t)clear_error << 4);
    return BIRA_INSN_RET(BIRA_FUNCT_STATUS, request, 0);
}

static inline uint64_t bira_tlb_flush(void)
{
    return BIRA_INSN_RET(BIRA_FUNCT_TLB_FLUSH, 0, 0);
}

#else
#define BIRA_HAS_ROCC 0
#endif

#endif
