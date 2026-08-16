#ifndef BIRA_RUNTIME_H
#define BIRA_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bira_rocc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BIRA_CONTEXT_AUTO UINT8_C(0xff)
#define BIRA_MAX_CONTEXTS 8u
#define BIRA_ADDR_ROLE_COUNT 9u

enum bira_result {
    BIRA_OK = 0,
    BIRA_ERR_ARGUMENT = -1,
    BIRA_ERR_RANGE = -2,
    BIRA_ERR_UNSUPPORTED = -3,
    BIRA_ERR_DRIVER = -4,
    BIRA_ERR_CAPACITY = -5,
    BIRA_ERR_IO = -6,
};

typedef struct {
    uint8_t funct;
    bool returns_value;
    uint64_t rs1;
    uint64_t rs2;
} bira_command_t;

typedef int (*bira_issue_fn)(
    void *user,
    const bira_command_t *command,
    uint64_t *result);

typedef struct {
    bira_issue_fn issue;
    void *user;
} bira_driver_t;

typedef struct {
    uint16_t dim;
    uint16_t bank_rows;
    uint16_t parameter_rows;
    uint8_t full_banks;
    uint8_t binary_banks;
    uint8_t accumulator_banks;
    uint8_t contexts;
    uint16_t max_image_height;
    uint16_t max_image_width;
} bira_capabilities_t;

typedef struct {
    bira_driver_t driver;
    bira_capabilities_t capabilities;
    uint8_t next_context;
} bira_runtime_t;

typedef struct {
    const void *dram_address;
    uint8_t role;
    uint16_t local_row_offset;
    uint16_t rows;
    uint8_t bytes_per_row;
    uint16_t dram_stride_bytes;
    uint8_t local_stride_rows;
} bira_load_t;

typedef struct {
    void *dram_address;
    uint8_t role;
    uint16_t local_row_offset;
    uint16_t rows;
    uint8_t bytes_per_row;
    uint16_t dram_stride_bytes;
    uint8_t local_stride_rows;
} bira_store_t;

typedef struct {
    uint16_t input_height;
    uint16_t input_width;
    uint16_t input_channels;
    uint16_t output_height;
    uint16_t output_width;
    uint16_t output_channels;
    uint8_t kernel_height;
    uint8_t kernel_width;
    uint8_t padding_height;
    uint8_t padding_width;
} bira_shape_t;

typedef struct {
    uint8_t array_mode;
    uint8_t weight_precision;
    bool input_signed;
    uint8_t post_mode;
    bool shuffle_pack2;
    bool write_full;
    bool write_binary;
} bira_mode_t;

/*
 * A layer is one hardware EXEC_CONV transaction plus its associated DMA.
 * Weight and parameter blobs must already use the native BIRA layout.
 *
 * address_mask bit r indicates that base_rows[r] is part of this context.
 * Loads and stores are submitted before/after EXEC without an
 * implicit fence, so the hardware reservation station may overlap them.
 */
typedef struct {
    const char *name;
    uint8_t context_id;
    bira_shape_t shape;
    bira_mode_t mode;
    uint16_t address_mask;
    uint16_t base_rows[BIRA_ADDR_ROLE_COUNT];
    const bira_load_t *loads;
    size_t load_count;
    const bira_store_t *stores;
    size_t store_count;
} bira_layer_t;

typedef struct {
    uint8_t context_id;
} bira_event_t;

void bira_runtime_init(
    bira_runtime_t *runtime,
    bira_driver_t driver,
    const bira_capabilities_t *capabilities);

bira_capabilities_t bira_default_capabilities(void);

int bira_issue(
    bira_runtime_t *runtime,
    uint8_t funct,
    uint64_t rs1,
    uint64_t rs2,
    bool returns_value,
    uint64_t *result);

int bira_validate_layer(
    const bira_runtime_t *runtime,
    const bira_layer_t *layer);

int bira_submit_layer(
    bira_runtime_t *runtime,
    const bira_layer_t *layer,
    bira_event_t *event);

int bira_wait_event(
    bira_runtime_t *runtime,
    const bira_event_t *event);

int bira_wait_all(bira_runtime_t *runtime);

int bira_run_layer(
    bira_runtime_t *runtime,
    const bira_layer_t *layer);

/*
 * Hardware-independent entry used by the RISC-V driver. On a non-RISC-V host
 * it returns BIRA_ERR_UNSUPPORTED, allowing the same operator sources to be
 * compiled with the Trace backend.
 */
int bira_rocc_issue(
    void *user,
    const bira_command_t *command,
    uint64_t *result);

bira_driver_t bira_rocc_driver(void);

#ifdef __cplusplus
}
#endif

#endif
