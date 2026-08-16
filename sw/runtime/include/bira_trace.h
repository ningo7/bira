#ifndef BIRA_TRACE_H
#define BIRA_TRACE_H

#include <stddef.h>
#include <stdint.h>

#include "bira_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uintptr_t host_address;
    size_t size;
    uint64_t device_address;
    bool output;
} bira_trace_mapping_t;

typedef struct {
    bira_command_t *commands;
    size_t command_count;
    size_t command_capacity;

    uint8_t *memory;
    size_t memory_size;
    size_t memory_capacity;
    uint64_t memory_base;

    bira_trace_mapping_t *mappings;
    size_t mapping_count;
    size_t mapping_capacity;
} bira_trace_t;

int bira_trace_init(bira_trace_t *trace, uint64_t memory_base);

void bira_trace_destroy(bira_trace_t *trace);

bira_driver_t bira_trace_driver(bira_trace_t *trace);

int bira_trace_issue(
    void *user,
    const bira_command_t *command,
    uint64_t *result);

int bira_trace_write_files(
    const bira_trace_t *trace,
    const char *command_path,
    const char *memory_path,
    const char *manifest_path);

#ifdef __cplusplus
}
#endif

#endif
