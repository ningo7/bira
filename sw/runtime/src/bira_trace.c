#include "bira_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIRA_TRACE_DEFAULT_BASE UINT64_C(0x10000)
#define BIRA_TRACE_ALIGNMENT 64u
#define BIRA_TRACE_VERSION 1u

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static int grow_array(
    void **buffer,
    size_t *capacity,
    size_t element_size,
    size_t required)
{
    size_t new_capacity;
    void *new_buffer;

    if (*capacity >= required) {
        return BIRA_OK;
    }
    new_capacity = *capacity == 0u ? 16u : *capacity;
    while (new_capacity < required) {
        if (new_capacity > SIZE_MAX / 2u) {
            return BIRA_ERR_CAPACITY;
        }
        new_capacity *= 2u;
    }
    if (new_capacity > SIZE_MAX / element_size) {
        return BIRA_ERR_CAPACITY;
    }
    new_buffer = realloc(*buffer, new_capacity * element_size);
    if (new_buffer == NULL) {
        return BIRA_ERR_CAPACITY;
    }
    *buffer = new_buffer;
    *capacity = new_capacity;
    return BIRA_OK;
}

static int grow_memory(bira_trace_t *trace, size_t required)
{
    size_t old_capacity = trace->memory_capacity;
    int status = grow_array(
        (void **)&trace->memory,
        &trace->memory_capacity,
        sizeof(uint8_t),
        required);
    if (status == BIRA_OK && trace->memory_capacity > old_capacity) {
        memset(
            trace->memory + old_capacity,
            0,
            trace->memory_capacity - old_capacity);
    }
    return status;
}

static size_t dma_span(uint64_t descriptor)
{
    size_t rows = (size_t)((descriptor >> 21) & UINT64_C(0x3fff));
    size_t bytes_per_row =
        (size_t)((descriptor >> 35) & UINT64_C(0x7f));
    size_t stride = (size_t)((descriptor >> 42) & UINT64_C(0xffff));

    if (rows == 0u || bytes_per_row == 0u || stride < bytes_per_row) {
        return 0u;
    }
    if (rows - 1u > (SIZE_MAX - bytes_per_row) / stride) {
        return 0u;
    }
    return (rows - 1u) * stride + bytes_per_row;
}

static bira_trace_mapping_t *find_mapping(
    bira_trace_t *trace,
    uintptr_t host_address,
    size_t size)
{
    size_t index;
    for (index = 0u; index < trace->mapping_count; ++index) {
        bira_trace_mapping_t *mapping = &trace->mappings[index];
        if (mapping->host_address == host_address && mapping->size >= size) {
            return mapping;
        }
    }
    return NULL;
}

static int map_dma_address(
    bira_trace_t *trace,
    uint64_t host_address_value,
    size_t size,
    bool output,
    uint64_t *device_address)
{
    uintptr_t host_address = (uintptr_t)host_address_value;
    bira_trace_mapping_t *mapping;
    size_t offset;
    size_t required;
    int status;

    if (host_address == (uintptr_t)0 || size == 0u
        || device_address == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    mapping = find_mapping(trace, host_address, size);
    if (mapping != NULL) {
        if (!output) {
            memcpy(
                trace->memory
                    + (size_t)(mapping->device_address - trace->memory_base),
                (const void *)host_address,
                size);
        }
        mapping->output = mapping->output || output;
        *device_address = mapping->device_address;
        return BIRA_OK;
    }

    offset = align_up(trace->memory_size, BIRA_TRACE_ALIGNMENT);
    if (offset > SIZE_MAX - size) {
        return BIRA_ERR_CAPACITY;
    }
    required = offset + size;
    status = grow_memory(trace, required);
    if (status != BIRA_OK) {
        return status;
    }
    status = grow_array(
        (void **)&trace->mappings,
        &trace->mapping_capacity,
        sizeof(*trace->mappings),
        trace->mapping_count + 1u);
    if (status != BIRA_OK) {
        return status;
    }
    memset(trace->memory + offset, 0, size);
    if (!output) {
        memcpy(trace->memory + offset, (const void *)host_address, size);
    }

    mapping = &trace->mappings[trace->mapping_count++];
    mapping->host_address = host_address;
    mapping->size = size;
    mapping->device_address = trace->memory_base + offset;
    mapping->output = output;
    trace->memory_size = required;
    *device_address = mapping->device_address;
    return BIRA_OK;
}

int bira_trace_init(bira_trace_t *trace, uint64_t memory_base)
{
    if (trace == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    memset(trace, 0, sizeof(*trace));
    trace->memory_base =
        memory_base == 0u ? BIRA_TRACE_DEFAULT_BASE : memory_base;
    return BIRA_OK;
}

void bira_trace_destroy(bira_trace_t *trace)
{
    if (trace == NULL) {
        return;
    }
    free(trace->commands);
    free(trace->memory);
    free(trace->mappings);
    memset(trace, 0, sizeof(*trace));
}

bira_driver_t bira_trace_driver(bira_trace_t *trace)
{
    bira_driver_t driver;
    driver.issue = bira_trace_issue;
    driver.user = trace;
    return driver;
}

int bira_trace_issue(
    void *user,
    const bira_command_t *command,
    uint64_t *result)
{
    bira_trace_t *trace = (bira_trace_t *)user;
    bira_command_t recorded;
    int status;

    if (trace == NULL || command == NULL || result == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    recorded = *command;
    if (command->funct == BIRA_FUNCT_LOAD_2D
        || command->funct == BIRA_FUNCT_STORE_2D) {
        uint64_t device_address;
        size_t span = dma_span(command->rs2);
        status = map_dma_address(
            trace,
            command->rs1,
            span,
            command->funct == BIRA_FUNCT_STORE_2D,
            &device_address);
        if (status != BIRA_OK) {
            return status;
        }
        recorded.rs1 = device_address;
    }

    status = grow_array(
        (void **)&trace->commands,
        &trace->command_capacity,
        sizeof(*trace->commands),
        trace->command_count + 1u);
    if (status != BIRA_OK) {
        return status;
    }
    trace->commands[trace->command_count++] = recorded;

    if (command->funct == BIRA_FUNCT_FENCE) {
        *result = 1u;
    } else {
        *result = 0u;
    }
    return BIRA_OK;
}

static int write_u32(FILE *file, uint32_t value)
{
    uint8_t bytes[4];
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
    return fwrite(bytes, sizeof(bytes), 1u, file) == 1u
        ? BIRA_OK
        : BIRA_ERR_IO;
}

static int write_u64(FILE *file, uint64_t value)
{
    uint8_t bytes[8];
    unsigned int index;
    for (index = 0u; index < 8u; ++index) {
        bytes[index] = (uint8_t)(value >> (8u * index));
    }
    return fwrite(bytes, sizeof(bytes), 1u, file) == 1u
        ? BIRA_OK
        : BIRA_ERR_IO;
}

static int write_commands(
    const bira_trace_t *trace,
    const char *path)
{
    static const uint8_t magic[8] = {
        'B', 'I', 'R', 'A', '_', 'C', 'M', 'D'
    };
    FILE *file;
    size_t index;
    int status = BIRA_OK;

    file = fopen(path, "wb");
    if (file == NULL) {
        return BIRA_ERR_IO;
    }
    if (fwrite(magic, sizeof(magic), 1u, file) != 1u
        || write_u32(file, BIRA_TRACE_VERSION) != BIRA_OK
        || trace->command_count > UINT32_MAX
        || write_u32(file, (uint32_t)trace->command_count) != BIRA_OK
        || write_u64(file, trace->memory_base) != BIRA_OK) {
        status = BIRA_ERR_IO;
    }
    for (index = 0u; status == BIRA_OK
        && index < trace->command_count; ++index) {
        const bira_command_t *command = &trace->commands[index];
        uint8_t prefix[8] = {
            command->funct,
            command->returns_value ? 1u : 0u,
            0u, 0u, 0u, 0u, 0u, 0u
        };
        if (fwrite(prefix, sizeof(prefix), 1u, file) != 1u
            || write_u64(file, command->rs1) != BIRA_OK
            || write_u64(file, command->rs2) != BIRA_OK) {
            status = BIRA_ERR_IO;
        }
    }
    if (fclose(file) != 0) {
        status = BIRA_ERR_IO;
    }
    return status;
}

static int write_memory(
    const bira_trace_t *trace,
    const char *path)
{
    FILE *file = fopen(path, "wb");
    int status = BIRA_OK;
    if (file == NULL) {
        return BIRA_ERR_IO;
    }
    if (trace->memory_size != 0u
        && fwrite(trace->memory, trace->memory_size, 1u, file) != 1u) {
        status = BIRA_ERR_IO;
    }
    if (fclose(file) != 0) {
        status = BIRA_ERR_IO;
    }
    return status;
}

static int write_manifest(
    const bira_trace_t *trace,
    const char *path)
{
    FILE *file = fopen(path, "w");
    size_t index;
    int status = BIRA_OK;
    if (file == NULL) {
        return BIRA_ERR_IO;
    }
    if (fprintf(
            file,
            "format=bira-trace-v%u\n"
            "memory_base=0x%llx\n"
            "memory_size=%zu\n"
            "command_count=%zu\n"
            "mapping_count=%zu\n",
            BIRA_TRACE_VERSION,
            (unsigned long long)trace->memory_base,
            trace->memory_size,
            trace->command_count,
            trace->mapping_count) < 0) {
        status = BIRA_ERR_IO;
    }
    for (index = 0u; status == BIRA_OK
        && index < trace->mapping_count; ++index) {
        const bira_trace_mapping_t *mapping = &trace->mappings[index];
        if (fprintf(
                file,
                "mapping.%zu.address=0x%llx\n"
                "mapping.%zu.size=%zu\n"
                "mapping.%zu.output=%u\n",
                index,
                (unsigned long long)mapping->device_address,
                index,
                mapping->size,
                index,
                mapping->output ? 1u : 0u) < 0) {
            status = BIRA_ERR_IO;
        }
    }
    if (fclose(file) != 0) {
        status = BIRA_ERR_IO;
    }
    return status;
}

int bira_trace_write_files(
    const bira_trace_t *trace,
    const char *command_path,
    const char *memory_path,
    const char *manifest_path)
{
    int status;
    if (trace == NULL || command_path == NULL || memory_path == NULL
        || manifest_path == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    status = write_commands(trace, command_path);
    if (status != BIRA_OK) {
        return status;
    }
    status = write_memory(trace, memory_path);
    return status == BIRA_OK
        ? write_manifest(trace, manifest_path)
        : status;
}
