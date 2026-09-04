#include "bira_runtime.h"

#include <limits.h>
#include <string.h>

static bool nonzero_u16(uint16_t value)
{
    return value != 0u;
}

static bool valid_shape(const bira_shape_t *shape)
{
    return nonzero_u16(shape->input_height)
        && nonzero_u16(shape->input_width)
        && nonzero_u16(shape->input_channels)
        && nonzero_u16(shape->output_height)
        && nonzero_u16(shape->output_width)
        && nonzero_u16(shape->output_channels)
        && shape->kernel_height != 0u
        && shape->kernel_width != 0u;
}

static bool valid_dma(
    uint8_t context_id,
    uint8_t role,
    uint16_t local_row_offset,
    uint16_t rows,
    uint8_t bytes_per_row,
    uint16_t dram_stride_bytes,
    uint8_t local_stride_rows)
{
    return bira_dma_desc_is_valid(
        context_id,
        role,
        local_row_offset,
        rows,
        bytes_per_row,
        dram_stride_bytes,
        local_stride_rows);
}

static uint64_t dma_descriptor(
    uint8_t context_id,
    uint8_t role,
    uint16_t local_row_offset,
    uint16_t rows,
    uint8_t bytes_per_row,
    uint16_t dram_stride_bytes,
    uint8_t local_stride_rows)
{
    return bira_pack_dma_desc(
        context_id,
        role,
        local_row_offset,
        rows,
        bytes_per_row,
        dram_stride_bytes,
        local_stride_rows);
}

bira_capabilities_t bira_default_capabilities(void)
{
    bira_capabilities_t capabilities;
    capabilities.dim = 16u;
    capabilities.bank_rows = 1024u;
    capabilities.parameter_rows = 512u;
    capabilities.full_banks = 10u;
    capabilities.binary_banks = 4u;
    capabilities.accumulator_banks = 1u;
    /*
     * The ISA context-id field can encode eight values, while the current
     * AccelParams hardware instance implements four physical contexts.
     */
    capabilities.contexts = 4u;
    capabilities.max_image_height = 128u;
    capabilities.max_image_width = 128u;
    return capabilities;
}

void bira_runtime_init(
    bira_runtime_t *runtime,
    bira_driver_t driver,
    const bira_capabilities_t *capabilities)
{
    if (runtime == NULL) {
        return;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->driver = driver;
    runtime->capabilities = capabilities == NULL
        ? bira_default_capabilities()
        : *capabilities;
}

int bira_issue(
    bira_runtime_t *runtime,
    uint8_t funct,
    uint64_t rs1,
    uint64_t rs2,
    bool returns_value,
    uint64_t *result)
{
    bira_command_t command;
    uint64_t ignored_result = 0u;

    if (runtime == NULL || runtime->driver.issue == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    command.funct = funct;
    command.returns_value = returns_value;
    command.rs1 = rs1;
    command.rs2 = rs2;
    if (runtime->driver.issue(
            runtime->driver.user,
            &command,
            result == NULL ? &ignored_result : result) != BIRA_OK) {
        return BIRA_ERR_DRIVER;
    }
    return BIRA_OK;
}

static int validate_mode(const bira_layer_t *layer)
{
    const bira_mode_t *mode = &layer->mode;
    const bira_shape_t *shape = &layer->shape;

    if (mode->array_mode > BIRA_ARRAY_COLUMN_REDUCE
        || mode->weight_precision > BIRA_WEIGHT_W16
        || mode->post_mode > BIRA_POST_FINAL_BILINEAR_RESIDUAL) {
        return BIRA_ERR_RANGE;
    }
    if (mode->array_mode == BIRA_ARRAY_DEPTHWISE
        && shape->input_channels != shape->output_channels) {
        return BIRA_ERR_UNSUPPORTED;
    }
    if (mode->array_mode == BIRA_ARRAY_BINARY
        && mode->weight_precision != BIRA_WEIGHT_W2) {
        return BIRA_ERR_UNSUPPORTED;
    }
    if (mode->array_mode == BIRA_ARRAY_COLUMN_REDUCE
        && shape->output_channels != 1u) {
        return BIRA_ERR_UNSUPPORTED;
    }
    if (mode->shuffle_pack2 && shape->output_channels % 16u != 0u) {
        return BIRA_ERR_UNSUPPORTED;
    }
    return BIRA_OK;
}

int bira_validate_layer(
    const bira_runtime_t *runtime,
    const bira_layer_t *layer)
{
    size_t index;
    int mode_status;
    uint8_t context_id;

    if (runtime == NULL || layer == NULL || !valid_shape(&layer->shape)) {
        return BIRA_ERR_ARGUMENT;
    }
    if (runtime->capabilities.contexts == 0u
        || runtime->capabilities.contexts > BIRA_MAX_CONTEXTS) {
        return BIRA_ERR_CAPACITY;
    }
    context_id = layer->context_id == BIRA_CONTEXT_AUTO
        ? 0u
        : layer->context_id;
    if (context_id >= runtime->capabilities.contexts) {
        return BIRA_ERR_RANGE;
    }
    if (layer->shape.input_height > runtime->capabilities.max_image_height
        || layer->shape.input_width > runtime->capabilities.max_image_width
        || layer->shape.output_height > runtime->capabilities.max_image_height
        || layer->shape.output_width > runtime->capabilities.max_image_width
        || layer->shape.input_height >= (1u << 12)
        || layer->shape.input_width >= (1u << 12)
        || layer->shape.input_channels >= (1u << 12)
        || layer->shape.output_height >= (1u << 12)
        || layer->shape.output_width >= (1u << 12)
        || layer->shape.output_channels >= (1u << 12)
        || layer->shape.kernel_height >= (1u << 4)
        || layer->shape.kernel_width >= (1u << 4)
        || layer->shape.padding_height >= (1u << 4)
        || layer->shape.padding_width >= (1u << 4)) {
        return BIRA_ERR_RANGE;
    }
    mode_status = validate_mode(layer);
    if (mode_status != BIRA_OK) {
        return mode_status;
    }
    if ((layer->address_mask >> BIRA_ADDR_ROLE_COUNT) != 0u) {
        return BIRA_ERR_RANGE;
    }
    for (index = 0u; index < BIRA_ADDR_ROLE_COUNT; ++index) {
        if (((layer->address_mask >> index) & 1u) != 0u
            && layer->base_rows[index] == UINT16_MAX) {
            return BIRA_ERR_RANGE;
        }
    }
    if (layer->load_count != 0u && layer->loads == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    for (index = 0u; index < layer->load_count; ++index) {
        const bira_load_t *load = &layer->loads[index];
        if (load->dram_address == NULL
            || !valid_dma(
                context_id,
                load->role,
                load->local_row_offset,
                load->rows,
                load->bytes_per_row,
                load->dram_stride_bytes,
                load->local_stride_rows)) {
            return BIRA_ERR_ARGUMENT;
        }
    }
    if (layer->store_count != 0u && layer->stores == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    for (index = 0u; index < layer->store_count; ++index) {
        const bira_store_t *store = &layer->stores[index];
        if (store->dram_address == NULL
            || !valid_dma(
                context_id,
                store->role,
                store->local_row_offset,
                store->rows,
                store->bytes_per_row,
                store->dram_stride_bytes,
                store->local_stride_rows)) {
            return BIRA_ERR_ARGUMENT;
        }
    }
    return BIRA_OK;
}

static uint8_t choose_context(
    bira_runtime_t *runtime,
    const bira_layer_t *layer)
{
    uint8_t context_id;
    if (layer->context_id != BIRA_CONTEXT_AUTO) {
        return layer->context_id;
    }
    context_id = runtime->next_context;
    runtime->next_context = (uint8_t)(
        (runtime->next_context + 1u) % runtime->capabilities.contexts);
    return context_id;
}

int bira_submit_layer(
    bira_runtime_t *runtime,
    const bira_layer_t *layer,
    bira_event_t *event)
{
    size_t index;
    uint8_t context_id;
    int status = bira_validate_layer(runtime, layer);
    if (status != BIRA_OK) {
        return status;
    }

    context_id = choose_context(runtime, layer);
    status = bira_issue(
        runtime,
        BIRA_FUNCT_CFG_SHAPE,
        bira_pack_shape_a(
            context_id,
            layer->shape.input_height,
            layer->shape.input_width,
            layer->shape.input_channels,
            layer->shape.output_height),
        bira_pack_shape_b(
            layer->shape.output_width,
            layer->shape.output_channels,
            layer->shape.kernel_height,
            layer->shape.kernel_width,
            layer->shape.padding_height,
            layer->shape.padding_width),
        false,
        NULL);
    if (status != BIRA_OK) {
        return status;
    }

    for (index = 0u; index < BIRA_ADDR_ROLE_COUNT; ++index) {
        if (((layer->address_mask >> index) & 1u) == 0u) {
            continue;
        }
        status = bira_issue(
            runtime,
            BIRA_FUNCT_CFG_ADDR,
            bira_pack_addr(
                context_id,
                (uint32_t)index,
                layer->base_rows[index]),
            0u,
            false,
            NULL);
        if (status != BIRA_OK) {
            return status;
        }
    }

    status = bira_issue(
        runtime,
        BIRA_FUNCT_CFG_MODE,
        bira_pack_mode(
            context_id,
            layer->mode.array_mode,
            layer->mode.weight_precision,
            layer->mode.input_signed,
            layer->mode.post_mode,
            layer->mode.shuffle_pack2,
            layer->mode.write_full,
            layer->mode.write_binary),
        0u,
        false,
        NULL);
    if (status != BIRA_OK) {
        return status;
    }
    status = bira_issue(
        runtime,
        BIRA_FUNCT_CFG_COMMIT,
        context_id,
        0u,
        false,
        NULL);
    if (status != BIRA_OK) {
        return status;
    }

    for (index = 0u; index < layer->load_count; ++index) {
        const bira_load_t *load = &layer->loads[index];
        status = bira_issue(
            runtime,
            BIRA_FUNCT_LOAD_2D,
            (uint64_t)(uintptr_t)load->dram_address,
            dma_descriptor(
                context_id,
                load->role,
                load->local_row_offset,
                load->rows,
                load->bytes_per_row,
                load->dram_stride_bytes,
                load->local_stride_rows),
            false,
            NULL);
        if (status != BIRA_OK) {
            return status;
        }
    }

    status = bira_issue(
        runtime,
        BIRA_FUNCT_EXEC_CONV,
        context_id,
        0u,
        false,
        NULL);
    if (status != BIRA_OK) {
        return status;
    }

    for (index = 0u; index < layer->store_count; ++index) {
        const bira_store_t *store = &layer->stores[index];
        status = bira_issue(
            runtime,
            BIRA_FUNCT_STORE_2D,
            (uint64_t)(uintptr_t)store->dram_address,
            dma_descriptor(
                context_id,
                store->role,
                store->local_row_offset,
                store->rows,
                store->bytes_per_row,
                store->dram_stride_bytes,
                store->local_stride_rows),
            false,
            NULL);
        if (status != BIRA_OK) {
            return status;
        }
    }

    if (event != NULL) {
        event->context_id = context_id;
    }
    return BIRA_OK;
}

int bira_wait_event(
    bira_runtime_t *runtime,
    const bira_event_t *event)
{
    uint64_t result = 0u;
    int status;
    if (runtime == NULL || event == NULL
        || event->context_id >= runtime->capabilities.contexts) {
        return BIRA_ERR_ARGUMENT;
    }
    status = bira_issue(
        runtime,
        BIRA_FUNCT_FENCE,
        1u | ((uint64_t)event->context_id << 1),
        0u,
        true,
        &result);
    if (status != BIRA_OK) {
        return status;
    }
    return (result & 1u) != 0u && ((result >> 1) & 0xffu) == 0u
        ? BIRA_OK
        : BIRA_ERR_DRIVER;
}

int bira_wait_all(bira_runtime_t *runtime)
{
    uint64_t result = 0u;
    int status = bira_issue(
        runtime,
        BIRA_FUNCT_FENCE,
        0u,
        0u,
        true,
        &result);
    if (status != BIRA_OK) {
        return status;
    }
    return (result & 1u) != 0u && ((result >> 1) & 0xffu) == 0u
        ? BIRA_OK
        : BIRA_ERR_DRIVER;
}

int bira_run_layer(
    bira_runtime_t *runtime,
    const bira_layer_t *layer)
{
    bira_event_t event;
    int status = bira_submit_layer(runtime, layer, &event);
    return status == BIRA_OK
        ? bira_wait_event(runtime, &event)
        : status;
}

int bira_rocc_issue(
    void *user,
    const bira_command_t *command,
    uint64_t *result)
{
    (void)user;
    if (command == NULL || result == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
#if BIRA_HAS_ROCC
    switch (command->funct) {
    case BIRA_FUNCT_CFG_SHAPE:
        BIRA_INSN_NORET(
            BIRA_FUNCT_CFG_SHAPE, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_CFG_ADDR:
        BIRA_INSN_NORET(
            BIRA_FUNCT_CFG_ADDR, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_CFG_MODE:
        BIRA_INSN_NORET(
            BIRA_FUNCT_CFG_MODE, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_CFG_COMMIT:
        BIRA_INSN_NORET(
            BIRA_FUNCT_CFG_COMMIT, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_LOAD_2D:
        BIRA_INSN_NORET(
            BIRA_FUNCT_LOAD_2D, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_EXEC_CONV:
        BIRA_INSN_NORET(
            BIRA_FUNCT_EXEC_CONV, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_STORE_2D:
        BIRA_INSN_NORET(
            BIRA_FUNCT_STORE_2D, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_FENCE:
        *result = BIRA_INSN_RET(
            BIRA_FUNCT_FENCE, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_STATUS:
        *result = BIRA_INSN_RET(
            BIRA_FUNCT_STATUS, command->rs1, command->rs2);
        break;
    case BIRA_FUNCT_TLB_FLUSH:
        *result = BIRA_INSN_RET(
            BIRA_FUNCT_TLB_FLUSH, command->rs1, command->rs2);
        break;
    default:
        return BIRA_ERR_UNSUPPORTED;
    }
    if (!command->returns_value) {
        *result = 0u;
    }
    return BIRA_OK;
#else
    (void)command;
    *result = 0u;
    return BIRA_ERR_UNSUPPORTED;
#endif
}

bira_driver_t bira_rocc_driver(void)
{
    bira_driver_t driver;
    driver.issue = bira_rocc_issue;
    driver.user = NULL;
    return driver;
}
