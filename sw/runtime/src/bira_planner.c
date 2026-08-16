#include "bira_inference.h"

#include <limits.h>
#include <string.h>

typedef struct {
    uint16_t first;
    uint16_t last;
    bool referenced;
    bool produced;
    bool consumed;
} tensor_lifetime_t;

static uint32_t divide_up_u32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static bool valid_tensor_id(
    const bira_inference_desc_t *inference,
    uint16_t tensor)
{
    return tensor == BIRA_TENSOR_NONE
        || tensor < inference->tensor_count;
}

static uint16_t blob_stride(const bira_native_blob_t *blob)
{
    return blob->stride_bytes == 0u
        ? blob->bytes_per_row
        : blob->stride_bytes;
}

static bool lifetimes_overlap(
    uint16_t first_a,
    uint16_t last_a,
    uint16_t first_b,
    uint16_t last_b)
{
    return first_a <= last_b && first_b <= last_a;
}

static uint8_t tensor_space(const bira_tensor_desc_t *tensor)
{
    return tensor->layout == BIRA_LAYOUT_BINARY_HWC16
        ? BIRA_SPACE_BINARY
        : BIRA_SPACE_FULL;
}

static void mark_use(
    tensor_lifetime_t *lifetimes,
    uint16_t tensor,
    uint16_t layer_offset,
    bool produced)
{
    tensor_lifetime_t *lifetime;
    if (tensor == BIRA_TENSOR_NONE) {
        return;
    }
    lifetime = &lifetimes[tensor];
    if (!lifetime->referenced) {
        lifetime->first = layer_offset;
        lifetime->last = layer_offset;
        lifetime->referenced = true;
    } else {
        if (layer_offset < lifetime->first) {
            lifetime->first = layer_offset;
        }
        if (layer_offset > lifetime->last) {
            lifetime->last = layer_offset;
        }
    }
    if (produced) {
        lifetime->produced = true;
    } else {
        lifetime->consumed = true;
    }
}

static bool tensor_has_binding(
    const bira_tensor_binding_t *bindings,
    size_t binding_count,
    uint16_t tensor)
{
    size_t index;
    for (index = 0u; index < binding_count; ++index) {
        if (bindings[index].tensor == tensor
            && bindings[index].data != NULL) {
            return true;
        }
    }
    return false;
}

static int build_lifetimes(
    bira_inference_workspace_t *workspace,
    tensor_lifetime_t *lifetimes,
    const bira_tensor_binding_t *bindings,
    size_t binding_count)
{
    const bira_inference_desc_t *inference =
        workspace->inference;
    uint16_t offset;
    size_t tensor;

    memset(
        lifetimes,
        0,
        sizeof(*lifetimes) * BIRA_INFERENCE_MAX_TENSORS);
    for (offset = 0u; offset < workspace->layer_count; ++offset) {
        const bira_layer_desc_t *op =
            &inference->layers[offset];
        if (!valid_tensor_id(inference, op->input_tensor)
            || !valid_tensor_id(inference, op->residual_tensor)
            || !valid_tensor_id(inference, op->output_tensor)
            || !valid_tensor_id(
                inference, op->binary_output_tensor)
            || op->input_tensor == BIRA_TENSOR_NONE
            || op->output_tensor == BIRA_TENSOR_NONE) {
            return BIRA_ERR_RANGE;
        }
        mark_use(lifetimes, op->input_tensor, offset, false);
        mark_use(lifetimes, op->residual_tensor, offset, false);
        mark_use(lifetimes, op->output_tensor, offset, true);
        mark_use(
            lifetimes, op->binary_output_tensor, offset, true);
    }

    for (tensor = 0u;
         tensor < inference->tensor_count;
         ++tensor) {
        bira_tensor_plan_t *plan = &workspace->tensors[tensor];
        const tensor_lifetime_t *lifetime = &lifetimes[tensor];
        uint32_t rows;
        if (!lifetime->referenced) {
            continue;
        }
        rows = bira_tensor_rows(&inference->tensors[tensor]);
        if (rows == 0u || rows > UINT16_MAX
            || bira_tensor_bytes_per_row(
                &inference->tensors[tensor]) == 0u) {
            return BIRA_ERR_RANGE;
        }
        plan->allocated = true;
        plan->space = tensor_space(
            &inference->tensors[tensor]);
        plan->first_layer = lifetime->first;
        plan->last_layer = lifetime->last;
        plan->rows = (uint16_t)rows;
        plan->external_input = lifetime->consumed
            && !lifetime->produced;
        plan->external_output = lifetime->produced
            && (!lifetime->consumed
                || tensor_has_binding(
                    bindings,
                    binding_count,
                    (uint16_t)tensor));
    }
    return BIRA_OK;
}

static uint16_t maximum_weight_rows(
    const bira_inference_workspace_t *workspace,
    bool binary)
{
    uint16_t maximum = 0u;
    uint16_t offset;
    for (offset = 0u; offset < workspace->layer_count; ++offset) {
        const bira_layer_desc_t *op =
            &workspace->inference->layers[offset];
        uint32_t rows;
        bool is_binary = op->type == BIRA_OP_BINARY_CONV2D;
        if (is_binary != binary) {
            continue;
        }
        rows = op->weight_low.rows + op->weight_high.rows;
        if (rows > maximum && rows <= UINT16_MAX) {
            maximum = (uint16_t)rows;
        }
    }
    return maximum;
}

static bool range_available(
    const bira_inference_workspace_t *workspace,
    size_t tensor,
    uint8_t space,
    uint16_t base,
    uint16_t rows)
{
    size_t other;
    const bira_tensor_plan_t *candidate =
        &workspace->tensors[tensor];
    for (other = 0u;
         other < workspace->inference->tensor_count;
         ++other) {
        const bira_tensor_plan_t *allocated =
            &workspace->tensors[other];
        if (other == tensor || !allocated->allocated
            || allocated->base_row == UINT16_MAX
            || allocated->space != space
            || !lifetimes_overlap(
                candidate->first_layer,
                candidate->last_layer,
                allocated->first_layer,
                allocated->last_layer)) {
            continue;
        }
        if (base < allocated->base_row + allocated->rows
            && allocated->base_row < base + rows) {
            return false;
        }
    }
    return true;
}

static int allocate_tensor_rows(
    bira_inference_workspace_t *workspace,
    const bira_runtime_t *runtime,
    size_t tensor,
    uint16_t full_weight_rows,
    uint16_t binary_weight_rows,
    uint16_t *full_tail_cursor,
    uint16_t *binary_tail_cursor)
{
    bira_tensor_plan_t *plan = &workspace->tensors[tensor];
    uint16_t bank_rows = runtime->capabilities.bank_rows;
    uint16_t data_banks;
    uint16_t scratch_base;
    uint16_t scratch_end;
    uint16_t *tail_cursor;
    uint16_t weight_rows;
    uint16_t bank;
    uint16_t needed_banks;

    if (!plan->allocated) {
        return BIRA_OK;
    }
    if (plan->space == BIRA_SPACE_FULL) {
        data_banks = runtime->capabilities.full_banks - 1u;
        scratch_base = data_banks * bank_rows;
        scratch_end =
            runtime->capabilities.full_banks * bank_rows;
        tail_cursor = full_tail_cursor;
        weight_rows = full_weight_rows;
    } else {
        data_banks = runtime->capabilities.binary_banks - 1u;
        scratch_base = data_banks * bank_rows;
        scratch_end =
            runtime->capabilities.binary_banks * bank_rows;
        tail_cursor = binary_tail_cursor;
        weight_rows = binary_weight_rows;
    }

    /*
     * Small group inputs with long lifetimes share the unused tail of the
     * weight scratch bank. This keeps skip/residual inputs resident without
     * taking a complete data bank.
     */
    if (plan->external_input && plan->rows < bank_rows
        && *tail_cursor >= scratch_base + weight_rows + plan->rows) {
        *tail_cursor = (uint16_t)(*tail_cursor - plan->rows);
        plan->base_row = *tail_cursor;
        return BIRA_OK;
    }

    needed_banks = (uint16_t)divide_up_u32(plan->rows, bank_rows);
    for (bank = 0u; bank + needed_banks <= data_banks; ++bank) {
        uint16_t base = bank * bank_rows;
        uint16_t reserved_rows = needed_banks * bank_rows;
        if (range_available(
                workspace,
                tensor,
                plan->space,
                base,
                reserved_rows)) {
            plan->base_row = base;
            plan->rows = reserved_rows;
            return BIRA_OK;
        }
    }
    (void)scratch_end;
    return BIRA_ERR_CAPACITY;
}

static int allocate_tensors(
    bira_inference_workspace_t *workspace,
    const bira_runtime_t *runtime)
{
    uint16_t full_weight_rows = maximum_weight_rows(workspace, false);
    uint16_t binary_weight_rows = maximum_weight_rows(workspace, true);
    uint32_t full_total =
        (uint32_t)runtime->capabilities.full_banks
        * runtime->capabilities.bank_rows;
    uint32_t binary_total =
        (uint32_t)runtime->capabilities.binary_banks
        * runtime->capabilities.bank_rows;
    uint16_t full_tail;
    uint16_t binary_tail;
    size_t tensor;
    size_t remaining;

    if (runtime->capabilities.bank_rows == 0u
        || runtime->capabilities.full_banks < 2u
        || runtime->capabilities.binary_banks < 2u
        || full_total > UINT16_MAX || binary_total > UINT16_MAX
        || full_weight_rows > runtime->capabilities.bank_rows
        || binary_weight_rows > runtime->capabilities.bank_rows) {
        return BIRA_ERR_CAPACITY;
    }
    full_tail = (uint16_t)full_total;
    binary_tail = (uint16_t)binary_total;
    /*
     * Place the largest live ranges first. This avoids fragmenting the data
     * banks before layout-changing layers (for example pixel shuffle) request
     * a wide contiguous output region.
     */
    for (remaining = 0u;
         remaining < workspace->inference->tensor_count;
         ++remaining) {
        size_t selected = SIZE_MAX;
        uint16_t largest = 0u;
        int status;
        for (tensor = 0u;
             tensor < workspace->inference->tensor_count;
             ++tensor) {
            const bira_tensor_plan_t *candidate =
                &workspace->tensors[tensor];
            if (candidate->allocated
                && candidate->base_row == UINT16_MAX
                && (selected == SIZE_MAX
                    || candidate->rows > largest)) {
                selected = tensor;
                largest = candidate->rows;
            }
        }
        if (selected == SIZE_MAX) {
            break;
        }
        status = allocate_tensor_rows(
            workspace,
            runtime,
            selected,
            full_weight_rows,
            binary_weight_rows,
            &full_tail,
            &binary_tail);
        if (status != BIRA_OK) {
            return status;
        }
    }
    workspace->full_rows_used = (uint16_t)full_total;
    workspace->binary_rows_used = (uint16_t)binary_total;
    return BIRA_OK;
}

static uint8_t array_mode_for(const bira_layer_desc_t *op)
{
    switch (op->type) {
    case BIRA_OP_DEPTHWISE_CONV2D:
        return BIRA_ARRAY_DEPTHWISE;
    case BIRA_OP_BINARY_CONV2D:
        return BIRA_ARRAY_BINARY;
    case BIRA_OP_COLUMN_REDUCE_CONV2D:
        return BIRA_ARRAY_COLUMN_REDUCE;
    case BIRA_OP_CONV2D:
    case BIRA_OP_LINEAR:
    default:
        return BIRA_ARRAY_DENSE;
    }
}

static void set_address(
    bira_planned_layer_t *compiled,
    uint8_t role,
    uint16_t row)
{
    compiled->address_mask |= (uint16_t)(UINT16_C(1) << role);
    compiled->base_rows[role] = row;
}

static int append_load(
    bira_planned_layer_t *compiled,
    uint16_t tensor,
    const void *address,
    uint8_t role,
    uint16_t rows,
    uint8_t bytes_per_row,
    uint16_t stride)
{
    bira_planned_load_t *load;
    if (compiled->load_count
        >= BIRA_INFERENCE_MAX_LOADS_PER_LAYER) {
        return BIRA_ERR_CAPACITY;
    }
    load = &compiled->loads[compiled->load_count++];
    memset(load, 0, sizeof(*load));
    load->tensor = tensor;
    load->static_address = address;
    load->role = role;
    load->rows = rows;
    load->bytes_per_row = bytes_per_row;
    load->dram_stride_bytes = stride;
    load->local_stride_rows = 1u;
    return BIRA_OK;
}

static int append_blob_load(
    bira_planned_layer_t *compiled,
    const bira_native_blob_t *blob,
    uint8_t role)
{
    if (blob->data == NULL || blob->rows == 0u) {
        return BIRA_OK;
    }
    return append_load(
        compiled,
        BIRA_TENSOR_NONE,
        blob->data,
        role,
        blob->rows,
        blob->bytes_per_row,
        blob_stride(blob));
}

static int append_store(
    bira_planned_layer_t *compiled,
    uint16_t tensor,
    uint8_t role,
    uint16_t rows,
    uint8_t bytes_per_row)
{
    bira_planned_store_t *store;
    if (compiled->store_count
        >= BIRA_INFERENCE_MAX_STORES_PER_LAYER) {
        return BIRA_ERR_CAPACITY;
    }
    store = &compiled->stores[compiled->store_count++];
    memset(store, 0, sizeof(*store));
    store->tensor = tensor;
    store->role = role;
    store->rows = rows;
    store->bytes_per_row = bytes_per_row;
    store->dram_stride_bytes = bytes_per_row;
    store->local_stride_rows = 1u;
    return BIRA_OK;
}

static bool correction_already_loaded(
    const bira_inference_workspace_t *workspace,
    uint16_t layer_offset,
    const void *address)
{
    uint16_t previous;
    for (previous = 0u; previous < layer_offset; ++previous) {
        const bira_planned_layer_t *compiled =
            &workspace->layers[previous];
        size_t load;
        for (load = 0u; load < compiled->load_count; ++load) {
            if (compiled->loads[load].role == BIRA_ROLE_CORRECTION
                && compiled->loads[load].static_address == address) {
                return true;
            }
        }
    }
    return false;
}

static int compile_layer(
    bira_inference_workspace_t *workspace,
    const bira_runtime_t *runtime,
    uint16_t offset)
{
    const bira_inference_desc_t *inference =
        workspace->inference;
    const bira_layer_desc_t *op =
        &inference->layers[offset];
    const bira_tensor_desc_t *input =
        &inference->tensors[op->input_tensor];
    const bira_tensor_desc_t *output =
        &inference->tensors[op->output_tensor];
    bira_planned_layer_t *compiled =
        &workspace->layers[offset];
    const bira_tensor_plan_t *input_plan =
        &workspace->tensors[op->input_tensor];
    const bira_tensor_plan_t *output_plan =
        &workspace->tensors[op->output_tensor];
    uint16_t full_weight_base =
        (runtime->capabilities.full_banks - 1u)
        * runtime->capabilities.bank_rows;
    uint16_t binary_weight_base =
        (runtime->capabilities.binary_banks - 1u)
        * runtime->capabilities.bank_rows;
    uint16_t weight_base = op->type == BIRA_OP_BINARY_CONV2D
        ? binary_weight_base : full_weight_base;
    int status;

    memset(compiled, 0, sizeof(*compiled));
    compiled->layer_desc = op;
    compiled->shape.input_height = input->height;
    compiled->shape.input_width = input->width;
    compiled->shape.input_channels = input->channels;
    compiled->shape.output_height =
        op->execution_output_height == 0u
        ? output->height : op->execution_output_height;
    compiled->shape.output_width =
        op->execution_output_width == 0u
        ? output->width : op->execution_output_width;
    compiled->shape.output_channels =
        op->execution_output_channels == 0u
        ? output->channels : op->execution_output_channels;
    compiled->shape.kernel_height = op->kernel_height;
    compiled->shape.kernel_width = op->kernel_width;
    compiled->shape.padding_height = op->padding_height;
    compiled->shape.padding_width = op->padding_width;
    compiled->mode.array_mode = array_mode_for(op);
    compiled->mode.weight_precision = op->weight_precision;
    compiled->mode.input_signed = op->input_signed;
    compiled->mode.post_mode = op->post_mode;
    compiled->mode.shuffle_pack2 = op->shuffle_pack2;
    compiled->mode.write_full = true;
    compiled->mode.write_binary =
        op->binary_output_tensor != BIRA_TENSOR_NONE;

    set_address(
        compiled, BIRA_ROLE_INPUT, input_plan->base_row);
    set_address(
        compiled, BIRA_ROLE_OUTPUT_FULL, output_plan->base_row);
    set_address(
        compiled,
        BIRA_ROLE_WEIGHT_LOW,
        weight_base);
    set_address(compiled, BIRA_ROLE_PARAM, 0u);
    set_address(
        compiled,
        BIRA_ROLE_ACCUMULATOR,
        runtime->capabilities.bank_rows);
    if (op->weight_high.data != NULL) {
        set_address(
            compiled,
            BIRA_ROLE_WEIGHT_HIGH,
            (uint16_t)(weight_base + op->weight_low.rows));
    }
    if (op->residual_tensor != BIRA_TENSOR_NONE) {
        set_address(
            compiled,
            BIRA_ROLE_RESIDUAL,
            workspace->tensors[op->residual_tensor].base_row);
    }
    if (op->binary_output_tensor != BIRA_TENSOR_NONE) {
        set_address(
            compiled,
            BIRA_ROLE_OUTPUT_BINARY,
            workspace->tensors[
                op->binary_output_tensor].base_row);
    }
    if (op->correction.data != NULL) {
        if (op->correction.rows + 256u
            > runtime->capabilities.parameter_rows) {
            return BIRA_ERR_CAPACITY;
        }
        set_address(compiled, BIRA_ROLE_CORRECTION, 256u);
    }

    if (input_plan->external_input
        && input_plan->first_layer == offset) {
        status = append_load(
            compiled,
            op->input_tensor,
            NULL,
            BIRA_ROLE_INPUT,
            (uint16_t)bira_tensor_rows(input),
            bira_tensor_bytes_per_row(input),
            bira_tensor_bytes_per_row(input));
        if (status != BIRA_OK) {
            return status;
        }
    }
    if (op->residual_tensor != BIRA_TENSOR_NONE) {
        const bira_tensor_plan_t *residual_plan =
            &workspace->tensors[op->residual_tensor];
        const bira_tensor_desc_t *residual =
            &inference->tensors[op->residual_tensor];
        if (residual_plan->external_input
            && residual_plan->first_layer == offset) {
            status = append_load(
                compiled,
                op->residual_tensor,
                NULL,
                BIRA_ROLE_RESIDUAL,
                (uint16_t)bira_tensor_rows(residual),
                bira_tensor_bytes_per_row(residual),
                bira_tensor_bytes_per_row(residual));
            if (status != BIRA_OK) {
                return status;
            }
        }
    }
    status = append_blob_load(
        compiled, &op->weight_low, BIRA_ROLE_WEIGHT_LOW);
    if (status == BIRA_OK) {
        status = append_blob_load(
            compiled, &op->weight_high, BIRA_ROLE_WEIGHT_HIGH);
    }
    if (status == BIRA_OK) {
        status = append_blob_load(
            compiled, &op->parameters, BIRA_ROLE_PARAM);
    }
    if (status == BIRA_OK && op->correction.data != NULL
        && !correction_already_loaded(
            workspace, offset, op->correction.data)) {
        status = append_blob_load(
            compiled, &op->correction, BIRA_ROLE_CORRECTION);
    }
    if (status != BIRA_OK) {
        return status;
    }

    if (output_plan->external_output) {
        status = append_store(
            compiled,
            op->output_tensor,
            BIRA_ROLE_OUTPUT_FULL,
            (uint16_t)bira_tensor_rows(output),
            bira_tensor_bytes_per_row(output));
        if (status != BIRA_OK) {
            return status;
        }
    } else {
        ++workspace->eliminated_tensor_transfers;
    }
    if (op->binary_output_tensor != BIRA_TENSOR_NONE) {
        const bira_tensor_desc_t *binary =
            &inference->tensors[op->binary_output_tensor];
        const bira_tensor_plan_t *binary_plan =
            &workspace->tensors[op->binary_output_tensor];
        if (binary_plan->external_output) {
            status = append_store(
                compiled,
                op->binary_output_tensor,
                BIRA_ROLE_OUTPUT_BINARY,
                (uint16_t)bira_tensor_rows(binary),
                bira_tensor_bytes_per_row(binary));
            if (status != BIRA_OK) {
                return status;
            }
        } else {
            ++workspace->eliminated_tensor_transfers;
        }
    }
    return BIRA_OK;
}

static int prepare_inference(
    bira_inference_workspace_t *workspace,
    const bira_inference_desc_t *inference,
    const bira_runtime_t *runtime,
    const bira_tensor_binding_t *bindings,
    size_t binding_count)
{
    tensor_lifetime_t lifetimes[BIRA_INFERENCE_MAX_TENSORS];
    uint16_t offset;
    int status;

    if (workspace == NULL || inference == NULL || runtime == NULL
        || inference->name == NULL
        || inference->tensors == NULL
        || inference->layers == NULL
        || inference->tensor_count == 0u
        || inference->tensor_count > BIRA_INFERENCE_MAX_TENSORS
        || inference->layer_count == 0u
        || inference->layer_count > BIRA_INFERENCE_MAX_LAYERS
        || inference->layer_count > UINT16_MAX) {
        return BIRA_ERR_ARGUMENT;
    }
    memset(workspace, 0, sizeof(*workspace));
    memset(
        workspace->tensors,
        0xff,
        sizeof(workspace->tensors));
    for (offset = 0u;
         offset < inference->tensor_count;
         ++offset) {
        workspace->tensors[offset].allocated = false;
        workspace->tensors[offset].external_input = false;
        workspace->tensors[offset].external_output = false;
        workspace->tensors[offset].space = BIRA_SPACE_NONE;
    }
    workspace->inference = inference;
    workspace->name = inference->name;
    workspace->layer_count = (uint16_t)inference->layer_count;
    status = build_lifetimes(
        workspace, lifetimes, bindings, binding_count);
    if (status == BIRA_OK) {
        status = allocate_tensors(workspace, runtime);
    }
    for (offset = 0u;
         status == BIRA_OK
            && offset < workspace->layer_count;
         ++offset) {
        status = compile_layer(workspace, runtime, offset);
    }
    if (status == BIRA_OK) {
        workspace->prepared = true;
    }
    return status;
}

static void *find_binding(
    const bira_inference_context_t *context,
    uint16_t tensor)
{
    size_t index;
    for (index = 0u; index < context->binding_count; ++index) {
        if (context->bindings[index].tensor == tensor) {
            return context->bindings[index].data;
        }
    }
    return NULL;
}

static int initialize_instance(
    bira_inference_context_t *context,
    const bira_inference_workspace_t *workspace,
    bira_runtime_t *runtime,
    const bira_tensor_binding_t *bindings,
    size_t binding_count)
{
    size_t tensor;
    if (context == NULL || workspace == NULL || !workspace->prepared
        || runtime == NULL || bindings == NULL
        || binding_count == 0u) {
        return BIRA_ERR_ARGUMENT;
    }
    memset(context, 0, sizeof(*context));
    context->workspace = workspace;
    context->runtime = runtime;
    context->bindings = bindings;
    context->binding_count = binding_count;
    for (tensor = 0u;
         tensor < workspace->inference->tensor_count;
         ++tensor) {
        const bira_tensor_plan_t *plan = &workspace->tensors[tensor];
        if ((plan->external_input || plan->external_output)
            && find_binding(context, (uint16_t)tensor) == NULL) {
            return BIRA_ERR_ARGUMENT;
        }
    }
    return BIRA_OK;
}

static void set_hook(
    bira_inference_context_t *context,
    bira_inference_layer_hook_t hook,
    void *user)
{
    if (context != NULL) {
        context->hook = hook;
        context->hook_user = user;
    }
}

static int run_planned_layer(
    bira_inference_context_t *context,
    size_t layer_offset)
{
    const bira_planned_layer_t *compiled;
    bira_layer_t layer;
    bira_load_t loads[BIRA_INFERENCE_MAX_LOADS_PER_LAYER];
    bira_store_t stores[BIRA_INFERENCE_MAX_STORES_PER_LAYER];
    size_t index;
    int status;

    if (context == NULL || context->workspace == NULL
        || context->runtime == NULL
        || layer_offset >= context->workspace->layer_count) {
        return BIRA_ERR_ARGUMENT;
    }
    compiled = &context->workspace->layers[layer_offset];
    memset(&layer, 0, sizeof(layer));
    memset(loads, 0, sizeof(loads));
    memset(stores, 0, sizeof(stores));
    layer.name = compiled->layer_desc->name;
    layer.context_id = BIRA_CONTEXT_AUTO;
    layer.shape = compiled->shape;
    layer.mode = compiled->mode;
    layer.address_mask = compiled->address_mask;
    memcpy(layer.base_rows, compiled->base_rows, sizeof(layer.base_rows));
    for (index = 0u; index < compiled->load_count; ++index) {
        const bira_planned_load_t *source =
            &compiled->loads[index];
        const void *address = source->tensor == BIRA_TENSOR_NONE
            ? source->static_address
            : find_binding(context, source->tensor);
        if (address == NULL) {
            return BIRA_ERR_ARGUMENT;
        }
        loads[index].dram_address = address;
        loads[index].role = source->role;
        loads[index].local_row_offset = source->local_row_offset;
        loads[index].rows = source->rows;
        loads[index].bytes_per_row = source->bytes_per_row;
        loads[index].dram_stride_bytes =
            source->dram_stride_bytes;
        loads[index].local_stride_rows =
            source->local_stride_rows;
    }
    for (index = 0u; index < compiled->store_count; ++index) {
        const bira_planned_store_t *source =
            &compiled->stores[index];
        void *address = find_binding(context, source->tensor);
        if (address == NULL) {
            return BIRA_ERR_ARGUMENT;
        }
        stores[index].dram_address = address;
        stores[index].role = source->role;
        stores[index].local_row_offset = source->local_row_offset;
        stores[index].rows = source->rows;
        stores[index].bytes_per_row = source->bytes_per_row;
        stores[index].dram_stride_bytes =
            source->dram_stride_bytes;
        stores[index].local_stride_rows =
            source->local_stride_rows;
    }
    layer.loads = loads;
    layer.load_count = compiled->load_count;
    layer.stores = stores;
    layer.store_count = compiled->store_count;

    if (context->hook != NULL) {
        context->hook(
            context,
            layer_offset,
            true,
            context->hook_user);
    }
    status = bira_run_layer(context->runtime, &layer);
    if (context->hook != NULL) {
        context->hook(
            context,
            layer_offset,
            false,
            context->hook_user);
    }
    return status;
}

static int run_inference(bira_inference_context_t *context)
{
    size_t offset;
    if (context == NULL || context->workspace == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    for (offset = 0u;
         offset < context->workspace->layer_count;
         ++offset) {
        int status = run_planned_layer(context, offset);
        if (status != BIRA_OK) {
            return status;
        }
    }
    return BIRA_OK;
}

int bira_inference(
    bira_runtime_t *runtime,
    bira_inference_workspace_t *workspace,
    const bira_inference_desc_t *inference,
    const bira_tensor_binding_t *bindings,
    size_t binding_count,
    bira_inference_layer_hook_t hook,
    void *hook_user)
{
    bira_inference_context_t context;
    int status = prepare_inference(
        workspace,
        inference,
        runtime,
        bindings,
        binding_count);
    if (status == BIRA_OK) {
        status = initialize_instance(
            &context,
            workspace,
            runtime,
            bindings,
            binding_count);
    }
    if (status == BIRA_OK) {
        set_hook(&context, hook, hook_user);
        status = run_inference(&context);
    }
    return status;
}
