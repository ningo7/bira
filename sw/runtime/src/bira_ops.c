#include "bira_ops.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint16_t output_y;
    uint16_t output_x;
    uint16_t output_height;
    uint16_t output_width;
    uint16_t input_y;
    uint16_t input_x;
    uint16_t input_height;
    uint16_t input_width;
    uint8_t local_padding_height;
    uint8_t local_padding_width;
} tile_window_t;

static uint32_t divide_up_u32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static uint32_t tensor_row_stride(const bira_tensor_t *tensor)
{
    uint32_t packed = tensor->format == BIRA_TENSOR_BINARY_HWC16
        ? divide_up_u32(tensor->channels, 16u) * 2u
        : tensor->channels;
    return tensor->row_stride_bytes == 0u
        ? (uint32_t)tensor->width * packed
        : tensor->row_stride_bytes;
}

static uint32_t tensor_pixel_bytes(const bira_tensor_t *tensor)
{
    return tensor->format == BIRA_TENSOR_BINARY_HWC16
        ? divide_up_u32(tensor->channels, 16u) * 2u
        : tensor->channels;
}

static uint32_t tensor_blocks(const bira_tensor_t *tensor)
{
    return divide_up_u32(tensor->channels, 16u);
}

static bool channel_layout_supported(const bira_tensor_t *tensor)
{
    return tensor->channels <= 16u || tensor->channels % 16u == 0u;
}

static bool valid_blob(
    const bira_native_blob_t *blob,
    uint8_t maximum_row_bytes)
{
    return blob->data != NULL
        && blob->rows != 0u
        && blob->bytes_per_row != 0u
        && blob->bytes_per_row <= maximum_row_bytes
        && (blob->stride_bytes == 0u
            || blob->stride_bytes >= blob->bytes_per_row);
}

static uint16_t blob_stride(const bira_native_blob_t *blob)
{
    return blob->stride_bytes == 0u
        ? blob->bytes_per_row
        : blob->stride_bytes;
}

static uint32_t full_rows_for_tensor(
    const bira_tensor_t *tensor,
    uint16_t height,
    uint16_t width)
{
    return (uint32_t)height * width * tensor_blocks(tensor);
}

static uint32_t dense_input_rows(
    const bira_tensor_t *tensor,
    uint16_t height,
    uint16_t width)
{
    uint32_t elements = (uint32_t)height * width * tensor->channels;
    return divide_up_u32(elements, 16u);
}

static int validate_tensor(const bira_tensor_t *tensor, uint8_t format)
{
    uint32_t minimum_stride;
    if (tensor == NULL || tensor->data == NULL
        || tensor->height == 0u || tensor->width == 0u
        || tensor->channels == 0u || tensor->format != format
        || !channel_layout_supported(tensor)) {
        return BIRA_ERR_ARGUMENT;
    }
    minimum_stride = (uint32_t)tensor->width
        * tensor_pixel_bytes(tensor);
    if (tensor_row_stride(tensor) < minimum_stride) {
        return BIRA_ERR_RANGE;
    }
    return BIRA_OK;
}

static bool shape_matches_stride1(
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    uint8_t kernel_height,
    uint8_t kernel_width,
    uint8_t padding_height,
    uint8_t padding_width)
{
    uint32_t expected_height =
        (uint32_t)input->height + 2u * padding_height - kernel_height + 1u;
    uint32_t expected_width =
        (uint32_t)input->width + 2u * padding_width - kernel_width + 1u;
    return output->height == expected_height
        && output->width == expected_width;
}

static uint32_t banks_for_rows(uint32_t rows, uint16_t bank_rows)
{
    return divide_up_u32(rows, bank_rows);
}

static bool multi_tile_fits(
    const bira_runtime_t *runtime,
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    const bira_conv2d_params_t *params,
    uint16_t tile_height,
    uint16_t tile_width)
{
    uint32_t maximum_input_height =
        (uint32_t)tile_height + params->kernel_height - 1u;
    uint32_t maximum_input_width =
        (uint32_t)tile_width + params->kernel_width - 1u;
    bool depthwise = params->depthwise;
    uint32_t input_rows = depthwise
        ? maximum_input_height * maximum_input_width * tensor_blocks(input)
        : divide_up_u32(
            maximum_input_height * maximum_input_width * input->channels,
            16u);
    uint32_t output_rows = (uint32_t)tile_height * tile_width
        * tensor_blocks(output);
    uint32_t read_rows = input_rows + params->weight_low.rows
        + params->weight_high.rows;
    uint32_t output_base = banks_for_rows(
        read_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    uint32_t full_rows = (uint32_t)runtime->capabilities.full_banks
        * runtime->capabilities.bank_rows;
    uint32_t binary_rows = (uint32_t)runtime->capabilities.binary_banks
        * runtime->capabilities.bank_rows;
    uint32_t accumulator_rows = (uint32_t)tile_height * tile_width;

    return output_base + output_rows <= full_rows
        && (!params->write_binary || output_rows <= binary_rows)
        && banks_for_rows(
            accumulator_rows,
            runtime->capabilities.bank_rows)
            <= runtime->capabilities.accumulator_banks;
}

int bira_plan_conv2d(
    const bira_runtime_t *runtime,
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    const bira_conv2d_params_t *params,
    bira_tile_plan_t *plan)
{
    uint16_t height;
    uint16_t width;
    uint32_t best_area = 0u;
    int status;

    if (runtime == NULL || params == NULL || plan == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    status = validate_tensor(input, BIRA_TENSOR_FULL_HWC);
    if (status != BIRA_OK) {
        return status;
    }
    status = validate_tensor(output, BIRA_TENSOR_FULL_HWC);
    if (status != BIRA_OK) {
        return status;
    }
    if (!shape_matches_stride1(
            input,
            output,
            params->kernel_height,
            params->kernel_width,
            params->padding_height,
            params->padding_width)
        || params->kernel_height == 0u
        || params->kernel_width == 0u
        || params->groups == 0u
        || input->channels % params->groups != 0u
        || output->channels % params->groups != 0u
        || !valid_blob(&params->weight_low, 16u)
        || !valid_blob(&params->parameters, 64u)
        || (params->weight_precision == BIRA_WEIGHT_W16
            && !valid_blob(&params->weight_high, 16u))) {
        return BIRA_ERR_ARGUMENT;
    }
    if ((!params->depthwise && params->groups != 1u)
        || (params->depthwise
            && (params->groups != input->channels
                || output->channels != input->channels))) {
        return BIRA_ERR_UNSUPPORTED;
    }
    if (params->write_binary) {
        if (params->binary_output == NULL) {
            return BIRA_ERR_ARGUMENT;
        }
        status = validate_tensor(
            params->binary_output, BIRA_TENSOR_BINARY_HWC16);
        if (status != BIRA_OK
            || params->binary_output->height != output->height
            || params->binary_output->width != output->width
            || params->binary_output->channels != output->channels) {
            return BIRA_ERR_ARGUMENT;
        }
    }

    memset(plan, 0, sizeof(*plan));
    for (height = 1u; height <= output->height
        && height <= runtime->capabilities.max_image_height; ++height) {
        for (width = 1u; width <= output->width
            && width <= runtime->capabilities.max_image_width; ++width) {
            uint32_t area = (uint32_t)height * width;
            if (area > best_area
                && multi_tile_fits(
                    runtime, input, output, params, height, width)) {
                plan->tile_height = height;
                plan->tile_width = width;
                best_area = area;
            }
        }
    }
    if (best_area == 0u) {
        return BIRA_ERR_CAPACITY;
    }
    plan->tile_count = (uint16_t)(
        divide_up_u32(output->height, plan->tile_height)
        * divide_up_u32(output->width, plan->tile_width));
    return BIRA_OK;
}

static tile_window_t make_window(
    const bira_tensor_t *input,
    uint16_t output_y,
    uint16_t output_x,
    uint16_t output_height,
    uint16_t output_width,
    uint8_t kernel_height,
    uint8_t kernel_width,
    uint8_t padding_height,
    uint8_t padding_width)
{
    tile_window_t window;
    uint32_t input_end_y;
    uint32_t input_end_x;

    memset(&window, 0, sizeof(window));
    window.output_y = output_y;
    window.output_x = output_x;
    window.output_height = output_height;
    window.output_width = output_width;
    window.input_y = output_y > padding_height
        ? (uint16_t)(output_y - padding_height)
        : 0u;
    window.input_x = output_x > padding_width
        ? (uint16_t)(output_x - padding_width)
        : 0u;
    window.local_padding_height = output_y < padding_height
        ? (uint8_t)(padding_height - output_y)
        : 0u;
    window.local_padding_width = output_x < padding_width
        ? (uint8_t)(padding_width - output_x)
        : 0u;
    input_end_y = (uint32_t)output_y + output_height
        + kernel_height - 1u;
    input_end_y = input_end_y > padding_height
        ? input_end_y - padding_height
        : 0u;
    input_end_x = (uint32_t)output_x + output_width
        + kernel_width - 1u;
    input_end_x = input_end_x > padding_width
        ? input_end_x - padding_width
        : 0u;
    if (input_end_y > input->height) {
        input_end_y = input->height;
    }
    if (input_end_x > input->width) {
        input_end_x = input->width;
    }
    window.input_height = (uint16_t)(input_end_y - window.input_y);
    window.input_width = (uint16_t)(input_end_x - window.input_x);
    return window;
}

static int append_tensor_load_rows(
    bira_load_t *loads,
    size_t capacity,
    size_t *count,
    const bira_tensor_t *tensor,
    const tile_window_t *window,
    uint8_t role,
    uint16_t local_base_offset)
{
    uint16_t row;
    uint32_t blocks = tensor_blocks(tensor);
    uint8_t bytes_per_row = tensor->format == BIRA_TENSOR_BINARY_HWC16
        ? 2u
        : (tensor->channels < 16u ? (uint8_t)tensor->channels : 16u);
    uint16_t stride = bytes_per_row;
    uint32_t pixel_bytes = tensor_pixel_bytes(tensor);
    uint32_t row_stride = tensor_row_stride(tensor);
    const uint8_t *base = (const uint8_t *)tensor->data;

    for (row = 0u; row < window->input_height; ++row) {
        bira_load_t *load;
        size_t offset;
        if (*count >= capacity) {
            return BIRA_ERR_CAPACITY;
        }
        load = &loads[(*count)++];
        memset(load, 0, sizeof(*load));
        offset = (size_t)(window->input_y + row) * row_stride
            + (size_t)window->input_x * pixel_bytes;
        load->dram_address = base + offset;
        load->role = role;
        load->local_row_offset = (uint16_t)(
            local_base_offset
            + (uint32_t)row * window->input_width * blocks);
        load->rows = (uint16_t)(window->input_width * blocks);
        load->bytes_per_row = bytes_per_row;
        load->dram_stride_bytes = stride;
        load->local_stride_rows = 1u;
    }
    return BIRA_OK;
}

static int append_tensor_store_rows(
    bira_store_t *stores,
    size_t capacity,
    size_t *count,
    const bira_tensor_t *tensor,
    const tile_window_t *window,
    uint8_t role,
    uint16_t local_base_offset)
{
    uint16_t row;
    uint32_t blocks = tensor_blocks(tensor);
    uint8_t bytes_per_row = tensor->format == BIRA_TENSOR_BINARY_HWC16
        ? 2u
        : (tensor->channels < 16u ? (uint8_t)tensor->channels : 16u);
    uint16_t stride = bytes_per_row;
    uint32_t pixel_bytes = tensor_pixel_bytes(tensor);
    uint32_t row_stride = tensor_row_stride(tensor);
    uint8_t *base = (uint8_t *)tensor->data;

    for (row = 0u; row < window->output_height; ++row) {
        bira_store_t *store;
        size_t offset;
        if (*count >= capacity) {
            return BIRA_ERR_CAPACITY;
        }
        store = &stores[(*count)++];
        memset(store, 0, sizeof(*store));
        offset = (size_t)(window->output_y + row) * row_stride
            + (size_t)window->output_x * pixel_bytes;
        store->dram_address = base + offset;
        store->role = role;
        store->local_row_offset = (uint16_t)(
            local_base_offset
            + (uint32_t)row * window->output_width * blocks);
        store->rows = (uint16_t)(window->output_width * blocks);
        store->bytes_per_row = bytes_per_row;
        store->dram_stride_bytes = stride;
        store->local_stride_rows = 1u;
    }
    return BIRA_OK;
}

static void set_blob_load(
    bira_load_t *load,
    const bira_native_blob_t *blob,
    uint8_t role)
{
    memset(load, 0, sizeof(*load));
    load->dram_address = blob->data;
    load->role = role;
    load->rows = blob->rows;
    load->bytes_per_row = blob->bytes_per_row;
    load->dram_stride_bytes = blob_stride(blob);
    load->local_stride_rows = 1u;
}

static int pack_dense_input(
    const bira_tensor_t *input,
    const tile_window_t *window,
    uint8_t **packed,
    uint16_t *rows)
{
    uint32_t elements =
        (uint32_t)window->input_height
        * window->input_width
        * input->channels;
    uint32_t packed_rows = divide_up_u32(elements, 16u);
    uint8_t *data;
    uint16_t row;
    uint32_t destination = 0u;
    const uint8_t *source = (const uint8_t *)input->data;
    uint32_t source_stride = tensor_row_stride(input);

    if (packed_rows == 0u || packed_rows > UINT16_MAX) {
        return BIRA_ERR_RANGE;
    }
    data = calloc((size_t)packed_rows, 16u);
    if (data == NULL) {
        return BIRA_ERR_CAPACITY;
    }
    for (row = 0u; row < window->input_height; ++row) {
        size_t source_offset =
            (size_t)(window->input_y + row) * source_stride
            + (size_t)window->input_x * input->channels;
        size_t bytes =
            (size_t)window->input_width * input->channels;
        memcpy(data + destination, source + source_offset, bytes);
        destination += (uint32_t)bytes;
    }
    *packed = data;
    *rows = (uint16_t)packed_rows;
    return BIRA_OK;
}

static int run_multi_tile(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_conv2d_params_t *params,
    const tile_window_t *window)
{
    bool depthwise = params->depthwise;
    bool needs_dense_pack = !depthwise && input->channels % 16u != 0u;
    size_t load_capacity =
        (needs_dense_pack ? 1u : (size_t)window->input_height) + 3u;
    size_t store_capacity = (size_t)window->output_height
        * (params->write_binary ? 2u : 1u);
    bira_load_t *loads = calloc(load_capacity, sizeof(*loads));
    bira_store_t *stores = calloc(store_capacity, sizeof(*stores));
    bira_layer_t layer;
    uint8_t *packed_input = NULL;
    uint16_t packed_input_rows = 0u;
    uint32_t input_rows;
    uint32_t output_rows;
    uint32_t read_rows;
    uint32_t output_base;
    size_t load_count = 0u;
    size_t store_count = 0u;
    int status;

    if (loads == NULL || stores == NULL) {
        free(loads);
        free(stores);
        return BIRA_ERR_CAPACITY;
    }
    input_rows = depthwise
        ? full_rows_for_tensor(
            input, window->input_height, window->input_width)
        : dense_input_rows(
            input, window->input_height, window->input_width);
    output_rows = full_rows_for_tensor(
        output, window->output_height, window->output_width);
    read_rows = input_rows + params->weight_low.rows
        + params->weight_high.rows;
    output_base = banks_for_rows(
        read_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    if (output_base + output_rows
        > (uint32_t)runtime->capabilities.full_banks
            * runtime->capabilities.bank_rows) {
        status = BIRA_ERR_CAPACITY;
        goto done;
    }

    if (needs_dense_pack) {
        status = pack_dense_input(
            input, window, &packed_input, &packed_input_rows);
        if (status == BIRA_OK) {
            memset(&loads[load_count], 0, sizeof(loads[load_count]));
            loads[load_count].dram_address = packed_input;
            loads[load_count].role = BIRA_ROLE_INPUT;
            loads[load_count].rows = packed_input_rows;
            loads[load_count].bytes_per_row = 16u;
            loads[load_count].dram_stride_bytes = 16u;
            loads[load_count].local_stride_rows = 1u;
            ++load_count;
        }
    } else {
        status = append_tensor_load_rows(
            loads,
            load_capacity,
            &load_count,
            input,
            window,
            BIRA_ROLE_INPUT,
            0u);
    }
    if (status != BIRA_OK) {
        goto done;
    }
    set_blob_load(
        &loads[load_count++],
        &params->weight_low,
        BIRA_ROLE_WEIGHT_LOW);
    if (params->weight_precision == BIRA_WEIGHT_W16) {
        set_blob_load(
            &loads[load_count++],
            &params->weight_high,
            BIRA_ROLE_WEIGHT_HIGH);
    }
    set_blob_load(
        &loads[load_count++],
        &params->parameters,
        BIRA_ROLE_PARAM);
    status = append_tensor_store_rows(
        stores,
        store_capacity,
        &store_count,
        output,
        window,
        BIRA_ROLE_OUTPUT_FULL,
        0u);
    if (status != BIRA_OK) {
        goto done;
    }
    if (params->write_binary) {
        status = append_tensor_store_rows(
            stores,
            store_capacity,
            &store_count,
            params->binary_output,
            window,
            BIRA_ROLE_OUTPUT_BINARY,
            0u);
        if (status != BIRA_OK) {
            goto done;
        }
    }

    memset(&layer, 0, sizeof(layer));
    layer.name = "conv2d";
    layer.context_id = BIRA_CONTEXT_AUTO;
    layer.shape.input_height = window->input_height;
    layer.shape.input_width = window->input_width;
    layer.shape.input_channels = input->channels;
    layer.shape.output_height = window->output_height;
    layer.shape.output_width = window->output_width;
    layer.shape.output_channels = output->channels;
    layer.shape.kernel_height = params->kernel_height;
    layer.shape.kernel_width = params->kernel_width;
    layer.shape.padding_height = window->local_padding_height;
    layer.shape.padding_width = window->local_padding_width;
    layer.mode.array_mode = params->depthwise
        ? BIRA_ARRAY_DEPTHWISE
        : BIRA_ARRAY_DENSE;
    layer.mode.weight_precision = params->weight_precision;
    layer.mode.input_signed = input->is_signed;
    layer.mode.post_mode = params->post_mode;
    layer.mode.write_full = true;
    layer.mode.write_binary = params->write_binary;
    layer.address_mask =
        (UINT16_C(1) << BIRA_ROLE_INPUT)
        | (UINT16_C(1) << BIRA_ROLE_WEIGHT_LOW)
        | (UINT16_C(1) << BIRA_ROLE_PARAM)
        | (UINT16_C(1) << BIRA_ROLE_ACCUMULATOR)
        | (UINT16_C(1) << BIRA_ROLE_OUTPUT_FULL);
    layer.base_rows[BIRA_ROLE_INPUT] = 0u;
    layer.base_rows[BIRA_ROLE_WEIGHT_LOW] = (uint16_t)input_rows;
    if (params->weight_precision == BIRA_WEIGHT_W16) {
        layer.address_mask |= UINT16_C(1) << BIRA_ROLE_WEIGHT_HIGH;
        layer.base_rows[BIRA_ROLE_WEIGHT_HIGH] = (uint16_t)(
            input_rows + params->weight_low.rows);
    }
    layer.base_rows[BIRA_ROLE_PARAM] = 0u;
    layer.base_rows[BIRA_ROLE_ACCUMULATOR] = 0u;
    layer.base_rows[BIRA_ROLE_OUTPUT_FULL] = (uint16_t)output_base;
    if (params->write_binary) {
        layer.address_mask |= UINT16_C(1) << BIRA_ROLE_OUTPUT_BINARY;
        layer.base_rows[BIRA_ROLE_OUTPUT_BINARY] = 0u;
    }
    layer.loads = loads;
    layer.load_count = load_count;
    layer.stores = stores;
    layer.store_count = store_count;
    status = bira_run_layer(runtime, &layer);

done:
    free(packed_input);
    free(loads);
    free(stores);
    return status;
}

int bira_conv2d(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_conv2d_params_t *params)
{
    bira_tile_plan_t plan;
    uint16_t output_y;
    int status = bira_plan_conv2d(
        runtime, input, output, params, &plan);
    if (status != BIRA_OK) {
        return status;
    }
    for (output_y = 0u; output_y < output->height;
        output_y = (uint16_t)(output_y + plan.tile_height)) {
        uint16_t output_x;
        uint16_t tile_height = (uint16_t)(
            output->height - output_y < plan.tile_height
                ? output->height - output_y
                : plan.tile_height);
        for (output_x = 0u; output_x < output->width;
            output_x = (uint16_t)(output_x + plan.tile_width)) {
            uint16_t tile_width = (uint16_t)(
                output->width - output_x < plan.tile_width
                    ? output->width - output_x
                    : plan.tile_width);
            tile_window_t window = make_window(
                input,
                output_y,
                output_x,
                tile_height,
                tile_width,
                params->kernel_height,
                params->kernel_width,
                params->padding_height,
                params->padding_width);
            status = run_multi_tile(
                runtime, input, output, params, &window);
            if (status != BIRA_OK) {
                return status;
            }
        }
    }
    return BIRA_OK;
}

int bira_linear(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_conv2d_params_t *params)
{
    if (input == NULL || output == NULL || params == NULL
        || input->height != 1u || input->width != 1u
        || output->height != 1u || output->width != 1u
        || params->kernel_height != 1u || params->kernel_width != 1u
        || params->padding_height != 0u || params->padding_width != 0u
        || params->groups != 1u) {
        return BIRA_ERR_ARGUMENT;
    }
    return bira_conv2d(runtime, input, output, params);
}

static bool binary_tile_fits(
    const bira_runtime_t *runtime,
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    const bira_binary_conv2d_params_t *params,
    uint16_t tile_height,
    uint16_t tile_width)
{
    uint32_t maximum_input_height =
        (uint32_t)tile_height + params->kernel_height - 1u;
    uint32_t maximum_input_width =
        (uint32_t)tile_width + params->kernel_width - 1u;
    uint32_t input_rows = maximum_input_height * maximum_input_width
        * tensor_blocks(input);
    uint32_t output_rows = (uint32_t)tile_height * tile_width
        * tensor_blocks(output);
    uint32_t binary_read_rows = input_rows + params->weight.rows;
    uint32_t binary_output_base = banks_for_rows(
        binary_read_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    uint32_t binary_rows = (uint32_t)runtime->capabilities.binary_banks
        * runtime->capabilities.bank_rows;
    uint32_t full_read_rows = params->residual == NULL ? 0u : output_rows;
    uint32_t full_output_base = banks_for_rows(
        full_read_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    uint32_t full_rows = (uint32_t)runtime->capabilities.full_banks
        * runtime->capabilities.bank_rows;
    uint32_t correction_rows = divide_up_u32(
        (uint32_t)tile_height * tile_width, 16u);
    uint32_t parameter_rows =
        (uint32_t)params->parameters.rows + correction_rows;
    uint32_t accumulator_rows =
        (uint32_t)runtime->capabilities.accumulator_banks
        * runtime->capabilities.bank_rows;

    return (!params->write_binary
            || binary_output_base + output_rows <= binary_rows)
        && full_output_base + output_rows <= full_rows
        && parameter_rows <= runtime->capabilities.parameter_rows
        && output_rows <= accumulator_rows;
}

static int plan_binary_conv2d(
    const bira_runtime_t *runtime,
    const bira_tensor_t *input,
    const bira_tensor_t *output,
    const bira_binary_conv2d_params_t *params,
    bira_tile_plan_t *plan)
{
    uint16_t height;
    uint16_t width;
    uint32_t best_area = 0u;
    int status;

    if (runtime == NULL || params == NULL || plan == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    status = validate_tensor(input, BIRA_TENSOR_BINARY_HWC16);
    if (status != BIRA_OK) {
        return status;
    }
    status = validate_tensor(output, BIRA_TENSOR_FULL_HWC);
    if (status != BIRA_OK) {
        return status;
    }
    if (!shape_matches_stride1(
            input,
            output,
            params->kernel_height,
            params->kernel_width,
            params->padding_height,
            params->padding_width)
        || input->channels != output->channels
        || input->channels % 16u != 0u
        || params->kernel_height == 0u
        || params->kernel_width == 0u
        || !valid_blob(&params->weight, 2u)
        || !valid_blob(&params->parameters, 64u)
        || params->correction == NULL) {
        return BIRA_ERR_ARGUMENT;
    }
    if (params->residual != NULL) {
        status = validate_tensor(
            params->residual, BIRA_TENSOR_FULL_HWC);
        if (status != BIRA_OK
            || params->residual->height != output->height
            || params->residual->width != output->width
            || params->residual->channels != output->channels) {
            return BIRA_ERR_ARGUMENT;
        }
    }
    if (params->write_binary) {
        if (params->binary_output == NULL) {
            return BIRA_ERR_ARGUMENT;
        }
        status = validate_tensor(
            params->binary_output, BIRA_TENSOR_BINARY_HWC16);
        if (status != BIRA_OK
            || params->binary_output->height != output->height
            || params->binary_output->width != output->width
            || params->binary_output->channels != output->channels) {
            return BIRA_ERR_ARGUMENT;
        }
    }

    memset(plan, 0, sizeof(*plan));
    for (height = 1u; height <= output->height
        && height <= runtime->capabilities.max_image_height; ++height) {
        for (width = 1u; width <= output->width
            && width <= runtime->capabilities.max_image_width; ++width) {
            uint32_t area = (uint32_t)height * width;
            if (area > best_area
                && binary_tile_fits(
                    runtime, input, output, params, height, width)) {
                plan->tile_height = height;
                plan->tile_width = width;
                best_area = area;
            }
        }
    }
    if (best_area == 0u) {
        return BIRA_ERR_CAPACITY;
    }
    plan->tile_count = (uint16_t)(
        divide_up_u32(output->height, plan->tile_height)
        * divide_up_u32(output->width, plan->tile_width));
    return BIRA_OK;
}

static void pack_correction_tile(
    uint8_t *packed,
    const bira_binary_conv2d_params_t *params,
    const tile_window_t *window,
    uint16_t full_output_width)
{
    uint16_t row;
    uint32_t row_stride = params->correction_row_stride_bytes == 0u
        ? (uint32_t)full_output_width * sizeof(int32_t)
        : params->correction_row_stride_bytes;
    const uint8_t *base = (const uint8_t *)params->correction;
    for (row = 0u; row < window->output_height; ++row) {
        uint16_t column;
        for (column = 0u; column < window->output_width; ++column) {
            size_t source_offset =
                (size_t)(window->output_y + row) * row_stride
                + (size_t)(window->output_x + column)
                    * sizeof(int32_t);
            size_t destination_pixel =
                (size_t)row * window->output_width + column;
            memcpy(
                packed + destination_pixel * sizeof(int32_t),
                base + source_offset,
                sizeof(int32_t));
        }
    }
}

static tile_window_t output_as_input_window(const tile_window_t *window)
{
    tile_window_t result = *window;
    result.input_y = window->output_y;
    result.input_x = window->output_x;
    result.input_height = window->output_height;
    result.input_width = window->output_width;
    return result;
}

static int run_binary_tile(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_binary_conv2d_params_t *params,
    const tile_window_t *window)
{
    size_t load_capacity = (size_t)window->input_height
        + (size_t)window->output_height + 3u;
    size_t store_capacity = (size_t)window->output_height
        * (params->write_binary ? 2u : 1u);
    bira_load_t *loads = calloc(load_capacity, sizeof(*loads));
    bira_store_t *stores = calloc(store_capacity, sizeof(*stores));
    bira_layer_t layer;
    tile_window_t output_window = output_as_input_window(window);
    uint32_t input_rows;
    uint32_t output_rows;
    uint32_t binary_read_rows;
    uint32_t binary_output_base;
    uint32_t residual_rows;
    uint32_t full_output_base;
    uint32_t correction_rows;
    uint8_t *packed_correction = NULL;
    size_t load_count = 0u;
    size_t store_count = 0u;
    int status;

    if (loads == NULL || stores == NULL) {
        free(loads);
        free(stores);
        return BIRA_ERR_CAPACITY;
    }
    input_rows = full_rows_for_tensor(
        input, window->input_height, window->input_width);
    output_rows = full_rows_for_tensor(
        output, window->output_height, window->output_width);
    binary_read_rows = input_rows + params->weight.rows;
    binary_output_base = banks_for_rows(
        binary_read_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    residual_rows = params->residual == NULL ? 0u : output_rows;
    full_output_base = banks_for_rows(
        residual_rows, runtime->capabilities.bank_rows)
        * runtime->capabilities.bank_rows;
    correction_rows =
        divide_up_u32(
            (uint32_t)window->output_height * window->output_width,
            16u);
    packed_correction = calloc(correction_rows, 64u);
    if (packed_correction == NULL) {
        status = BIRA_ERR_CAPACITY;
        goto done;
    }
    pack_correction_tile(
        packed_correction,
        params,
        window,
        output->width);

    status = append_tensor_load_rows(
        loads,
        load_capacity,
        &load_count,
        input,
        window,
        BIRA_ROLE_INPUT,
        0u);
    if (status != BIRA_OK) {
        goto done;
    }
    set_blob_load(
        &loads[load_count++], &params->weight, BIRA_ROLE_WEIGHT_LOW);
    set_blob_load(
        &loads[load_count++], &params->parameters, BIRA_ROLE_PARAM);
    loads[load_count].dram_address = packed_correction;
    loads[load_count].role = BIRA_ROLE_CORRECTION;
    loads[load_count].rows = (uint16_t)correction_rows;
    loads[load_count].bytes_per_row = 64u;
    loads[load_count].dram_stride_bytes = 64u;
    loads[load_count].local_stride_rows = 1u;
    ++load_count;
    if (params->residual != NULL) {
        status = append_tensor_load_rows(
            loads,
            load_capacity,
            &load_count,
            params->residual,
            &output_window,
            BIRA_ROLE_RESIDUAL,
            0u);
        if (status != BIRA_OK) {
            goto done;
        }
    }
    status = append_tensor_store_rows(
        stores,
        store_capacity,
        &store_count,
        output,
        window,
        BIRA_ROLE_OUTPUT_FULL,
        0u);
    if (status != BIRA_OK) {
        goto done;
    }
    if (params->write_binary) {
        status = append_tensor_store_rows(
            stores,
            store_capacity,
            &store_count,
            params->binary_output,
            window,
            BIRA_ROLE_OUTPUT_BINARY,
            0u);
        if (status != BIRA_OK) {
            goto done;
        }
    }

    memset(&layer, 0, sizeof(layer));
    layer.name = "binary_conv2d";
    layer.context_id = BIRA_CONTEXT_AUTO;
    layer.shape.input_height = window->input_height;
    layer.shape.input_width = window->input_width;
    layer.shape.input_channels = input->channels;
    layer.shape.output_height = window->output_height;
    layer.shape.output_width = window->output_width;
    layer.shape.output_channels = output->channels;
    layer.shape.kernel_height = params->kernel_height;
    layer.shape.kernel_width = params->kernel_width;
    layer.shape.padding_height = window->local_padding_height;
    layer.shape.padding_width = window->local_padding_width;
    layer.mode.array_mode = BIRA_ARRAY_BINARY;
    layer.mode.weight_precision = BIRA_WEIGHT_W2;
    layer.mode.post_mode = params->post_mode;
    layer.mode.write_full = true;
    layer.mode.write_binary = params->write_binary;
    layer.address_mask =
        (UINT16_C(1) << BIRA_ROLE_INPUT)
        | (UINT16_C(1) << BIRA_ROLE_WEIGHT_LOW)
        | (UINT16_C(1) << BIRA_ROLE_PARAM)
        | (UINT16_C(1) << BIRA_ROLE_CORRECTION)
        | (UINT16_C(1) << BIRA_ROLE_ACCUMULATOR)
        | (UINT16_C(1) << BIRA_ROLE_OUTPUT_FULL);
    layer.base_rows[BIRA_ROLE_INPUT] = 0u;
    layer.base_rows[BIRA_ROLE_WEIGHT_LOW] = (uint16_t)input_rows;
    layer.base_rows[BIRA_ROLE_PARAM] = 0u;
    layer.base_rows[BIRA_ROLE_CORRECTION] =
        params->parameters.rows;
    layer.base_rows[BIRA_ROLE_ACCUMULATOR] = 0u;
    layer.base_rows[BIRA_ROLE_OUTPUT_FULL] =
        (uint16_t)full_output_base;
    if (params->residual != NULL) {
        layer.address_mask |= UINT16_C(1) << BIRA_ROLE_RESIDUAL;
        layer.base_rows[BIRA_ROLE_RESIDUAL] = 0u;
    }
    if (params->write_binary) {
        layer.address_mask |= UINT16_C(1) << BIRA_ROLE_OUTPUT_BINARY;
        layer.base_rows[BIRA_ROLE_OUTPUT_BINARY] =
            (uint16_t)binary_output_base;
    }
    layer.loads = loads;
    layer.load_count = load_count;
    layer.stores = stores;
    layer.store_count = store_count;
    status = bira_run_layer(runtime, &layer);

done:
    free(packed_correction);
    free(loads);
    free(stores);
    return status;
}

int bira_binary_conv2d(
    bira_runtime_t *runtime,
    const bira_tensor_t *input,
    bira_tensor_t *output,
    const bira_binary_conv2d_params_t *params)
{
    bira_tile_plan_t plan;
    uint16_t output_y;
    int status = plan_binary_conv2d(
        runtime, input, output, params, &plan);
    if (status != BIRA_OK) {
        return status;
    }
    for (output_y = 0u; output_y < output->height;
        output_y = (uint16_t)(output_y + plan.tile_height)) {
        uint16_t output_x;
        uint16_t tile_height = (uint16_t)(
            output->height - output_y < plan.tile_height
                ? output->height - output_y
                : plan.tile_height);
        for (output_x = 0u; output_x < output->width;
            output_x = (uint16_t)(output_x + plan.tile_width)) {
            uint16_t tile_width = (uint16_t)(
                output->width - output_x < plan.tile_width
                    ? output->width - output_x
                    : plan.tile_width);
            tile_window_t window = make_window(
                input,
                output_y,
                output_x,
                tile_height,
                tile_width,
                params->kernel_height,
                params->kernel_width,
                params->padding_height,
                params->padding_width);
            status = run_binary_tile(
                runtime, input, output, params, &window);
            if (status != BIRA_OK) {
                return status;
            }
        }
    }
    return BIRA_OK;
}
