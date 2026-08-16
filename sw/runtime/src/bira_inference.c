#include "bira_inference.h"

#include <limits.h>

static uint32_t divide_up_u32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

uint32_t bira_tensor_rows(const bira_tensor_desc_t *tensor)
{
    uint32_t pixels;
    uint32_t elements;
    if (tensor == NULL || tensor->height == 0u
        || tensor->width == 0u || tensor->channels == 0u) {
        return 0u;
    }
    pixels = (uint32_t)tensor->height * tensor->width;
    switch (tensor->layout) {
    case BIRA_LAYOUT_FULL_DENSE16:
        elements = pixels * tensor->channels;
        return divide_up_u32(elements, 16u);
    case BIRA_LAYOUT_FULL_HWC16:
    case BIRA_LAYOUT_BINARY_HWC16:
        return pixels * divide_up_u32(tensor->channels, 16u);
    case BIRA_LAYOUT_FULL_PACKED_PIXEL_PAIR:
        return divide_up_u32(pixels, 2u)
            * divide_up_u32(tensor->channels, 8u);
    default:
        return 0u;
    }
}

uint8_t bira_tensor_bytes_per_row(
    const bira_tensor_desc_t *tensor)
{
    if (tensor == NULL) {
        return 0u;
    }
    switch (tensor->layout) {
    case BIRA_LAYOUT_BINARY_HWC16:
        return 2u;
    case BIRA_LAYOUT_FULL_DENSE16:
    case BIRA_LAYOUT_FULL_HWC16:
    case BIRA_LAYOUT_FULL_PACKED_PIXEL_PAIR:
        return 16u;
    default:
        return 0u;
    }
}

size_t bira_tensor_bytes(const bira_tensor_desc_t *tensor)
{
    uint32_t rows = bira_tensor_rows(tensor);
    uint8_t bytes = bira_tensor_bytes_per_row(tensor);
    if (rows == 0u || bytes == 0u
        || rows > SIZE_MAX / bytes) {
        return 0u;
    }
    return (size_t)rows * bytes;
}

const char *bira_inference_layer_name(
    const bira_inference_desc_t *inference,
    size_t layer_index)
{
    return inference != NULL && inference->layers != NULL
        && layer_index < inference->layer_count
        && inference->layers[layer_index].name != NULL
        ? inference->layers[layer_index].name
        : "invalid";
}
