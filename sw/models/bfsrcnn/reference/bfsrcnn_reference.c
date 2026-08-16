/* Model-side integer golden reference. */
#include "bfsrcnn_reference.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef BFSRCNN_HW_MODEL_HEADER
#define BFSRCNN_HW_MODEL_HEADER "bfsrcnn_hardware.h"
#endif
#include BFSRCNN_HW_MODEL_HEADER

#ifndef BFSRCNN_HW_PIXEL_SHUFFLE_PAIR_LAYOUT
#define BFSRCNN_HW_PIXEL_SHUFFLE_PAIR_LAYOUT 0
#endif
#ifndef BFSRCNN_HW_FINAL_COLUMN_REDUCE_LAYOUT
#define BFSRCNN_HW_FINAL_COLUMN_REDUCE_LAYOUT 0
#endif

#if BFSRCNN_HW_S != 16
#error "binary inference requires exactly 16 state channels per packed word"
#endif

#if BFSRCNN_HW_M != 8
#error "mapping_blocks initialization must be updated when BFSRCNN_HW_M changes"
#endif

typedef struct {
    const int32_t *move0_threshold;
    const uint16_t *weight;
    const int8_t *rprelu_threshold;
    const uint8_t *branch_coefficients;
    const int8_t *positive_left_shift1;
    const int8_t *positive_left_shift2;
    const int8_t *positive_common_shift;
    const int32_t *positive_bias;
    const int8_t *negative_left_shift1;
    const int8_t *negative_left_shift2;
    const int8_t *negative_common_shift;
    const int32_t *negative_bias;
} BinaryBlock;

static const BinaryBlock mapping_blocks[BFSRCNN_HW_M] = {
    {
        bfsrcnn_hw_mapping_0_move0_threshold,
        bfsrcnn_hw_mapping_0_weight,
        bfsrcnn_hw_mapping_0_rprelu_threshold,
        bfsrcnn_hw_mapping_0_branch_coefficients,
        bfsrcnn_hw_mapping_0_positive_left_shift1,
        bfsrcnn_hw_mapping_0_positive_left_shift2,
        bfsrcnn_hw_mapping_0_positive_common_shift,
        bfsrcnn_hw_mapping_0_positive_bias,
        bfsrcnn_hw_mapping_0_negative_left_shift1,
        bfsrcnn_hw_mapping_0_negative_left_shift2,
        bfsrcnn_hw_mapping_0_negative_common_shift,
        bfsrcnn_hw_mapping_0_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_1_move0_threshold,
        bfsrcnn_hw_mapping_1_weight,
        bfsrcnn_hw_mapping_1_rprelu_threshold,
        bfsrcnn_hw_mapping_1_branch_coefficients,
        bfsrcnn_hw_mapping_1_positive_left_shift1,
        bfsrcnn_hw_mapping_1_positive_left_shift2,
        bfsrcnn_hw_mapping_1_positive_common_shift,
        bfsrcnn_hw_mapping_1_positive_bias,
        bfsrcnn_hw_mapping_1_negative_left_shift1,
        bfsrcnn_hw_mapping_1_negative_left_shift2,
        bfsrcnn_hw_mapping_1_negative_common_shift,
        bfsrcnn_hw_mapping_1_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_2_move0_threshold,
        bfsrcnn_hw_mapping_2_weight,
        bfsrcnn_hw_mapping_2_rprelu_threshold,
        bfsrcnn_hw_mapping_2_branch_coefficients,
        bfsrcnn_hw_mapping_2_positive_left_shift1,
        bfsrcnn_hw_mapping_2_positive_left_shift2,
        bfsrcnn_hw_mapping_2_positive_common_shift,
        bfsrcnn_hw_mapping_2_positive_bias,
        bfsrcnn_hw_mapping_2_negative_left_shift1,
        bfsrcnn_hw_mapping_2_negative_left_shift2,
        bfsrcnn_hw_mapping_2_negative_common_shift,
        bfsrcnn_hw_mapping_2_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_3_move0_threshold,
        bfsrcnn_hw_mapping_3_weight,
        bfsrcnn_hw_mapping_3_rprelu_threshold,
        bfsrcnn_hw_mapping_3_branch_coefficients,
        bfsrcnn_hw_mapping_3_positive_left_shift1,
        bfsrcnn_hw_mapping_3_positive_left_shift2,
        bfsrcnn_hw_mapping_3_positive_common_shift,
        bfsrcnn_hw_mapping_3_positive_bias,
        bfsrcnn_hw_mapping_3_negative_left_shift1,
        bfsrcnn_hw_mapping_3_negative_left_shift2,
        bfsrcnn_hw_mapping_3_negative_common_shift,
        bfsrcnn_hw_mapping_3_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_4_move0_threshold,
        bfsrcnn_hw_mapping_4_weight,
        bfsrcnn_hw_mapping_4_rprelu_threshold,
        bfsrcnn_hw_mapping_4_branch_coefficients,
        bfsrcnn_hw_mapping_4_positive_left_shift1,
        bfsrcnn_hw_mapping_4_positive_left_shift2,
        bfsrcnn_hw_mapping_4_positive_common_shift,
        bfsrcnn_hw_mapping_4_positive_bias,
        bfsrcnn_hw_mapping_4_negative_left_shift1,
        bfsrcnn_hw_mapping_4_negative_left_shift2,
        bfsrcnn_hw_mapping_4_negative_common_shift,
        bfsrcnn_hw_mapping_4_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_5_move0_threshold,
        bfsrcnn_hw_mapping_5_weight,
        bfsrcnn_hw_mapping_5_rprelu_threshold,
        bfsrcnn_hw_mapping_5_branch_coefficients,
        bfsrcnn_hw_mapping_5_positive_left_shift1,
        bfsrcnn_hw_mapping_5_positive_left_shift2,
        bfsrcnn_hw_mapping_5_positive_common_shift,
        bfsrcnn_hw_mapping_5_positive_bias,
        bfsrcnn_hw_mapping_5_negative_left_shift1,
        bfsrcnn_hw_mapping_5_negative_left_shift2,
        bfsrcnn_hw_mapping_5_negative_common_shift,
        bfsrcnn_hw_mapping_5_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_6_move0_threshold,
        bfsrcnn_hw_mapping_6_weight,
        bfsrcnn_hw_mapping_6_rprelu_threshold,
        bfsrcnn_hw_mapping_6_branch_coefficients,
        bfsrcnn_hw_mapping_6_positive_left_shift1,
        bfsrcnn_hw_mapping_6_positive_left_shift2,
        bfsrcnn_hw_mapping_6_positive_common_shift,
        bfsrcnn_hw_mapping_6_positive_bias,
        bfsrcnn_hw_mapping_6_negative_left_shift1,
        bfsrcnn_hw_mapping_6_negative_left_shift2,
        bfsrcnn_hw_mapping_6_negative_common_shift,
        bfsrcnn_hw_mapping_6_negative_bias,
    },
    {
        bfsrcnn_hw_mapping_7_move0_threshold,
        bfsrcnn_hw_mapping_7_weight,
        bfsrcnn_hw_mapping_7_rprelu_threshold,
        bfsrcnn_hw_mapping_7_branch_coefficients,
        bfsrcnn_hw_mapping_7_positive_left_shift1,
        bfsrcnn_hw_mapping_7_positive_left_shift2,
        bfsrcnn_hw_mapping_7_positive_common_shift,
        bfsrcnn_hw_mapping_7_positive_bias,
        bfsrcnn_hw_mapping_7_negative_left_shift1,
        bfsrcnn_hw_mapping_7_negative_left_shift2,
        bfsrcnn_hw_mapping_7_negative_common_shift,
        bfsrcnn_hw_mapping_7_negative_bias,
    },
};

static size_t index_hwc(int y, int x, int c, int width, int channels)
{
    return ((size_t)y * (size_t)width + (size_t)x) * (size_t)channels
        + (size_t)c;
}

static int64_t shift_integer(int64_t value, int shift)
{
    if (shift >= 0) {
        return value * ((int64_t)1 << shift);
    }
    {
        int right = -shift;
        int64_t magnitude = value < 0 ? -value : value;
        magnitude = (magnitude + ((int64_t)1 << (right - 1))) >> right;
        return value < 0 ? -magnitude : magnitude;
    }
}

static int32_t clamp_i32(int64_t value, int32_t minimum, int32_t maximum)
{
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return (int32_t)value;
}

#if !BFSRCNN_HW_PIXEL_SHUFFLE_PAIR_LAYOUT \
    || !BFSRCNN_HW_FINAL_COLUMN_REDUCE_LAYOUT
static int32_t weight_i16(
    const uint8_t *weight,
    int n,
    int ky,
    int kx,
    int c,
    int output_channels,
    int kernel_height,
    int kernel_width,
    int input_channels
)
{
    size_t plane = (size_t)output_channels * (size_t)kernel_height
        * (size_t)kernel_width * (size_t)input_channels;
    size_t index = (((size_t)n * (size_t)kernel_height + (size_t)ky)
        * (size_t)kernel_width + (size_t)kx) * (size_t)input_channels
        + (size_t)c;
    int32_t bits = (int32_t)weight[index]
        | ((int32_t)weight[plane + index] << 8);
    return bits >= 0x8000 ? bits - 0x10000 : bits;
}
#endif

static int32_t head_weight_i16(int n, int ky, int kx)
{
    const size_t output_lanes = BFSRCNN_HW_HEAD_WEIGHT_DIM_3;
    const size_t kernel_elements = BFSRCNN_HW_HEAD_WEIGHT_DIM_2;
    const size_t block = (size_t)n / output_lanes;
    const size_t lane = (size_t)n % output_lanes;
    const size_t tap = (size_t)ky * 3u + (size_t)kx;
    const size_t plane = (size_t)BFSRCNN_HW_HEAD_WEIGHT_DIM_1
        * kernel_elements * output_lanes;
    const size_t index = (block * kernel_elements + tap) * output_lanes
        + lane;
    int32_t bits = (int32_t)bfsrcnn_hw_head_weight[index]
        | ((int32_t)bfsrcnn_hw_head_weight[plane + index] << 8);
    return bits >= 0x8000 ? bits - 0x10000 : bits;
}

static int32_t final_weight_i16(int ky, int kx, int c)
{
#if BFSRCNN_HW_FINAL_COLUMN_REDUCE_LAYOUT
    const size_t taps = BFSRCNN_HW_FINAL_WEIGHT_DIM_1;
    const size_t lanes = BFSRCNN_HW_FINAL_WEIGHT_DIM_2;
    const size_t tap = (size_t)ky * 3u + (size_t)kx;
    const size_t plane = taps * lanes;
    const size_t index = tap * lanes + (size_t)c;
    int32_t bits = (int32_t)bfsrcnn_hw_final_weight[index]
        | ((int32_t)bfsrcnn_hw_final_weight[plane + index] << 8);
    return bits >= 0x8000 ? bits - 0x10000 : bits;
#else
    return weight_i16(
        bfsrcnn_hw_final_weight,
        0, ky, kx, c,
        1, 3, 3, BFSRCNN_HW_TAIL_CHANNELS
    );
#endif
}

static int32_t shrink1_weight_i8(int n, int c)
{
    const size_t output_lanes = BFSRCNN_HW_SHRINK1_WEIGHT_DIM_5;
    const size_t input_groups = BFSRCNN_HW_SHRINK1_WEIGHT_DIM_3;
    const size_t operands = BFSRCNN_HW_SHRINK1_WEIGHT_DIM_4;
    const size_t block = (size_t)n / output_lanes;
    const size_t lane = (size_t)n % output_lanes;
    const size_t group = (size_t)c / operands;
    const size_t operand = (size_t)c % operands;
    const size_t index = (((block * input_groups + group) * operands
        + operand) * output_lanes) + lane;
    return bfsrcnn_hw_shrink1_weight[index];
}

static int32_t shrink2_weight_i16(int n, int ky, int kx)
{
    const size_t lanes = BFSRCNN_HW_SHRINK2_WEIGHT_DIM_3;
    const size_t taps = BFSRCNN_HW_SHRINK2_WEIGHT_DIM_2;
    const size_t blocks = BFSRCNN_HW_SHRINK2_WEIGHT_DIM_1;
    const size_t block = (size_t)n / lanes;
    const size_t lane = (size_t)n % lanes;
    const size_t tap = (size_t)ky * 3u + (size_t)kx;
    const size_t plane = blocks * taps * lanes;
    const size_t index = (block * taps + tap) * lanes + lane;
    int32_t bits = (int32_t)bfsrcnn_hw_shrink2_weight[index]
        | ((int32_t)bfsrcnn_hw_shrink2_weight[plane + index] << 8);
    return bits >= 0x8000 ? bits - 0x10000 : bits;
}

static int32_t shrink3_weight_i8(int n, int c)
{
    const size_t input_groups = BFSRCNN_HW_SHRINK3_WEIGHT_DIM_3;
    const size_t operands = BFSRCNN_HW_SHRINK3_WEIGHT_DIM_4;
    const size_t lanes = BFSRCNN_HW_SHRINK3_WEIGHT_DIM_5;
    const size_t block = (size_t)n / lanes;
    const size_t lane = (size_t)n % lanes;
    const size_t group = (size_t)c / operands;
    const size_t operand = (size_t)c % operands;
    const size_t index = (((block * input_groups + group) * operands
        + operand) * lanes) + lane;
    return bfsrcnn_hw_shrink3_weight[index];
}

static size_t expand_param_index(int n)
{
#if BFSRCNN_HW_PIXEL_SHUFFLE_PAIR_LAYOUT
    const int subpixels =
        BFSRCNN_HW_SCALE_FACTOR * BFSRCNN_HW_SCALE_FACTOR;
    const int channel = n / subpixels;
    const int subpixel = n % subpixels;
    const int pair = subpixel / 2;
    const int lane =
        (subpixel & 1) * BFSRCNN_HW_TAIL_CHANNELS + channel;
    return (size_t)pair
        * (size_t)(2 * BFSRCNN_HW_TAIL_CHANNELS)
        + (size_t)lane;
#else
    return (size_t)n;
#endif
}

static int32_t expand_weight_i8(int n, int c)
{
#if BFSRCNN_HW_PIXEL_SHUFFLE_PAIR_LAYOUT
    const int subpixels =
        BFSRCNN_HW_SCALE_FACTOR * BFSRCNN_HW_SCALE_FACTOR;
    const size_t input_groups = BFSRCNN_HW_EXPAND_WEIGHT_DIM_3;
    const size_t operands = BFSRCNN_HW_EXPAND_WEIGHT_DIM_4;
    const size_t lanes = BFSRCNN_HW_EXPAND_WEIGHT_DIM_5;
    const int channel = n / subpixels;
    const int subpixel = n % subpixels;
    const size_t block = (size_t)(subpixel / 2);
    const size_t group = (size_t)c / operands;
    const size_t operand = (size_t)c % operands;
    const size_t lane = (size_t)(
        (subpixel & 1) * BFSRCNN_HW_TAIL_CHANNELS + channel
    );
    const size_t index =
        (((block * input_groups + group) * operands + operand)
            * lanes) + lane;
    return bfsrcnn_hw_expand_weight[index];
#else
    const int output_channels =
        BFSRCNN_HW_TAIL_CHANNELS
        * BFSRCNN_HW_SCALE_FACTOR
        * BFSRCNN_HW_SCALE_FACTOR;
    const int input_groups = BFSRCNN_HW_EXPAND_WEIGHT_DIM_3;
    const int operands = BFSRCNN_HW_EXPAND_WEIGHT_DIM_4;
    const int lanes = BFSRCNN_HW_EXPAND_WEIGHT_DIM_5;
    const int block = n / lanes;
    const int lane = n % lanes;
    const int group = c / operands;
    const int operand = c % operands;
    const size_t index = (((size_t)block * (size_t)input_groups
        + (size_t)group) * (size_t)operands + (size_t)operand)
        * (size_t)lanes + (size_t)lane;
    (void)output_channels;
    return bfsrcnn_hw_expand_weight[index];
#endif
}

static int64_t two_term_shift(
    int64_t value,
    int coeff1,
    int coeff2,
    int left_shift1,
    int left_shift2,
    int common_shift
)
{
    int64_t term1 = value * ((int64_t)1 << left_shift1);
    int64_t term2 = value * ((int64_t)1 << left_shift2);
    return shift_integer(coeff1 * term1 + coeff2 * term2, common_shift);
}

static int32_t prelu_value(
    int64_t value,
    int positive_shift,
    int negative_coeff1,
    int negative_coeff2,
    int negative_left_shift1,
    int negative_left_shift2,
    int negative_common_shift,
    int32_t qmin,
    int32_t qmax
)
{
    int64_t output;
    if (value >= 0) {
        output = shift_integer(value, positive_shift);
    } else {
        output = two_term_shift(
            value,
            negative_coeff1,
            negative_coeff2,
            negative_left_shift1,
            negative_left_shift2,
            negative_common_shift
        );
    }
    return clamp_i32(output, qmin, qmax);
}

static unsigned popcount16(uint16_t value)
{
    unsigned count = 0;
    while (value != 0) {
        value &= (uint16_t)(value - 1);
        ++count;
    }
    return count;
}

static int unpack_ternary_coefficient(uint8_t packed, int index)
{
    unsigned code = (packed >> (2 * index)) & 3u;
    if (code == 1u) {
        return 1;
    }
    if (code == 2u) {
        return -1;
    }
    return 0;
}

static void head_layer(
    const uint8_t *input,
    int height,
    int width,
    int32_t *output
)
{
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < BFSRCNN_HW_D; ++n) {
                int ky;
                int kx;
                int64_t accumulator = bfsrcnn_hw_head_bias[n];
                for (ky = 0; ky < 3; ++ky) {
                    int iy = y + ky - 1;
                    if (iy < 0 || iy >= height) {
                        continue;
                    }
                    for (kx = 0; kx < 3; ++kx) {
                        int ix = x + kx - 1;
                        if (ix < 0 || ix >= width) {
                            continue;
                        }
                        accumulator +=
                            (int64_t)input[(size_t)iy * (size_t)width + (size_t)ix]
                            * head_weight_i16(n, ky, kx);
                    }
                }
                output[index_hwc(y, x, n, width, BFSRCNN_HW_D)] =
                    prelu_value(
                        accumulator,
                        bfsrcnn_hw_head_prelu_positive_shift[n],
                        bfsrcnn_hw_head_prelu_negative_coeff1[n],
                        bfsrcnn_hw_head_prelu_negative_coeff2[n],
                        bfsrcnn_hw_head_prelu_negative_left_shift1[n],
                        bfsrcnn_hw_head_prelu_negative_left_shift2[n],
                        bfsrcnn_hw_head_prelu_negative_common_shift[n],
                        BFSRCNN_HW_HEAD_QMIN,
                        BFSRCNN_HW_HEAD_QMAX
                    );
            }
        }
    }
}

static void shrink1_layer(
    const int32_t *input,
    int height,
    int width,
    int32_t *output
)
{
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < BFSRCNN_HW_SHRINK_CHANNELS; ++n) {
                int c;
                int64_t accumulator = bfsrcnn_hw_shrink1_bias[n];
                for (c = 0; c < BFSRCNN_HW_D; ++c) {
                    accumulator +=
                        (int64_t)input[index_hwc(
                            y, x, c, width, BFSRCNN_HW_D
                        )] * shrink1_weight_i8(n, c);
                }
                if (accumulator < 0) {
                    accumulator = 0;
                }
                output[index_hwc(
                    y, x, n, width, BFSRCNN_HW_SHRINK_CHANNELS
                )] = clamp_i32(
                    shift_integer(
                        accumulator,
                        bfsrcnn_hw_shrink1_requant_shift[n]
                    ),
                    BFSRCNN_HW_SHRINK1_QMIN,
                    BFSRCNN_HW_SHRINK1_QMAX
                );
            }
        }
    }
}

static void shrink2_layer(
    const int32_t *input,
    int height,
    int width,
    int32_t *output
)
{
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < BFSRCNN_HW_SHRINK_CHANNELS; ++n) {
                int ky;
                int kx;
                int64_t accumulator = bfsrcnn_hw_shrink2_bias[n];
                for (ky = 0; ky < 3; ++ky) {
                    int iy = y + ky - 1;
                    if (iy < 0 || iy >= height) {
                        continue;
                    }
                    for (kx = 0; kx < 3; ++kx) {
                        int ix = x + kx - 1;
                        if (ix < 0 || ix >= width) {
                            continue;
                        }
                        accumulator +=
                            (int64_t)input[index_hwc(
                                iy, ix, n, width,
                                BFSRCNN_HW_SHRINK_CHANNELS
                            )] * shrink2_weight_i16(n, ky, kx);
                    }
                }
                if (accumulator < 0) {
                    accumulator = 0;
                }
                output[index_hwc(
                    y, x, n, width, BFSRCNN_HW_SHRINK_CHANNELS
                )] = clamp_i32(
                    shift_integer(
                        accumulator,
                        bfsrcnn_hw_shrink2_requant_shift[n]
                    ),
                    BFSRCNN_HW_SHRINK2_QMIN,
                    BFSRCNN_HW_SHRINK2_QMAX
                );
            }
        }
    }
}

static void shrink3_layer(
    const int32_t *input,
    int height,
    int width,
    int32_t *output
)
{
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < BFSRCNN_HW_S; ++n) {
                int c;
                int64_t accumulator = bfsrcnn_hw_shrink3_bias[n];
                for (c = 0; c < BFSRCNN_HW_SHRINK_CHANNELS; ++c) {
                    accumulator +=
                        (int64_t)input[index_hwc(
                            y, x, c, width,
                            BFSRCNN_HW_SHRINK_CHANNELS
                        )] * shrink3_weight_i8(n, c);
                }
                output[index_hwc(y, x, n, width, BFSRCNN_HW_S)] =
                    clamp_i32(
                        shift_integer(
                            accumulator,
                            bfsrcnn_hw_shrink3_requant_shift[n]
                        ),
                        BFSRCNN_HW_STATE_QMIN,
                        BFSRCNN_HW_STATE_QMAX
                    );
            }
        }
    }
}

static void binary_block(
    int32_t *state,
    int height,
    int width,
    const BinaryBlock *block,
    uint16_t *binary
)
{
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            uint16_t packed = 0;
            for (n = 0; n < BFSRCNN_HW_S; ++n) {
                if (state[index_hwc(y, x, n, width, BFSRCNN_HW_S)]
                        >= block->move0_threshold[n]) {
                    packed |= (uint16_t)((uint16_t)1 << n);
                }
            }
            binary[(size_t)y * (size_t)width + (size_t)x] = packed;
        }
    }

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < BFSRCNN_HW_S; ++n) {
                int ky;
                int kx;
                int64_t accumulator = 0;
                for (ky = 0; ky < 3; ++ky) {
                    int iy = y + ky - 1;
                    if (iy < 0 || iy >= height) {
                        continue;
                    }
                    for (kx = 0; kx < 3; ++kx) {
                        int ix = x + kx - 1;
                        uint16_t activation;
                        uint16_t weight;
                        unsigned equal_bits;
                        if (ix < 0 || ix >= width) {
                            continue;
                        }
                        activation = binary[
                            (size_t)iy * (size_t)width + (size_t)ix
                        ];
                        weight = block->weight[
                            ((size_t)n * 3u + (size_t)ky) * 3u + (size_t)kx
                        ];
                        equal_bits = popcount16(
                            (uint16_t)~(activation ^ weight)
                        );
                        accumulator += (int64_t)(2 * (int)equal_bits
                            - BFSRCNN_HW_S);
                    }
                }
                {
                    int64_t delta;
                    uint8_t coefficients = block->branch_coefficients[n];
                    if (accumulator >= block->rprelu_threshold[n]) {
                        delta = two_term_shift(
                            accumulator,
                            1,
                            unpack_ternary_coefficient(coefficients, 0),
                            block->positive_left_shift1[n],
                            block->positive_left_shift2[n],
                            block->positive_common_shift[n]
                        ) + block->positive_bias[n];
                    } else {
                        delta = two_term_shift(
                            accumulator,
                            unpack_ternary_coefficient(coefficients, 1),
                            unpack_ternary_coefficient(coefficients, 2),
                            block->negative_left_shift1[n],
                            block->negative_left_shift2[n],
                            block->negative_common_shift[n]
                        ) + block->negative_bias[n];
                    }
                    state[index_hwc(y, x, n, width, BFSRCNN_HW_S)] =
                        clamp_i32(
                            (int64_t)state[index_hwc(
                                y, x, n, width, BFSRCNN_HW_S
                            )] + delta,
                            BFSRCNN_HW_STATE_QMIN,
                            BFSRCNN_HW_STATE_QMAX
                        );
                }
            }
        }
    }
}

static void expand_layer(
    const int32_t *input,
    int height,
    int width,
    int32_t *output
)
{
    const int output_channels =
        BFSRCNN_HW_TAIL_CHANNELS
        * BFSRCNN_HW_SCALE_FACTOR
        * BFSRCNN_HW_SCALE_FACTOR;
    int y;
    int x;
    int n;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (n = 0; n < output_channels; ++n) {
                int c;
                const size_t parameter = expand_param_index(n);
                int64_t accumulator =
                    bfsrcnn_hw_expand_bias[parameter];
                for (c = 0; c < BFSRCNN_HW_S; ++c) {
                    accumulator +=
                        (int64_t)input[index_hwc(
                            y, x, c, width, BFSRCNN_HW_S
                        )] * expand_weight_i8(n, c);
                }
                output[index_hwc(y, x, n, width, output_channels)] =
                    prelu_value(
                        accumulator,
                        bfsrcnn_hw_tail_prelu_positive_shift[parameter],
                        bfsrcnn_hw_tail_prelu_negative_coeff1[parameter],
                        bfsrcnn_hw_tail_prelu_negative_coeff2[parameter],
                        bfsrcnn_hw_tail_prelu_negative_left_shift1[parameter],
                        bfsrcnn_hw_tail_prelu_negative_left_shift2[parameter],
                        bfsrcnn_hw_tail_prelu_negative_common_shift[parameter],
                        BFSRCNN_HW_TAIL_QMIN,
                        BFSRCNN_HW_TAIL_QMAX
                    );
            }
        }
    }
}

static void pixel_shuffle(
    const int32_t *input,
    int height,
    int width,
    int32_t *output
)
{
    const int scale = BFSRCNN_HW_SCALE_FACTOR;
    const int input_channels = BFSRCNN_HW_TAIL_CHANNELS * scale * scale;
    const int output_width = width * scale;
    int y;
    int x;
    int c;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (c = 0; c < BFSRCNN_HW_TAIL_CHANNELS; ++c) {
                int sy;
                int sx;
                for (sy = 0; sy < scale; ++sy) {
                    for (sx = 0; sx < scale; ++sx) {
                        int input_channel =
                            c * scale * scale + sy * scale + sx;
                        output[index_hwc(
                            y * scale + sy,
                            x * scale + sx,
                            c,
                            output_width,
                            BFSRCNN_HW_TAIL_CHANNELS
                        )] = input[index_hwc(
                            y, x, input_channel, width, input_channels
                        )];
                    }
                }
            }
        }
    }
}

static int floor_divide(int numerator, int denominator)
{
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -((-numerator + denominator - 1) / denominator);
}

static int clamp_coordinate(int value, int maximum)
{
    if (value < 0) {
        return 0;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

static int32_t bilinear_pixel(
    const uint8_t *input,
    int height,
    int width,
    int oy,
    int ox
)
{
    const int denominator = BFSRCNN_HW_BILINEAR_DENOMINATOR;
    const int divisor = BFSRCNN_HW_BILINEAR_DIVISOR;
    int y_numerator = 2 * oy + 1 - BFSRCNN_HW_SCALE_FACTOR;
    int x_numerator = 2 * ox + 1 - BFSRCNN_HW_SCALE_FACTOR;
    int y0_unclamped = floor_divide(y_numerator, denominator);
    int x0_unclamped = floor_divide(x_numerator, denominator);
    int wy1 = y_numerator - y0_unclamped * denominator;
    int wx1 = x_numerator - x0_unclamped * denominator;
    int wy0 = denominator - wy1;
    int wx0 = denominator - wx1;
    int y0 = clamp_coordinate(y0_unclamped, height - 1);
    int y1 = clamp_coordinate(y0_unclamped + 1, height - 1);
    int x0 = clamp_coordinate(x0_unclamped, width - 1);
    int x1 = clamp_coordinate(x0_unclamped + 1, width - 1);
    int64_t numerator =
        (int64_t)input[(size_t)y0 * (size_t)width + (size_t)x0] * wy0 * wx0
        + (int64_t)input[(size_t)y0 * (size_t)width + (size_t)x1] * wy0 * wx1
        + (int64_t)input[(size_t)y1 * (size_t)width + (size_t)x0] * wy1 * wx0
        + (int64_t)input[(size_t)y1 * (size_t)width + (size_t)x1] * wy1 * wx1;
    return (int32_t)((numerator + divisor / 2) / divisor);
}

static void final_layer(
    const uint8_t *input,
    int input_height,
    int input_width,
    const int32_t *tail,
    uint8_t *output
)
{
    const int scale = BFSRCNN_HW_SCALE_FACTOR;
    const int height = input_height * scale;
    const int width = input_width * scale;
    int y;
    int x;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            int ky;
            int kx;
            int64_t accumulator = bfsrcnn_hw_final_bias[0];
            for (ky = 0; ky < 3; ++ky) {
                int iy = y + ky - 1;
                if (iy < 0 || iy >= height) {
                    continue;
                }
                for (kx = 0; kx < 3; ++kx) {
                    int ix = x + kx - 1;
                    int c;
                    if (ix < 0 || ix >= width) {
                        continue;
                    }
                    for (c = 0; c < BFSRCNN_HW_TAIL_CHANNELS; ++c) {
                        accumulator +=
                            (int64_t)tail[index_hwc(
                                iy, ix, c, width,
                                BFSRCNN_HW_TAIL_CHANNELS
                            )] * final_weight_i16(ky, kx, c);
                    }
                }
            }
            {
                int64_t correction = shift_integer(
                    accumulator,
                    bfsrcnn_hw_final_correction_shift[0]
                );
                int64_t value = bilinear_pixel(
                    input, input_height, input_width, y, x
                ) + correction;
                output[(size_t)y * (size_t)width + (size_t)x] =
                    (uint8_t)clamp_i32(
                        value,
                        BFSRCNN_HW_INPUT_QMIN,
                        BFSRCNN_HW_INPUT_QMAX
                    );
            }
        }
    }
}

int bfsrcnn_hw_infer_stages(
    const uint8_t *input,
    int input_height,
    int input_width,
    uint8_t *output,
    bfsrcnn_hw_stage_callback_t callback,
    void *callback_user
)
{
    const size_t pixels = (size_t)input_height * (size_t)input_width;
    const size_t output_pixels =
        pixels * BFSRCNN_HW_SCALE_FACTOR * BFSRCNN_HW_SCALE_FACTOR;
    const int expanded_channels =
        BFSRCNN_HW_TAIL_CHANNELS
        * BFSRCNN_HW_SCALE_FACTOR
        * BFSRCNN_HW_SCALE_FACTOR;
    int32_t *head = NULL;
    int32_t *shrink1 = NULL;
    int32_t *shrink2 = NULL;
    int32_t *state = NULL;
    int32_t *expanded = NULL;
    int32_t *tail = NULL;
    uint16_t *binary = NULL;
    int block;

    head = (int32_t *)calloc(
        pixels * BFSRCNN_HW_D, sizeof(*head)
    );
    shrink1 = (int32_t *)calloc(
        pixels * BFSRCNN_HW_SHRINK_CHANNELS, sizeof(*shrink1)
    );
    shrink2 = (int32_t *)calloc(
        pixels * BFSRCNN_HW_SHRINK_CHANNELS, sizeof(*shrink2)
    );
    state = (int32_t *)calloc(
        pixels * BFSRCNN_HW_S, sizeof(*state)
    );
    expanded = (int32_t *)calloc(
        pixels * (size_t)expanded_channels, sizeof(*expanded)
    );
    tail = (int32_t *)calloc(
        output_pixels * BFSRCNN_HW_TAIL_CHANNELS, sizeof(*tail)
    );
    binary = (uint16_t *)calloc(pixels, sizeof(*binary));
    if (head == NULL || shrink1 == NULL || shrink2 == NULL || state == NULL
            || expanded == NULL || tail == NULL || binary == NULL) {
        free(head);
        free(shrink1);
        free(shrink2);
        free(state);
        free(expanded);
        free(tail);
        free(binary);
        return -1;
    }

    head_layer(input, input_height, input_width, head);
    if (callback != NULL && head != NULL) {
        callback(
            BFSRCNN_HW_STAGE_HEAD,
            head,
            input_height,
            input_width,
            BFSRCNN_HW_D,
            callback_user
        );
    }
    shrink1_layer(head, input_height, input_width, shrink1);
    if (callback != NULL && shrink1 != NULL) {
        callback(
            BFSRCNN_HW_STAGE_SHRINK1,
            shrink1,
            input_height,
            input_width,
            BFSRCNN_HW_SHRINK_CHANNELS,
            callback_user
        );
    }
    shrink2_layer(shrink1, input_height, input_width, shrink2);
    if (callback != NULL && shrink2 != NULL) {
        callback(
            BFSRCNN_HW_STAGE_SHRINK2,
            shrink2,
            input_height,
            input_width,
            BFSRCNN_HW_SHRINK_CHANNELS,
            callback_user
        );
    }
    shrink3_layer(shrink2, input_height, input_width, state);
    if (callback != NULL && state != NULL) {
        callback(
            BFSRCNN_HW_STAGE_SHRINK3,
            state,
            input_height,
            input_width,
            BFSRCNN_HW_S,
            callback_user
        );
    }
    for (block = 0; block < BFSRCNN_HW_M; ++block) {
        binary_block(
            state,
            input_height,
            input_width,
            &mapping_blocks[block],
            binary
        );
        if (callback != NULL && state != NULL) {
            callback(
                (bfsrcnn_hw_stage_t)(
                    BFSRCNN_HW_STAGE_MAPPING0 + block
                ),
                state,
                input_height,
                input_width,
                BFSRCNN_HW_S,
                callback_user
            );
        }
    }
    expand_layer(state, input_height, input_width, expanded);
    pixel_shuffle(expanded, input_height, input_width, tail);
    if (callback != NULL && tail != NULL) {
        callback(
            BFSRCNN_HW_STAGE_EXPAND,
            tail,
            input_height * BFSRCNN_HW_SCALE_FACTOR,
            input_width * BFSRCNN_HW_SCALE_FACTOR,
            BFSRCNN_HW_TAIL_CHANNELS,
            callback_user
        );
    }
    final_layer(input, input_height, input_width, tail, output);

    free(head);
    free(shrink1);
    free(shrink2);
    free(state);
    free(expanded);
    free(tail);
    free(binary);
    return 0;
}

int bfsrcnn_hw_infer(
    const uint8_t *input,
    int input_height,
    int input_width,
    uint8_t *output
)
{
    return bfsrcnn_hw_infer_stages(
        input,
        input_height,
        input_width,
        output,
        NULL,
        NULL
    );
}
