"""Export BFSRCNN integer tensors in BIRA controller-native layouts."""

import argparse
from collections import OrderedDict
from pathlib import Path
import re

import torch

from models.bfsrcnn.quantization.integer_model import integer_range
from models.bfsrcnn.quantization.checkpoint import (
    load_pixel_integer_checkpoint,
)
from models.common.deployment.c_data import (
    DTYPE_TO_C,
    c_identifier,
    tensor_values,
)
from models.common.deployment.weight_layout import (
    pack_binary_last_axis,
    split_int16_byte_planes,
)


def checked_shift(name, value):
    value = torch.as_tensor(value, dtype=torch.int64).contiguous()
    if value.numel():
        minimum = int(value.min().item())
        maximum = int(value.max().item())
        if minimum < -128 or maximum > 127:
            raise OverflowError(
                f'{name} shift range [{minimum}, {maximum}] does not fit int8'
            )
    return value.to(torch.int8)


def pack_ternary_coefficients(name, *coefficients):
    """Pack up to four {-1, 0, +1} tensors into one uint8 tensor."""
    if not coefficients or len(coefficients) > 4:
        raise ValueError('one to four coefficient tensors are required')
    packed = torch.zeros_like(
        torch.as_tensor(coefficients[0]),
        dtype=torch.uint8,
    )
    for index, coefficient in enumerate(coefficients):
        coefficient = torch.as_tensor(coefficient, dtype=torch.int8)
        if coefficient.shape != packed.shape:
            raise ValueError(f'{name} coefficient shapes do not match')
        if not bool(((coefficient >= -1) & (coefficient <= 1)).all()):
            raise ValueError(f'{name} coefficients must be -1, 0, or +1')
        code = torch.where(
            coefficient > 0,
            1,
            torch.where(coefficient < 0, 2, 0),
        ).to(torch.uint8)
        packed |= code << (2 * index)
    return packed.contiguous()


def decompose_two_term_shifts(name, shift1, shift2):
    """Represent two signed shifts as two left shifts and one common shift."""
    shift1 = torch.as_tensor(shift1, dtype=torch.int64)
    shift2 = torch.as_tensor(shift2, dtype=torch.int64)
    common = torch.minimum(
        torch.minimum(shift1, shift2),
        torch.zeros_like(shift1),
    )
    return (
        checked_shift(f'{name}_left_shift1', shift1 - common),
        checked_shift(f'{name}_left_shift2', shift2 - common),
        checked_shift(f'{name}_common_shift', common),
    )


def prelu_derived_shifts(
    name,
    accumulator_exponent,
    output_exponent,
    exponent1,
    exponent2,
):
    accumulator_exponent = torch.as_tensor(
        accumulator_exponent, dtype=torch.int64
    )
    output_exponent = torch.as_tensor(output_exponent, dtype=torch.int64)
    positive = checked_shift(
        f'{name}_positive_shift',
        accumulator_exponent - output_exponent,
    )
    negative1 = accumulator_exponent + exponent1.to(torch.int64) - output_exponent
    negative2 = accumulator_exponent + exponent2.to(torch.int64) - output_exponent
    left1, left2, common = decompose_two_term_shifts(
        f'{name}_negative', negative1, negative2
    )
    return positive, left1, left2, common


def hardware_tensors(model):
    """Return only tensors required by a forward path with precomputed shifts."""
    tensors = OrderedDict()

    tensors['head_weight'] = model.head_weight
    tensors['head_bias'] = model.head_bias
    tensors['head_prelu_negative_coeff1'] = model.head_prelu_slope_sign
    tensors['head_prelu_negative_coeff2'] = (
        model.head_prelu_slope_sign * model.head_prelu_term2_sign
    )
    head_accumulator_exponent = (
        model.head_weight_exponent + model.input_exponent
    )
    (
        tensors['head_prelu_positive_shift'],
        tensors['head_prelu_negative_left_shift1'],
        tensors['head_prelu_negative_left_shift2'],
        tensors['head_prelu_negative_common_shift'],
    ) = prelu_derived_shifts(
        'head_prelu',
        head_accumulator_exponent,
        model.head_output_exponent,
        model.head_prelu_exponent1,
        model.head_prelu_exponent2,
    )

    for name, weight, bias, shift in (
        (
            'shrink1',
            model.shrink1_weight,
            model.shrink1_bias,
            model.head_output_exponent
            + model.shrink1_weight_exponent
            - model.shrink1_output_exponent,
        ),
        (
            'shrink2',
            model.shrink2_weight,
            model.shrink2_bias,
            model.shrink1_output_exponent
            + model.shrink2_weight_exponent
            - model.shrink2_output_exponent,
        ),
        (
            'shrink3',
            model.shrink3_weight,
            model.shrink3_bias,
            model.shrink2_output_exponent
            + model.shrink3_weight_exponent
            - model.state_exponent,
        ),
    ):
        tensors[f'{name}_weight'] = weight
        tensors[f'{name}_bias'] = bias
        tensors[f'{name}_requant_shift'] = checked_shift(
            f'{name}_requant_shift', shift
        )

    for index, block in enumerate(model.mapping):
        prefix = f'mapping_{index}'
        tensors[f'{prefix}_move0_threshold'] = block.move0_threshold
        tensors[f'{prefix}_weight'] = block.weight
        tensors[f'{prefix}_rprelu_threshold'] = checked_shift(
            f'{prefix}_rprelu_threshold',
            block.rprelu_threshold,
        )
        tensors[f'{prefix}_branch_coefficients'] = pack_ternary_coefficients(
            f'{prefix}_branch_coefficients',
            block.positive_term2_sign,
            block.negative_slope_sign,
            block.negative_slope_sign * block.negative_term2_sign,
        )
        (
            tensors[f'{prefix}_positive_left_shift1'],
            tensors[f'{prefix}_positive_left_shift2'],
            tensors[f'{prefix}_positive_common_shift'],
        ) = decompose_two_term_shifts(
            f'{prefix}_positive',
            block.positive_exponent1.to(torch.int64)
            - block.state_exponent.to(torch.int64),
            block.positive_exponent2.to(torch.int64)
            - block.state_exponent.to(torch.int64),
        )
        tensors[f'{prefix}_positive_bias'] = block.positive_bias
        (
            tensors[f'{prefix}_negative_left_shift1'],
            tensors[f'{prefix}_negative_left_shift2'],
            tensors[f'{prefix}_negative_common_shift'],
        ) = decompose_two_term_shifts(
            f'{prefix}_negative',
            block.negative_exponent1.to(torch.int64)
            - block.state_exponent.to(torch.int64),
            block.negative_exponent2.to(torch.int64)
            - block.state_exponent.to(torch.int64),
        )
        tensors[f'{prefix}_negative_bias'] = block.negative_bias

    tensors['expand_weight'] = model.expand_weight
    tensors['expand_bias'] = model.expand_bias
    tensors['tail_prelu_negative_coeff1'] = model.tail_prelu_slope_sign
    tensors['tail_prelu_negative_coeff2'] = (
        model.tail_prelu_slope_sign * model.tail_prelu_term2_sign
    )
    (
        tensors['tail_prelu_positive_shift'],
        tensors['tail_prelu_negative_left_shift1'],
        tensors['tail_prelu_negative_left_shift2'],
        tensors['tail_prelu_negative_common_shift'],
    ) = prelu_derived_shifts(
        'tail_prelu',
        model.expand_weight_exponent,
        model.tail_output_exponent,
        model.tail_prelu_exponent1,
        model.tail_prelu_exponent2,
    )

    tensors['final_weight'] = model.final_weight
    tensors['final_bias'] = model.final_bias
    tensors['final_correction_shift'] = checked_shift(
        'final_correction_shift',
        model.tail_output_exponent
        + model.final_weight_exponent[0]
        - model.input_exponent,
    ).reshape(1)
    return tensors


HEAD_OUTPUT_LANES = 16
ARRAY_WEIGHT_BITS = 16


def head_weight_layout(tensor, output_lanes=HEAD_OUTPUT_LANES):
    """Pack OIHW head weights into rows consumed by ConvolutionController.

    The head layer has one input channel. One hardware row contains the same
    kernel tap for `output_lanes` output channels. Output channels are padded
    with zero weights when their count is not a multiple of the array width.

    int16 result dimensions are KBTL:
      K: byte plane (0=low, 1=high)
      B: output-channel block
      T: flattened kernel tap (ky * kernel_width + kx)
      L: output-channel lane within the block
    """
    if tensor.ndim != 4:
        raise ValueError(f'head_weight must be OIHW, got rank {tensor.ndim}')
    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    if input_channels != 1:
        raise ValueError(
            'the current head dataflow requires exactly one input channel, '
            f'got {input_channels}'
        )
    if output_lanes <= 0:
        raise ValueError('output_lanes must be positive')

    kernel_elements = kernel_height * kernel_width
    output_blocks = (output_channels + output_lanes - 1) // output_lanes
    padded = torch.zeros(
        output_blocks * output_lanes,
        kernel_elements,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:output_channels] = tensor[:, 0].reshape(
        output_channels, kernel_elements
    )
    block_tap_lane = padded.reshape(
        output_blocks, output_lanes, kernel_elements
    ).permute(0, 2, 1).contiguous()

    if tensor.dtype == torch.int16:
        return (
            split_int16_byte_planes(block_tap_lane),
            'KBTL, K0=low byte, K1=high byte, '
            'B=output block, T=kernel tap, L=output lane',
        )

    return block_tap_lane, (
        'BTL, B=output block, T=kernel tap, L=output lane'
    )


def int8_grouped_weight_layout(
    tensor,
    weight_bits,
    output_lanes=HEAD_OUTPUT_LANES,
    array_weight_bits=ARRAY_WEIGHT_BITS,
):
    """Transpose W8/W4/W2 OIHW weights into controller-native vector rows.

    Each int8 row contains one input-channel operand for `output_lanes`
    output channels. The number of operands held stationary at once is the
    number of weights that fit in one array column.
    """
    if tensor.ndim != 4:
        raise ValueError(f'weight must be OIHW, got rank {tensor.ndim}')
    if tensor.dtype != torch.int8:
        raise ValueError(
            f'grouped sub-byte weights must use int8 storage, got {tensor.dtype}'
        )
    if weight_bits not in (2, 4, 8):
        raise ValueError(
            f'int8 grouped layout requires W2/W4/W8, got W{weight_bits}'
        )

    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    operands = array_weight_bits // weight_bits
    output_blocks = (output_channels + output_lanes - 1) // output_lanes
    input_groups = (input_channels + operands - 1) // operands
    padded = torch.zeros(
        output_blocks * output_lanes,
        input_groups * operands,
        kernel_height,
        kernel_width,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:output_channels, :input_channels] = tensor
    grouped = padded.reshape(
        output_blocks,
        output_lanes,
        input_groups,
        operands,
        kernel_height,
        kernel_width,
    ).permute(0, 4, 5, 2, 3, 1).contiguous()
    return grouped, (
        f'BHWGIL, W{weight_bits} int8 rows, B=output block, '
        'G=input group, I=operand, L=output lane'
    )


def int16_dense_weight_layout(tensor, output_lanes=HEAD_OUTPUT_LANES):
    """Transpose dense W16 OIHW into low/high controller-native rows."""
    if tensor.ndim != 4 or tensor.dtype != torch.int16:
        raise ValueError('dense W16 layout requires an int16 OIHW tensor')
    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    output_blocks = (output_channels + output_lanes - 1) // output_lanes
    padded = torch.zeros(
        output_blocks * output_lanes,
        input_channels,
        kernel_height,
        kernel_width,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:output_channels] = tensor
    block_hw_channel_lane = padded.reshape(
        output_blocks,
        output_lanes,
        input_channels,
        kernel_height,
        kernel_width,
    ).permute(0, 3, 4, 2, 1).contiguous()
    return (
        split_int16_byte_planes(block_hw_channel_lane),
        'KBHWCL, K0=low byte, K1=high byte, '
        'B=output block, C=input channel, L=output lane',
    )


def pixel_shuffle_pair_weight_layout(
    tensor,
    tail_channels,
    output_lanes=HEAD_OUTPUT_LANES,
):
    """Arrange expand W16 rows for two adjacent shuffled pixels per tile.

    Lane 0..tail_channels-1 contains subpixel 2*B. The upper half contains
    subpixel 2*B+1. Within each half, lanes are the final HWI channels.
    """
    if tensor.ndim != 4 or tensor.dtype != torch.int16:
        raise ValueError('PixelShuffle pair layout requires int16 OIHW')
    if tail_channels * 2 != output_lanes:
        raise ValueError(
            'pair layout requires two shuffled pixels to fill one row'
        )
    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    if output_channels % tail_channels != 0:
        raise ValueError('expand outputs must be divisible by tail channels')
    subpixels = output_channels // tail_channels
    if subpixels % 2 != 0:
        raise ValueError('PixelShuffle subpixel count must be even')
    pairs = subpixels // 2
    channel_subpixel = tensor.reshape(
        tail_channels,
        subpixels,
        input_channels,
        kernel_height,
        kernel_width,
    )
    even = channel_subpixel[:, 0::2].permute(
        1, 3, 4, 2, 0
    )
    odd = channel_subpixel[:, 1::2].permute(
        1, 3, 4, 2, 0
    )
    pair_hw_channel_lane = torch.cat((even, odd), dim=-1).contiguous()
    if pair_hw_channel_lane.shape != (
        pairs,
        kernel_height,
        kernel_width,
        input_channels,
        output_lanes,
    ):
        raise RuntimeError('unexpected PixelShuffle pair weight shape')
    return (
        split_int16_byte_planes(pair_hw_channel_lane),
        'KBHWCL-P2, K0=low byte, K1=high byte, '
        'B=subpixel pair, L=[even-pixel channels, odd-pixel channels]',
    )


def pixel_shuffle_pair_grouped_weight_layout(
    tensor,
    weight_bits,
    tail_channels,
    output_lanes=HEAD_OUTPUT_LANES,
    array_weight_bits=ARRAY_WEIGHT_BITS,
):
    """Arrange W8/W4/W2 expand rows in pair lanes and input groups."""
    if tensor.ndim != 4 or tensor.dtype != torch.int8:
        raise ValueError('grouped PixelShuffle pair layout requires int8 OIHW')
    if weight_bits not in (2, 4, 8):
        raise ValueError(
            f'grouped PixelShuffle layout requires W2/W4/W8, got W{weight_bits}'
        )
    if tail_channels * 2 != output_lanes:
        raise ValueError(
            'pair layout requires two shuffled pixels to fill one row'
        )
    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    if output_channels % tail_channels != 0:
        raise ValueError('expand outputs must be divisible by tail channels')
    subpixels = output_channels // tail_channels
    if subpixels % 2 != 0:
        raise ValueError('PixelShuffle subpixel count must be even')

    operands = array_weight_bits // weight_bits
    input_groups = (input_channels + operands - 1) // operands
    padded = torch.zeros(
        output_channels,
        input_groups * operands,
        kernel_height,
        kernel_width,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :input_channels] = tensor
    channel_subpixel = padded.reshape(
        tail_channels,
        subpixels,
        input_groups,
        operands,
        kernel_height,
        kernel_width,
    )
    even = channel_subpixel[:, 0::2].permute(
        1, 4, 5, 2, 3, 0
    )
    odd = channel_subpixel[:, 1::2].permute(
        1, 4, 5, 2, 3, 0
    )
    paired = torch.cat((even, odd), dim=-1).contiguous()
    return paired, (
        f'BHWGIL-P2, W{weight_bits} int8 rows, B=subpixel pair, '
        'G=input group, I=operand, '
        'L=[even-pixel channels, odd-pixel channels]'
    )


def pixel_shuffle_pair_param_layout(
    tensor,
    tail_channels,
    output_lanes=HEAD_OUTPUT_LANES,
):
    """Apply the same two-pixel lane order to bias/PReLU vectors."""
    if tensor.ndim != 1:
        raise ValueError('PixelShuffle pair parameters must be vectors')
    if tail_channels * 2 != output_lanes:
        raise ValueError(
            'pair parameter layout requires two pixels per row'
        )
    if tensor.numel() % tail_channels != 0:
        raise ValueError('parameter count must be divisible by tail channels')
    subpixels = tensor.numel() // tail_channels
    if subpixels % 2 != 0:
        raise ValueError('PixelShuffle subpixel count must be even')
    channel_subpixel = tensor.reshape(tail_channels, subpixels)
    paired = torch.cat(
        (
            channel_subpixel[:, 0::2].transpose(0, 1),
            channel_subpixel[:, 1::2].transpose(0, 1),
        ),
        dim=1,
    ).contiguous()
    return paired, (
        'BL-P2, B=subpixel pair, '
        'L=[even-pixel channels, odd-pixel channels]'
    )


def int16_depthwise_weight_layout(tensor, output_lanes=HEAD_OUTPUT_LANES):
    """Transpose depthwise W16 OIHW into low/high block-tap rows."""
    if tensor.ndim != 4 or tensor.dtype != torch.int16:
        raise ValueError('depthwise W16 layout requires an int16 OIHW tensor')
    if tensor.shape[1] != 1:
        raise ValueError('depthwise OIHW must contain one input per group')
    output_channels, _, kernel_height, kernel_width = tensor.shape
    output_blocks = (output_channels + output_lanes - 1) // output_lanes
    kernel_elements = kernel_height * kernel_width
    padded = torch.zeros(
        output_blocks * output_lanes,
        kernel_elements,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:output_channels] = tensor[:, 0].reshape(
        output_channels, kernel_elements
    )
    block_tap_lane = padded.reshape(
        output_blocks, output_lanes, kernel_elements
    ).permute(0, 2, 1).contiguous()
    return (
        split_int16_byte_planes(block_tap_lane),
        'KBTL, K0=low byte, K1=high byte, '
        'B=output block, T=kernel tap, L=channel lane',
    )


def final_weight_layout(tensor, input_lanes=HEAD_OUTPUT_LANES):
    """Pack one-output W16 OIHW weights for cross-column input reduction.

    Each kernel tap occupies one scratchpad row. Input channels are placed in
    array columns and the unused columns are zero padded.
    """
    if tensor.ndim != 4 or tensor.dtype != torch.int16:
        raise ValueError('final W16 layout requires an int16 OIHW tensor')
    output_channels, input_channels, kernel_height, kernel_width = tensor.shape
    if output_channels != 1:
        raise ValueError('final column-reduce layout requires one output channel')
    if input_channels > input_lanes:
        raise ValueError(
            f'final input channels {input_channels} exceed {input_lanes} lanes'
        )
    kernel_elements = kernel_height * kernel_width
    tap_lane = torch.zeros(
        kernel_elements,
        input_lanes,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    tap_lane[:, :input_channels] = tensor[0].permute(
        1, 2, 0
    ).reshape(kernel_elements, input_channels)
    return (
        split_int16_byte_planes(tap_lane),
        'KTL-CRED, K0=low byte, K1=high byte, '
        'T=kernel tap, L=input-channel column',
    )


def hardware_layout_tensor(
    name,
    tensor,
    weight_bits=None,
    tail_channels=None,
):
    """Convert an OIHW convolution weight to its accelerator storage layout."""
    if not name.endswith('_weight') or tensor.ndim != 4:
        return tensor.contiguous(), 'row-major'

    if name == 'head_weight':
        return head_weight_layout(tensor)

    if name == 'final_weight' and weight_bits == 16:
        return final_weight_layout(tensor)

    if (
        name == 'expand_weight'
        and weight_bits in (2, 4, 8)
        and tail_channels is not None
        and tail_channels * 2 == HEAD_OUTPUT_LANES
    ):
        return pixel_shuffle_pair_grouped_weight_layout(
            tensor,
            weight_bits,
            tail_channels,
        )

    if weight_bits in (2, 4, 8):
        return int8_grouped_weight_layout(tensor, weight_bits)

    if name == 'shrink2_weight' and weight_bits == 16:
        return int16_depthwise_weight_layout(tensor)

    if name == 'shrink3_weight' and weight_bits == 16:
        return int16_dense_weight_layout(tensor)

    if (
        name == 'expand_weight'
        and weight_bits == 16
        and tail_channels is not None
        and tail_channels * 2 == HEAD_OUTPUT_LANES
    ):
        return pixel_shuffle_pair_weight_layout(
            tensor, tail_channels
        )

    # PyTorch convolution weights are OIHW. The accelerator calls the output
    # channel dimension N and consumes weights in N,H,W,C order.
    nhwc = tensor.permute(0, 2, 3, 1).contiguous()

    if re.fullmatch(r'mapping_[0-9]+_weight', name):
        input_channels = nhwc.shape[-1]
        if input_channels != 16:
            raise ValueError(
                f'{name} binary input channel count must be 16 for uint16 '
                f'packing, got {input_channels}'
            )
        packed = pack_binary_last_axis(nhwc, 16)
        return packed, 'NHW, binary C packed into uint16 (C0=LSB)'

    if tensor.dtype == torch.int16:
        return (
            split_int16_byte_planes(nhwc),
            'KNHWC, K0=low byte, K1=high byte',
        )

    return nhwc, 'NHWC'


def hardware_layout_tensors(tensors, model_config=None):
    """Apply accelerator layouts and retain a description for each tensor."""
    laid_out = OrderedDict()
    layouts = {}
    tail_channels = None
    if model_config is not None:
        tail_channels = int(model_config.get('tail_channels', 0))
    pair_param_names = {
        'expand_bias',
        'tail_prelu_negative_coeff1',
        'tail_prelu_negative_coeff2',
        'tail_prelu_positive_shift',
        'tail_prelu_negative_left_shift1',
        'tail_prelu_negative_left_shift2',
        'tail_prelu_negative_common_shift',
    }
    pair_layout = (
        tail_channels is not None
        and tail_channels * 2 == HEAD_OUTPUT_LANES
    )
    for name, tensor in tensors.items():
        if pair_layout and name in pair_param_names:
            laid_out[name], layouts[name] = (
                pixel_shuffle_pair_param_layout(
                    tensor, tail_channels
                )
            )
            continue
        prefix = name[:-len('_weight')] if name.endswith('_weight') else None
        weight_bits = None
        if prefix is not None and model_config is not None:
            weight_bits = model_config.get(f'{prefix}_weight_bits')
        laid_out[name], layouts[name] = hardware_layout_tensor(
            name,
            tensor,
            weight_bits,
            tail_channels=tail_channels,
        )
    return laid_out, layouts


def render_hardware_header(
    model,
    metadata,
    checkpoint_path,
    output_path,
    symbol_prefix='bfsrcnn_hw',
    include_guard=None,
    values_per_line=16,
):
    if values_per_line < 1:
        raise ValueError('values_per_line must be positive')
    prefix = c_identifier(symbol_prefix)
    macro_prefix = c_identifier(symbol_prefix, uppercase=True)
    if include_guard is None:
        include_guard = c_identifier(
            f'{macro_prefix}_{Path(output_path).name}', uppercase=True
        )
    elif not re.fullmatch(r'[A-Za-z_][0-9A-Za-z_]*', include_guard):
        raise ValueError('include_guard must be a valid C identifier')

    tensors, layouts = hardware_layout_tensors(
        hardware_tensors(model), model.config
    )
    total_elements = sum(tensor.numel() for tensor in tensors.values())
    total_bytes = 0
    for name, tensor in tensors.items():
        if tensor.dtype not in DTYPE_TO_C:
            raise TypeError(f'unsupported tensor dtype for {name}: {tensor.dtype}')
        total_bytes += tensor.numel() * DTYPE_TO_C[tensor.dtype][1]

    lines = [
        '/* Auto-generated by models.bfsrcnn.deployment.hardware_params. */',
        f'/* Source checkpoint: {Path(checkpoint_path).as_posix()} */',
        '/* Runtime exponent arithmetic has been folded into shift tensors. */',
        '/* Two-term APoT: (coeff1*(x << left1) + coeff2*(x << left2)) */',
        '/* is passed through the signed common shift. */',
        '/* Mapping scale+RPReLU is fused into one affine transform per branch. */',
        '/* Mapping branch thresholds are in the unscaled convolution domain. */',
        '/* Packed branch coeff code: 0=zero, 1=+1, 2=-1 (2 bits each). */',
        '/* Convolution weights use accelerator-native layouts documented below. */',
        '/* Binary weight channel C0 is packed into the uint16_t least-significant bit. */',
        '/* 16-bit weights are byte-split with K0=low byte and K1=high byte. */',
        '/* Head KBTL rows can be copied directly into the accelerator scratchpad. */',
        f'#ifndef {include_guard}',
        f'#define {include_guard}',
        '',
        '#include <stdint.h>',
        '',
    ]
    for name, value in model.config.items():
        lines.append(f'#define {macro_prefix}_{c_identifier(name, True)} {int(value)}')
    pair_layout = int(
        int(model.tail_channels) * 2 == HEAD_OUTPUT_LANES
    )
    lines.append(
        f'#define {macro_prefix}_PIXEL_SHUFFLE_PAIR_LAYOUT '
        f'{pair_layout}'
    )
    lines.append(
        f'#define {macro_prefix}_FINAL_COLUMN_REDUCE_LAYOUT 1'
    )

    input_qmin, input_qmax = integer_range(model.input_bits, signed=False)
    head_qmin, head_qmax = integer_range(
        model.head_activation_bits, signed=True
    )
    shrink1_qmin, shrink1_qmax = integer_range(
        model.shrink1_activation_bits, signed=False
    )
    shrink2_qmin, shrink2_qmax = integer_range(
        model.shrink2_activation_bits, signed=False
    )
    state_qmin, state_qmax = integer_range(model.state_bits, signed=True)
    tail_qmin, tail_qmax = integer_range(
        model.tail_activation_bits, signed=True
    )
    scale = int(model.scale_factor)
    lines.extend([
        f'#define {macro_prefix}_INPUT_QMIN {input_qmin}',
        f'#define {macro_prefix}_INPUT_QMAX {input_qmax}',
        f'#define {macro_prefix}_HEAD_QMIN {head_qmin}',
        f'#define {macro_prefix}_HEAD_QMAX {head_qmax}',
        f'#define {macro_prefix}_SHRINK1_QMIN {shrink1_qmin}',
        f'#define {macro_prefix}_SHRINK1_QMAX {shrink1_qmax}',
        f'#define {macro_prefix}_SHRINK2_QMIN {shrink2_qmin}',
        f'#define {macro_prefix}_SHRINK2_QMAX {shrink2_qmax}',
        f'#define {macro_prefix}_STATE_QMIN {state_qmin}',
        f'#define {macro_prefix}_STATE_QMAX {state_qmax}',
        f'#define {macro_prefix}_TAIL_QMIN {tail_qmin}',
        f'#define {macro_prefix}_TAIL_QMAX {tail_qmax}',
        f'#define {macro_prefix}_BILINEAR_DENOMINATOR {2 * scale}',
        f'#define {macro_prefix}_BILINEAR_DIVISOR {4 * scale * scale}',
        f"#define {macro_prefix}_CALIBRATION_TILE_SIZE "
        f"{int(metadata.get('calibration_tile_size', 0))}",
        f"#define {macro_prefix}_CALIBRATION_TILE_PAD "
        f"{int(metadata.get('calibration_tile_pad', 0))}",
        f'#define {macro_prefix}_TENSOR_COUNT {len(tensors)}',
        f'#define {macro_prefix}_TOTAL_ELEMENTS {total_elements}',
        f'#define {macro_prefix}_TOTAL_BYTES {total_bytes}',
        '',
    ])

    for tensor_name, tensor in tensors.items():
        c_type, _ = DTYPE_TO_C[tensor.dtype]
        name = c_identifier(f'{prefix}_{tensor_name}')
        macro = c_identifier(
            f'{macro_prefix}_{tensor_name}', uppercase=True
        )
        shape = tuple(tensor.shape)
        lines.append(
            f'/* {tensor_name}: shape={shape}, dtype={tensor.dtype}, '
            f'layout={layouts[tensor_name]} */'
        )
        lines.append(f'#define {macro}_RANK {tensor.ndim}')
        for index, dimension in enumerate(shape):
            lines.append(f'#define {macro}_DIM_{index} {dimension}')
        lines.append(f'#define {macro}_SIZE {tensor.numel()}')
        lines.append(f'static const {c_type} {name}[{macro}_SIZE] = {{')
        lines.append(tensor_values(tensor, values_per_line))
        lines.extend(['};', ''])

    lines.extend([f'#endif /* {include_guard} */', ''])
    return '\n'.join(lines), {
        'tensor_count': len(tensors),
        'total_elements': total_elements,
        'total_bytes': total_bytes,
    }


def export_c_hardware_header(
    checkpoint_path,
    output_path,
    symbol_prefix='bfsrcnn_hw',
    include_guard=None,
    values_per_line=16,
):
    model, metadata = load_pixel_integer_checkpoint(checkpoint_path, 'cpu')
    text, summary = render_hardware_header(
        model,
        metadata,
        checkpoint_path,
        output_path,
        symbol_prefix,
        include_guard,
        values_per_line,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Export BFSRCNN integer parameters with all runtime shifts '
            'precomputed as a C99 header'
        )
    )
    parser.add_argument('--integer-weights-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--symbol-prefix', default='bfsrcnn_hw')
    parser.add_argument('--include-guard')
    parser.add_argument('--values-per-line', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = export_c_hardware_header(
        args.integer_weights_file,
        args.output_file,
        args.symbol_prefix,
        args.include_guard,
        args.values_per_line,
    )
    print(f'hardware C header written to {args.output_file}')
    print(
        f"exported {summary['tensor_count']} tensors, "
        f"{summary['total_elements']} elements, "
        f"{summary['total_bytes']} data bytes"
    )


if __name__ == '__main__':
    main()
