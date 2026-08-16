"""Export BFSRCNN's quantized weights as native BIRA C data arrays.

This model-owned adapter contains no inference schedule. The programmer-written
``applications/bfsrcnn/bfsrcnn_inference.c`` consumes the exported arrays through the
generic Runtime layer descriptors.
"""

import argparse
import json
from pathlib import Path

from models.bfsrcnn.deployment.hardware_params import (
    hardware_layout_tensors,
    hardware_tensors,
)
from models.bfsrcnn.quantization.checkpoint import (
    load_pixel_integer_checkpoint,
)
from models.bfsrcnn.quantization.integer_model import integer_range
from models.common.deployment.c_data import DTYPE_TO_C, c_identifier
from models.common.deployment.parameter_layout import (
    BinaryRecord,
    MultiBitRecord,
    binary_correction,
    pack_parameter_rows,
)


def model_export_tables(model):
    """Return the macros and native tensors consumed by final C rendering."""

    tensors, _ = hardware_layout_tensors(
        hardware_tensors(model), model.config
    )
    macros = {
        f'BFSRCNN_HW_{c_identifier(name, uppercase=True)}': int(value)
        for name, value in model.config.items()
    }
    ranges = {
        'INPUT': integer_range(model.input_bits, signed=False),
        'HEAD': integer_range(model.head_activation_bits, signed=True),
        'SHRINK1': integer_range(
            model.shrink1_activation_bits, signed=False
        ),
        'SHRINK2': integer_range(
            model.shrink2_activation_bits, signed=False
        ),
        'STATE': integer_range(model.state_bits, signed=True),
        'TAIL': integer_range(model.tail_activation_bits, signed=True),
    }
    for name, (qmin, qmax) in ranges.items():
        macros[f'BFSRCNN_HW_{name}_QMIN'] = qmin
        macros[f'BFSRCNN_HW_{name}_QMAX'] = qmax
    arrays = {}
    for name, tensor in tensors.items():
        c_type, _ = DTYPE_TO_C[tensor.dtype]
        arrays[f'bfsrcnn_hw_{name}'] = {
            'type': c_type,
            'values': [
                int(value)
                for value in tensor.detach().cpu().contiguous().view(-1)
            ],
        }
    return macros, arrays


def _values(arrays, suffix):
    name = f'bfsrcnn_hw_{suffix}'
    try:
        return arrays[name]['values']
    except KeyError as error:
        raise KeyError(f'missing tensor {name}') from error


def _macro(macros, suffix):
    name = f'BFSRCNN_HW_{suffix}'
    try:
        return macros[name]
    except KeyError as error:
        raise KeyError(f'missing macro {name}') from error


def _semantic_coefficient(code):
    """Decode the BFSRCNN hardware tensor's two-bit ternary coefficient."""

    code &= 0b11
    if code == 0:
        return 0
    if code == 1:
        return 1
    if code == 2:
        return -1
    raise ValueError('BFSRCNN coefficient code 3 is reserved')


def _multi_records(
    bias,
    positive_shift,
    negative_coeff1,
    negative_coeff2,
    negative_left_shift1,
    negative_left_shift2,
    negative_common_shift,
    qmin,
    qmax,
    binary_threshold=None,
):
    count = len(bias)
    fields = (
        positive_shift,
        negative_coeff1,
        negative_coeff2,
        negative_left_shift1,
        negative_left_shift2,
        negative_common_shift,
    )
    if any(len(field) != count for field in fields):
        raise ValueError('multi-bit parameter tensor lengths do not match')
    if binary_threshold is None:
        binary_threshold = [0] * count
    if len(binary_threshold) != count:
        raise ValueError('binary threshold tensor length does not match')
    return [
        MultiBitRecord(
            bias=bias[index],
            positive_shift=positive_shift[index],
            negative_coeff1=negative_coeff1[index],
            negative_coeff2=negative_coeff2[index],
            negative_left_shift1=negative_left_shift1[index],
            negative_left_shift2=negative_left_shift2[index],
            negative_common_shift=negative_common_shift[index],
            qmin=qmin,
            qmax=qmax,
            binary_threshold=binary_threshold[index],
        )
        for index in range(count)
    ]


def build_parameter_blobs(macros, arrays):
    state_qmin = _macro(macros, 'STATE_QMIN')
    state_qmax = _macro(macros, 'STATE_QMAX')
    blobs = {}

    blobs['head_parameters'] = pack_parameter_rows(
        _multi_records(
            _values(arrays, 'head_bias'),
            _values(arrays, 'head_prelu_positive_shift'),
            _values(arrays, 'head_prelu_negative_coeff1'),
            _values(arrays, 'head_prelu_negative_coeff2'),
            _values(arrays, 'head_prelu_negative_left_shift1'),
            _values(arrays, 'head_prelu_negative_left_shift2'),
            _values(arrays, 'head_prelu_negative_common_shift'),
            _macro(macros, 'HEAD_QMIN'),
            _macro(macros, 'HEAD_QMAX'),
        )
    )

    for name, qmin_name, qmax_name in (
        ('shrink1', 'SHRINK1_QMIN', 'SHRINK1_QMAX'),
        ('shrink2', 'SHRINK2_QMIN', 'SHRINK2_QMAX'),
    ):
        bias = _values(arrays, f'{name}_bias')
        shift = _values(arrays, f'{name}_requant_shift')
        zeros = [0] * len(bias)
        blobs[f'{name}_parameters'] = pack_parameter_rows(
            _multi_records(
                bias,
                shift,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                _macro(macros, qmin_name),
                _macro(macros, qmax_name),
            )
        )

    shrink3_bias = _values(arrays, 'shrink3_bias')
    shrink3_shift = _values(arrays, 'shrink3_requant_shift')
    zeros = [0] * len(shrink3_bias)
    ones = [1] * len(shrink3_bias)
    blobs['shrink3_parameters'] = pack_parameter_rows(
        _multi_records(
            shrink3_bias,
            shrink3_shift,
            ones,
            zeros,
            zeros,
            zeros,
            shrink3_shift,
            state_qmin,
            state_qmax,
            _values(arrays, 'mapping_0_move0_threshold'),
        )
    )

    mapping_count = _macro(macros, 'M')
    for mapping in range(mapping_count):
        prefix = f'mapping_{mapping}'
        coefficients = _values(arrays, f'{prefix}_branch_coefficients')
        next_threshold = (
            _values(arrays, f'mapping_{mapping + 1}_move0_threshold')
            if mapping + 1 < mapping_count
            else [0] * len(coefficients)
        )
        records = []
        for channel, packed in enumerate(coefficients):
            records.append(
                BinaryRecord(
                    threshold=_values(
                        arrays, f'{prefix}_rprelu_threshold'
                    )[channel],
                    positive_coeff2=_semantic_coefficient(packed),
                    positive_left_shift1=_values(
                        arrays, f'{prefix}_positive_left_shift1'
                    )[channel],
                    positive_left_shift2=_values(
                        arrays, f'{prefix}_positive_left_shift2'
                    )[channel],
                    positive_common_shift=_values(
                        arrays, f'{prefix}_positive_common_shift'
                    )[channel],
                    positive_bias=_values(
                        arrays, f'{prefix}_positive_bias'
                    )[channel],
                    negative_coeff1=_semantic_coefficient(packed >> 2),
                    negative_coeff2=_semantic_coefficient(packed >> 4),
                    negative_left_shift1=_values(
                        arrays, f'{prefix}_negative_left_shift1'
                    )[channel],
                    negative_left_shift2=_values(
                        arrays, f'{prefix}_negative_left_shift2'
                    )[channel],
                    negative_common_shift=_values(
                        arrays, f'{prefix}_negative_common_shift'
                    )[channel],
                    negative_bias=_values(
                        arrays, f'{prefix}_negative_bias'
                    )[channel],
                    qmin=state_qmin,
                    qmax=state_qmax,
                    output_sign_threshold=next_threshold[channel],
                )
            )
        blobs[f'{prefix}_parameters'] = pack_parameter_rows(records)

    expand_bias = _values(arrays, 'expand_bias')
    blobs['expand_parameters'] = pack_parameter_rows(
        _multi_records(
            expand_bias,
            _values(arrays, 'tail_prelu_positive_shift'),
            _values(arrays, 'tail_prelu_negative_coeff1'),
            _values(arrays, 'tail_prelu_negative_coeff2'),
            _values(arrays, 'tail_prelu_negative_left_shift1'),
            _values(arrays, 'tail_prelu_negative_left_shift2'),
            _values(arrays, 'tail_prelu_negative_common_shift'),
            _macro(macros, 'TAIL_QMIN'),
            _macro(macros, 'TAIL_QMAX'),
        )
    )

    final_bias = _values(arrays, 'final_bias')
    final_shift = _values(arrays, 'final_correction_shift')
    final_record = _multi_records(
        final_bias,
        final_shift,
        [1],
        [0],
        [0],
        [0],
        final_shift,
        -(1 << 31),
        (1 << 31) - 1,
    )[0]
    # COLUMN_REDUCE packs 16 consecutive output pixels into accumulator
    # lanes. IntPostProc consumes one parameter record per lane, so the
    # single output-channel record must be broadcast in the static image.
    blobs['final_parameters'] = pack_parameter_rows(
        [final_record] * 16
    )
    return blobs


WEIGHT_SUFFIXES = [
    'head_weight',
    'shrink1_weight',
    'shrink2_weight',
    'shrink3_weight',
    *[f'mapping_{index}_weight' for index in range(8)],
    'expand_weight',
    'final_weight',
]


def _format_values(values, values_per_line=16):
    return '\n'.join(
        '    ' + ', '.join(str(value) for value in values[index:index + values_per_line]) + ','
        for index in range(0, len(values), values_per_line)
    )


def _format_bytes(values, values_per_line=16):
    return '\n'.join(
        '    ' + ', '.join(
            f'0x{value:02x}'
            for value in values[index:index + values_per_line]
        ) + ','
        for index in range(0, len(values), values_per_line)
    )


def _array_declaration(c_type, name, size):
    return f'extern const {c_type} bfsrcnn_bira_{name}[{size}];'


def render_data_files(macros, arrays, input_height, input_width):
    if input_height <= 0 or input_width <= 0:
        raise ValueError('input dimensions must be positive')
    if input_height * input_width % 16:
        raise ValueError(
            'ISA BFSRCNN bundle requires input H*W divisible by 16 '
            'for packed head/residual DMA'
        )
    if _macro(macros, 'S') != 16 or _macro(macros, 'M') != 8:
        raise ValueError('the ISA BFSRCNN template requires S=16 and M=8')

    parameter_blobs = build_parameter_blobs(macros, arrays)
    correction = binary_correction(
        input_height,
        input_width,
        3,
        3,
        1,
        1,
        16,
    )
    parameter_blobs['mapping_correction'] = correction

    weights = {}
    for suffix in WEIGHT_SUFFIXES:
        source_name = f'bfsrcnn_hw_{suffix}'
        if source_name not in arrays:
            raise KeyError(f'missing weight tensor {source_name}')
        weights[suffix] = arrays[source_name]

    output_height = input_height * _macro(macros, 'SCALE_FACTOR')
    output_width = input_width * _macro(macros, 'SCALE_FACTOR')
    header_lines = [
        '#ifndef BFSRCNN_BIRA_DATA_H',
        '#define BFSRCNN_BIRA_DATA_H',
        '',
        '#include <stdint.h>',
        '',
        f'#define BFSRCNN_BIRA_INPUT_HEIGHT {input_height}u',
        f'#define BFSRCNN_BIRA_INPUT_WIDTH {input_width}u',
        f'#define BFSRCNN_BIRA_OUTPUT_HEIGHT {output_height}u',
        f'#define BFSRCNN_BIRA_OUTPUT_WIDTH {output_width}u',
        f'#define BFSRCNN_BIRA_INPUT_PACKED_ROWS '
        f'{input_height * input_width // 16}u',
        f'#define BFSRCNN_BIRA_OUTPUT_PACKED_ROWS '
        f'{output_height * output_width // 16}u',
        '',
    ]

    source_lines = [
        '#include "bfsrcnn_bira_data.h"',
        '',
        '#if defined(__GNUC__)',
        '#define BIRA_DATA_ALIGN __attribute__((aligned(64)))',
        '#else',
        '#define BIRA_DATA_ALIGN',
        '#endif',
        '',
    ]
    for suffix, array in weights.items():
        c_type = array['type']
        values = array['values']
        header_lines.append(_array_declaration(c_type, suffix, len(values)))
        source_lines.extend([
            f'const {c_type} bfsrcnn_bira_{suffix}[{len(values)}] '
            f'BIRA_DATA_ALIGN = {{',
            _format_values(values),
            '};',
            '',
        ])

    plane_specs = {
        'HEAD': len(weights['head_weight']['values']) // 2,
        'SHRINK2': len(weights['shrink2_weight']['values']) // 2,
        'FINAL': len(weights['final_weight']['values']) // 2,
    }
    for name, plane_bytes in plane_specs.items():
        header_lines.extend([
            f'#define BFSRCNN_BIRA_{name}_WEIGHT_PLANE_BYTES '
            f'{plane_bytes}u',
            f'#define BFSRCNN_BIRA_{name}_WEIGHT_LOW_ROWS '
            f'{plane_bytes // 16}u',
        ])
    header_lines.extend([
        '#define BFSRCNN_BIRA_SHRINK1_WEIGHT_ROWS '
        f'{len(weights["shrink1_weight"]["values"]) // 16}u',
        '#define BFSRCNN_BIRA_SHRINK3_WEIGHT_ROWS '
        f'{len(weights["shrink3_weight"]["values"]) // 16}u',
        '#define BFSRCNN_BIRA_EXPAND_WEIGHT_ROWS '
        f'{len(weights["expand_weight"]["values"]) // 16}u',
        '#define BFSRCNN_BIRA_MAPPING_WEIGHT_ROWS '
        f'{len(weights["mapping_0_weight"]["values"])}u',
        '',
    ])

    for name, data in parameter_blobs.items():
        header_lines.extend([
            f'#define BFSRCNN_BIRA_{name.upper()}_SIZE {len(data)}u',
            _array_declaration('uint8_t', name, len(data)),
        ])
        source_lines.extend([
            f'const uint8_t bfsrcnn_bira_{name}[{len(data)}] '
            f'BIRA_DATA_ALIGN = {{',
            _format_bytes(data),
            '};',
            '',
        ])

    header_lines.extend(['', '#endif', ''])
    return (
        '\n'.join(header_lines),
        '\n'.join(source_lines),
        {
            'format': 'bira-model-data-v2',
            'input_shape': [input_height, input_width, 1],
            'output_shape': [output_height, output_width, 1],
            'layers': [
                'head',
                'shrink1',
                'shrink2',
                'shrink3',
                *[f'mapping{index}' for index in range(8)],
                'expand',
                'final',
            ],
            'weights': {
                name: len(array['values'])
                for name, array in weights.items()
            },
            'parameter_bytes': {
                name: len(data)
                for name, data in parameter_blobs.items()
            },
        },
    )


def write_bfsrcnn_data(model, output_dir, input_height, input_width):
    """Write native model data without emitting network control code."""

    macros, arrays = model_export_tables(model)
    data_header, data_source, manifest = render_data_files(
        macros, arrays, input_height, input_width
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'bfsrcnn_bira_data.h': data_header,
        'bfsrcnn_bira_data.c': data_source,
        'bfsrcnn_bira_data_manifest.json': json.dumps(
            manifest, indent=2, sort_keys=True
        ) + '\n',
    }
    for name, text in files.items():
        (output_dir / name).write_text(text)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='generate native BFSRCNN parameters for BIRA operators'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--input-height', type=int, default=32)
    parser.add_argument('--input-width', type=int, default=32)
    args = parser.parse_args()
    model, _ = load_pixel_integer_checkpoint(args.checkpoint, 'cpu')
    summary = write_bfsrcnn_data(
        model,
        args.output_dir,
        args.input_height,
        args.input_width,
    )
    print(
        f'generated data for {len(summary["layers"])} layers in '
        f'{args.output_dir}'
    )


if __name__ == '__main__':
    main()
