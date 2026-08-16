"""BIRA Parameter Buffer layout shared by model deployment adapters.

The quantization frontend produces semantic signed integers. This module is
the only place that knows the frozen 256-bit lane records and 64-byte native
row layout, keeping bit positions out of model-specific code.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


LANES = 16
RECORD_BITS = 256
ROW_BYTES = 64
RECORDS_PER_ROW = 2
ROWS_PER_BLOCK = LANES // RECORDS_PER_ROW


def _unsigned(value: int, bits: int, name: str) -> int:
    if value < 0 or value >= 1 << bits:
        raise ValueError(f'{name}={value} does not fit uint{bits}')
    return value


def _signed(value: int, bits: int, name: str) -> int:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if value < minimum or value > maximum:
        raise ValueError(f'{name}={value} does not fit int{bits}')
    return value & ((1 << bits) - 1)


def _coefficient(value: int, name: str) -> int:
    if value not in (-1, 0, 1):
        raise ValueError(f'{name} must be -1, 0, or +1')
    # Hardware decodes the field with asSInt: -1 is two's-complement 0b11.
    return value & 0b11


def _field(record: int, value: int, offset: int, bits: int) -> int:
    return record | (value << offset)


@dataclass(frozen=True)
class MultiBitRecord:
    bias: int = 0
    positive_shift: int = 0
    negative_coeff1: int = 0
    negative_coeff2: int = 0
    negative_left_shift1: int = 0
    negative_left_shift2: int = 0
    negative_common_shift: int = 0
    qmin: int = -(1 << 31)
    qmax: int = (1 << 31) - 1
    binary_threshold: int = 0

    def pack(self) -> int:
        record = 0
        record = _field(record, _signed(self.bias, 32, 'bias'), 0, 32)
        record = _field(
            record,
            _signed(self.positive_shift, 8, 'positive_shift'),
            32,
            8,
        )
        record = _field(
            record,
            _coefficient(self.negative_coeff1, 'negative_coeff1'),
            40,
            2,
        )
        record = _field(
            record,
            _coefficient(self.negative_coeff2, 'negative_coeff2'),
            42,
            2,
        )
        record = _field(
            record,
            _unsigned(
                self.negative_left_shift1, 5, 'negative_left_shift1'
            ),
            44,
            5,
        )
        record = _field(
            record,
            _unsigned(
                self.negative_left_shift2, 5, 'negative_left_shift2'
            ),
            49,
            5,
        )
        record = _field(
            record,
            _signed(
                self.negative_common_shift, 8, 'negative_common_shift'
            ),
            54,
            8,
        )
        record = _field(record, _signed(self.qmin, 32, 'qmin'), 62, 32)
        record = _field(record, _signed(self.qmax, 32, 'qmax'), 94, 32)
        record = _field(
            record,
            _signed(self.binary_threshold, 32, 'binary_threshold'),
            126,
            32,
        )
        return record


@dataclass(frozen=True)
class BinaryRecord:
    threshold: int = 0
    positive_coeff2: int = 0
    positive_left_shift1: int = 0
    positive_left_shift2: int = 0
    positive_common_shift: int = 0
    positive_bias: int = 0
    negative_coeff1: int = 0
    negative_coeff2: int = 0
    negative_left_shift1: int = 0
    negative_left_shift2: int = 0
    negative_common_shift: int = 0
    negative_bias: int = 0
    qmin: int = -(1 << 31)
    qmax: int = (1 << 31) - 1
    output_sign_threshold: int = 0

    def pack(self) -> int:
        record = 0
        record = _field(
            record, _signed(self.threshold, 32, 'threshold'), 0, 32
        )
        record = _field(
            record,
            _coefficient(self.positive_coeff2, 'positive_coeff2'),
            32,
            2,
        )
        record = _field(
            record,
            _unsigned(
                self.positive_left_shift1, 5, 'positive_left_shift1'
            ),
            34,
            5,
        )
        record = _field(
            record,
            _unsigned(
                self.positive_left_shift2, 5, 'positive_left_shift2'
            ),
            39,
            5,
        )
        record = _field(
            record,
            _signed(
                self.positive_common_shift, 8, 'positive_common_shift'
            ),
            44,
            8,
        )
        record = _field(
            record,
            _signed(self.positive_bias, 32, 'positive_bias'),
            52,
            32,
        )
        record = _field(
            record,
            _coefficient(self.negative_coeff1, 'negative_coeff1'),
            84,
            2,
        )
        record = _field(
            record,
            _coefficient(self.negative_coeff2, 'negative_coeff2'),
            86,
            2,
        )
        record = _field(
            record,
            _unsigned(
                self.negative_left_shift1, 5, 'negative_left_shift1'
            ),
            88,
            5,
        )
        record = _field(
            record,
            _unsigned(
                self.negative_left_shift2, 5, 'negative_left_shift2'
            ),
            93,
            5,
        )
        record = _field(
            record,
            _signed(
                self.negative_common_shift, 8, 'negative_common_shift'
            ),
            98,
            8,
        )
        record = _field(
            record,
            _signed(self.negative_bias, 32, 'negative_bias'),
            106,
            32,
        )
        record = _field(record, _signed(self.qmin, 32, 'qmin'), 138, 32)
        record = _field(record, _signed(self.qmax, 32, 'qmax'), 170, 32)
        record = _field(
            record,
            _signed(
                self.output_sign_threshold, 32, 'output_sign_threshold'
            ),
            202,
            32,
        )
        return record


def pack_parameter_rows(
    records: Sequence[MultiBitRecord | BinaryRecord],
) -> bytes:
    """Pack records into complete 16-lane blocks.

    Even a one-channel layer consumes all eight rows of its first block,
    because the hardware reads a complete 16-lane block.
    """

    if not records:
        raise ValueError('at least one lane record is required')
    block_count = (len(records) + LANES - 1) // LANES
    padded_count = block_count * LANES
    packed = bytearray()
    for lane in range(0, padded_count, RECORDS_PER_ROW):
        low = records[lane].pack() if lane < len(records) else 0
        high = records[lane + 1].pack() if lane + 1 < len(records) else 0
        row = low | (high << RECORD_BITS)
        packed.extend(row.to_bytes(ROW_BYTES, 'little', signed=False))
    return bytes(packed)


def binary_correction(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    padding_height: int,
    padding_width: int,
    input_channels: int,
    lanes: int = LANES,
) -> bytes:
    """Pack compiler-computed per-pixel ``-N`` values into Parameter rows.

    Corrections do not vary across output channels, so one native 64-byte row
    stores 16 signed-int32 pixel values instead of repeating one value across
    all 16 lanes. The final row is zero-padded.
    """

    if min(
        height,
        width,
        kernel_height,
        kernel_width,
        input_channels,
        lanes,
    ) <= 0:
        raise ValueError('dimensions and channel counts must be positive')
    if lanes != ROW_BYTES // 4:
        raise ValueError('binary correction rows contain exactly 16 int32 values')
    corrections: list[int] = []
    for output_y in range(height):
        for output_x in range(width):
            valid_taps = 0
            for kernel_y in range(kernel_height):
                input_y = output_y + kernel_y - padding_height
                if input_y < 0 or input_y >= height:
                    continue
                for kernel_x in range(kernel_width):
                    input_x = output_x + kernel_x - padding_width
                    if 0 <= input_x < width:
                        valid_taps += 1
            correction = -(valid_taps * input_channels)
            corrections.append(correction)
    rows = bytearray()
    for start in range(0, len(corrections), lanes):
        packed = corrections[start:start + lanes]
        packed.extend([0] * (lanes - len(packed)))
        for correction in packed:
            rows.extend(correction.to_bytes(4, 'little', signed=True))
    return bytes(rows)
