from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


INTEGER_FORMAT = 'bfsrcnn_integer_v4'
ALLOWED_WEIGHT_BITS = (2, 4, 8, 16)
MAX_ACTIVATION_BITS = 8


def validate_quantization_bits(**widths):
    for name, bits in widths.items():
        if name.endswith('weight_bits'):
            if bits not in ALLOWED_WEIGHT_BITS:
                raise ValueError(
                    f'{name} must be one of {ALLOWED_WEIGHT_BITS}, got {bits}'
                )
        elif name.endswith('activation_bits') or name in ('input_bits', 'state_bits'):
            if not 2 <= bits <= MAX_ACTIVATION_BITS:
                raise ValueError(
                    f'{name} must be between 2 and {MAX_ACTIVATION_BITS}, got {bits}'
                )
        else:
            raise ValueError(f'unknown quantization width {name}')


def integer_range(bits, signed):
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def weight_storage_dtype(bits):
    return torch.int16 if bits > 8 else torch.int8


def saturate(x, bits, signed):
    minimum, maximum = integer_range(bits, signed)
    return x.clamp(minimum, maximum)


def rounded_right_shift(x, amount):
    if amount <= 0:
        return x << (-amount)
    magnitude = (x.abs() + (1 << (amount - 1))) >> amount
    return torch.where(x < 0, -magnitude, magnitude)


def shift_integer(x, amount):
    if amount >= 0:
        return x << amount
    return rounded_right_shift(x, -amount)


def requantize_channels(x, shifts, bits, signed):
    outputs = []
    for channel in range(x.shape[1]):
        shifted = shift_integer(x[:, channel], int(shifts[channel].item()))
        outputs.append(shifted.unsqueeze(1))
    return saturate(torch.cat(outputs, dim=1), bits, signed).to(torch.int32)


def apot_shift_channels(x, exponent1, exponent2, sign2, base_exponent):
    outputs = []
    for channel in range(x.shape[1]):
        k1 = int(exponent1[channel].item()) - int(base_exponent[channel].item())
        k2 = int(exponent2[channel].item()) - int(base_exponent[channel].item())
        common = min(k1, k2, 0)
        term1 = x[:, channel].to(torch.int64) << (k1 - common)
        term2 = x[:, channel].to(torch.int64) << (k2 - common)
        combined = term1 + int(sign2[channel].item()) * term2
        outputs.append(shift_integer(combined, common).unsqueeze(1))
    return torch.cat(outputs, dim=1)


def apot_prelu_integer(
    accumulator,
    accumulator_exponent,
    output_exponent,
    slope_sign,
    exponent1,
    exponent2,
    term2_sign,
    output_bits,
):
    outputs = []
    for channel in range(accumulator.shape[1]):
        value = accumulator[:, channel].to(torch.int64)
        acc_exp = int(accumulator_exponent[channel].item())
        positive = shift_integer(value, acc_exp - int(output_exponent))

        k1 = acc_exp + int(exponent1[channel].item()) - int(output_exponent)
        k2 = acc_exp + int(exponent2[channel].item()) - int(output_exponent)
        common = min(k1, k2, 0)
        term1 = value << (k1 - common)
        term2 = value << (k2 - common)
        negative = term1 + int(term2_sign[channel].item()) * term2
        negative = shift_integer(negative, common)
        if int(slope_sign[channel].item()) < 0:
            negative = -negative
        outputs.append(torch.where(value >= 0, positive, negative).unsqueeze(1))
    output = torch.cat(outputs, dim=1)
    return saturate(output, output_bits, signed=True).to(torch.int32)


def integer_bilinear(image, scale):
    """Integer align_corners=False bilinear resize with round-to-nearest."""
    if scale < 1:
        raise ValueError('scale must be positive')
    if scale == 1:
        return image.clone()
    _, _, height, width = image.shape
    out_height, out_width = height * scale, width * scale
    denominator = 2 * scale

    oy = torch.arange(out_height, device=image.device, dtype=torch.int64)
    ox = torch.arange(out_width, device=image.device, dtype=torch.int64)
    y_numerator = 2 * oy + 1 - scale
    x_numerator = 2 * ox + 1 - scale
    y0 = torch.div(y_numerator, denominator, rounding_mode='floor')
    x0 = torch.div(x_numerator, denominator, rounding_mode='floor')
    wy1 = y_numerator - y0 * denominator
    wx1 = x_numerator - x0 * denominator
    wy0 = denominator - wy1
    wx0 = denominator - wx1
    y1 = y0 + 1
    x1 = x0 + 1
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)
    x0 = x0.clamp(0, width - 1)
    x1 = x1.clamp(0, width - 1)

    source = image.to(torch.int64)
    top_left = source[:, :, y0[:, None], x0[None, :]]
    top_right = source[:, :, y0[:, None], x1[None, :]]
    bottom_left = source[:, :, y1[:, None], x0[None, :]]
    bottom_right = source[:, :, y1[:, None], x1[None, :]]
    numerator = (
        top_left * wy0[:, None] * wx0[None, :]
        + top_right * wy0[:, None] * wx1[None, :]
        + bottom_left * wy1[:, None] * wx0[None, :]
        + bottom_right * wy1[:, None] * wx1[None, :]
    )
    divisor = denominator * denominator
    return ((numerator + divisor // 2) // divisor).to(torch.int32)


class IntegerBinaryBlock(nn.Module):
    def __init__(self, channels, state_bits):
        super().__init__()
        self.channels = channels
        self.state_bits = state_bits
        self.register_buffer('state_exponent', torch.zeros(channels, dtype=torch.int16))
        self.register_buffer('move0_threshold', torch.zeros(channels, dtype=torch.int32))
        self.register_buffer(
            'weight', torch.ones(channels, channels, 3, 3, dtype=torch.int8)
        )
        self.register_buffer(
            'rprelu_threshold', torch.zeros(channels, dtype=torch.int32)
        )
        self.register_buffer(
            'positive_exponent1', torch.zeros(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'positive_exponent2', torch.zeros(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'positive_term2_sign', torch.ones(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'positive_bias', torch.zeros(channels, dtype=torch.int32)
        )
        self.register_buffer(
            'negative_slope_sign', torch.ones(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'negative_exponent1', torch.zeros(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'negative_exponent2', torch.zeros(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'negative_term2_sign', torch.ones(channels, dtype=torch.int8)
        )
        self.register_buffer(
            'negative_bias', torch.zeros(channels, dtype=torch.int32)
        )

    def forward(self, state):
        threshold = self.move0_threshold.view(1, -1, 1, 1)
        binary = torch.where(state >= threshold, 1, -1).to(torch.int32)
        accumulator = F.conv2d(
            binary,
            self.weight.to(torch.int32),
            padding=1,
        )
        positive = apot_shift_channels(
            accumulator,
            self.positive_exponent1,
            self.positive_exponent2,
            self.positive_term2_sign,
            self.state_exponent,
        )
        positive = (
            positive
            + self.positive_bias.view(1, -1, 1, 1).to(torch.int64)
        )
        negative = apot_shift_channels(
            accumulator,
            self.negative_exponent1,
            self.negative_exponent2,
            self.negative_term2_sign,
            self.state_exponent,
        )
        negative = (
            negative * self.negative_slope_sign.view(1, -1, 1, 1)
            + self.negative_bias.view(1, -1, 1, 1).to(torch.int64)
        )
        delta = torch.where(
            accumulator
            >= self.rprelu_threshold.view(1, -1, 1, 1),
            positive,
            negative,
        )
        return saturate(
            state.to(torch.int64) + delta,
            self.state_bits,
            signed=True,
        ).to(torch.int32)


class IntegerBFSRCNN(nn.Module):
    def __init__(
        self,
        scale_factor=4,
        d=48,
        s=16,
        m=8,
        shrink_channels=32,
        tail_channels=8,
        input_bits=8,
        head_weight_bits=8,
        head_activation_bits=8,
        shrink1_weight_bits=8,
        shrink1_activation_bits=8,
        shrink2_weight_bits=8,
        shrink2_activation_bits=8,
        shrink3_weight_bits=8,
        state_bits=8,
        expand_weight_bits=8,
        tail_activation_bits=8,
        final_weight_bits=8,
    ):
        super().__init__()
        validate_quantization_bits(
            input_bits=input_bits,
            head_weight_bits=head_weight_bits,
            head_activation_bits=head_activation_bits,
            shrink1_weight_bits=shrink1_weight_bits,
            shrink1_activation_bits=shrink1_activation_bits,
            shrink2_weight_bits=shrink2_weight_bits,
            shrink2_activation_bits=shrink2_activation_bits,
            shrink3_weight_bits=shrink3_weight_bits,
            state_bits=state_bits,
            expand_weight_bits=expand_weight_bits,
            tail_activation_bits=tail_activation_bits,
            final_weight_bits=final_weight_bits,
        )
        self.scale_factor = scale_factor
        self.d = d
        self.s = s
        self.m = m
        self.shrink_channels = shrink_channels
        self.tail_channels = tail_channels
        self.input_bits = input_bits
        self.head_weight_bits = head_weight_bits
        self.head_activation_bits = head_activation_bits
        self.shrink1_weight_bits = shrink1_weight_bits
        self.shrink1_activation_bits = shrink1_activation_bits
        self.shrink2_weight_bits = shrink2_weight_bits
        self.shrink2_activation_bits = shrink2_activation_bits
        self.shrink3_weight_bits = shrink3_weight_bits
        self.state_bits = state_bits
        self.expand_weight_bits = expand_weight_bits
        self.tail_activation_bits = tail_activation_bits
        self.final_weight_bits = final_weight_bits

        self.register_buffer('input_exponent', torch.tensor(-input_bits, dtype=torch.int16))
        self.register_buffer('head_output_exponent', torch.tensor(0, dtype=torch.int16))
        self.register_buffer('shrink1_output_exponent', torch.tensor(0, dtype=torch.int16))
        self.register_buffer('shrink2_output_exponent', torch.tensor(0, dtype=torch.int16))
        self.register_buffer('state_exponent', torch.zeros(s, dtype=torch.int16))
        self.register_buffer('tail_output_exponent', torch.tensor(0, dtype=torch.int16))

        self.register_buffer('head_weight', torch.zeros(d, 1, 3, 3, dtype=weight_storage_dtype(head_weight_bits)))
        self.register_buffer('head_weight_exponent', torch.zeros(d, dtype=torch.int16))
        self.register_buffer('head_bias', torch.zeros(d, dtype=torch.int32))
        self._register_apot('head_prelu', d)

        self.register_buffer(
            'shrink1_weight',
            torch.zeros(
                shrink_channels, d, 1, 1,
                dtype=weight_storage_dtype(shrink1_weight_bits),
            ),
        )
        self.register_buffer(
            'shrink1_weight_exponent',
            torch.zeros(shrink_channels, dtype=torch.int16),
        )
        self.register_buffer(
            'shrink1_bias', torch.zeros(shrink_channels, dtype=torch.int32)
        )
        self.register_buffer(
            'shrink2_weight',
            torch.zeros(
                shrink_channels, 1, 3, 3,
                dtype=weight_storage_dtype(shrink2_weight_bits),
            ),
        )
        self.register_buffer(
            'shrink2_weight_exponent',
            torch.zeros(shrink_channels, dtype=torch.int16),
        )
        self.register_buffer(
            'shrink2_bias', torch.zeros(shrink_channels, dtype=torch.int32)
        )
        self.register_buffer(
            'shrink3_weight',
            torch.zeros(
                s, shrink_channels, 1, 1,
                dtype=weight_storage_dtype(shrink3_weight_bits),
            ),
        )
        self.register_buffer('shrink3_weight_exponent', torch.zeros(s, dtype=torch.int16))
        self.register_buffer('shrink3_bias', torch.zeros(s, dtype=torch.int32))

        self.mapping = nn.ModuleList(
            [IntegerBinaryBlock(s, state_bits) for _ in range(m)]
        )

        expanded_channels = tail_channels * scale_factor**2
        self.register_buffer(
            'expand_weight',
            torch.zeros(
                expanded_channels, s, 1, 1,
                dtype=weight_storage_dtype(expand_weight_bits),
            ),
        )
        self.register_buffer(
            'expand_weight_exponent',
            torch.zeros(expanded_channels, dtype=torch.int16),
        )
        self.register_buffer(
            'expand_bias', torch.zeros(expanded_channels, dtype=torch.int32)
        )
        self._register_apot('tail_prelu', expanded_channels)
        self.register_buffer(
            'final_weight',
            torch.zeros(
                1, tail_channels, 3, 3,
                dtype=weight_storage_dtype(final_weight_bits),
            ),
        )
        self.register_buffer('final_weight_exponent', torch.zeros(1, dtype=torch.int16))
        self.register_buffer('final_bias', torch.zeros(1, dtype=torch.int32))
        self.last_stats = {}

    def _register_apot(self, prefix, channels):
        self.register_buffer(f'{prefix}_slope_sign', torch.ones(channels, dtype=torch.int8))
        self.register_buffer(f'{prefix}_exponent1', torch.zeros(channels, dtype=torch.int8))
        self.register_buffer(f'{prefix}_exponent2', torch.zeros(channels, dtype=torch.int8))
        self.register_buffer(f'{prefix}_term2_sign', torch.ones(channels, dtype=torch.int8))

    @property
    def config(self):
        return {
            'scale_factor': self.scale_factor,
            'd': self.d,
            's': self.s,
            'm': self.m,
            'shrink_channels': self.shrink_channels,
            'tail_channels': self.tail_channels,
            'input_bits': self.input_bits,
            'head_weight_bits': self.head_weight_bits,
            'head_activation_bits': self.head_activation_bits,
            'shrink1_weight_bits': self.shrink1_weight_bits,
            'shrink1_activation_bits': self.shrink1_activation_bits,
            'shrink2_weight_bits': self.shrink2_weight_bits,
            'shrink2_activation_bits': self.shrink2_activation_bits,
            'shrink3_weight_bits': self.shrink3_weight_bits,
            'state_bits': self.state_bits,
            'expand_weight_bits': self.expand_weight_bits,
            'tail_activation_bits': self.tail_activation_bits,
            'final_weight_bits': self.final_weight_bits,
        }

    def _conv_accumulator(self, x, weight, bias, stride=1, padding=0, groups=1):
        out = F.conv2d(
            x.to(torch.int32),
            weight.to(torch.int32),
            stride=stride,
            padding=padding,
            groups=groups,
        ).to(torch.int64)
        return out + bias.to(torch.int64).view(1, -1, 1, 1)

    def _record_saturation(self, name, raw, bits, signed):
        minimum, maximum = integer_range(bits, signed)
        clipped = int(((raw < minimum) | (raw > maximum)).sum().item())
        total = raw.numel()
        previous = self.last_stats.get(name, (0, 0))
        self.last_stats[name] = (previous[0] + clipped, previous[1] + total)

    def reset_stats(self):
        self.last_stats = {}

    def forward_int(self, input_integer):
        residual = integer_bilinear(input_integer, self.scale_factor)
        head_acc = self._conv_accumulator(
            input_integer,
            self.head_weight,
            self.head_bias,
            padding=1,
        )
        head_acc_exp = self.head_weight_exponent + self.input_exponent + 6
        head = apot_prelu_integer(
            head_acc,
            head_acc_exp,
            self.head_output_exponent,
            self.head_prelu_slope_sign,
            self.head_prelu_exponent1,
            self.head_prelu_exponent2,
            self.head_prelu_term2_sign,
            self.head_activation_bits,
        )

        shrink1_acc = self._conv_accumulator(
            head, self.shrink1_weight, self.shrink1_bias
        )
        shrink1_acc.clamp_(min=0)
        shift1 = (
            self.head_output_exponent
            + self.shrink1_weight_exponent
            - self.shrink1_output_exponent
        )
        shrink1 = requantize_channels(
            shrink1_acc, shift1, self.shrink1_activation_bits, signed=False
        )

        shrink2_acc = self._conv_accumulator(
            shrink1,
            self.shrink2_weight,
            self.shrink2_bias,
            padding=1,
            groups=self.shrink_channels,
        )
        shrink2_acc.clamp_(min=0)
        shift2 = (
            self.shrink1_output_exponent
            + self.shrink2_weight_exponent
            - self.shrink2_output_exponent
        )
        shrink2 = requantize_channels(
            shrink2_acc, shift2, self.shrink2_activation_bits, signed=False
        )

        shrink3_acc = self._conv_accumulator(
            shrink2, self.shrink3_weight, self.shrink3_bias
        )
        shift3 = (
            self.shrink2_output_exponent
            + self.shrink3_weight_exponent
            - self.state_exponent
        )
        state = requantize_channels(
            shrink3_acc, shift3, self.state_bits, signed=True
        )
        for block in self.mapping:
            state = block(state)

        expand_acc = self._conv_accumulator(
            state, self.expand_weight, self.expand_bias
        )
        expand_acc_exp = self.expand_weight_exponent
        expanded = apot_prelu_integer(
            expand_acc,
            expand_acc_exp,
            self.tail_output_exponent,
            self.tail_prelu_slope_sign,
            self.tail_prelu_exponent1,
            self.tail_prelu_exponent2,
            self.tail_prelu_term2_sign,
            self.tail_activation_bits,
        )
        expanded = F.pixel_shuffle(expanded, self.scale_factor)

        final_acc = self._conv_accumulator(
            expanded, self.final_weight, self.final_bias, padding=1
        )
        correction_shift = int(
            (
                self.tail_output_exponent
                + self.final_weight_exponent[0]
                - 6
                - self.input_exponent
            ).item()
        )
        correction = shift_integer(final_acc, correction_shift)
        output_raw = residual.to(torch.int64) + correction
        self._record_saturation(
            'output', output_raw, self.input_bits, signed=False
        )
        return saturate(
            output_raw, self.input_bits, signed=False
        ).to(torch.int32)

    def forward(self, x):
        scale = 2.0 ** int(self.input_exponent.item())
        minimum, maximum = integer_range(self.input_bits, signed=False)
        input_integer = torch.round(x / scale).clamp(minimum, maximum)
        output_integer = self.forward_int(input_integer.to(torch.int32))
        return output_integer.to(torch.float32) * scale


def save_integer_checkpoint(model, output_file, metadata=None):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format': INTEGER_FORMAT,
            'config': model.config,
            'state_dict': model.state_dict(),
            'metadata': {} if metadata is None else metadata,
        },
        output_file,
    )


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_integer_checkpoint(path, map_location='cpu'):
    checkpoint = _torch_load(path, map_location)
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != INTEGER_FORMAT:
        raise ValueError(f'{path} is not an {INTEGER_FORMAT} checkpoint')
    model = IntegerBFSRCNN(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint.get('metadata', {})
