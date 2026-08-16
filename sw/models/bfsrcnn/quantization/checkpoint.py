from pathlib import Path

import torch
import torch.nn.functional as F

from models.bfsrcnn.quantization.integer_model import (
    IntegerBFSRCNN,
    INTEGER_FORMAT,
    apot_prelu_integer,
    integer_bilinear,
    integer_range,
    requantize_channels,
    saturate,
    shift_integer,
)


PIXEL_INTEGER_FORMAT = f'{INTEGER_FORMAT}_pixel_v2'


class PixelIntegerBFSRCNN(IntegerBFSRCNN):
    """Integer BFSRCNN whose external representation is native 0..255 pixels.

    The converter folds 64/255 into the head weights and 255/64 into the
    final weights, so neither factor is evaluated by inference hardware.
    """

    def forward_int(self, input_integer):
        residual = integer_bilinear(input_integer, self.scale_factor)
        head_acc = self._conv_accumulator(
            input_integer,
            self.head_weight,
            self.head_bias,
            padding=1,
        )
        head_acc_exp = self.head_weight_exponent + self.input_exponent
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
        shrink1 = requantize_channels(
            shrink1_acc,
            self.head_output_exponent
            + self.shrink1_weight_exponent
            - self.shrink1_output_exponent,
            self.shrink1_activation_bits,
            signed=False,
        )

        shrink2_acc = self._conv_accumulator(
            shrink1,
            self.shrink2_weight,
            self.shrink2_bias,
            padding=1,
            groups=self.shrink_channels,
        )
        shrink2_acc.clamp_(min=0)
        shrink2 = requantize_channels(
            shrink2_acc,
            self.shrink1_output_exponent
            + self.shrink2_weight_exponent
            - self.shrink2_output_exponent,
            self.shrink2_activation_bits,
            signed=False,
        )

        shrink3_acc = self._conv_accumulator(
            shrink2, self.shrink3_weight, self.shrink3_bias
        )
        state = requantize_channels(
            shrink3_acc,
            self.shrink2_output_exponent
            + self.shrink3_weight_exponent
            - self.state_exponent,
            self.state_bits,
            signed=True,
        )
        for block in self.mapping:
            state = block(state)

        expand_acc = self._conv_accumulator(
            state, self.expand_weight, self.expand_bias
        )
        expanded = apot_prelu_integer(
            expand_acc,
            self.expand_weight_exponent,
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
                - self.input_exponent
            ).item()
        )
        correction = shift_integer(final_acc, correction_shift)
        output_raw = residual.to(torch.int64) + correction
        self._record_saturation('output', output_raw, self.input_bits, False)
        return saturate(output_raw, self.input_bits, signed=False).to(torch.int32)

    def forward(self, x):
        minimum, maximum = integer_range(self.input_bits, signed=False)
        input_integer = torch.round(x * maximum).clamp(minimum, maximum)
        output_integer = self.forward_int(input_integer.to(torch.int32))
        return output_integer.to(torch.float32) / maximum


def save_pixel_integer_checkpoint(model, output_file, metadata=None):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format': PIXEL_INTEGER_FORMAT,
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


def load_pixel_integer_checkpoint(path, map_location='cpu'):
    checkpoint = _torch_load(path, map_location)
    if checkpoint.get('format') != PIXEL_INTEGER_FORMAT:
        raise ValueError(f'{path} is not a {PIXEL_INTEGER_FORMAT} checkpoint')
    model = PixelIntegerBFSRCNN(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint.get('metadata', {})
