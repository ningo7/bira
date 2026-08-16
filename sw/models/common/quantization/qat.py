"""Drop-in PyTorch QAT layers for ordinary binary/integer networks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .core import (
    SymmetricQuantSpec,
    fake_binary_sign,
    fake_quantize_symmetric,
)


class QuantConv2d(nn.Conv2d):
    """Conv2d whose forward pass emulates symmetric integer arithmetic.

    This layer deliberately does not prescribe a network. It can replace an
    ordinary ``nn.Conv2d`` in any topology supported by the later BIRA
    exporter. The integer export still records the returned scales explicitly.
    """

    def __init__(
        self,
        *args,
        weight_bits: int = 8,
        activation_bits: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_spec = SymmetricQuantSpec(weight_bits, channel_axis=0)
        self.activation_spec = SymmetricQuantSpec(activation_bits)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        activation, _ = fake_quantize_symmetric(
            input_tensor, self.activation_spec)
        weight, _ = fake_quantize_symmetric(self.weight, self.weight_spec)
        return F.conv2d(
            activation,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinaryConv2d(nn.Conv2d):
    """Conv2d with {-1,+1} activations and weights using clipped STE."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            fake_binary_sign(input_tensor),
            fake_binary_sign(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLinear(nn.Linear):
    """Linear layer with model-independent symmetric QAT."""

    def __init__(
        self,
        *args,
        weight_bits: int = 8,
        activation_bits: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_spec = SymmetricQuantSpec(weight_bits, channel_axis=0)
        self.activation_spec = SymmetricQuantSpec(activation_bits)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        activation, _ = fake_quantize_symmetric(
            input_tensor, self.activation_spec)
        weight, _ = fake_quantize_symmetric(self.weight, self.weight_spec)
        return F.linear(activation, weight, self.bias)
