"""Quantization handlers for basic PyTorch neural-network layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .core import (
    binary_sign,
    power_of_two_activation_exponent,
    quantize_accumulator_bias,
    quantize_double_power_of_two,
    quantize_power_of_two_weight,
)


@dataclass(frozen=True)
class LayerQuantSpec:
    """Hardware-aware policy for one Conv/Linear/Binary layer."""

    weight_bits: int = 8
    activation_bits: int = 8
    activation_signed: bool = True
    binary: bool = False
    minimum_exponent: int = -24
    maximum_exponent: int = 8

    def __post_init__(self) -> None:
        if self.binary:
            if self.weight_bits not in (1, 2):
                raise ValueError('binary weights use semantic 1 bit or ISA W2')
        elif self.weight_bits not in (2, 4, 8, 16):
            raise ValueError('integer weights must be W2/W4/W8/W16')
        if not 2 <= self.activation_bits <= 8:
            raise ValueError('BIRA activation width must be 2..8')


@dataclass
class QuantizedLayer:
    """Semantic integer parameters produced before backend layout."""

    kind: str
    weight: torch.Tensor
    weight_exponent: torch.Tensor
    bias: torch.Tensor
    input_exponent: torch.Tensor
    output_exponent: torch.Tensor
    weight_bits: int
    activation_bits: int
    activation_signed: bool
    binary_scale: torch.Tensor | None = None


def sample_tensor(
    tensor: torch.Tensor,
    maximum: int = 16384,
    *,
    per_channel: bool = False,
) -> torch.Tensor:
    """Deterministically subsample activation magnitudes for calibration."""

    if maximum <= 0:
        raise ValueError('maximum sample count must be positive')
    values = tensor.detach().abs().to(torch.float32).cpu()
    if per_channel:
        if values.ndim < 2:
            raise ValueError('per-channel samples require N,C,... tensor')
        order = [1, 0, *range(2, values.ndim)]
        values = values.permute(order).reshape(values.shape[1], -1)
        step = max(1, (values.shape[1] + maximum - 1) // maximum)
        return values[:, ::step][:, :maximum]
    values = values.reshape(-1)
    step = max(1, (values.numel() + maximum - 1) // maximum)
    return values[::step][:maximum]


class ModuleOutputCalibrator:
    """Collect output samples from named modules using forward hooks."""

    def __init__(
        self,
        model: nn.Module,
        module_names=None,
        *,
        maximum_per_call: int = 16384,
        per_channel: bool = False,
    ):
        modules = dict(model.named_modules())
        if module_names is None:
            module_names = [
                name for name, module in modules.items()
                if name and isinstance(module, (nn.Conv2d, nn.Linear))
            ]
        self.maximum_per_call = maximum_per_call
        self.per_channel = per_channel
        self.samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in module_names
        }
        self._handles = []
        for name in module_names:
            if name not in modules:
                raise KeyError(f'module {name!r} does not exist')
            self._handles.append(
                modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def observe(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f'module {name} output is not a tensor')
            self.samples[name].append(sample_tensor(
                output,
                self.maximum_per_call,
                per_channel=self.per_channel,
            ))
        return observe

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def values(self, name: str) -> torch.Tensor:
        samples = self.samples.get(name)
        if not samples:
            raise RuntimeError(f'module {name!r} has no calibration samples')
        dimension = 1 if self.per_channel else 0
        return torch.cat(samples, dim=dimension)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def quantize_basic_layer(
    module: nn.Module,
    spec: LayerQuantSpec,
    *,
    input_exponent,
    output_samples: torch.Tensor,
) -> QuantizedLayer:
    """Quantize Conv2d or Linear without depending on a network topology."""

    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        raise TypeError('basic integer handler supports Conv2d and Linear')
    input_exponent = torch.as_tensor(
        input_exponent, dtype=torch.int16)
    output_exponent = power_of_two_activation_exponent(
        output_samples,
        spec.activation_bits,
        spec.activation_signed,
        spec.minimum_exponent,
        spec.maximum_exponent,
    )
    if spec.binary:
        weight = binary_sign(module.weight.detach()).to(torch.int8)
        flattened = module.weight.detach().abs().reshape(
            module.weight.shape[0], -1)
        binary_scale = flattened.mean(dim=1)
        weight_exponent = torch.zeros(
            module.weight.shape[0], dtype=torch.int16)
        # Binary affine fusion consumes the floating scale and bias later.
        bias = (
            torch.zeros(module.weight.shape[0], dtype=torch.int32)
            if module.bias is None else
            module.bias.detach().to(torch.float32)
        )
        return QuantizedLayer(
            'binary_conv2d' if isinstance(module, nn.Conv2d) else 'binary_linear',
            weight,
            weight_exponent,
            bias,
            input_exponent,
            output_exponent,
            1,
            spec.activation_bits,
            spec.activation_signed,
            binary_scale,
        )

    weight, weight_exponent = quantize_power_of_two_weight(
        module.weight,
        spec.weight_bits,
        spec.minimum_exponent,
        spec.maximum_exponent,
    )
    accumulator_exponent = weight_exponent + input_exponent
    bias = quantize_accumulator_bias(module.bias, accumulator_exponent)
    return QuantizedLayer(
        'conv2d' if isinstance(module, nn.Conv2d) else 'linear',
        weight,
        weight_exponent,
        bias,
        input_exponent,
        output_exponent,
        spec.weight_bits,
        spec.activation_bits,
        spec.activation_signed,
    )


def quantize_prelu_slope(
    module: nn.PReLU,
    *,
    minimum_exponent: int = -16,
    maximum_exponent: int = 4,
    allow_subtract: bool = True,
) -> dict[str, torch.Tensor]:
    """Lower an ordinary PReLU slope to the two-term shift representation."""

    slope = module.weight.detach()
    exponent1, exponent2, term2_sign, quantized = (
        quantize_double_power_of_two(
            slope.abs(),
            minimum_exponent,
            maximum_exponent,
            allow_subtract,
        )
    )
    return {
        'slope_sign': slope.sign().to(torch.int8),
        'exponent1': exponent1,
        'exponent2': exponent2,
        'term2_sign': term2_sign,
        'quantized_slope': quantized * slope.sign(),
    }


def discover_basic_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Return modules handled by the generic quantization registry."""

    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(
            module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.PReLU)
        )
    }
