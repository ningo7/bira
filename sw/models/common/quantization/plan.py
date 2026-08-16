"""Declarative, model-independent quantization-plan executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn

from .core import (
    fused_binary_rprelu,
    power_of_two_activation_exponent,
    quantize_accumulator_bias,
    quantize_double_power_of_two,
    quantize_power_of_two_weight,
)


TensorValue = str | int | float | torch.Tensor


def _source_tensor(model: nn.Module, path: str) -> torch.Tensor | None:
    if path == '':
        raise ValueError('source tensor path is empty')
    try:
        return model.get_parameter(path)
    except AttributeError:
        pass
    try:
        return model.get_buffer(path)
    except AttributeError:
        pass
    owner, separator, name = path.rpartition('.')
    module = model.get_submodule(owner) if separator else model
    value = getattr(module, name)
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f'{path!r} is not a tensor')
    return value


def _value(
    state: dict[str, torch.Tensor],
    value: TensorValue,
    *,
    dtype=None,
) -> torch.Tensor:
    result = state[value] if isinstance(value, str) else torch.as_tensor(value)
    return result if dtype is None else result.to(dtype)


def _apot_fields(
    value: torch.Tensor,
    minimum_exponent: int,
    maximum_exponent: int,
    allow_subtract: bool,
    *,
    include_slope_sign: bool,
    require_nonnegative: bool = False,
) -> dict[str, torch.Tensor]:
    detached = value.detach()
    if require_nonnegative and bool((detached < 0).any()):
        raise ValueError('unsigned APoT value must be non-negative')
    exponent1, exponent2, term2_sign, _ = quantize_double_power_of_two(
        detached.abs(),
        minimum_exponent,
        maximum_exponent,
        allow_subtract,
    )
    result = {
        'exponent1': exponent1,
        'exponent2': exponent2,
        'term2_sign': term2_sign,
    }
    if include_slope_sign:
        result['slope_sign'] = detached.sign().to(torch.int8)
    return result


class QuantizationRule:
    """Base class for one reusable quantization transformation."""

    def apply(
        self,
        model: nn.Module,
        calibration: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantRule(QuantizationRule):
    target: str
    value: TensorValue
    dtype: torch.dtype | None = None

    def apply(self, model, calibration, state) -> None:
        del model, calibration
        state[self.target] = _value(state, self.value, dtype=self.dtype).clone()


@dataclass(frozen=True)
class ActivationRule(QuantizationRule):
    target: str
    calibration_group: str
    bits: int
    signed: bool
    minimum_exponent: int = -24
    maximum_exponent: int = 8

    def apply(self, model, calibration, state) -> None:
        del model
        if self.calibration_group not in calibration:
            raise KeyError(
                f'calibration group {self.calibration_group!r} is absent')
        state[self.target] = power_of_two_activation_exponent(
            calibration[self.calibration_group],
            self.bits,
            self.signed,
            self.minimum_exponent,
            self.maximum_exponent,
        )


@dataclass(frozen=True)
class IntegerConvRule(QuantizationRule):
    """Quantize an ordinary Conv/Linear weight and accumulator bias."""

    target_prefix: str
    weight: str
    bias: str | None
    weight_bits: int
    bias_input_exponent: TensorValue = 0
    weight_scale: float = 1.0
    bias_scale: float = 1.0
    input_channel_exponent: str | None = None

    def apply(self, model, calibration, state) -> None:
        del calibration
        weight = _source_tensor(model, self.weight).detach()
        if self.input_channel_exponent is not None:
            exponent = _value(
                state, self.input_channel_exponent, dtype=torch.float32)
            if exponent.numel() != weight.shape[1]:
                raise ValueError(
                    f'{self.target_prefix} input exponent/channel mismatch')
            weight = weight * torch.pow(2.0, exponent).view(
                1, -1, *([1] * (weight.ndim - 2)))
        weight = weight * self.weight_scale
        integer_weight, weight_exponent = quantize_power_of_two_weight(
            weight, self.weight_bits)
        state[f'{self.target_prefix}_weight'] = integer_weight
        state[f'{self.target_prefix}_weight_exponent'] = weight_exponent
        source_bias = (
            None
            if self.bias is None else
            _source_tensor(model, self.bias)
        )
        if source_bias is not None:
            source_bias = source_bias.detach() * self.bias_scale
        accumulator_exponent = (
            weight_exponent
            + _value(
                state,
                self.bias_input_exponent,
                dtype=weight_exponent.dtype,
            )
        )
        state[f'{self.target_prefix}_bias'] = quantize_accumulator_bias(
            source_bias, accumulator_exponent)


@dataclass(frozen=True)
class ApotRule(QuantizationRule):
    target_prefix: str
    value: str
    minimum_exponent: int = -16
    maximum_exponent: int = 4
    allow_subtract: bool = True
    require_nonnegative: bool = False

    def apply(self, model, calibration, state) -> None:
        del calibration
        fields = _apot_fields(
            _source_tensor(model, self.value),
            self.minimum_exponent,
            self.maximum_exponent,
            self.allow_subtract,
            include_slope_sign=True,
            require_nonnegative=self.require_nonnegative,
        )
        for suffix, value in fields.items():
            state[f'{self.target_prefix}_{suffix}'] = value


@dataclass(frozen=True)
class BinaryAffineResidualRule(QuantizationRule):
    """Fuse sign, binary convolution scale and RPReLU affine branches."""

    target_prefix: str
    state_exponent: str
    activation_bias: str
    weight: str
    convolution_bias: str | None
    prelu_bias0: str
    prelu_slope: str
    prelu_bias1: str
    minimum_exponent: int = -16
    maximum_exponent: int = 4
    allow_subtract: bool = True

    def apply(self, model, calibration, state) -> None:
        del calibration
        exponent = _value(state, self.state_exponent)
        scale = torch.pow(2.0, exponent.to(torch.float32))
        activation_bias = _source_tensor(
            model, self.activation_bias).detach().flatten()
        state[f'{self.target_prefix}.state_exponent'] = exponent.clone()
        state[f'{self.target_prefix}.move0_threshold'] = torch.ceil(
            -activation_bias / scale).to(torch.int32)

        real_weight = _source_tensor(model, self.weight).detach()
        state[f'{self.target_prefix}.weight'] = (
            real_weight.sign().to(torch.int8))
        convolution_scale = real_weight.abs().flatten(1).mean(1)
        source_bias = (
            None
            if self.convolution_bias is None else
            _source_tensor(model, self.convolution_bias)
        )
        convolution_bias = (
            torch.zeros_like(convolution_scale)
            if source_bias is None else
            source_bias.detach().flatten()
        )
        fused = fused_binary_rprelu(
            convolution_scale,
            convolution_bias,
            _source_tensor(model, self.prelu_bias0).detach().flatten(),
            _source_tensor(model, self.prelu_slope).detach().flatten(),
            _source_tensor(model, self.prelu_bias1).detach().flatten(),
        )
        state[f'{self.target_prefix}.rprelu_threshold'] = fused['threshold']

        positive = _apot_fields(
            fused['positive_scale'],
            self.minimum_exponent,
            self.maximum_exponent,
            self.allow_subtract,
            include_slope_sign=False,
            require_nonnegative=True,
        )
        for suffix, value in positive.items():
            state[f'{self.target_prefix}.positive_{suffix}'] = value
        state[f'{self.target_prefix}.positive_bias'] = (
            quantize_accumulator_bias(
                fused['positive_bias'], exponent))

        negative = _apot_fields(
            fused['negative_scale'],
            self.minimum_exponent,
            self.maximum_exponent,
            self.allow_subtract,
            include_slope_sign=True,
        )
        for suffix, value in negative.items():
            state[f'{self.target_prefix}.negative_{suffix}'] = value
        state[f'{self.target_prefix}.negative_bias'] = (
            quantize_accumulator_bias(
                fused['negative_bias'], exponent))


@dataclass
class QuantizationPlan:
    """Ordered reusable transformations with no model-class conditionals."""

    rules: list[QuantizationRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, rule: QuantizationRule) -> None:
        self.rules.append(rule)

    def execute(
        self,
        model: nn.Module,
        calibration: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for rule in self.rules:
            rule.apply(model, calibration, state)
        return state


def apply_quantized_state(
    target: nn.Module,
    quantized: dict[str, torch.Tensor],
) -> nn.Module:
    """Strictly merge generated tensors into a target reference module."""

    target_state = target.state_dict()
    unknown = sorted(set(quantized) - set(target_state))
    if unknown:
        raise KeyError(f'quantization generated unknown targets: {unknown}')
    for name, value in quantized.items():
        expected = target_state[name]
        if value.shape != expected.shape:
            raise ValueError(
                f'{name} shape {tuple(value.shape)} != '
                f'{tuple(expected.shape)}')
        target_state[name] = value.to(expected.dtype)
    missing = sorted(set(target_state) - set(quantized))
    if missing:
        raise KeyError(f'quantization did not generate targets: {missing}')
    target.load_state_dict(target_state, strict=True)
    return target.eval()
