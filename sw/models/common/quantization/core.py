"""Reusable quantization primitives for binary/integer PyTorch networks.

Nothing in this module knows a model topology. A model-specific quantizer
uses these functions during calibration and then exports its logical integer
tensors to a model-owned native-data exporter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SymmetricQuantSpec:
    """Signed symmetric integer quantization settings.

    ``bits`` is the stored arithmetic precision. ``channel_axis=None`` selects
    one scale for the complete tensor. Giving an axis selects one scale per
    channel along that dimension; convolution weights normally use axis 0
    (one scale per output channel).
    """

    bits: int
    channel_axis: int | None = None
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.bits < 2 or self.bits > 16:
            raise ValueError('symmetric quantization supports 2..16 bits')
        if self.epsilon <= 0.0:
            raise ValueError('epsilon must be positive')

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1


@dataclass(frozen=True)
class PowerOfTwoQuantSpec:
    """Shift-exact quantization settings understood by BIRA.

    Unlike arbitrary symmetric scales, every scale is exactly ``2**exponent``
    and can therefore be propagated into accumulator and post-process shifts.
    """

    bits: int
    signed: bool = True
    channel_axis: int | None = None
    minimum_exponent: int = -24
    maximum_exponent: int = 8

    def __post_init__(self) -> None:
        if self.bits < 2 or self.bits > 16:
            raise ValueError('power-of-two quantization supports 2..16 bits')
        if self.minimum_exponent > self.maximum_exponent:
            raise ValueError('minimum_exponent exceeds maximum_exponent')

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (
            (1 << (self.bits - 1)) - 1
            if self.signed else
            (1 << self.bits) - 1
        )


def _scale_shape(tensor: torch.Tensor, axis: int) -> list[int]:
    normalized = axis if axis >= 0 else tensor.ndim + axis
    if normalized < 0 or normalized >= tensor.ndim:
        raise ValueError(f'channel axis {axis} is out of range')
    shape = [1] * tensor.ndim
    shape[normalized] = tensor.shape[normalized]
    return shape


def symmetric_scale(
    tensor: torch.Tensor,
    spec: SymmetricQuantSpec,
) -> torch.Tensor:
    """Calculate a tensor-wide or per-channel symmetric scale."""

    if not tensor.is_floating_point():
        raise TypeError('symmetric_scale expects a floating-point tensor')
    magnitude = tensor.detach().abs()
    if spec.channel_axis is None:
        maximum = magnitude.amax()
    else:
        axis = (
            spec.channel_axis
            if spec.channel_axis >= 0
            else tensor.ndim + spec.channel_axis
        )
        reduction = tuple(index for index in range(tensor.ndim) if index != axis)
        maximum = magnitude.amax(dim=reduction).reshape(
            _scale_shape(tensor, spec.channel_axis)
        )
    denominator = float(spec.qmax)
    return torch.clamp(maximum / denominator, min=spec.epsilon)


def quantize_symmetric(
    tensor: torch.Tensor,
    spec: SymmetricQuantSpec,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed integer tensor and the scale used to produce it."""

    if scale is None:
        scale = symmetric_scale(tensor, spec)
    quantized = torch.round(tensor / scale).clamp(spec.qmin, spec.qmax)
    dtype = torch.int16 if spec.bits > 8 else torch.int8
    return quantized.to(dtype), scale.detach()


def fake_quantize_symmetric(
    tensor: torch.Tensor,
    spec: SymmetricQuantSpec,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize/dequantize with a straight-through gradient estimator."""

    integers, used_scale = quantize_symmetric(tensor, spec, scale)
    dequantized = integers.to(tensor.dtype) * used_scale
    return tensor + (dequantized - tensor).detach(), used_scale


def binary_sign(tensor: torch.Tensor) -> torch.Tensor:
    """Map nonnegative values to +1 and negative values to -1."""

    return torch.where(
        tensor >= 0,
        torch.ones_like(tensor),
        -torch.ones_like(tensor),
    )


def fake_binary_sign(tensor: torch.Tensor) -> torch.Tensor:
    """Binary sign in the forward pass and clipped STE in the backward pass."""

    clipped = tensor.clamp(-1.0, 1.0)
    return clipped + (binary_sign(tensor) - clipped).detach()


def power_of_two_activation_exponent(
    values: torch.Tensor,
    bits: int,
    signed: bool,
    minimum_exponent: int = -24,
    maximum_exponent: int = 8,
) -> torch.Tensor:
    """Choose MSE-optimal shift scales for sampled activation magnitudes.

    A one-dimensional input produces one tensor-wide exponent. A two-
    dimensional ``[channels, samples]`` input produces one exponent/channel.
    The operation is independent of model topology and module names.
    """

    spec = PowerOfTwoQuantSpec(
        bits,
        signed,
        minimum_exponent=minimum_exponent,
        maximum_exponent=maximum_exponent,
    )
    samples = torch.as_tensor(values, dtype=torch.float32).abs()
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    elif samples.ndim != 2:
        raise ValueError('activation samples must be one- or two-dimensional')
    exponents: list[int] = []
    for channel_values in samples:
        if not bool((channel_values > 0).any()):
            exponents.append(0)
            continue
        best_error = None
        best_exponent = None
        for exponent in range(
            spec.minimum_exponent, spec.maximum_exponent + 1
        ):
            scale = 2.0**exponent
            reconstructed = (
                torch.round(channel_values / scale)
                .clamp(0, spec.qmax)
                * scale
            )
            error = torch.mean(
                (channel_values - reconstructed) ** 2
            ).item()
            if best_error is None or error < best_error:
                best_error = error
                best_exponent = exponent
        exponents.append(int(best_exponent))
    result = torch.tensor(exponents, dtype=torch.int16)
    return result[0] if result.numel() == 1 else result


def quantize_power_of_two_weight(
    weight: torch.Tensor,
    bits: int,
    minimum_exponent: int = -24,
    maximum_exponent: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Conv/Linear weights per output channel with shift scales."""

    spec = PowerOfTwoQuantSpec(
        bits,
        True,
        channel_axis=0,
        minimum_exponent=minimum_exponent,
        maximum_exponent=maximum_exponent,
    )
    if weight.ndim < 2 or not weight.is_floating_point():
        raise TypeError('weights must be a floating tensor with output axis 0')
    outputs = []
    exponents = []
    for output_channel in range(weight.shape[0]):
        values = weight[output_channel].detach()
        best_error = None
        best_quantized = None
        best_exponent = None
        for exponent in range(
            spec.minimum_exponent, spec.maximum_exponent + 1
        ):
            scale = 2.0**exponent
            # Use the symmetric narrow range [-qmax, qmax]. This avoids one
            # unmatched negative code in per-channel weight scales and keeps
            # the arithmetic symmetric around zero.
            quantized = torch.round(values / scale).clamp(
                -spec.qmax, spec.qmax)
            error = torch.mean(
                (values - quantized * scale) ** 2
            ).item()
            if best_error is None or error < best_error:
                best_error = error
                best_quantized = quantized
                best_exponent = exponent
        outputs.append(best_quantized.unsqueeze(0))
        exponents.append(int(best_exponent))
    storage = torch.int16 if bits > 8 else torch.int8
    return (
        torch.cat(outputs, dim=0).to(storage),
        torch.tensor(exponents, dtype=torch.int16),
    )


def quantize_accumulator_bias(
    bias: torch.Tensor | None,
    accumulator_exponent: torch.Tensor,
) -> torch.Tensor:
    """Convert floating bias into each output channel's accumulator domain."""

    if bias is None:
        return torch.zeros_like(accumulator_exponent, dtype=torch.int32)
    scale = torch.pow(2.0, accumulator_exponent.to(torch.float32))
    return torch.round(bias.detach() / scale).to(torch.int32)


def fake_quantize_power_of_two(
    tensor: torch.Tensor,
    exponent: torch.Tensor,
    bits: int,
    *,
    signed: bool,
    channel_axis: int | None = None,
    narrow_range: bool = False,
) -> torch.Tensor:
    """Fake-quantize with a supplied shift exponent and STE."""

    spec = PowerOfTwoQuantSpec(bits, signed, channel_axis)
    scale = torch.pow(2.0, exponent.to(tensor.dtype))
    if channel_axis is not None:
        scale = scale.reshape(_scale_shape(tensor, channel_axis))
    qmin = -spec.qmax if signed and narrow_range else spec.qmin
    quantized = torch.round(tensor / scale).clamp(qmin, spec.qmax)
    dequantized = quantized * scale
    return tensor + (dequantized - tensor).detach()


def quantize_double_power_of_two(
    scale: torch.Tensor,
    minimum_exponent: int = -12,
    maximum_exponent: int = 1,
    allow_subtract: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate positive values by ``2**e1 +/- 2**e2``."""

    if minimum_exponent >= maximum_exponent:
        raise ValueError('minimum exponent must be smaller than maximum')
    values = []
    exponent1 = []
    exponent2 = []
    signs = []
    sign_options = (-1, 1) if allow_subtract else (1,)
    for e1 in range(minimum_exponent + 1, maximum_exponent + 1):
        for e2 in range(minimum_exponent, e1):
            for sign2 in sign_options:
                value = 2.0**e1 + sign2 * 2.0**e2
                if value > 0:
                    values.append(value)
                    exponent1.append(e1)
                    exponent2.append(e2)
                    signs.append(sign2)
    candidates = scale.new_tensor(values)
    flat = scale.detach().flatten()
    nearest = (flat[:, None] - candidates[None, :]).abs().argmin(dim=1)
    e1 = torch.tensor(
        exponent1, device=scale.device, dtype=torch.int8)[nearest]
    e2 = torch.tensor(
        exponent2, device=scale.device, dtype=torch.int8)[nearest]
    sign2 = torch.tensor(
        signs, device=scale.device, dtype=torch.int8)[nearest]
    quantized = candidates[nearest]
    return (
        e1.reshape(scale.shape),
        e2.reshape(scale.shape),
        sign2.reshape(scale.shape),
        quantized.reshape(scale.shape),
    )


def fused_binary_rprelu(
    convolution_scale: torch.Tensor,
    convolution_bias: torch.Tensor,
    bias0: torch.Tensor,
    slope: torch.Tensor,
    bias1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fold binary-convolution scale and RPReLU into two affine branches."""

    convolution_scale = convolution_scale.detach()
    if not bool((convolution_scale > 0).all()):
        raise ValueError('binary convolution scale must be strictly positive')
    effective_bias0 = (
        bias0.detach() + convolution_bias.detach() * convolution_scale)
    slope = slope.detach()
    bias1 = bias1.detach()
    return {
        'threshold': torch.ceil(
            -effective_bias0 / convolution_scale).to(torch.int32),
        'positive_scale': convolution_scale,
        'positive_bias': effective_bias0 + bias1,
        'negative_scale': convolution_scale * slope,
        'negative_bias': slope * effective_bias0 + bias1,
    }


class MinMaxObserver:
    """Small model-independent PTQ activation observer.

    Call ``observe`` on representative tensors. ``scale`` then returns the
    symmetric scale for the accumulated range. Per-channel observation is
    supported as long as the channel count does not change.
    """

    def __init__(self, spec: SymmetricQuantSpec):
        self.spec = spec
        self._maximum: torch.Tensor | None = None

    def observe(self, tensor: torch.Tensor) -> None:
        candidate = symmetric_scale(tensor, self.spec) * self.spec.qmax
        candidate = candidate.detach().cpu()
        if self._maximum is None:
            self._maximum = candidate
        else:
            if self._maximum.shape != candidate.shape:
                raise ValueError('observed per-channel shape changed')
            self._maximum = torch.maximum(self._maximum, candidate)

    @property
    def calibrated(self) -> bool:
        return self._maximum is not None

    def scale(self, device=None) -> torch.Tensor:
        if self._maximum is None:
            raise RuntimeError('observer has not seen calibration data')
        result = torch.clamp(
            self._maximum / float(self.spec.qmax),
            min=self.spec.epsilon,
        )
        return result.to(device=device) if device is not None else result
