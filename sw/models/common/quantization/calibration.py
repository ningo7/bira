"""Model-independent activation collection driven by module-path rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .handlers import sample_tensor


@dataclass(frozen=True)
class ObservationSpec:
    """Collect one module output into a named calibration group.

    Multiple modules may write the same ``group``. This is useful for repeated
    residual blocks that share one activation scale.
    """

    module: str
    group: str
    maximum_per_call: int = 16384
    per_channel: bool = False


class RuleCalibrator:
    """Attach hooks described by :class:`ObservationSpec`."""

    def __init__(self, model: nn.Module, specs: Iterable[ObservationSpec]):
        self.model = model
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError('at least one observation rule is required')
        modules = dict(model.named_modules())
        self.samples: dict[str, list[torch.Tensor]] = {}
        self.per_channel: dict[str, bool] = {}
        self._handles = []
        for spec in self.specs:
            if spec.module not in modules:
                raise KeyError(f'module {spec.module!r} does not exist')
            previous = self.per_channel.setdefault(
                spec.group, spec.per_channel)
            if previous != spec.per_channel:
                raise ValueError(
                    f'group {spec.group!r} mixes per-channel modes')
            self.samples.setdefault(spec.group, [])
            self._handles.append(
                modules[spec.module].register_forward_hook(
                    self._make_hook(spec)
                )
            )

    def _make_hook(self, spec: ObservationSpec):
        def observe(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f'module {spec.module!r} output is not a tensor')
            self.samples[spec.group].append(sample_tensor(
                output,
                spec.maximum_per_call,
                per_channel=spec.per_channel,
            ))
        return observe

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self) -> dict[str, torch.Tensor]:
        result = {}
        for group, samples in self.samples.items():
            if not samples:
                raise RuntimeError(
                    f'calibration group {group!r} has no samples')
            result[group] = torch.cat(
                samples,
                dim=1 if self.per_channel[group] else 0,
            )
        return result

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


@torch.no_grad()
def calibrate_model(
    model: nn.Module,
    inputs: Iterable,
    specs: Iterable[ObservationSpec],
) -> dict[str, torch.Tensor]:
    """Run arbitrary model inputs and return grouped activation samples.

    An item may be one tensor, a positional-argument tuple, or a dictionary of
    keyword arguments. Dataset decoding and preprocessing stay outside this
    function.
    """

    with RuleCalibrator(model, specs) as calibrator:
        count = 0
        for item in inputs:
            if isinstance(item, dict):
                model(**item)
            elif isinstance(item, tuple):
                model(*item)
            else:
                model(item)
            count += 1
        if count == 0:
            raise ValueError('calibration input iterable is empty')
        return calibrator.result()
