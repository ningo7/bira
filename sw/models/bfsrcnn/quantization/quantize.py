"""BFSRCNN-owned quantization flow built from reusable quantization math.

The model owns module paths, calibration preprocessing, precision policy and
checkpoint format. Only low-level quantization mathematics is shared.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from models.common.quantization import (
    ActivationRule,
    ApotRule,
    BinaryAffineResidualRule,
    ConstantRule,
    IntegerConvRule,
    ObservationSpec,
    QuantizationPlan,
    apply_quantized_state,
    calibrate_model,
)
from models.bfsrcnn.common.patch_utils import iter_padded_tiles, load_model_weights
from models.bfsrcnn.common.utils import preprocess
from models.bfsrcnn.network.bfsrcnn import BFSRCNN
from models.bfsrcnn.quantization.integer_model import (
    ALLOWED_WEIGHT_BITS,
    MAX_ACTIVATION_BITS,
)
from models.bfsrcnn.quantization.checkpoint import (
    PixelIntegerBFSRCNN,
    save_pixel_integer_checkpoint,
)


PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
BICUBIC = (
    Image.Resampling.BICUBIC
    if hasattr(Image, 'Resampling') else
    Image.BICUBIC
)


@dataclass(frozen=True)
class BfsrcnnQuantConfig:
    """Only architecture and policy choices; no quantization implementation."""

    scale: int = 4
    d: int = 48
    s: int = 16
    mapping_blocks: int = 8
    input_bits: int = 8
    head_weight_bits: int = 16
    head_activation_bits: int = 8
    shrink1_weight_bits: int = 4
    shrink1_activation_bits: int = 8
    shrink2_weight_bits: int = 16
    shrink2_activation_bits: int = 8
    shrink3_weight_bits: int = 8
    state_bits: int = 8
    expand_weight_bits: int = 8
    tail_activation_bits: int = 8
    final_weight_bits: int = 16
    apot_min_exp: int = -16
    apot_max_exp: int = 4
    apot_add_only: bool = False


def calibration_paths(
    directory,
    *,
    recursive: bool = False,
    maximum: int = 0,
) -> list[Path]:
    """Select model calibration files; image decoding remains model-side."""

    root = Path(directory)
    paths = []
    for pattern in PATTERNS:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    paths = sorted(set(paths))
    if maximum > 0:
        paths = paths[:maximum]
    if not paths:
        raise ValueError(f'no calibration images found in {directory}')
    return paths


def _low_resolution_image(path: Path, scale: int) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    width = image.width // scale * scale
    height = image.height // scale * scale
    if min(width, height) == 0:
        raise ValueError(f'{path} is smaller than scale x{scale}')
    high_resolution = image.crop((0, 0, width, height))
    low_resolution = high_resolution.resize(
        (width // scale, height // scale), BICUBIC)
    return preprocess(low_resolution, 'cpu')[0]


def calibration_inputs(
    paths,
    config: BfsrcnnQuantConfig,
    *,
    tile_size: int = 64,
    tile_pad: int = 16,
):
    """Yield ordinary BFSRCNN forward inputs, one padded tile at a time."""

    for path in paths:
        low_resolution = _low_resolution_image(Path(path), config.scale)
        yield from iter_padded_tiles(low_resolution, tile_size, tile_pad)


def observation_specs(config: BfsrcnnQuantConfig) -> list[ObservationSpec]:
    """Declare which ordinary module outputs share an activation scale."""

    specs = [
        ObservationSpec('feature_extraction', 'head', 16384),
        ObservationSpec('shrinking.1', 'shrink1', 16384),
        ObservationSpec('shrinking.3', 'shrink2', 16384),
        ObservationSpec('shrinking.4', 'state', 2048, True),
    ]
    specs.extend(
        ObservationSpec(f'mapping.{index}', 'state', 2048, True)
        for index in range(config.mapping_blocks)
    )
    specs.append(ObservationSpec('expanding.1', 'tail', 16384))
    return specs


def build_plan(config: BfsrcnnQuantConfig) -> QuantizationPlan:
    """Describe BFSRCNN with reusable quantization rules."""

    allow_subtract = not config.apot_add_only
    plan = QuantizationPlan(metadata={
        'adapter': 'bfsrcnn-pixel-v1',
        'policy': asdict(config),
    })
    plan.rules.extend([
        ConstantRule('input_exponent', 0, torch.int16),
        ActivationRule(
            'head_output_exponent', 'head',
            config.head_activation_bits, True),
        ActivationRule(
            'shrink1_output_exponent', 'shrink1',
            config.shrink1_activation_bits, False),
        ActivationRule(
            'shrink2_output_exponent', 'shrink2',
            config.shrink2_activation_bits, False),
        ActivationRule(
            'state_exponent', 'state', config.state_bits, True),
        ActivationRule(
            'tail_output_exponent', 'tail',
            config.tail_activation_bits, True),
        IntegerConvRule(
            'head',
            'feature_extraction.0.weight',
            'feature_extraction.0.bias',
            config.head_weight_bits,
            bias_input_exponent=0,
            weight_scale=64.0 / 255.0,
        ),
        ApotRule(
            'head_prelu',
            'feature_extraction.1.weight',
            config.apot_min_exp,
            config.apot_max_exp,
            allow_subtract,
        ),
        IntegerConvRule(
            'shrink1',
            'shrinking.0.weight',
            'shrinking.0.bias',
            config.shrink1_weight_bits,
            bias_input_exponent='head_output_exponent',
        ),
        IntegerConvRule(
            'shrink2',
            'shrinking.2.weight',
            'shrinking.2.bias',
            config.shrink2_weight_bits,
            bias_input_exponent='shrink1_output_exponent',
        ),
        IntegerConvRule(
            'shrink3',
            'shrinking.4.weight',
            'shrinking.4.bias',
            config.shrink3_weight_bits,
            bias_input_exponent='shrink2_output_exponent',
        ),
    ])
    for index in range(config.mapping_blocks):
        source = f'mapping.{index}'
        plan.append(BinaryAffineResidualRule(
            target_prefix=source,
            state_exponent='state_exponent',
            activation_bias=f'{source}.move0.bias',
            weight=f'{source}.binary_conv.weight',
            convolution_bias=None,
            prelu_bias0=f'{source}.relu.pr_bias0.bias',
            prelu_slope=f'{source}.relu.pr_prelu.weight',
            prelu_bias1=f'{source}.relu.pr_bias1.bias',
            minimum_exponent=config.apot_min_exp,
            maximum_exponent=config.apot_max_exp,
            allow_subtract=allow_subtract,
        ))
    plan.rules.extend([
        IntegerConvRule(
            'expand',
            'expanding.0.weight',
            'expanding.0.bias',
            config.expand_weight_bits,
            bias_input_exponent=0,
            input_channel_exponent='state_exponent',
        ),
        ApotRule(
            'tail_prelu',
            'expanding.1.weight',
            config.apot_min_exp,
            config.apot_max_exp,
            allow_subtract,
        ),
        IntegerConvRule(
            'final',
            'expanding.3.weight',
            'expanding.3.bias',
            config.final_weight_bits,
            bias_input_exponent='tail_output_exponent',
            weight_scale=255.0 / 64.0,
            bias_scale=255.0 / 64.0,
        ),
    ])
    return plan


def create_source(config: BfsrcnnQuantConfig, weights_file) -> BFSRCNN:
    model = BFSRCNN(
        config.scale,
        d=config.d,
        s=config.s,
        m=config.mapping_blocks,
    ).cpu().eval()
    load_model_weights(model, weights_file, 'cpu')
    return model


def create_target(
    source: BFSRCNN,
    config: BfsrcnnQuantConfig,
) -> PixelIntegerBFSRCNN:
    return PixelIntegerBFSRCNN(
        scale_factor=config.scale,
        d=config.d,
        s=config.s,
        m=config.mapping_blocks,
        shrink_channels=source.shrinking[0].out_channels,
        tail_channels=source.expanding[3].in_channels,
        input_bits=config.input_bits,
        head_weight_bits=config.head_weight_bits,
        head_activation_bits=config.head_activation_bits,
        shrink1_weight_bits=config.shrink1_weight_bits,
        shrink1_activation_bits=config.shrink1_activation_bits,
        shrink2_weight_bits=config.shrink2_weight_bits,
        shrink2_activation_bits=config.shrink2_activation_bits,
        shrink3_weight_bits=config.shrink3_weight_bits,
        state_bits=config.state_bits,
        expand_weight_bits=config.expand_weight_bits,
        tail_activation_bits=config.tail_activation_bits,
        final_weight_bits=config.final_weight_bits,
    )


def quantize(
    source: BFSRCNN,
    calibration: dict[str, torch.Tensor],
    config: BfsrcnnQuantConfig,
) -> PixelIntegerBFSRCNN:
    quantized = build_plan(config).execute(source, calibration)
    return apply_quantized_state(
        create_target(source, config), quantized)


def config_from_namespace(namespace) -> BfsrcnnQuantConfig:
    """Build policy from any CLI/training namespace with matching names."""

    config_fields = set(BfsrcnnQuantConfig.__dataclass_fields__)
    return BfsrcnnQuantConfig(**{
        name: value
        for name, value in vars(namespace).items()
        if name in config_fields
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description='quantize BFSRCNN through the generic BIRA engine')
    parser.add_argument('--weights-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--calibration-dir', required=True)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--max-calibration-images', type=int, default=0)
    parser.add_argument('--calibration-tile-size', type=int, default=64)
    parser.add_argument('--calibration-tile-pad', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--d', type=int, default=48)
    parser.add_argument('--s', type=int, default=16)
    parser.add_argument('--mapping-blocks', type=int, default=8)
    parser.add_argument(
        '--input-bits', type=int, choices=range(2, MAX_ACTIVATION_BITS + 1),
        default=8)
    for name, default in (
        ('head-weight-bits', 16),
        ('shrink1-weight-bits', 4),
        ('shrink2-weight-bits', 16),
        ('shrink3-weight-bits', 8),
        ('expand-weight-bits', 8),
        ('final-weight-bits', 16),
    ):
        parser.add_argument(
            f'--{name}', type=int, choices=ALLOWED_WEIGHT_BITS,
            default=default)
    for name in (
        'head-activation-bits',
        'shrink1-activation-bits',
        'shrink2-activation-bits',
        'state-bits',
        'tail-activation-bits',
    ):
        parser.add_argument(
            f'--{name}', type=int,
            choices=range(2, MAX_ACTIVATION_BITS + 1), default=8)
    parser.add_argument('--apot-min-exp', type=int, default=-16)
    parser.add_argument('--apot-max-exp', type=int, default=4)
    parser.add_argument('--apot-add-only', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_namespace(args)
    paths = calibration_paths(
        args.calibration_dir,
        recursive=args.recursive,
        maximum=args.max_calibration_images,
    )
    source = create_source(config, args.weights_file)
    samples = calibrate_model(
        source,
        calibration_inputs(
            paths,
            config,
            tile_size=args.calibration_tile_size,
            tile_pad=args.calibration_tile_pad,
        ),
        observation_specs(config),
    )
    target = quantize(source, samples, config)
    metadata = {
        'source_weights': args.weights_file,
        'calibration_dir': args.calibration_dir,
        'calibration_images': [str(path) for path in paths],
        'activation_calibration': 'generic_hook_sampled_mse_power_of_two',
        'calibration_tile_size': args.calibration_tile_size,
        'calibration_tile_pad': args.calibration_tile_pad,
        'quantization_engine': 'models.common.quantization',
        'model_quantizer': 'models.bfsrcnn.quantization',
        'policy': asdict(config),
        'boundary_folding': {
            'head_weight': '64/255 folded offline',
            'final_weight_and_bias': '255/64 folded offline',
        },
    }
    save_pixel_integer_checkpoint(target, args.output_file, metadata)
    print(f'quantized checkpoint written to {args.output_file}')


if __name__ == '__main__':
    main()
