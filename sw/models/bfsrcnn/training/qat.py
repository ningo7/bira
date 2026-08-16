import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bfsrcnn.common.patch_utils import load_model_weights
from models.bfsrcnn.network.bfsrcnn import BFSRCNN
from models.bfsrcnn.quantization.integer_model import (
    ALLOWED_WEIGHT_BITS,
    MAX_ACTIVATION_BITS,
)
from models.bfsrcnn.quantization.quantize import (
    calibration_inputs,
    calibration_paths,
    config_from_namespace,
    observation_specs,
    quantize,
)
from models.bfsrcnn.training.datasets import TrainDataset
from models.common.quantization import (
    calibrate_model,
    fake_quantize_power_of_two,
    power_of_two_activation_exponent,
    quantize_power_of_two_weight,
)


class FakeQuantBFSRCNN(nn.Module):
    def __init__(self, model, calibration, args):
        super().__init__()
        self.model = model
        self.args = args
        self.register_buffer(
            'head_output_exponent',
            power_of_two_activation_exponent(
                calibration['head'], args.head_activation_bits, True),
        )
        self.register_buffer(
            'shrink1_output_exponent',
            power_of_two_activation_exponent(
                calibration['shrink1'],
                args.shrink1_activation_bits,
                False,
            ),
        )
        self.register_buffer(
            'shrink2_output_exponent',
            power_of_two_activation_exponent(
                calibration['shrink2'],
                args.shrink2_activation_bits,
                False,
            ),
        )
        self.register_buffer(
            'state_exponent',
            power_of_two_activation_exponent(
                calibration['state'], args.state_bits, True),
        )
        self.register_buffer(
            'tail_output_exponent',
            power_of_two_activation_exponent(
                calibration['tail'], args.tail_activation_bits, True),
        )
        layers = self.full_precision_layers()
        bits = self.weight_bits()
        for name, layer in layers.items():
            _, exponent = quantize_power_of_two_weight(
                layer.weight, bits[name])
            self.register_buffer(f'{name}_weight_exponent', exponent)

    def full_precision_layers(self):
        model = self.model
        return {
            'head': model.feature_extraction[0],
            'shrink1': model.shrinking[0],
            'shrink2': model.shrinking[2],
            'shrink3': model.shrinking[4],
            'expand': model.expanding[0],
            'final': model.expanding[3],
        }

    def weight_bits(self):
        args = self.args
        return {
            'head': args.head_weight_bits,
            'shrink1': args.shrink1_weight_bits,
            'shrink2': args.shrink2_weight_bits,
            'shrink3': args.shrink3_weight_bits,
            'expand': args.expand_weight_bits,
            'final': args.final_weight_bits,
        }

    def refresh_weight_exponents(self):
        with torch.no_grad():
            for name, layer in self.full_precision_layers().items():
                _, exponent = quantize_power_of_two_weight(
                    layer.weight, self.weight_bits()[name])
                getattr(self, f'{name}_weight_exponent').copy_(exponent)

    @staticmethod
    def fake_activation(value, exponent, bits, signed):
        return fake_quantize_power_of_two(
            value,
            exponent,
            bits,
            signed=signed,
            channel_axis=1 if exponent.ndim else None,
        )

    @staticmethod
    def fake_weight(weight, exponent, bits):
        return fake_quantize_power_of_two(
            weight,
            exponent,
            bits,
            signed=True,
            channel_axis=0,
            narrow_range=True,
        )

    def quantized_conv(self, name, value):
        layer = self.full_precision_layers()[name]
        weight = self.fake_weight(
            layer.weight,
            getattr(self, f'{name}_weight_exponent'),
            self.weight_bits()[name],
        )
        return F.conv2d(
            value,
            weight,
            layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )

    def forward(self, value):
        args = self.args
        residual = F.interpolate(
            value,
            scale_factor=self.model.scale_factor,
            mode='bilinear',
            align_corners=False,
        )
        value = self.quantized_conv('head', value * 64.0)
        value = self.model.feature_extraction[1](value)
        value = self.fake_activation(
            value, self.head_output_exponent, args.head_activation_bits, True
        )

        value = F.relu(self.quantized_conv('shrink1', value))
        value = self.fake_activation(
            value, self.shrink1_output_exponent,
            args.shrink1_activation_bits, False,
        )
        value = F.relu(self.quantized_conv('shrink2', value))
        value = self.fake_activation(
            value, self.shrink2_output_exponent,
            args.shrink2_activation_bits, False,
        )
        value = self.quantized_conv('shrink3', value)
        value = self.fake_activation(value, self.state_exponent, args.state_bits, True)
        for block in self.model.mapping:
            value = block(value)
            value = self.fake_activation(value, self.state_exponent, args.state_bits, True)

        value = self.quantized_conv('expand', value)
        value = self.model.expanding[1](value)
        value = self.fake_activation(
            value, self.tail_output_exponent, args.tail_activation_bits, True
        )
        value = F.pixel_shuffle(value, self.model.scale_factor)
        value = self.quantized_conv('final', value)
        return value / 64.0 + residual


def parse_args():
    parser = argparse.ArgumentParser(
        description='Short quantization-aware fine-tuning for mixed-bit BFSRCNN'
    )
    parser.add_argument('--weights-file', required=True)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--calibration-dir', required=True)
    parser.add_argument('--calibration-tile-size', type=int, default=64)
    parser.add_argument('--calibration-tile-pad', type=int, default=16)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--distill-weight', type=float, default=0.5)
    parser.add_argument(
        '--float-distill-weight',
        type=float,
        default=0.0,
        help='Preserve the unquantized student path by distilling the teacher',
    )
    parser.add_argument('--refresh-interval', type=int, default=100)
    parser.add_argument(
        '--exact-integer-ste',
        action='store_true',
        help='Use the exact integer backend for the forward value and fake quantization for gradients',
    )
    parser.add_argument('--apot-min-exp', type=int, default=-16)
    parser.add_argument('--apot-max-exp', type=int, default=4)
    parser.add_argument('--apot-add-only', action='store_true')
    parser.add_argument('--d', type=int, default=48)
    parser.add_argument('--s', type=int, default=16)
    parser.add_argument('--mapping-blocks', type=int, default=8)
    parser.add_argument(
        '--input-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument(
        '--head-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=8)
    parser.add_argument(
        '--head-activation-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=8)
    parser.add_argument(
        '--shrink1-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=8)
    parser.add_argument(
        '--shrink1-activation-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=6)
    parser.add_argument(
        '--shrink2-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=8)
    parser.add_argument(
        '--shrink2-activation-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=6)
    parser.add_argument(
        '--shrink3-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=8)
    parser.add_argument(
        '--state-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=8)
    parser.add_argument(
        '--expand-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=8)
    parser.add_argument(
        '--tail-activation-bits', type=int,
        choices=range(2, MAX_ACTIVATION_BITS + 1), default=7)
    parser.add_argument(
        '--final-weight-bits', type=int,
        choices=ALLOWED_WEIGHT_BITS, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('steps, batch size, and learning rate must be positive')
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    teacher = BFSRCNN(args.scale, d=48, s=16, m=8).to(device).eval()
    load_model_weights(teacher, args.weights_file, device)
    student = BFSRCNN(args.scale, d=48, s=16, m=8).to(device)
    load_model_weights(student, args.weights_file, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for block in student.mapping:
        block.binary_conv.weight.requires_grad_(False)

    config = config_from_namespace(args)
    paths = calibration_paths(args.calibration_dir)
    teacher.cpu()
    calibration = calibrate_model(
        teacher,
        calibration_inputs(
            paths,
            config,
            tile_size=args.calibration_tile_size,
            tile_pad=args.calibration_tile_pad,
        ),
        observation_specs(config),
    )
    teacher.to(device)
    fake_model = FakeQuantBFSRCNN(student, calibration, args).to(device).train()
    integer_model = None
    if args.exact_integer_ste:
        integer_model = quantize(
            student.cpu().eval(), calibration, config).cpu().eval()
        student.to(device).train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.L1Loss()
    loader = DataLoader(
        TrainDataset(args.train_file),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    step = 0
    while step < args.steps:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_output = teacher(inputs).clamp(0.0, 1.0)
            fake_prediction = fake_model(inputs)
            if integer_model is None:
                prediction = fake_prediction
            else:
                with torch.no_grad():
                    integer_prediction = integer_model(inputs.cpu()).to(device)
                prediction = fake_prediction + (
                    integer_prediction - fake_prediction
                ).detach()
            loss_reconstruction = criterion(prediction, labels)
            loss_distillation = criterion(prediction, teacher_output)
            if args.float_distill_weight > 0:
                float_prediction = student(inputs).clamp(0.0, 1.0)
                loss_float_distillation = criterion(
                    float_prediction, teacher_output
                )
            else:
                loss_float_distillation = prediction.new_zeros(())
            loss = (
                loss_reconstruction
                + args.distill_weight * loss_distillation
                + args.float_distill_weight * loss_float_distillation
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if args.refresh_interval > 0 and step % args.refresh_interval == 0:
                fake_model.refresh_weight_exponents()
                if args.exact_integer_ste:
                    integer_model = quantize(
                        student.cpu().eval(), calibration, config
                    ).cpu().eval()
                    student.to(device).train()
            if step == 1 or step % 25 == 0 or step == args.steps:
                print(
                    f'step {step}/{args.steps}: loss={loss.item():.7f} '
                    f'reconstruction={loss_reconstruction.item():.7f} '
                    f'distillation={loss_distillation.item():.7f} '
                    f'float_distillation={loss_float_distillation.item():.7f}'
                )
            if step >= args.steps:
                break

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.cpu().state_dict(), output)
    print(f'QAT weights written to {output}')


if __name__ == '__main__':
    main()
