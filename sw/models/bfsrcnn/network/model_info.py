"""Print the BFSRCNN architecture and a per-layer model summary."""

import argparse
from pathlib import Path

import torch

from models.bfsrcnn.common.patch_utils import load_model_weights
from models.bfsrcnn.network.bfsrcnn import BFSRCNN


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def tensor_shape(value):
    """Return a compact description of a tensor or nested tensor output."""
    if isinstance(value, torch.Tensor):
        return 'x'.join(str(size) for size in value.shape)
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(tensor_shape(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(
            f'{key}: {tensor_shape(item)}' for key, item in value.items()
        ) + '}'
    return type(value).__name__


def human_bytes(size):
    units = ('B', 'KiB', 'MiB', 'GiB')
    size = float(size)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f'{size:.2f} {unit}'
        size /= 1024.0


def collect_layer_info(model, sample):
    """Run one inference and collect information from every leaf module."""
    rows = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            parameters = sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
            )
            trainable = sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad
            )
            rows.append({
                'name': name,
                'type': module.__class__.__name__,
                'input': tensor_shape(inputs[0] if len(inputs) == 1 else inputs),
                'output': tensor_shape(output),
                'parameters': parameters,
                'trainable': trainable,
            })

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.inference_mode():
            output = model(sample)
    finally:
        for handle in handles:
            handle.remove()

    return rows, output


def print_table(rows):
    columns = (
        ('Layer', 'name'),
        ('Type', 'type'),
        ('Input shape', 'input'),
        ('Output shape', 'output'),
        ('Parameters', 'parameters'),
    )
    display_rows = []
    for row in rows:
        display_rows.append({
            **row,
            'parameters': f"{row['parameters']:,}",
        })

    widths = {
        key: max(len(title), *(len(str(row[key])) for row in display_rows))
        for title, key in columns
    }
    header = ' | '.join(title.ljust(widths[key]) for title, key in columns)
    separator = '-+-'.join('-' * widths[key] for _, key in columns)
    print(header)
    print(separator)
    for row in display_rows:
        print(' | '.join(
            str(row[key]).ljust(widths[key]) for _, key in columns
        ))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Print BFSRCNN architecture, layer shapes and parameter counts.',
    )
    parser.add_argument('--scale', type=positive_int, default=4)
    parser.add_argument('--height', type=positive_int, default=32,
                        help='low-resolution input height (default: 32)')
    parser.add_argument('--width', type=positive_int, default=32,
                        help='low-resolution input width (default: 32)')
    parser.add_argument('--batch-size', type=positive_int, default=1)
    parser.add_argument('--d', type=positive_int, default=48,
                        help='feature channel count (default: 48)')
    parser.add_argument('--s', type=positive_int, default=16,
                        help='shrinking channel count (default: 16)')
    parser.add_argument('--mapping-blocks', type=positive_int, default=8)
    parser.add_argument('--weights-file', type=Path,
                        help='optional state_dict/checkpoint to load')
    parser.add_argument('--device', default='cpu',
                        help='PyTorch device used for the sample forward pass')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = BFSRCNN(
        scale_factor=args.scale,
        d=args.d,
        s=args.s,
        m=args.mapping_blocks,
    ).to(device)

    if args.weights_file is not None:
        load_model_weights(model, args.weights_file, map_location=device)

    sample = torch.zeros(
        args.batch_size, 1, args.height, args.width, device=device
    )
    rows, output = collect_layer_info(model, sample)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    buffer_elements = sum(buffer.numel() for buffer in model.buffers())
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in (*model.parameters(), *model.buffers())
    )

    print('BFSRCNN architecture')
    print('====================')
    print(model)
    print('\nLayer summary')
    print('=============')
    print_table(rows)
    print('\nTotals')
    print('======')
    print(f'Input shape:          {tensor_shape(sample)}')
    print(f'Output shape:         {tensor_shape(output)}')
    print(f'Total parameters:     {total_parameters:,}')
    print(f'Trainable parameters: {trainable_parameters:,}')
    print(f'Non-trainable params: {total_parameters - trainable_parameters:,}')
    print(f'Buffer elements:      {buffer_elements:,}')
    print(f'Parameter/buffer size:{human_bytes(model_bytes):>12}')
    if args.weights_file is not None:
        print(f'Loaded weights:       {args.weights_file}')


if __name__ == '__main__':
    main()
