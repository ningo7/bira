"""Small model-independent helpers for rendering generated C data."""

import re

import torch


DTYPE_TO_C = {
    torch.int8: ('int8_t', 1),
    torch.uint8: ('uint8_t', 1),
    torch.int16: ('int16_t', 2),
    torch.uint16: ('uint16_t', 2),
    torch.int32: ('int32_t', 4),
    torch.int64: ('int64_t', 8),
}


def c_identifier(value, uppercase=False):
    identifier = re.sub(r'[^0-9A-Za-z_]', '_', str(value))
    if not identifier or identifier[0].isdigit():
        identifier = f'_{identifier}'
    return identifier.upper() if uppercase else identifier.lower()


def tensor_values(tensor, values_per_line):
    values = tensor.detach().cpu().contiguous().view(-1).tolist()
    lines = []
    for start in range(0, len(values), values_per_line):
        chunk = values[start:start + values_per_line]
        lines.append(
            '    ' + ', '.join(str(int(value)) for value in chunk)
        )
    return ',\n'.join(lines)
