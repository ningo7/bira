from collections import OrderedDict

import torch


def load_model_weights(model, weights_file, map_location='cpu'):
    """Load a plain state_dict or a common checkpoint dictionary."""
    try:
        state = torch.load(
            weights_file,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(weights_file, map_location=map_location)
    if isinstance(state, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError('weights file does not contain a state_dict')
    if state and all(name.startswith('module.') for name in state):
        state = OrderedDict(
            (name[len('module.'):], value) for name, value in state.items()
        )
    model.load_state_dict(state, strict=True)


def iter_padded_tiles(image, tile_size=64, tile_pad=16):
    """Yield LR tiles with halo; small images are yielded once in full."""
    if image.ndim != 4:
        raise ValueError('image must have shape [N, C, H, W]')
    if tile_size < 1 or tile_pad < 0:
        raise ValueError('tile_size must be positive and tile_pad non-negative')
    height, width = image.shape[-2:]
    if height <= tile_size and width <= tile_size:
        yield image.clone()
        return
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        py0, py1 = max(y0 - tile_pad, 0), min(y1 + tile_pad, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            px0, px1 = max(x0 - tile_pad, 0), min(x1 + tile_pad, width)
            yield image[:, :, py0:py1, px0:px1].clone()


@torch.no_grad()
def tiled_forward(model, image, scale, tile_size=64, tile_pad=16):
    """Overlap-crop inference; tile_size is core and tile_pad is LR halo."""
    if image.ndim != 4:
        raise ValueError('image must have shape [N, C, H, W]')
    if scale < 1 or tile_size < 1 or tile_pad < 0:
        raise ValueError('scale and tile_size must be positive; tile_pad cannot be negative')
    batch, _, height, width = image.shape
    output = None
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        py0, py1 = max(y0 - tile_pad, 0), min(y1 + tile_pad, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            px0, px1 = max(x0 - tile_pad, 0), min(x1 + tile_pad, width)
            tile = image[:, :, py0:py1, px0:px1].clone()
            tile_output = model(tile)
            expected = ((py1 - py0) * scale, (px1 - px0) * scale)
            if tile_output.shape[-2:] != expected:
                raise RuntimeError(
                    f'model returned {tuple(tile_output.shape[-2:])}, '
                    f'expected {expected} for scale x{scale}'
                )
            if output is None:
                output = tile_output.new_empty(
                    batch, tile_output.shape[1], height * scale, width * scale
                )

            crop_y = (y0 - py0) * scale
            crop_x = (x0 - px0) * scale
            core_h = (y1 - y0) * scale
            core_w = (x1 - x0) * scale
            output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                tile_output[
                    :, :,
                    crop_y:crop_y + core_h,
                    crop_x:crop_x + core_w,
                ]
            )
    return output


def crop_border(tensor, border):
    if border < 0:
        raise ValueError('border must be non-negative')
    if border == 0:
        return tensor
    if min(tensor.shape[-2:]) <= 2 * border:
        raise ValueError('border is too large for the tensor')
    return tensor[..., border:-border, border:-border]
