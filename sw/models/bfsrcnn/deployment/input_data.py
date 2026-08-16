"""Create reproducible integer BFSRCNN inputs from ordinary images."""

from pathlib import Path

import numpy as np
from PIL import Image


BICUBIC = (
    Image.Resampling.BICUBIC
    if hasattr(Image, "Resampling")
    else Image.BICUBIC
)


def make_integer_input_patch(
    image_path,
    *,
    scale: int = 4,
    tile_size: int = 24,
    tile_pad: int = 4,
    tile_row: int = 1,
    tile_column: int = 1,
):
    """Return one uint8 Y-channel patch and its source metadata."""

    if scale < 1 or tile_size < 1 or tile_pad < 0:
        raise ValueError(
            "scale and tile size must be positive; tile pad cannot be negative"
        )
    if tile_row < 0 or tile_column < 0:
        raise ValueError("tile row and column cannot be negative")

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    width = image.width // scale * scale
    height = image.height // scale * scale
    if min(width, height) == 0:
        raise ValueError(f"{image_path} is smaller than scale x{scale}")

    high_resolution = image.crop((0, 0, width, height))
    low_resolution = high_resolution.resize(
        (width // scale, height // scale), BICUBIC
    )
    rgb = np.asarray(low_resolution, dtype=np.float32)
    y = (
        16.0
        + (
            64.738 * rgb[..., 0]
            + 129.057 * rgb[..., 1]
            + 25.064 * rgb[..., 2]
        )
        / 256.0
    )
    input_integer = np.rint(y).clip(0, 255).astype(np.uint8)

    lr_height, lr_width = input_integer.shape
    core_y = tile_row * tile_size
    core_x = tile_column * tile_size
    if core_y >= lr_height or core_x >= lr_width:
        raise ValueError(
            f"tile ({tile_row}, {tile_column}) starts outside "
            f"low-resolution shape {(lr_height, lr_width)}"
        )

    core_end_y = min(core_y + tile_size, lr_height)
    core_end_x = min(core_x + tile_size, lr_width)
    tile_y = max(core_y - tile_pad, 0)
    tile_x = max(core_x - tile_pad, 0)
    tile_end_y = min(core_end_y + tile_pad, lr_height)
    tile_end_x = min(core_end_x + tile_pad, lr_width)
    patch = np.ascontiguousarray(
        input_integer[tile_y:tile_end_y, tile_x:tile_end_x]
    )
    return patch, {
        "source_image": str(image_path),
        "scale": scale,
        "lr_height": int(lr_height),
        "lr_width": int(lr_width),
        "tile_size": tile_size,
        "tile_pad": tile_pad,
        "tile_row": tile_row,
        "tile_column": tile_column,
        "core_y": int(core_y),
        "core_x": int(core_x),
        "tile_y": int(tile_y),
        "tile_x": int(tile_x),
        "input_height": int(patch.shape[0]),
        "input_width": int(patch.shape[1]),
    }
