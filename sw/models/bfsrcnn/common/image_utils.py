from pathlib import Path

from PIL import Image


PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


def image_paths(image_file=None, image_dir=None, recursive=False):
    if image_file:
        return [Path(image_file)]
    if not image_dir:
        raise ValueError('either image_file or image_dir is required')
    root = Path(image_dir)
    paths = []
    for pattern in PATTERNS:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    if not paths:
        raise ValueError(f'no images found in {image_dir}')
    return sorted(set(paths))
