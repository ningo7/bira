import argparse
import glob
import os
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
import torch.backends.cudnn as cudnn

from models.bfsrcnn.network.bfsrcnn import BFSRCNN
from models.bfsrcnn.common.patch_utils import crop_border, load_model_weights, tiled_forward
from models.bfsrcnn.common.utils import calc_psnr, convert_ycbcr_to_rgb, preprocess


PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


def image_paths(image_file, image_dir):
    if image_file:
        return [image_file]
    paths = []
    for pattern in PATTERNS:
        paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    if not paths:
        raise ValueError(f'no images found in {image_dir}')
    return sorted(set(paths))


@torch.no_grad()
def process(path, model, device, args):
    image = Image.open(path).convert('RGB')
    width = image.width // args.scale * args.scale
    height = image.height // args.scale * args.scale
    if min(width, height) == 0:
        raise ValueError(f'{path} is smaller than scale x{args.scale}')

    # Mod-crop preserves the reference pixels instead of resizing the HR.
    hr_image = image.crop((0, 0, width, height))
    lr_image = hr_image.resize((width // args.scale, height // args.scale), BICUBIC)
    bicubic = lr_image.resize((width, height), BICUBIC)
    lr, _ = preprocess(lr_image, device)
    hr, _ = preprocess(hr_image, device)
    _, ycbcr = preprocess(bicubic, device)

    prediction = tiled_forward(
        model, lr, args.scale, args.tile_size, args.tile_pad
    ).clamp(0.0, 1.0)
    metric_hr = crop_border(hr, args.crop_border)
    tiled_psnr = calc_psnr(
        metric_hr, crop_border(prediction, args.crop_border)
    ).item()

    full_psnr = None
    max_error = None
    if (
        args.verify_full
        and lr.shape[-2] <= args.tile_size
        and lr.shape[-1] <= args.tile_size
    ):
        full = model(lr.clone()).clamp(0.0, 1.0)
        full_psnr = calc_psnr(
            metric_hr, crop_border(full, args.crop_border)
        ).item()
        max_error = (full - prediction).abs().max().item()

    y = prediction.mul(255.0).cpu().numpy()[0, 0]
    ycbcr = np.stack((y, ycbcr[..., 1], ycbcr[..., 2]), axis=-1)
    rgb = np.clip(convert_ycbcr_to_rgb(ycbcr), 0, 255).astype(np.uint8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(path)
    output_path = (
        output_dir
        / f'{source.stem}_bfsrcnn_tiled_x{args.scale}{source.suffix}'
    )
    Image.fromarray(rgb).save(output_path)
    if args.save_bicubic:
        bicubic.save(
            output_dir / f'{source.stem}_bicubic_x{args.scale}{source.suffix}'
        )
    return tiled_psnr, full_psnr, max_error, output_path


def parse_args():
    parser = argparse.ArgumentParser(description='BFSRCNN overlap-crop tiled test')
    parser.add_argument('--weights-file', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image-file')
    source.add_argument('--image-dir')
    parser.add_argument('--output-dir', default='models/bfsrcnn/outputs/tiled')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument(
        '--tile-size', type=int, default=64,
        help='non-overlapping LR core; small images use one tile',
    )
    parser.add_argument(
        '--tile-pad', type=int, default=16, help='LR halo on every side'
    )
    parser.add_argument(
        '--crop-border', type=int, default=0,
        help='HR border ignored only when calculating PSNR',
    )
    parser.add_argument('--verify-full', action='store_true')
    parser.add_argument('--save-bicubic', action='store_true')
    parser.add_argument('--d', type=int, default=48)
    parser.add_argument('--s', type=int, default=16)
    parser.add_argument('--mapping-blocks', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scale < 1 or args.tile_size < 1 or args.tile_pad < 0:
        raise ValueError(
            'scale and tile-size must be positive; tile-pad cannot be negative'
        )
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BFSRCNN(
        args.scale, d=args.d, s=args.s, m=args.mapping_blocks
    ).to(device)
    load_model_weights(model, args.weights_file, device)
    model.eval()

    scores = []
    full_scores = []
    errors = []
    for path in image_paths(args.image_file, args.image_dir):
        tiled_psnr, full_psnr, max_error, output_path = process(
            path, model, device, args
        )
        scores.append(tiled_psnr)
        if args.verify_full and full_psnr is not None:
            full_scores.append(full_psnr)
            errors.append(max_error)
            print(f'{path}:')
            print(f'  tiled PSNR: {tiled_psnr:.6f} dB')
            print(f'  full PSNR:  {full_psnr:.6f} dB')
            print(f'  tiled/full max abs error: {max_error:.8g}')
            print(f'  output: {output_path}')
        elif args.verify_full:
            print(f'{path}: PSNR {tiled_psnr:.2f} dB')
            print('  full-image verification skipped for large LR image')
            print(f'  output: {output_path}')
        else:
            print(f'{path}: PSNR {tiled_psnr:.2f} dB -> {output_path}')
    if len(scores) > 1:
        print(f'average tiled PSNR: {sum(scores) / len(scores):.6f} dB')
        if full_scores:
            average_full = sum(full_scores) / len(full_scores)
            print(f'average full PSNR:  {average_full:.6f} dB')
            print(f'max tiled/full abs error: {max(errors):.8g}')


if __name__ == '__main__':
    main()
