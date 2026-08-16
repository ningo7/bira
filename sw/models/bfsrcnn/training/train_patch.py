import argparse
import copy
import glob
import os
import random

import h5py
import numpy as np
import PIL.Image as Image
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.bfsrcnn.network.bfsrcnn import BFSRCNN
from models.bfsrcnn.training.datasets import EvalDataset
from models.bfsrcnn.common.patch_utils import crop_border, load_model_weights, tiled_forward
from models.bfsrcnn.common.utils import AverageMeter, calc_psnr, convert_rgb_to_y


PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    return array / 255.0 if array.size and array.max() > 1.0 else array


def crop_pair(lr, hr, scale, core, halo):
    size = core + 2 * halo
    height, width = lr.shape
    if height < size or width < size:
        raise ValueError(
            f'LR image {lr.shape} is smaller than required patch {size}x{size}'
        )
    if hr.shape != (height * scale, width * scale):
        raise ValueError(
            f'unaligned LR {lr.shape} and HR {hr.shape} for x{scale}'
        )
    y = int(np.random.randint(height - size + 1))
    x = int(np.random.randint(width - size + 1))
    lr = lr[y:y + size, x:x + size]
    target_y, target_x = (y + halo) * scale, (x + halo) * scale
    target_size = core * scale
    hr = hr[
        target_y:target_y + target_size,
        target_x:target_x + target_size,
    ]
    return lr, hr


def augment(lr, hr):
    if np.random.randint(2):
        lr, hr = np.flip(lr, 0), np.flip(hr, 0)
    if np.random.randint(2):
        lr, hr = np.flip(lr, 1), np.flip(hr, 1)
    rotations = int(np.random.randint(4))
    if rotations:
        lr, hr = np.rot90(lr, rotations), np.rot90(hr, rotations)
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)


class PatchDatasetBase(Dataset):
    def __init__(self, scale, core, halo, patches_per_image, use_augment):
        self.scale = scale
        self.core = core
        self.halo = halo
        self.repeats = patches_per_image
        self.use_augment = use_augment

    def make_sample(self, lr, hr):
        lr, hr = crop_pair(
            normalize(lr), normalize(hr),
            self.scale, self.core, self.halo,
        )
        if self.use_augment:
            lr, hr = augment(lr, hr)
        return torch.from_numpy(lr[None]), torch.from_numpy(hr[None])


class H5PatchDataset(PatchDatasetBase):
    """Random patches from a full-image aligned LR/HR HDF5 file."""
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        with h5py.File(path, 'r') as handle:
            if 'lr' not in handle or 'hr' not in handle:
                raise KeyError('HDF5 must contain lr and hr')
            self.grouped = isinstance(handle['lr'], h5py.Group)
            if self.grouped:
                self.keys = sorted(
                    handle['lr'].keys(),
                    key=lambda key: int(key) if key.isdigit() else key,
                )
                if set(self.keys) != set(handle['hr'].keys()):
                    raise ValueError('lr and hr keys differ')
                self.count = len(self.keys)
            else:
                self.keys = None
                self.count = len(handle['lr'])
                if self.count != len(handle['hr']):
                    raise ValueError('lr and hr lengths differ')
        if not self.count:
            raise ValueError('empty training HDF5')

    def __len__(self):
        return self.count * self.repeats

    def __getitem__(self, index):
        index %= self.count
        with h5py.File(self.path, 'r') as handle:
            key = self.keys[index] if self.grouped else index
            lr, hr = handle['lr'][key][:], handle['hr'][key][:]
        return self.make_sample(lr, hr)


class ImagePatchDataset(PatchDatasetBase):
    """Create bicubic LR and random patches from original HR images."""
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        paths = []
        for pattern in PATTERNS:
            paths.extend(glob.glob(os.path.join(directory, pattern)))
        self.paths = sorted(set(paths))
        if not self.paths:
            raise ValueError(f'no images found in {directory}')

    def __len__(self):
        return len(self.paths) * self.repeats

    def __getitem__(self, index):
        image = Image.open(self.paths[index % len(self.paths)]).convert('RGB')
        width = image.width // self.scale * self.scale
        height = image.height // self.scale * self.scale
        hr_image = image.crop((0, 0, width, height))
        lr_image = hr_image.resize(
            (width // self.scale, height // self.scale), BICUBIC
        )
        hr = convert_rgb_to_y(np.asarray(hr_image, dtype=np.float32))
        lr = convert_rgb_to_y(np.asarray(lr_image, dtype=np.float32))
        return self.make_sample(lr, hr)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def center_crop(prediction, core, halo, scale):
    start, size = halo * scale, core * scale
    return prediction[..., start:start + size, start:start + size]


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    meter = AverageMeter()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = tiled_forward(
            model, inputs, args.scale,
            args.eval_tile_size, args.eval_tile_pad,
        ).clamp(0.0, 1.0)
        score = calc_psnr(
            crop_border(predictions, args.eval_crop_border),
            crop_border(labels, args.eval_crop_border),
        )
        meter.update(score.item(), len(inputs))
    return meter.avg


def parse_args():
    parser = argparse.ArgumentParser(
        description='BFSRCNN random context-patch training'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--train-file', help='full-image aligned LR/HR HDF5')
    source.add_argument('--train-images-dir', help='original HR image directory')
    parser.add_argument('--eval-file', required=True)
    parser.add_argument('--outputs-dir', required=True)
    parser.add_argument('--weights-file')
    parser.add_argument('--optimizer-state')
    parser.add_argument('--scheduler-state')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32, help='LR loss core')
    parser.add_argument('--halo', type=int, default=12, help='LR context per side')
    parser.add_argument('--patches-per-image', type=int, default=32)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--eval-tile-size', type=int, default=64)
    parser.add_argument('--eval-tile-pad', type=int, default=16)
    parser.add_argument('--eval-crop-border', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--d', type=int, default=48)
    parser.add_argument('--s', type=int, default=16)
    parser.add_argument('--mapping-blocks', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.scale, args.patch_size, args.patches_per_image) < 1:
        raise ValueError(
            'scale, patch size and patches per image must be positive'
        )
    if args.halo < 0:
        raise ValueError('halo must be non-negative')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.outputs_dir, f'x{args.scale}-patch')
    os.makedirs(output_dir, exist_ok=True)

    dataset_options = dict(
        scale=args.scale,
        core=args.patch_size,
        halo=args.halo,
        patches_per_image=args.patches_per_image,
        use_augment=not args.no_augment,
    )
    dataset = (
        H5PatchDataset(args.train_file, **dataset_options)
        if args.train_file
        else ImagePatchDataset(args.train_images_dir, **dataset_options)
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = DataLoader(EvalDataset(args.eval_file), batch_size=1)

    model = BFSRCNN(
        args.scale, d=args.d, s=args.s, m=args.mapping_blocks
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3,
        patience=3,
        threshold=0.01,
        threshold_mode='abs',
        cooldown=1,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    if args.weights_file:
        load_model_weights(model, args.weights_file, device)
    if args.optimizer_state:
        optimizer.load_state_dict(
            torch.load(args.optimizer_state, map_location=device)
        )
    if args.scheduler_state:
        scheduler.load_state_dict(
            torch.load(args.scheduler_state, map_location=device)
        )

    best_psnr, best_epoch = float('-inf'), -1
    for epoch in range(args.num_epochs):
        model.train()
        losses = AverageMeter()
        progress = tqdm(
            train_loader, ncols=100,
            desc=f'epoch {epoch}/{args.num_epochs - 1}',
        )
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = center_crop(
                model(inputs), args.patch_size, args.halo, args.scale
            )
            loss = criterion(predictions, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), len(inputs))
            progress.set_postfix(loss=f'{losses.avg:.6f}')

        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f'epoch_{epoch}.pth'),
        )
        psnr = evaluate(model, eval_loader, device, args)
        scheduler.step(psnr)
        rates = ', '.join(
            f"{group['lr']:.2e}" for group in optimizer.param_groups
        )
        print(f'eval PSNR: {psnr:.2f} dB; learning rates: {rates}')
        torch.save(
            optimizer.state_dict(),
            os.path.join(output_dir, 'optimizer_state.pth'),
        )
        torch.save(
            scheduler.state_dict(),
            os.path.join(output_dir, 'scheduler_state.pth'),
        )
        if psnr > best_psnr:
            best_psnr, best_epoch = psnr, epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(output_dir, 'best.pth'))

    print(f'best epoch: {best_epoch}, PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
