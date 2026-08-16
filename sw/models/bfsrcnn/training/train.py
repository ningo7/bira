import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models.bfsrcnn.network.bfsrcnn import BFSRCNN

from models.bfsrcnn.training.datasets import TrainDataset, EvalDataset
from models.bfsrcnn.common.patch_utils import tiled_forward
from models.bfsrcnn.common.utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--optimizer-state', type=str)
    parser.add_argument('--scheduler-state', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval-tile-size', type=int, default=64)
    parser.add_argument('--eval-tile-pad', type=int, default=16)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    # model = FSRCNN(scale_factor=args.scale).to(device)
    model = BFSRCNN(scale_factor=args.scale).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        # {'params': model.first_part.parameters()},
        # {'params': model.mid_part.parameters()},
        # {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
        {'params': model.feature_extraction.parameters()},
        {'params': model.shrinking.parameters()},
        {'params': model.mapping.parameters()},
        {'params': model.expanding.parameters()}
        # {'params': model.expanding.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

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

    if args.weights_file is not None:
        print('load weights file')
        weights_file = torch.load(args.weights_file)
        model.load_state_dict(weights_file)
    if args.optimizer_state is not None:
        print('load optimizer state')
        optimizer_state = torch.load(args.optimizer_state)
        optimizer.load_state_dict(optimizer_state)

    if args.scheduler_state is not None:
        print('load scheduler state')
        scheduler_state = torch.load(args.scheduler_state)
        scheduler.load_state_dict(scheduler_state)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                # print(inputs.shape, labels.shape, preds.shape)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = tiled_forward(
                    model, inputs, args.scale,
                    args.eval_tile_size, args.eval_tile_pad,
                ).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.8f}'.format(epoch_psnr.avg))

        scheduler.step(float(epoch_psnr.avg))
        print('learning rates: {}'.format(
            ', '.join('{:.2e}'.format(group['lr']) for group in optimizer.param_groups)
        ))

        torch.save(optimizer.state_dict(), os.path.join(args.outputs_dir, 'optimizer_state.pth'))
        torch.save(scheduler.state_dict(), os.path.join(args.outputs_dir, 'scheduler_state.pth'))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
