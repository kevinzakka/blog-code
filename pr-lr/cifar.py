import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

from models import WideResNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--data_dir', default='./data/', type=str,
                    help='data path')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, # 5e-4
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')

best_prec1 = 0

def get_data_loader(data_dir, batch_size, num_workers=3, pin_memory=False):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader

def get_weighted_loader(data_dir, batch_size, weights, num_workers=3, pin_memory=False):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    sampler = WeightedRandomSampler(weights, len(weights), False)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, sampler=sampler
    )
    return loader

def get_test_loader(data_dir, batch_size, num_workers=3, pin_memory=False):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_train)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader

def main():
    global args, best_prec1
    args = parser.parse_args()

    # ensuring reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

    kwargs = {'num_workers': 1, 'pin_memory': True}
    device = torch.device("cuda")

    num_epochs_transient = 2
    num_epochs_steady = 7
    perc_to_remove = 10

    torch.manual_seed(SEED)

    # create model
    model = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay
    )

    # instantiate loaders
    train_loader = get_data_loader(args.data_dir, args.batch_size, **kwargs)
    test_loader = get_test_loader(args.data_dir, 128, **kwargs)

    tic = time.time()
    seen_losses = None
    for epoch in range(1, 3):
        if epoch == 1:
            seen_losses = train_transient(model, device, train_loader, optimizer, epoch, track=True)
        else:
            train_transient(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    for epoch in range(3, 4):
        seen_losses = [v for sublist in seen_losses for v in sublist]
        sorted_loss_idx = sorted(range(len(seen_losses)), key=lambda k: seen_losses[k][1], reverse=True)
        removed = sorted_loss_idx[-int((perc_to_remove / 100) * len(sorted_loss_idx)):]
        sorted_loss_idx = sorted_loss_idx[:-int((perc_to_remove / 100) * len(sorted_loss_idx))]
        to_add = list(np.random.choice(removed, int(0.33*len(sorted_loss_idx)), replace=False))
        sorted_loss_idx = sorted_loss_idx + to_add
        sorted_loss_idx.sort()
        weights = [seen_losses[idx][1] for idx in sorted_loss_idx]
        train_loader = get_weighted_loader(args.data_dir, 64*2, weights, **kwargs)
        seen_losses = train_steady_state(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    for epoch in range(4, 8):
        train_transient(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
    toc = time.time()
    print("Time Elapsed: {}s".format(toc-tic))


def train_transient(model, device, train_loader, optimizer, epoch, track=False):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    epoch_stats = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # compute output
        output = model(data)
        losses_ = F.nll_loss(output, target, reduction='none')

        if track:
            indices = [batch_idx*train_loader.batch_size + i for i in range(len(data))]
            batch_stats = []
            for i, l in zip(indices, losses_):
                batch_stats.append([i, l.item()])
            epoch_stats.append(batch_stats)

        loss = losses_.mean()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), loss=losses, top1=top1))
    if track:
        return epoch_stats
    return None

def train_steady_state(model, device, train_loader, optimizer, epoch):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    epoch_stats = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # compute output
        output = model(data)
        losses_ = F.nll_loss(output, target, reduction='none')
        indices = [batch_idx*train_loader.batch_size + i for i in range(len(data))]
        batch_stats = []
        for i, l in zip(indices, losses_):
            batch_stats.append([i, l.item()])
        epoch_stats.append(batch_stats)

        loss = losses_.mean()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), loss=losses, top1=top1))
    return epoch_stats

def test(model, device, test_loader, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # compute output
        with torch.no_grad():
            output = model(data)
        loss = F.nll_loss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        if batch_idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      batch_idx, len(test_loader), loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
