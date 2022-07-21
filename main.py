#!/usr/bin/env python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from models.model import ConvNet
from datetime import timedelta
import os
import sys
import argparse
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from typing import List, Tuple
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--train', '-t', action='store_true', default=True)
    parser.add_argument('--resume', '-r', action='store_true', default=False)
    parser.add_argument('--evaluate', '-e', action='store_true', default=False)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--workers', '-w', type=int, default= 4*4)
    parser.add_argument('--checkpoint_path', '-cp', type=str, default=os.path.basename(sys.argv[0]))
    
    return parser.parse_args()
def initialize_model(single_model, lr, device_id):
    print(f"=> creating model ... ")
    model = single_model
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer

def initialize_data_loader(batch_size, num_workers) -> Tuple[DataLoader, DataLoader]:
    print(f"=> creating dataloader ... ")
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    # CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='path', train=True, download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(root='path', train=False, download=False, transform=transform)


    train_sampler = ElasticDistributedSampler(trainset)

    train_loader = DataLoader(trainset, 
                            batch_size = batch_size, 
                            num_workers=num_workers,
                            pin_memory= True,
                            sampler = train_sampler)

    val_loader = DataLoader(validset, 
                            batch_size = batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True)

    return train_loader, val_loader

def adjust_learing_rate(optimizer, epoch, lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def train(
        train_loader : DataLoader,
        model : DistributedDataParallel,
        criterion,
        optimizer,
        epoch : int,
        device_id : int,
        print_freq : int
    ):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        #* measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(device_id, non_blocking=True)
        target = target.cuda(device_id, non_blocking=True)

        #compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if device_id == 0:
                progress.display(i)

def validate(
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if device_id is not None:
                images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                if device_id == 0:
                    progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if device_id == 0:
            print(
                " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
            )

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    # parse arguments
    args = parse_args()
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )
    model = ConvNet()
    model, criterion, optimizer = initialize_model(model, args.learning_rate, device_id)

    train_loader, val_loader = initialize_data_loader(args.batch_size, args.workers)


    start_epoch = 0
    end_epoch = 100
    if device_id == 0:
        print(f"=> start_epoch: {start_epoch+1}")

    print_freq = 100

    for epoch in range(start_epoch, end_epoch):
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        adjust_learing_rate(optimizer, epoch, args.learning_rate)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device_id, print_freq)

        #evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device_id, print_freq)

        if device_id == 0:
            print(f"=> Epoch : {epoch + 1} accuracy : {acc1}")

    
if __name__ == "__main__":
    main()

