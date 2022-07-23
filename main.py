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

from typing import List, Tuple
import time
from data.data import initialize_data_loader
from utils.meters import AverageMeter, ProgressMeter
from torch.utils.data import DataLoader
from utils.train_util import *
from tqdm import tqdm
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--train', '-t', action='store_true', default=True)
    parser.add_argument('--resume', '-r', action='store_true', default=False)
    parser.add_argument('--evaluate', '-e', action='store_true', default=False)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--workers', '-w', type=int, default= 4*4)
    parser.add_argument('--project_name', '-n', type=str, default="default")    
    
    return parser.parse_args()


def train(
        args,
        train_loader : DataLoader,
        model : DistributedDataParallel,
        criterion,
        optimizer,
        epoch : int,
        device_id : int,
        print_freq : int,
        logger
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
    for i, (images, target)in tqdm(enumerate(train_loader), total=len(train_loader),leave=False):
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
                progress.display(i, logger)

def validate(
    args,
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
    logger
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
        for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader),leave=False):
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
                    progress.display(i, logger)

        # TODO: this should also be done with the ProgressMeter
        if device_id == 0:
            print(
                " * Acc@1 {top1.avg:.3f}".format(top1=top1)
            )

    return top1.avg

def main():
    random_seed = 8967
    # parse arguments
    args = parse_args()
    device_id = int(os.environ["LOCAL_RANK"])
    # Check for conflict project save files
    if device_id == 0:
        default_path = 'result'
        name = args.project_name
        if os.path.exists(default_path + '/' + name):
            print(f"\033[31m ************** WARNING : the \033[92m{default_path + '/' + name}\033[31m is already exist. **************\033[0m")
            print(f"\033[31m ************** the Training results all overwrite. **************\033[0m")
    torch.cuda.set_device(device_id)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 

    path = create_save_dir(args.project_name, device_id)

    logger = get_log(path)

    torch.cuda.set_device(device_id)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )
    model = ConvNet()

    logger.info(f"=> creating model ... ")
    model, criterion, optimizer = initialize_model(model, args.learning_rate, device_id)
    
    logger.info(f"=> creating dataloader ... ")
    train_loader, val_loader = initialize_data_loader(args.batch_size, args.workers)


    start_epoch = 0
    end_epoch = 100
    if device_id == 0:
        logger.info(f"=> start_epoch: {start_epoch+1}")

    print_freq = 100

    for epoch in range(start_epoch, end_epoch):
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        adjust_learing_rate(optimizer, epoch, args.learning_rate)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, device_id, print_freq, logger)

        #evaluate on validation set
        acc1 = validate(args, val_loader, model, criterion, device_id, print_freq, logger)

        if device_id == 0:
            logger.info(f"=> Epoch : {epoch + 1} accuracy : {acc1}")

    
if __name__ == "__main__":
    main()

