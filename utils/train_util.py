import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn


def adjust_learing_rate(optimizer, epoch, lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def initialize_model(single_model, lr, device_id):
    print(f"=> creating model ... ")
    model = single_model
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer

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