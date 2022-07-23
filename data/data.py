from torch.utils.data import DataLoader
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
# from torch.utils.data.distributed import DistributedSampler

from typing import List, Tuple
import torchvision.transforms as transforms
import torchvision

def initialize_data_loader(batch_size, num_workers) -> Tuple[DataLoader, DataLoader]:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (1.0,))
    ])
    # CIFAR-10
    # download_root = './MNIST_DATASET'

    # train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
    # valid_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True)
    
    trainset = torchvision.datasets.CIFAR10(root='path', train=True, download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(root='path', train=False, download=False, transform=transform)


    train_sampler = ElasticDistributedSampler(trainset)
    # train_sampler = DistributedSampler(train_dataset)
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