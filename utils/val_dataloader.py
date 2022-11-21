import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from utils.food_dataset import Food101


def normalize(x):
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(x)

def __get_val_dataloader(valdir, batch_size, image_size):    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    )
    

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )
    
    return val_loader
    
def get_comic_val_dataloader(path='./dataset/Comic Books', batch_size=100, image_size=224):
    valdir = os.path.join(path, 'test')
    return __get_val_dataloader(valdir, batch_size, image_size)

def get_flower_val_dataloader(path='./dataset/Oxford 102 Flower', batch_size=100, image_size=224):
    valdir = os.path.join(path, 'valid')
    return __get_val_dataloader(valdir, batch_size, image_size)

def get_bird_val_dataloader(path='./dataset/BIRDS-400', batch_size=100, image_size=224):
    valdir = os.path.join(path, 'test')
    return __get_val_dataloader(valdir, batch_size, image_size)

def get_food_val_dataloader(root='./dataset', batch_size=100, image_size=224):
    test_dataset = Food101(
        root=root,
        split='test',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    return test_loader

def get_val_dataloader(name, batch_size=100, image_size=224):
    assert name in ['BIRDS-400', 'Comic Books', 'Food-101', 'Oxford 102 Flower'], print(name)
    
    if name == 'Comic Books':
        return get_comic_val_dataloader(batch_size=batch_size, image_size=image_size)
    elif name == 'Oxford 102 Flower':
        return get_flower_val_dataloader(batch_size=batch_size, image_size=image_size)
    elif name == 'BIRDS-400':
        return get_bird_val_dataloader(batch_size=batch_size, image_size=image_size)
    elif name == 'Food-101':
        return get_food_val_dataloader(batch_size=batch_size, image_size=image_size)

