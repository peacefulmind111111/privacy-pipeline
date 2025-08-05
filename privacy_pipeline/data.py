import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from .config import ExperimentConfig


class IndexedSubset(Subset):
    """Subset that also returns the original index."""

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, idx


def get_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Creates train and test dataloaders with augmentation."""
    mean, std = cfg.dataset_mean, cfg.dataset_std
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    n = len(train_ds)
    train_sub = IndexedSubset(train_ds, range(n))
    train_full = ConcatDataset([train_sub] * cfg.self_aug_factor)

    train_loader = DataLoader(train_full, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
