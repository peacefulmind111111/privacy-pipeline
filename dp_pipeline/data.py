from __future__ import annotations

from typing import Tuple

try:
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
except ImportError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    transforms = None  # type: ignore
    torchvision = None  # type: ignore


def get_cifar10_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare CIFAR-10 train and test dataloaders."""
    if torch is None or torchvision is None or transforms is None:
        raise RuntimeError("PyTorch and torchvision must be installed to load CIFAR-10.")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
