import pytest
from dp_pipeline import get_cifar10_dataloaders


def test_get_cifar10_dataloaders_requires_torch():
    with pytest.raises(RuntimeError):
        get_cifar10_dataloaders(batch_size=4)
