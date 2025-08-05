import torch
from dataclasses import dataclass, field
from typing import List, Tuple


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    """Holds hyperparameters for a DP experiment."""

    # training hyperparameters
    seed: int = 0
    batch_size: int = 512
    lr: float = 0.05
    outer_momentum: float = 0.9
    inner_momentum: float = 0.10
    noise_mult: float = 1.0
    delta: float = 1e-5
    num_epochs: int = 20

    # clipping schedule
    c_start: float = 4.0
    c_end: float = 2.0

    # outlier clipping
    default_clip: float = 1.0
    outlier_clip: float = 0.5
    high_err_threshold: float = 0.95
    drop_after_frac: float = 0.6

    # dataset and augmentation
    self_aug_factor: int = 3
    dataset_mean: Tuple[float, float, float] = field(
        default_factory=lambda: (0.4914, 0.4822, 0.4465)
    )
    dataset_std: Tuple[float, float, float] = field(
        default_factory=lambda: (0.2470, 0.2435, 0.2616)
    )

    # LR schedule
    schedule_milestones: List[int] = field(default_factory=lambda: [12, 18])
    schedule_gamma: float = 0.1

    # momentum dictionary size
    max_momentum_size: int = 10000

    # device
    device: torch.device = field(default_factory=default_device)
