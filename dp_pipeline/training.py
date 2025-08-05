from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    F = None  # type: ignore

try:
    from opacus import PrivacyEngine
except ImportError:  # pragma: no cover - handled at runtime
    PrivacyEngine = None  # type: ignore

from .config import ExperimentConfig, current_timestamp
from .models import resnet20


def train_dp_sgd(
    config: ExperimentConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Train a model with DP-SGD and return comprehensive results."""
    if torch is None or PrivacyEngine is None:
        raise RuntimeError("PyTorch and Opacus must be installed for DP training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model if model is not None else resnet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )

    total_train = len(train_loader.dataset)
    if config.sample_rate is None:
        config.sample_rate = config.batch_size / total_train

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=None,
        target_delta=config.delta,
        epochs=config.epochs,
        max_grad_norm=config.clip_norm,
        noise_multiplier=config.noise_multiplier,
    )

    start_time = time.time()
    for _ in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    runtime = time.time() - start_time

    def evaluate(loader: DataLoader) -> Tuple[float, float, float]:
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_top1 += predicted.eq(labels).sum().item()
                top5_vals, top5_idx = outputs.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += sum([labels[i] in top5_idx[i] for i in range(labels.size(0))])
        avg_loss = total_loss / total
        acc_top1 = correct_top1 / total
        acc_top5 = correct_top5 / total
        return avg_loss, acc_top1, acc_top5

    train_loss, train_acc, train_top5 = evaluate(train_loader)
    test_loss, test_acc, test_top5 = evaluate(test_loader)
    epsilon, _ = privacy_engine.get_privacy_spent(delta=config.delta)

    results = {
        "experiment_name": config.experiment_name,
        "dataset": "CIFAR10",
        "model": "ResNet20",
        "method": config.method,
        "hyperparameters": config.to_dict(),
        "privacy": {
            "epsilon": epsilon,
            "delta": config.delta,
            "noise_multiplier": config.noise_multiplier,
            "sample_rate": config.sample_rate,
            "clip_norm": config.clip_norm,
        },
        "metrics": {
            "train_loss_final": train_loss,
            "train_accuracy_final": train_acc,
            "train_top5_accuracy_final": train_top5,
            "test_loss_final": test_loss,
            "test_accuracy_final": test_acc,
            "test_top5_accuracy_final": test_top5,
            "runtime_sec": runtime,
        },
        "timestamp": current_timestamp(),
        "comments": config.comments,
    }
    return results
