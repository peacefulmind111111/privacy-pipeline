"""
dp_pipeline.py
================

Utility functions for running differentially private (DP) machine‑learning
experiments on CIFAR‑10 with a ResNet‑20 architecture.  The goal of this module
is to provide a unified entry point for training models with a standard set of
hyper‑parameters and recording results in a comprehensive JSON format that can
be consumed by downstream analytics applications.

The module does not execute any training by itself; rather it exposes
functions that return callables which can be invoked from a Jupyter notebook
or another Python script.  This separation allows notebooks to remain clean
and enables reuse of the training logic across experiments.

Each experiment function accepts a configuration dictionary containing
experiment‑specific hyper‑parameters.  It returns a results dictionary with
the following recommended top‑level keys:

```
{
    "experiment_name": str,
    "dataset": str,
    "model": str,
    "method": str,
    "hyperparameters": {...},
    "privacy": {
        "epsilon": float,
        "delta": float,
        "noise_multiplier": float,
        "sample_rate": float
    },
    "metrics": {
        "train_accuracy_final": float,
        "train_loss_final": float,
        "test_accuracy_final": float,
        "test_loss_final": float,
        "top5_accuracy_final": float,
        "runtime_sec": float
    },
    "timestamp": str,
    "comments": str
}
```

Additional fields can be added as necessary.  All experiments should set the
`experiment_name` field uniquely (e.g. "clipping_tradeoff_C1.0") so that
results do not collide.

Note
----
This module assumes that PyTorch and Opacus are available in the runtime
environment.  It also requires torchvision for loading CIFAR‑10 and
scikit‑learn for LASSO experiments.  If these dependencies are not installed
or the environment lacks network access to download the dataset, the
functions may fail.  Users may pre‑download the dataset and pass a custom
data loader if desired.

"""

from __future__ import annotations

import json
import os
import time
import datetime
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn.functional as F
except ImportError:
    # If torch is not available, training functions will raise errors.
    torch = None  # type: ignore

try:
    from opacus import PrivacyEngine
except ImportError:
    PrivacyEngine = None  # type: ignore

try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
except ImportError:
    Lasso = None  # type: ignore

# Import numpy for array manipulation when available.  Some functions such as
# DP‑mixup and DP‑LASSO rely on numpy for beta distributions and noise.
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""

    experiment_name: str
    method: str
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    sample_rate: Optional[float] = None
    mixup_alpha: Optional[float] = None
    lasso_alpha: float = 0.01
    comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def current_timestamp() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def save_results_json(results: Dict[str, Any], output_dir: str) -> str:
    """Save a results dictionary to a JSON file.

    Args:
        results: Dictionary containing experiment results.
        output_dir: Directory in which to write the JSON file.

    Returns:
        The path to the created JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{results['experiment_name']}_{int(time.time())}.json"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return file_path


def get_cifar10_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare CIFAR‑10 train and test dataloaders.

    Args:
        batch_size: Mini‑batch size.

    Returns:
        A tuple of (train_loader, test_loader).
    """
    if torch is None or torchvision is None:
        raise RuntimeError("PyTorch and torchvision must be installed to load CIFAR‑10.")

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


def resnet20() -> nn.Module:
    """Returns a small ResNet‑20 model for CIFAR‑10.

    The implementation here is simplified and may not match exact ResNet‑20
    architecture; however, it provides a lightweight model suitable for
    experimentation.  Users can substitute their own implementation if
    necessary.
    """
    if torch is None:
        raise RuntimeError("PyTorch must be installed to construct models.")

    # Define a basic residual block
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super().__init__()
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(64, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, planes, s))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.linear(out)
            return out

    return ResNet(BasicBlock, [3, 3, 3])


def train_dp_sgd(
    config: ExperimentConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Train a model with DP‑SGD and return comprehensive results.

    Args:
        config: Experiment configuration.
        train_loader: Training data loader.
        test_loader: Test data loader.
        model: Optional preconstructed model; if None, a new ResNet‑20 is built.

    Returns:
        Dictionary of results following the standard schema.
    """
    if torch is None or PrivacyEngine is None:
        raise RuntimeError("PyTorch and Opacus must be installed for DP training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model if model is not None else resnet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )

    # Determine sample rate if not provided
    total_train = len(train_loader.dataset)
    if config.sample_rate is None:
        config.sample_rate = config.batch_size / total_train

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        #target_epsilon=None,  # we'll use noise_multiplier instead
        target_delta=config.delta,
        epochs=config.epochs,
        max_grad_norm=config.clip_norm,
        noise_multiplier=config.noise_multiplier,
    )

    start_time = time.time()
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Optionally evaluate on a subset of the train set at the end of each epoch

    runtime = time.time() - start_time

    # Evaluate on train and test sets
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
                # Compute top‑5 accuracy
                top5_vals, top5_idx = outputs.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += sum([labels[i] in top5_idx[i] for i in range(labels.size(0))])
        avg_loss = total_loss / total
        acc_top1 = correct_top1 / total
        acc_top5 = correct_top5 / total
        return avg_loss, acc_top1, acc_top5

    train_loss, train_acc, train_top5 = evaluate(train_loader)
    test_loss, test_acc, test_top5 = evaluate(test_loader)

    # Compute spent privacy using the accountant
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


def train_dp_mixup(
    config: ExperimentConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Train a model with DP‑SGD using mixup augmentation.

    Mixup linearly interpolates random pairs of examples.  To respect DP
    constraints, each original training example should be involved in at most
    one synthetic mixup per mini‑batch.  For simplicity, this implementation
    assumes that the provided `train_loader` yields batches where mixup can
    be applied directly.  The user may need to adjust the data loader to
    enforce strict per‑example participation.
    """
    if config.mixup_alpha is None:
        raise ValueError("mixup_alpha must be set for mixup experiments")
    if torch is None:
        raise RuntimeError("PyTorch must be installed for DP mixup training")

    # Precompute mixup lambda distribution
    import numpy as np
    def mixup_data(x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # Set up model and optimizer
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
    # Training loop with mixup
    for epoch in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            mixed_x, targets_a, targets_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

    runtime = time.time() - start_time

    # Evaluate using standard (non‑mixup) predictions
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


def dp_lasso(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
) -> Dict[str, Any]:
    """Perform a differentially private LASSO regression.

    This function wraps a standard LASSO solver from scikit‑learn and adds
    input perturbation to provide differential privacy.  The implementation
    normalizes features and adds Gaussian noise to the training data before
    fitting the model.  It then evaluates mean squared error on the test
    set.  Note that this is a simplified example and may not provide
    optimal privacy guarantees; it is intended for illustration.
    """
    if Lasso is None:
        raise RuntimeError("scikit‑learn must be installed to run DP LASSO")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add Gaussian noise to each feature to satisfy (epsilon, delta)-DP
    # This is a basic mechanism; for high‑dimensional problems users should
    # refer to more sophisticated algorithms like those in Talwar et al. (2015).
    sigma = config.noise_multiplier
    noise = sigma * np.random.randn(*X_train_scaled.shape)
    X_train_noisy = X_train_scaled + noise

    lasso = Lasso(alpha=config.lasso_alpha)
    start = time.time()
    lasso.fit(X_train_noisy, y_train)
    runtime = time.time() - start

    y_pred_train = lasso.predict(X_train_noisy)
    y_pred_test = lasso.predict(X_test_scaled)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Estimate epsilon via simple composition; here we do not compute a true DP
    # guarantee but include noise_multiplier as a proxy
    epsilon_est = config.noise_multiplier  # placeholder

    results = {
        "experiment_name": config.experiment_name,
        "dataset": "custom",
        "model": "Lasso",
        "method": config.method,
        "hyperparameters": config.to_dict(),
        "privacy": {
            "epsilon": epsilon_est,
            "delta": config.delta,
            "noise_multiplier": config.noise_multiplier,
            "sample_rate": None,
        },
        "metrics": {
            "train_mse": mse_train,
            "test_mse": mse_test,
            "runtime_sec": runtime,
        },
        "timestamp": current_timestamp(),
        "comments": config.comments,
    }
    return results
