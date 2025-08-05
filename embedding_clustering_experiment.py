#!/usr/bin/env python3
# dp_cifar10_clustering.py
"""
Differentially‑Private Clustering on CIFAR‑10 with Synthetic‑Data Training
===========================================================================

This script reproduces the simple DP‑clustering baseline described in
your snippet and extends it into a fully‑featured experiment runner.

1.  Load a *private* fraction of CIFAR‑10.
2.  Compute noisy per‑class means (Gaussian mechanism).
3.  Sample synthetic embeddings from the noisy means.
4.  Train an MLP classifier on the synthetic set.
5.  Evaluate on the real CIFAR‑10 test set.
6.  Report (ε, δ) using Opacus' RDP accountant.

© 2025  (MIT License)  — Feel free to adapt.
"""
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                               Utility helpers                               #
# --------------------------------------------------------------------------- #


def clear_memory() -> None:
    """Force‑release (GPU) memory to avoid fragmentation between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():  # Apple Silicon
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    """Make results (mostly) reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#                       Data preparation & noisy means                        #
# --------------------------------------------------------------------------- #


def _prepare_datasets(data_fraction: float) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
    """
    Load CIFAR‑10, flatten images, and split into private train / public test tensors.
    Output tensors are already scaled to [‑1, 1].
    """
    transform = transforms.Compose([transforms.ToTensor()])  # keep it simple
    cifar_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Draw a random private subset
    num_priv = int(len(cifar_train) * data_fraction)
    priv_idxs = torch.randperm(len(cifar_train))[:num_priv]
    priv_subset = Subset(cifar_train, priv_idxs)

    # Stack into tensors
    X_priv = torch.stack([priv_subset[i][0] for i in range(len(priv_subset))], 0)
    y_priv = torch.tensor([priv_subset[i][1] for i in range(len(priv_subset))], dtype=torch.long)
    X_test = torch.stack([cifar_test[i][0] for i in range(len(cifar_test))], 0)
    y_test = torch.tensor([cifar_test[i][1] for i in range(len(cifar_test))], dtype=torch.long)

    # Flatten + scale to [‑1, 1]
    X_priv = (X_priv * 2 - 1).view(X_priv.size(0), -1)
    X_test = (X_test * 2 - 1).view(X_test.size(0), -1)
    return X_priv, y_priv, X_test, y_test


def _compute_noisy_means(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    noise_mult: float,
    accountant: RDPAccountant,
) -> List[torch.Tensor]:
    """
    Gaussian mechanism on each class mean.  Sensitivity ≤ 2 / n_c because pixels ∈[‑1, 1].
    """
    means: List[torch.Tensor] = []
    n_total, _ = X.shape
    for c in range(n_classes):
        X_c = X[y == c]
        if X_c.numel() == 0:        # class may be empty in pathological splits
            means.append(torch.zeros_like(X[0]))
            continue

        n_c = X_c.size(0)
        true_mean = X_c.mean(0)
        sensitivity = 2.0 / n_c
        sigma = noise_mult * sensitivity
        noisy_mean = true_mean + torch.randn_like(true_mean) * sigma
        means.append(noisy_mean)

        # RDP accountant update — one query touching every example of class c.
        accountant.step(noise_multiplier=noise_mult,
                        sample_rate=n_c / n_total)  # Opacus uses q = m/N
    return means


# --------------------------------------------------------------------------- #
#                              Model — simple MLP                             #
# --------------------------------------------------------------------------- #


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --------------------------------------------------------------------------- #
#                                Core routine                                 #
# --------------------------------------------------------------------------- #


def run_experiment(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Entry point that can also be imported from other modules."""
    p = params or {}
    # Hyper‑parameters with sane defaults
    data_fraction: float = p.get("data_fraction", 0.20)
    n_synth_per_class: int = p.get("n_synth_per_class", 2_000)
    hidden_dim: int = p.get("hidden_dim", 128)
    noise_mult: float = p.get("noise_mult", 1.0)
    delta: float = p.get("delta", 1e-5)
    epochs: int = p.get("epochs", 5)
    batch_size: int = p.get("batch_size", 256)
    lr: float = p.get("lr", 1e-3)
    seed: int = p.get("seed", 42)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # 1. Data ----------------------------------------------------------------
    X_priv, y_priv, X_test, y_test = _prepare_datasets(data_fraction)
    p_dim = X_priv.size(1)

    # 2. Noisy per‑class means ----------------------------------------------
    accountant = RDPAccountant()
    noisy_means = _compute_noisy_means(X_priv, y_priv, 10, noise_mult, accountant)

    # 3. Synthetic dataset ---------------------------------------------------
    synth_X, synth_y = [], []
    for c, mu in enumerate(noisy_means):
        n_c = max((y_priv == c).sum().item(), 1)
        sigma = noise_mult * (2.0 / n_c)  # same heuristic as original snippet
        samples = mu + torch.randn(n_synth_per_class, p_dim) * sigma
        synth_X.append(samples)
        synth_y.append(torch.full((n_synth_per_class,), c, dtype=torch.long))
    X_synth = torch.cat(synth_X).to(device)
    y_synth = torch.cat(synth_y).to(device)

    # 4. Model, loss, optimiser ---------------------------------------------
    model = MLPClassifier(p_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training ------------------------------------------------------------
    dataset = torch.utils.data.TensorDataset(X_synth, y_synth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(dataset)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"‣ Epoch {epoch+1:>2}: train‑loss = {epoch_loss:.4f}")

    # 6. Evaluation ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        test_loss = criterion(logits, y_test.to(device)).item()
        test_acc = (logits.argmax(1)
                    == y_test.to(device)).float().mean().item() * 100
    epsilon = accountant.get_epsilon(delta)

    clear_memory()

    metrics = dict(
        final_accuracy=test_acc,
        final_loss=test_loss,
        epsilon=epsilon,
        delta=delta,
        epochs=epochs,
        history=history,
    )
    return metrics


# --------------------------------------------------------------------------- #
#                            Command‑line interface                           #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Differentially‑Private clustering baseline on CIFAR‑10")
    parser.add_argument("--data-fraction", type=float, default=0.20,
                        help="Fraction of CIFAR‑10 train images used as *private* data")
    parser.add_argument("--n-synth-per-class", type=int, default=2000,
                        help="# synthetic samples generated per class")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden width of the MLP")
    parser.add_argument("--noise-mult", type=float, default=1.0,
                        help="Gaussian noise multiplier for the mean query")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Target δ for (ε, δ)‑DP")
    parser.add_argument("--epochs", type=int, default=5,
                        help="# epochs to train the classifier")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(vars(args))

    # Pretty summary --------------------------------------------------------
    print("\n=================  Results  =================")
    print(f"Test accuracy : {metrics['final_accuracy']:.2f} %")
    print(f"Test loss     : {metrics['final_loss']:.4f}")
    print(f"Privacy (ε, δ): ({metrics['epsilon']:.3f}, {metrics['delta']})")
    print("============================================\n")

    # Optionally save metrics
    out = Path("dp_cifar10_results.json")
    try:
        import json

        out.write_text(json.dumps(metrics, indent=2))
        print(f"Saved full metrics to {out.resolve()}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not save JSON results: {e}")


if __name__ == "__main__":
    main()
