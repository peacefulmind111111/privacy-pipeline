"""
Differentially private clustering experiment for CIFAR‑10 embeddings.

This module implements a light‑weight differentially private clustering over
flattened CIFAR‑10 images.  The procedure is deliberately simple to keep the
runtime manageable in the context of this benchmark; it is **not** intended to
match the sophistication of the DP‑GMM approach described in the paper, but it
captures the key ideas: compute noisy class means, generate synthetic data from
those means, train a classifier on the synthetic data, and evaluate its
performance on the real test set.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant

from experiment_utils import clear_memory


def _prepare_datasets(data_fraction: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load CIFAR‑10 and return flattened private and test splits.

    Parameters
    ----------
    data_fraction : float
        Fraction of the CIFAR‑10 training set to use as private data.

    Returns
    -------
    X_priv : Tensor shape (n_priv, p)
        Flattened private examples scaled to [-1, 1].
    y_priv : Tensor shape (n_priv,)
        Integer labels for the private examples.
    X_test : Tensor shape (n_test, p)
        Flattened test examples scaled to [-1, 1].
    y_test : Tensor shape (n_test,)
        Integer labels for the test examples.
    """
    # Normalise images to roughly [-1, 1] to satisfy boundedness assumptions.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    # Subsample the training set to obtain private data.
    num_priv = int(len(train_set) * data_fraction)
    idxs = torch.randperm(len(train_set))[:num_priv]
    priv_subset = Subset(train_set, idxs)
    # Convert to tensors.
    X_priv = torch.stack([priv_subset[i][0] for i in range(len(priv_subset))], dim=0)
    y_priv = torch.tensor([priv_subset[i][1] for i in range(len(priv_subset))], dtype=torch.long)
    X_test = torch.stack([test_set[i][0] for i in range(len(test_set))], dim=0)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)
    # Flatten images.
    X_priv = X_priv.view(X_priv.size(0), -1)
    X_test = X_test.view(X_test.size(0), -1)
    # Scale to [-1, 1] (images are in [0,1]).
    X_priv = X_priv * 2.0 - 1.0
    X_test = X_test * 2.0 - 1.0
    return X_priv, y_priv, X_test, y_test


def _compute_noisy_means(
    X: torch.Tensor, y: torch.Tensor, n_classes: int, noise_mult: float, accnt: RDPAccountant, delta: float
) -> List[torch.Tensor]:
    """
    Compute class means with added Gaussian noise for differential privacy.

    Parameters
    ----------
    X : Tensor (n, p)
        Flattened feature matrix.
    y : Tensor (n,)
        Class labels.
    n_classes : int
        Number of classes.
    noise_mult : float
        Noise multiplier (std dev of noise relative to sensitivity).
    accnt : RDPAccountant
        Privacy accountant.
    delta : float
        Target δ for privacy accounting.

    Returns
    -------
    means_noisy : list of Tensors length n_classes
        A list of noisy class mean vectors.
    """
    n, p = X.size()
    means: List[torch.Tensor] = []
    # Sensitivity of a mean query in l2 is bounded by 2 / n_class when each feature lies in [-1,1].
    for c in range(n_classes):
        mask = (y == c)
        if mask.any():
            X_c = X[mask]
            mean_c = X_c.mean(dim=0)
            # Compute noise scale: sensitivity = 2 / n_class
            n_c = X_c.size(0)
            sensitivity = 2.0 / max(n_c, 1)
            sigma = noise_mult * sensitivity
            noise = torch.randn_like(mean_c) * sigma
            noisy_mean = mean_c + noise
            means.append(noisy_mean)
            # Advance privacy accountant: treat this as one Gaussian mechanism on the class data.
            accnt.step(noise_multiplier=noise_mult, sample_rate=(n_c / n),)
        else:
            means.append(torch.zeros(p))
    return means


class MLPClassifier(nn.Module):
    """Simple single hidden layer classifier for synthetic embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def run_experiment(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Run a simplified differentially private clustering and synthetic data experiment.

    The routine performs the following steps:
      1. Load a private subset of the CIFAR‑10 training data and flatten each image.
      2. Compute noisy per‑class means with Gaussian noise scaled by ``noise_mult``.
      3. Sample synthetic embeddings from a normal distribution centred at each noisy mean.
      4. Train a small classifier on the synthetic dataset.
      5. Evaluate the classifier on the real CIFAR‑10 test set.
      6. Estimate the privacy budget using the Opacus RDPAccountant.

    Parameters
    ----------
    params : dict, optional
        Hyper‑parameters for the experiment:
          data_fraction : float   fraction of CIFAR‑10 used as private data (default 0.2).
          n_synth_per_class : int number of synthetic samples to generate per class (default 2000).
          hidden_dim : int        hidden dimension of the classifier (default 128).
          noise_mult : float      multiplier controlling DP noise (default 1.0).
          delta : float           target δ for differential privacy (default 1e-5).

    Returns
    -------
    Dict[str, Any]
        A metrics dictionary with keys ``final_accuracy``, ``epsilon``,
        ``history`` (training loss trajectory), and ``final_loss``.
    """
    params = params or {}
    data_fraction: float = params.get("data_fraction", 0.2)
    n_synth_per_class: int = params.get("n_synth_per_class", 2000)
    hidden_dim: int = params.get("hidden_dim", 128)
    noise_mult: float = params.get("noise_mult", 1.0)
    delta: float = params.get("delta", 1e-5)
    num_classes: int = 10

    # Step 1: prepare data
    X_priv, y_priv, X_test, y_test = _prepare_datasets(data_fraction)
    n_priv, p = X_priv.size()

    # Step 2: compute noisy class means
    accnt = RDPAccountant()
    noisy_means = _compute_noisy_means(X_priv, y_priv, num_classes, noise_mult, accnt, delta)

    # Step 3: generate synthetic embeddings
    synth_features: List[torch.Tensor] = []
    synth_labels: List[torch.Tensor] = []
    for c in range(num_classes):
        mu = noisy_means[c]
        # standard deviation proportional to noise_mult to encourage diversity
        sigma = noise_mult * (2.0 / max((y_priv == c).sum().item(), 1))
        samples = mu + torch.randn(n_synth_per_class, p) * sigma
        synth_features.append(samples)
        synth_labels.append(torch.full((n_synth_per_class,), c, dtype=torch.long))
    X_synth = torch.cat(synth_features, dim=0)
    y_synth = torch.cat(synth_labels, dim=0)

    # Step 4: train classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=p, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Convert data to device
    X_synth = X_synth.to(device)
    y_synth = y_synth.to(device)
    X_test_d = X_test.to(device)
    y_test_d = y_test.to(device)
    history: List[Dict[str, float]] = []
    # one epoch is sufficient for demonstration; can extend if needed
    batch_size = 256
    loader = DataLoader(list(zip(X_synth, y_synth)), batch_size=batch_size, shuffle=True)
    for epoch in range(1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        history.append({"epoch": epoch, "loss": epoch_loss / len(loader.dataset)})

    # Step 5: evaluate on real test data
    with torch.no_grad():
        preds = model(X_test_d)
        final_loss = criterion(preds, y_test_d).item()
        final_accuracy = (preds.argmax(dim=1) == y_test_d).float().mean().item() * 100

    # Step 6: compute epsilon using accountant
    epsilon = accnt.get_epsilon(delta)

    # Clean up GPU memory after use
    clear_memory()

    return {
        "final_accuracy": float(final_accuracy),
        "epsilon": float(epsilon),
        "history": history,
        "final_loss": float(final_loss),
    }