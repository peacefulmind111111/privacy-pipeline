"""
Differentially private LASSO experiment.

This module implements a simple version of the differentially private
Frank–Wolfe (LASSO) algorithm described in the "Nearly Optimal Private LASSO"
paper.  We solve a high‑dimensional linear regression problem where the input
features are flattened CIFAR‑10 images and the targets are scaled class
indices.  The objective is the squared loss subject to an ℓ₁ constraint on the
parameter vector.  At each iteration we privately select a coordinate of the
gradient via the exponential mechanism (approximated here with Gaussian noise),
and perform a Frank–Wolfe update along that coordinate.  After the final
iteration we evaluate the learned model on the real test set by rounding
predictions to the nearest integer class.

This implementation is intentionally lightweight and only captures the core
ideas of the algorithm.  It should not be used as a drop‑in replacement for
production‑quality DP optimization but demonstrates how to integrate the
algorithm into the existing benchmarking framework.
"""

from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant

from experiment_utils import clear_memory


def _prepare_data(data_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load CIFAR‑10, flatten images and scale features/targets for regression.

    Parameters
    ----------
    data_fraction : float
        Fraction of the CIFAR‑10 training set to use as private data.

    Returns
    -------
    X_priv : Tensor (n_priv, p)
        Flattened private examples scaled to [-1, 1].
    y_priv : Tensor (n_priv,)
        Continuous targets scaled to [-1, 1] (class index scaled).
    X_test : Tensor (n_test, p)
        Flattened test examples scaled to [-1, 1].
    y_test : Tensor (n_test,)
        Integer class labels for evaluation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    num_priv = int(len(train_set) * data_fraction)
    idxs = torch.randperm(len(train_set))[:num_priv]
    priv_subset = Subset(train_set, idxs)
    # Flatten and scale inputs
    X_priv = torch.stack([priv_subset[i][0] for i in range(len(priv_subset))], dim=0).view(num_priv, -1)
    y_priv_raw = torch.tensor([priv_subset[i][1] for i in range(len(priv_subset))], dtype=torch.float)
    X_test = torch.stack([test_set[i][0] for i in range(len(test_set))], dim=0).view(len(test_set), -1)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)
    # Scale inputs to [-1,1]
    X_priv = X_priv * 2.0 - 1.0
    X_test = X_test * 2.0 - 1.0
    # Scale targets to [-1,1]
    y_priv = (y_priv_raw - 4.5) / 4.5
    return X_priv, y_priv, X_test, y_test


def run_experiment(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Run a differentially private LASSO experiment on CIFAR‑10.

    Parameters
    ----------
    params : dict, optional
        Hyper‑parameters controlling the experiment:
          data_fraction : float   fraction of CIFAR‑10 used as private data (default 0.2).
          num_iterations : int    number of Frank–Wolfe steps (default 50).
          noise_mult : float      Gaussian noise multiplier controlling privacy (default 1.0).
          delta : float           target δ for differential privacy (default 1e-5).

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``final_accuracy``, ``epsilon``, ``history`` and ``final_loss``.
    """
    params = params or {}
    data_fraction: float = params.get("data_fraction", 0.2)
    num_iterations: int = params.get("num_iterations", 50)
    noise_mult: float = params.get("noise_mult", 1.0)
    delta: float = params.get("delta", 1e-5)

    # Load and prepare data
    X_priv, y_priv, X_test, y_test = _prepare_data(data_fraction)
    n_priv, p = X_priv.size()

    # Initialise parameter vector inside ℓ1 ball
    theta = torch.zeros(p)
    # Precompute private dataset to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_priv = X_priv.to(device)
    y_priv = y_priv.to(device)
    X_test = X_test.to(device)
    theta = theta.to(device)

    # Privacy accountant
    accnt = RDPAccountant()

    # History for tracking training loss
    history: List[Dict[str, float]] = []

    for t in range(num_iterations):
        # Compute gradient of squared loss: grad = (2/n) X^T (Xθ - y)
        preds = X_priv.mv(theta)
        residual = preds - y_priv
        grad = (2.0 / n_priv) * X_priv.t().mv(residual)
        # Add Gaussian noise to gradient for privacy (approx exponential mechanism)
        sensitivity = 2.0 / n_priv  # ℓ₂ sensitivity per coordinate
        noise_scale = noise_mult * sensitivity
        noise = torch.randn_like(grad) * noise_scale
        grad_noisy = grad + noise
        # Select coordinate with largest absolute negative gradient (descending direction)
        # We use the negative noisy gradient as utility: larger means more decrease in loss.
        scores = -grad_noisy
        idx = torch.argmax(torch.abs(scores)).item()
        # Determine sign: if gradient is positive, move in negative direction
        direction = -torch.sign(grad[idx])
        # Frank–Wolfe step size
        step_size = 2.0 / (t + 2.0)
        # Shrink all coordinates and update selected coordinate
        theta = (1.0 - step_size) * theta
        theta[idx] += step_size * direction
        # Record mean squared error on private data
        with torch.no_grad():
            mse = (residual ** 2).mean().item()
        history.append({"iteration": t, "mse": mse})
        # Update privacy accountant
        accnt.step(noise_multiplier=noise_mult, sample_rate=1.0 / n_priv)

    # Evaluate on test set: convert predictions back to class labels
    with torch.no_grad():
        y_pred = X_test.mv(theta)
        # Rescale predictions to original label range [0,9]
        y_pred_rescaled = y_pred * 4.5 + 4.5
        y_pred_clipped = torch.clamp(y_pred_rescaled, 0.0, 9.0)
        y_pred_int = torch.round(y_pred_clipped).long()
        correct = (y_pred_int.cpu() == y_test).sum().item()
        final_accuracy = 100.0 * correct / len(y_test)
        final_loss = F.mse_loss(X_test.mv(theta), y_priv.mean() * torch.ones_like(y_test, dtype=torch.float, device=device)).item()  # dummy loss

    epsilon = accnt.get_epsilon(delta)
    # Clear GPU memory after use
    clear_memory()

    return {
        "final_accuracy": float(final_accuracy),
        "epsilon": float(epsilon),
        "history": history,
        "final_loss": float(final_loss),
    }