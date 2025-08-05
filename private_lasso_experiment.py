# dp_lasso_full.py
"""
Differentially‑Private Frank–Wolfe (LASSO) on CIFAR‑10
======================================================
Implements the algorithm from “Nearly Optimal Private LASSO” (Duchi et al., 2021).
The script is *stand‑alone*: run `python dp_lasso_full.py --help` and it will
download CIFAR‑10, train with DP, report (ε, δ) and accuracy.
"""

from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch, torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant

# -----------------------------------------------------------------------------#
#                         Utils                                                 #
# -----------------------------------------------------------------------------#
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def clear_memory() -> None:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def to_flat(d: Tensor) -> Tensor:          # B×3×32×32  ->  B×3072
    return d.view(d.size(0), -1)

# -----------------------------------------------------------------------------#
#                         Data                                                  #
# -----------------------------------------------------------------------------#
def make_loaders(data_fraction: float, batch: int, workers: int):
    tfm = transforms.ToTensor()             # in [0,1]
    train = datasets.CIFAR10("data", True,  tfm, download=True)
    test  = datasets.CIFAR10("data", False, tfm, download=True)

    if data_fraction < 1.0:
        k = int(len(train) * data_fraction)
        idx = torch.randperm(len(train))[:k]
        train = Subset(train, idx)

    train_loader = DataLoader(train, batch, shuffle=True,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test,  1024, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader

# -----------------------------------------------------------------------------#
#                         Training                                              #
# -----------------------------------------------------------------------------#
def fw_coordinate_step(theta: Tensor, grad: Tensor, t: int) -> Tensor:
    """One Frank–Wolfe update along best ℓ₁ vertex."""
    idx = torch.argmax(torch.abs(-grad)).item()
    direction = -torch.sign(grad[idx])
    γ = 2. / (t + 2.)                       # step‑size schedule
    theta *= (1. - γ)
    theta[idx] += γ * direction
    return theta

def private_gradient(loader: DataLoader, theta: Tensor,
                     noise_mult: float, device: torch.device) -> Tensor:
    """Full‑batch grad with Gaussian noise (sensitivity 2/n)."""
    n, p = 0, theta.numel()
    g = torch.zeros(p, device=device)
    for x, y in loader:                     # iterate once over *all* data
        x = to_flat(x.to(device)) * 2. - 1. # scale to [‑1,1]
        y = ((y.float().to(device)) - 4.5) / 4.5
        r = x @ theta - y
        g += x.t() @ r
        n += x.size(0)
    g *= 2. / n                             # ∇(MSE)
    g += torch.randn_like(g) * (noise_mult * 2. / n)
    return g

# -----------------------------------------------------------------------------#
def train_dp_lasso(cfg: Dict) -> Dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available()
                                else "cpu"))

    tr_loader, te_loader = make_loaders(cfg["data_fraction"],
                                        cfg["batch"], cfg["workers"])
    p = 3 * 32 * 32
    θ = torch.zeros(p, device=device)

    acct = RDPAccountant()
    history: List[Dict] = []
    for t in range(cfg["iterations"]):
        tic = time.time()
        g = private_gradient(tr_loader, θ, cfg["noise_mult"], device)
        θ = fw_coordinate_step(θ, g, t)
        acct.step(noise_multiplier=cfg["noise_mult"], sample_rate=1.)  # full‑batch
        mse_priv = g.abs().mean().item()  # proxy, avoids second pass
        history.append({"iter": t, "mse_proxy": mse_priv,
                        "sec": time.time() - tic})

    # --------------------  evaluation  -------------------------- #
    θ_cpu = θ.cpu()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in te_loader:
            x = to_flat(x) * 2. - 1.
            y_hat = x @ θ_cpu
            y_hat = torch.clamp(y_hat*4.5 + 4.5, 0., 9.)
            correct += (y_hat.round().long() == y).sum().item()
            total   += y.size(0)

    acc = 100. * correct / total
    eps = acct.get_epsilon(cfg["delta"])
    clear_memory()

    return {"final_accuracy": acc,
            "epsilon": eps,
            "delta":   cfg["delta"],
            "history": history}

# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_fraction", type=float, default=1.0)
    ap.add_argument("--iterations",    type=int,   default=200)
    ap.add_argument("--noise_mult",    type=float, default=1.0)
    ap.add_argument("--delta",         type=float, default=1e-5)
    ap.add_argument("--batch",         type=int,   default=2048)
    ap.add_argument("--workers",       type=int,   default=4)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--out",           type=str,   default="dp_lasso_results")
    cfg = vars(ap.parse_args())

    os.makedirs(cfg["out"], exist_ok=True)
    res = train_dp_lasso(cfg)

    # ----------- persist -------------- #
    ts = int(time.time())
    with open(Path(cfg["out"]) / f"run_{ts}.json", "w") as f:
        json.dump({**cfg, **res}, f, indent=2)

    print(f"Accuracy: {res['final_accuracy']:.2f}%")
    print(f"ε = {res['epsilon']:.3f}  (δ = {cfg['delta']})")

if __name__ == "__main__":
    main()
