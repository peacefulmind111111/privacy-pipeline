import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus.accountants import RDPAccountant

from .config import ExperimentConfig
from .data import get_dataloaders, set_random_seeds
from .grad_ops import (
    clip_sum_noise,
    clip_sum_noise_per_sample,
    measure_distribution,
    outer_step,
    plot_hist,
)
from .momentum import LRUOrderedDict, compute_per_id_momentum
from .model import build_model, evaluate


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def train(cfg: ExperimentConfig, output_dir: str, plot_every: int = 5) -> Dict[str, List[Dict[str, float]]]:
    """Runs a DP experiment and stores metrics in ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    set_random_seeds(cfg.seed)
    device = cfg.device

    train_loader, test_loader = get_dataloaders(cfg)
    dp_net = build_model(device).to(device)
    optimizer = optim.SGD(
        dp_net.parameters(),
        lr=cfg.lr,
        momentum=cfg.outer_momentum,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.schedule_milestones, gamma=cfg.schedule_gamma
    )
    accountant = RDPAccountant()
    momentum_dict = LRUOrderedDict(maxsize=cfg.max_momentum_size)

    total_steps = cfg.num_epochs * len(train_loader)
    step_count = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        dp_net.train()
        losses: List[float] = []
        epoch_preclip_l1: List[float] = []
        epoch_preclip_l2: List[float] = []
        epoch_preclip_cos: List[float] = []
        epoch_postclip_l1: List[float] = []
        epoch_postclip_l2: List[float] = []
        epoch_postclip_cos: List[float] = []

        for X, y, idxs in train_loader:
            step_count += 1
            frac = step_count / float(total_steps)
            clip_val = cfg.c_start + (cfg.c_end - cfg.c_start) * frac

            X, y = X.to(device), y.to(device)
            batch_v, unique_count = compute_per_id_momentum(
                dp_net, X, y, idxs, momentum_dict, cfg.inner_momentum, device
            )
            preclip_vecs = [g for (_, g) in batch_v]
            pre_l1, pre_l2, pre_cos = measure_distribution(preclip_vecs)
            epoch_preclip_l1.extend(pre_l1)
            epoch_preclip_l2.extend(pre_l2)
            epoch_preclip_cos.extend(pre_cos)

            final_grad, _ = clip_sum_noise(
                batch_v, clip_val, cfg.noise_mult, do_noise=True, return_postclip=True
            )
            clipped_vecs = []
            for (_, raw_v) in batch_v:
                nm = raw_v.norm(2).item()
                if nm > clip_val:
                    clipped_v = raw_v * (clip_val / (nm + 1e-9))
                else:
                    clipped_v = raw_v
                clipped_vecs.append(clipped_v)
            post_l1, post_l2, post_cos = measure_distribution(clipped_vecs)
            epoch_postclip_l1.extend(post_l1)
            epoch_postclip_l2.extend(post_l2)
            epoch_postclip_cos.extend(post_cos)

            grad_norm = final_grad.norm(2).item()
            outer_step(dp_net, optimizer, final_grad)

            with torch.no_grad():
                out = dp_net(X)
                loss_val = F.cross_entropy(out, y)
                batch_acc = (out.argmax(dim=1) == y).float().mean().item() * 100.0
            losses.append(loss_val.item())

            if step_count % cfg.log_every == 0:
                print(
                    f"    Step {step_count:05d}/{total_steps} "
                    f"loss={loss_val.item():.3f} acc={batch_acc:.2f}% grad_norm={grad_norm:.2f}"
                )

            sample_rate = unique_count / 50000.0
            accountant.step(noise_multiplier=cfg.noise_mult, sample_rate=sample_rate)

        scheduler.step()
        acc = evaluate(dp_net, test_loader, device)
        eps = accountant.get_epsilon(cfg.delta)

        epoch_record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(acc),
            "epsilon": float(eps),
            "preclip_l1": _summary_stats(epoch_preclip_l1),
            "preclip_l2": _summary_stats(epoch_preclip_l2),
            "preclip_cos": _summary_stats(epoch_preclip_cos),
            "postclip_l1": _summary_stats(epoch_postclip_l1),
            "postclip_l2": _summary_stats(epoch_postclip_l2),
            "postclip_cos": _summary_stats(epoch_postclip_cos),
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] clip~{clip_val:.2f} Loss={epoch_record['loss']:.3f} Acc={acc:.2f}% eps={eps:.2f}"
        )

        if plot_every and epoch % plot_every == 0:
            plot_hist(
                epoch_preclip_l1,
                title=f"Pre-clip L1 norms (epoch={epoch})",
                color="blue",
                path=os.path.join(output_dir, f"preclip_l1_epoch{epoch}.png"),
            )
            plot_hist(
                epoch_preclip_l2,
                title=f"Pre-clip L2 norms (epoch={epoch})",
                color="green",
                path=os.path.join(output_dir, f"preclip_l2_epoch{epoch}.png"),
            )
            plot_hist(
                epoch_preclip_cos,
                title=f"Pre-clip cos sim vs. mean (epoch={epoch})",
                color="red",
                path=os.path.join(output_dir, f"preclip_cos_epoch{epoch}.png"),
            )
            plot_hist(
                epoch_postclip_l1,
                title=f"Post-clip L1 norms (epoch={epoch})",
                color="blue",
                path=os.path.join(output_dir, f"postclip_l1_epoch{epoch}.png"),
            )
            plot_hist(
                epoch_postclip_l2,
                title=f"Post-clip L2 norms (epoch={epoch})",
                color="green",
                path=os.path.join(output_dir, f"postclip_l2_epoch{epoch}.png"),
            )
            plot_hist(
                epoch_postclip_cos,
                title=f"Post-clip cos sim vs. mean (epoch={epoch})",
                color="red",
                path=os.path.join(output_dir, f"postclip_cos_epoch{epoch}.png"),
            )

    final_acc = evaluate(dp_net, test_loader, device)
    final_eps = accountant.get_epsilon(cfg.delta)
    print(f"\nDone. Final test Acc={final_acc:.2f}% Eps={final_eps:.2f}")

    results = {
        "history": history,
        "final_accuracy": float(final_acc),
        "final_epsilon": float(final_eps),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def train_with_outlier_clipping(
    cfg: ExperimentConfig, output_dir: str
) -> Dict[str, List[Dict[str, float]]]:
    """Runs an experiment using per-sample outlier clipping."""
    os.makedirs(output_dir, exist_ok=True)

    set_random_seeds(cfg.seed)
    device = cfg.device

    train_loader, test_loader = get_dataloaders(cfg)
    base_size = len(train_loader.dataset) // cfg.self_aug_factor
    dp_net = build_model(device).to(device)
    optimizer = optim.SGD(
        dp_net.parameters(),
        lr=cfg.lr,
        momentum=cfg.outer_momentum,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.schedule_milestones, gamma=cfg.schedule_gamma
    )
    accountant = RDPAccountant()
    momentum_dict = LRUOrderedDict(maxsize=cfg.max_momentum_size)

    sample_correct = np.zeros(base_size, dtype=int)
    sample_total = np.zeros(base_size, dtype=int)

    total_steps = cfg.num_epochs * len(train_loader)
    step_count = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        dp_net.train()
        losses: List[float] = []
        for X, y, idxs in train_loader:
            step_count += 1
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                preds = dp_net(X).argmax(dim=1)
            for i, sidx in enumerate(idxs):
                sid = int(sidx.item()) % base_size
                sample_total[sid] += 1
                sample_correct[sid] += int(preds[i].item() == y[i].item())

            batch_v, unique_count = compute_per_id_momentum(
                dp_net, X, y, idxs, momentum_dict, cfg.inner_momentum, device, base_size
            )
            vecs = [g for (_, g) in batch_v]
            if step_count <= cfg.drop_after_frac * total_steps:
                clip_vals = [cfg.default_clip] * len(vecs)
            else:
                clip_vals = []
                for sid, _ in batch_v:
                    tot = sample_total[sid]
                    corr = sample_correct[sid]
                    err_rate = 1.0 - (corr / tot) if tot > 0 else 1.0
                    if err_rate >= cfg.high_err_threshold:
                        clip_vals.append(cfg.outlier_clip)
                    else:
                        clip_vals.append(cfg.default_clip)

            final_grad = clip_sum_noise_per_sample(
                vecs, clip_vals, cfg.noise_mult, do_noise=True
            )
            grad_norm = final_grad.norm(2).item()
            outer_step(dp_net, optimizer, final_grad)

            with torch.no_grad():
                out = dp_net(X)
                loss_val = F.cross_entropy(out, y)
                batch_acc = (out.argmax(dim=1) == y).float().mean().item() * 100.0
            losses.append(loss_val.item())

            if step_count % cfg.log_every == 0:
                print(
                    f"    Step {step_count:05d}/{total_steps} "
                    f"loss={loss_val.item():.3f} acc={batch_acc:.2f}% grad_norm={grad_norm:.2f}"
                )

            sample_rate = unique_count / float(base_size)
            accountant.step(noise_multiplier=cfg.noise_mult, sample_rate=sample_rate)

        scheduler.step()
        acc = evaluate(dp_net, test_loader, device)
        eps = accountant.get_epsilon(cfg.delta)
        epoch_record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(acc),
            "epsilon": float(eps),
        }
        history.append(epoch_record)
        print(
            f"[Epoch {epoch:02d}] Loss={epoch_record['loss']:.3f} Acc={acc:.2f}% eps={eps:.2f}"
        )

    final_acc = evaluate(dp_net, test_loader, device)
    final_eps = accountant.get_epsilon(cfg.delta)
    print(
        f"\nDone. (Outlier clipping) Final test Acc={final_acc:.2f}% eps={final_eps:.2f}"
    )
    results = {
        "history": history,
        "final_accuracy": float(final_acc),
        "final_epsilon": float(final_eps),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
