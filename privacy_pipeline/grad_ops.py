from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


def clip_sum_noise(
    batch_v: Sequence[Tuple[int, torch.Tensor]],
    clip_val: float,
    noise_mult: float,
    do_noise: bool = True,
    return_postclip: bool = False,
) -> Tuple[torch.Tensor, List[float] | None]:
    """Clips, sums and adds noise to gradients."""
    clipped_list = []
    postclip_norms: List[float] | None = [] if return_postclip else None
    for (_, g) in batch_v:
        norm_g = g.norm(2)
        if norm_g > clip_val:
            g = g * (clip_val / (norm_g + 1e-9))
        if return_postclip and postclip_norms is not None:
            postclip_norms.append(g.norm(2).item())
        clipped_list.append(g)

    grad_sum = torch.stack(clipped_list, dim=0).sum(dim=0)
    count = float(len(clipped_list))

    if do_noise:
        noise = torch.randn_like(grad_sum) * (noise_mult * clip_val)
        grad_sum += noise

    final_grad = grad_sum / count

    if return_postclip:
        return final_grad, postclip_norms
    return final_grad, None


def clip_sum_noise_per_sample(
    vecs: Sequence[torch.Tensor],
    clip_vals: Sequence[float],
    noise_mult: float,
    do_noise: bool = True,
) -> torch.Tensor:
    """Single-sum clipping with per-sample clip values and optional noise."""
    grad_sum = torch.stack(list(vecs), dim=0).sum(dim=0)
    total_clip = float(sum(clip_vals))
    norm_ = grad_sum.norm(2)
    if norm_ > total_clip:
        grad_sum = grad_sum * (total_clip / (norm_ + 1e-9))
    if do_noise and clip_vals:
        cmax = max(clip_vals)
        grad_sum += torch.randn_like(grad_sum) * (noise_mult * cmax)
    return grad_sum / float(len(clip_vals))


def outer_step(dp_model: torch.nn.Module, optimizer, final_grad: torch.Tensor) -> None:
    idx_start = 0
    for p in dp_model.parameters():
        numel = p.numel()
        chunk = final_grad[idx_start : idx_start + numel]
        p.grad = chunk.view_as(p)
        idx_start += numel
    optimizer.step()


def measure_distribution(vecs: Iterable[torch.Tensor]) -> Tuple[List[float], List[float], List[float]]:
    """Returns L1, L2 and cosine similarity with mean for each vector."""
    vecs_list = list(vecs)
    if not vecs_list:
        return [], [], []
    stacked = torch.stack(vecs_list, dim=0)
    mean_vec = stacked.mean(dim=0)
    mean_norm = mean_vec.norm(2).item() + 1e-9

    l1_list: List[float] = []
    l2_list: List[float] = []
    cos_list: List[float] = []
    for v in vecs_list:
        l1_list.append(v.norm(p=1).item())
        l2_list.append(v.norm(p=2).item())
        dot_val = (v * mean_vec).sum().item()
        denom = v.norm(2).item() * mean_norm + 1e-9
        cos_list.append(dot_val / denom)
    return l1_list, l2_list, cos_list


def plot_hist(data: Sequence[float], title: str, bins: int = 50, color: str = "blue", path: str | None = None) -> None:
    """Plots a histogram of the provided data and saves to ``path`` if given."""
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.7, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
