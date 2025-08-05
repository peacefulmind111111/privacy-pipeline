from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class LRUOrderedDict(OrderedDict):
    """OrderedDict with a maximum size implementing LRU eviction."""

    def __init__(self, *args, maxsize: int = 10000, **kwargs) -> None:
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):  # type: ignore[override]
        val = super().__getitem__(key)
        self.move_to_end(key)
        return val

    def __setitem__(self, key, val) -> None:  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, val)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def compute_per_id_momentum(
    dp_model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    idxs: torch.Tensor,
    momentum_dict: LRUOrderedDict,
    inner_momentum: float,
    device: torch.device,
    base_size: int = 50000,
) -> Tuple[List[Tuple[int, torch.Tensor]], int]:
    """Computes per-sample gradients with momentum grouped by ID."""

    dp_model.zero_grad()
    out = dp_model(X)
    loss = F.cross_entropy(out, y)
    loss.backward()

    bs = X.size(0)
    param_vecs: List[torch.Tensor | None] = [None] * bs

    for p in dp_model.parameters():
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            continue
        gs_flat = gs.view(bs, -1).detach()
        for i in range(bs):
            if param_vecs[i] is None:
                param_vecs[i] = gs_flat[i]
            else:
                param_vecs[i] = torch.cat([param_vecs[i], gs_flat[i]], dim=0)
        p.grad_sample = None

    group_dict: Dict[int, List[torch.Tensor]] = {}
    for i in range(bs):
        real_id = int(idxs[i].item()) % base_size
        v_i = param_vecs[i].to(device)  # type: ignore[index]
        group_dict.setdefault(real_id, []).append(v_i)

    results: List[Tuple[int, torch.Tensor]] = []
    for sample_id, v_list in group_dict.items():
        raw_v = torch.stack(v_list, dim=0).mean(dim=0)
        old_v = momentum_dict.get(sample_id, torch.zeros_like(raw_v))
        old_v = old_v.to(device)
        new_v = (inner_momentum * old_v) + ((1.0 - inner_momentum) * raw_v)
        momentum_dict[sample_id] = new_v.half().cpu()
        results.append((sample_id, new_v))
    return results, len(group_dict)
