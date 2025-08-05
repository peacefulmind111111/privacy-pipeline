##############################################################
# Filename: dp_virtual_projection_population_only.py
# (2025-07-03 patched version)
##############################################################
import os, psutil, random, math, time, statistics, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataclasses import dataclass, asdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Opacus
from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule

###############################################################################
# 0. Hyperparameter dataclass
###############################################################################


@dataclass
class ExperimentConfig:
    seed: int = 0
    batch_size: int = 1000
    lr: float = 0.1
    outer_momentum: float = 0.9
    inner_momentum: float = 0.0
    noise_mult: float = 0.0  # no DP noise in this script
    delta: float = 1e-5
    num_epochs: int = 10  # ← raise above warm_start_epochs for real runs
    M: int = 0  # momentum dictionary capacity
    self_aug_factor: int = 1
    # Projection experiment knobs
    warm_start_epochs: int = 2  # plain-SGD epochs before any projection
    proj_lr_multiplier: float = 3.0  # bump LR when projection begins
    scale_pprime_to_pL2: bool = True  # re-norm p′ so ‖p′‖₂ = ‖p‖₂
    trust_mix_alpha: float = 1.0  # 0→raw p ; 1→pure p′
    rebuild_basis_every: int = 3  # epochs; 0→never rebuild
    # Virtual-sample parameters
    subsets_per_class: int = 25
    subset_size: int = 200
    virtual_augment_factor: int = 10
    # “No-clipping” setup (huge C so clipping never triggers)
    c_start: float = 1e9
    c_end: float = 1e9


DEFAULT_CONFIG = ExperimentConfig()
for _k, _v in asdict(DEFAULT_CONFIG).items():
    globals()[_k] = _v

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_params(params: dict | ExperimentConfig | None) -> None:
    """Override module-level hyperparameters with ``params``."""
    if not params:
        return
    if isinstance(params, ExperimentConfig):
        params = asdict(params)
    for k, v in params.items():
        if k in globals():
            globals()[k] = v


DEFAULT_PARAMS = asdict(DEFAULT_CONFIG)

###############################################################################
# 0.1 Memory helper
###############################################################################
def print_memory_usage(msg=""):
    try:
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"[MEM] {msg} → {mem_mb:,.1f} MB")
    except Exception:
        pass

###############################################################################
# 1. Dataset
###############################################################################
mean, std = [0.4914,0.4822,0.4466], [0.2470,0.2435,0.2616]  # tiny typo fix
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.2)),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

train_ds = datasets.CIFAR10(root="./data", train=True , download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

class IndexedSubset(Subset):
    """Return (x, y, idx) for momentum tracking."""
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, idx

n = len(train_ds)
train_sub  = IndexedSubset(train_ds, range(n))
train_full = ConcatDataset([train_sub]*self_aug_factor)

train_loader = DataLoader(train_full,batch_size=batch_size,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_ds ,batch_size=batch_size,shuffle=False)

###############################################################################
# 2. ResNet-20 (GroupNorm)
###############################################################################
class BasicBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,3,stride,1,bias=False)
        self.gn1   = nn.GroupNorm(8,planes)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1,bias=False)
        self.gn2   = nn.GroupNorm(8,planes)
        self.short = nn.Sequential()
        if stride!=1 or in_planes!=planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes,planes,1,stride,bias=False),
                nn.GroupNorm(8,planes)
            )
    def forward(self,x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)

class ResNet20(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3,16,3,1,1,bias=False)
        self.gn1   = nn.GroupNorm(8,16)
        self.layer1 = self._make_layer(16,3,1)
        self.layer2 = self._make_layer(32,3,2)
        self.layer3 = self._make_layer(64,3,2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64,num_classes)
    def _make_layer(self,planes,blocks,stride):
        strides=[stride]+[1]*(blocks-1); layers=[]
        for s in strides:
            layers.append(BasicBlock(self.in_planes,planes,s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self,x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x); x=self.layer2(x); x=self.layer3(x)
        x = self.avgpool(x); x=torch.flatten(x,1)
        return self.fc(x)

def evaluate(model,loader):
    was_training = model.training
    model.eval(); correct=total=0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            correct += (preds==y).sum().item(); total += y.size(0)
    if was_training: model.train()
    return 100.*correct/total

###############################################################################
# 3. Build DP-ready network
###############################################################################
def build_model():
    net = ResNet20().to(device)
    if ModuleValidator.validate(net,strict=False):
        net = ModuleValidator.fix(net).to(device)
    return GradSampleModule(net)

###############################################################################
# 4. LRU-momentum dict
###############################################################################
class LRUOrderedDict(OrderedDict):
    def __init__(self,*a,maxsize=10_000,**kw): self.maxsize=maxsize; super().__init__(*a,**kw)
    def __getitem__(self,key):
        val=super().__getitem__(key); self.move_to_end(key); return val
    def __setitem__(self,key,val):
        if key in self: self.move_to_end(key)
        super().__setitem__(key,val)
        if len(self)>self.maxsize: self.popitem(last=False)

momentum_dict = LRUOrderedDict(maxsize=M)

###############################################################################
# 5. Per-sample momentum vector
###############################################################################
def compute_inner_momentum_grads_idxed(dp_model,X,y,idxs):
    dp_model.zero_grad()
    loss = F.cross_entropy(dp_model(X),y); loss.backward()
    bs = X.size(0); param_vecs=[None]*bs
    for p in dp_model.parameters():
        gs = getattr(p,"grad_sample",None)
        if gs is None: continue
        gs_flat = gs.view(bs,-1).detach()
        for i in range(bs):
            param_vecs[i] = gs_flat[i] if param_vecs[i] is None \
                            else torch.cat([param_vecs[i],gs_flat[i]],0)
        p.grad_sample=None
    sample_to_vecs={}
    for i in range(bs):
        sid = int(idxs[i]); sample_to_vecs.setdefault(sid,[]).append(param_vecs[i].to(device))
    batch_v=[]
    for sid,vecs in sample_to_vecs.items():
        g_i = torch.stack(vecs,0).mean(0)
        old_v = momentum_dict[sid].to(device) if sid in momentum_dict \
                 else torch.zeros_like(g_i)
        new_v = inner_momentum*old_v + (1-inner_momentum)*g_i
        momentum_dict[sid] = new_v.half().cpu();  batch_v.append(new_v)
    return batch_v, len(sample_to_vecs)

###############################################################################
# 6. Outer step (no clipping/no noise)
###############################################################################
def outer_step_noDP(dp_model,optimizer,batch_v):
    grad_sum = torch.stack(batch_v,0).sum(0)
    final_grad = grad_sum/len(batch_v)
    idx=0
    for p in dp_model.parameters():
        numel=p.numel(); p.grad=final_grad[idx:idx+numel].view_as(p); idx+=numel
    optimizer.step()

###############################################################################
# 7. Virtual subsets helpers
###############################################################################
def build_virtual_subsets(dataset,subsets_per_class,subset_size,augment_factor):
    cls_idx=[[] for _ in range(10)]
    for i in range(len(dataset)): _,c=dataset[i]; cls_idx[c].append(i)
    v_subsets=[]
    for c in range(10):
        random.shuffle(cls_idx[c])
        for k in range(subsets_per_class):
            subset=cls_idx[c][k*subset_size:(k+1)*subset_size]
            for _ in range(augment_factor): v_subsets.append((subset,c))
    return v_subsets

def compute_fullmodel_grad_for_subset(model,subset_idxs,dataset):
    model.zero_grad()
    X=torch.stack([dataset[i][0] for i in subset_idxs]).to(device)
    y=torch.tensor([dataset[i][1] for i in subset_idxs],device=device)
    loss = F.cross_entropy(model(X),y); loss.backward()
    parts=[p.grad.view(-1).detach() for p in model.parameters() if p.grad is not None]
    return torch.cat(parts).to(device)/len(subset_idxs)

###############################################################################
# 8. Gram–Schmidt
###############################################################################
def gram_schmidt(vecs,eps=1e-10):
    basis=[]
    for v in vecs:
        w=v.clone().float().to(device)
        for b in basis: w -= torch.dot(w,b)*b
        n=w.norm()
        if n>eps: basis.append(w/n)
    return basis

def build_e_basis_for(model):
    """Compute Gram–Schmidt-orthonormalised gradients of all virtual subsets
       using the *current* model parameters."""
    v_subsets = build_virtual_subsets(
        train_ds,
        subsets_per_class,
        subset_size,
        virtual_augment_factor,
    )
    e_list = [
        compute_fullmodel_grad_for_subset(model, idxs, train_ds)
        for idxs, _ in v_subsets
    ]
    return gram_schmidt(e_list)

###############################################################################
# 9. Unwrapped population grad
###############################################################################
def compute_population_grad_unwrapped(dp_model,X,y):
    net = dp_model._module
    net.zero_grad()
    F.cross_entropy(net(X),y).backward()
    parts=[p.grad.view(-1).detach() for p in net.parameters() if p.grad is not None]
    return torch.cat(parts).to(device) / X.size(0)

###############################################################################
# 10-A. Training loop
###############################################################################
def run_training(dp_net,
                 optimizer,
                 want_projection,
                 e_basis,
                 project,
                 train_loader,
                 test_loader,
                 diag_every:int=50):

    epoch_losses, epoch_accs = [], []
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    momentum_dict.clear()

    # <<< NEW -----------------------------------------------------------------
    # Local trackers to diagnose the projection run as well
    local_diff_L2, local_cos = [], []
    # -------------------------------------------------------------------------

    for epoch in range(1, num_epochs + 1):
        dp_net.train()
        losses, recent_losses = [], []
        epoch_start = time.time()

        # ---- experiment-level toggles -------------------------------------
        use_proj_this_epoch = want_projection and (epoch > warm_start_epochs)

        # (new) after warm-start, rebuild ONCE so the basis matches current weights
        if epoch == warm_start_epochs and want_projection:
            e_basis[:] = build_e_basis_for(dp_net._module)

        # optional periodic refresh
        if rebuild_basis_every and (epoch % rebuild_basis_every == 0):
            e_basis[:] = build_e_basis_for(dp_net._module)

        if epoch == warm_start_epochs + 1 and proj_lr_multiplier != 1.0 and want_projection:
            for pg in optimizer.param_groups:
                pg["lr"] *= proj_lr_multiplier
        if rebuild_basis_every and (epoch % rebuild_basis_every == 0):
            e_basis[:] = gram_schmidt(e_basis)  # quick refresh
        # -------------------------------------------------------------------

        total_batches = len(train_loader)
        for batch_idx, (X, y, idxs) in enumerate(train_loader, 1):
            X, y = X.to(device), y.to(device)

            # ---- forward / backward ---------------------------------------
            batch_v, _ = compute_inner_momentum_grads_idxed(dp_net, X, y, idxs)
            p          = compute_population_grad_unwrapped(dp_net, X, y)
            p_prime    = project(p, e_basis)

            # <<< NEW: dimensionality guard ---------------------------------
            assert p_prime.shape == p.shape, \
                f"[Bug] p′ dim {p_prime.shape} ≠ p dim {p.shape}"
            # ----------------------------------------------------------------

            if scale_pprime_to_pL2 and p_prime.norm() > 0:
                p_prime = p_prime * (p.norm() / p_prime.norm())

            g_update  = trust_mix_alpha * p_prime + (1 - trust_mix_alpha) * p
            # ----------------------------------------------------------------

            # choose update direction ---------------------------------------
            if use_proj_this_epoch:
                idx = 0
                for p_param in dp_net.parameters():
                    n = p_param.numel()
                    p_param.grad = g_update[idx: idx + n].view_as(p_param)
                    idx += n
                optimizer.step()
            else:
                outer_step_noDP(dp_net, optimizer, batch_v)
            # ----------------------------------------------------------------

            # <<< NEW: on-the-fly projection diagnostics (always on) --------
            with torch.no_grad():
                diff = (p - p_prime).norm().item()
                cos  = torch.dot(p, p_prime).item() / (
                       p.norm().item() * p_prime.norm().item() + 1e-12)
                local_diff_L2.append(diff)
                local_cos.append(cos)
            # ----------------------------------------------------------------

            # diagnostics (loss etc.) ---------------------------------------
            with torch.no_grad():
                batch_loss = F.cross_entropy(dp_net(X), y).item()
            losses.append(batch_loss)
            recent_losses.append(batch_loss)
            if len(recent_losses) > 10:
                recent_losses.pop(0)

            if batch_idx % diag_every == 0 or batch_idx == total_batches:
                secs_per_batch = (time.time() - epoch_start) / batch_idx
                tag = "ProjSGD" if use_proj_this_epoch else "SGD"
                fmt = (f"[{tag}] Epoch {epoch}/{num_epochs}  "
                       f"Batch {batch_idx:>4}/{total_batches}  "
                       f"η≈{secs_per_batch:.2f}s  "
                       f"run-loss={statistics.mean(recent_losses):.4f}  "
                       f"‖p-p′‖₂={diff:.4f}  cos={cos:.4f}")
                print_memory_usage(fmt)
        # -------------------------------------------------------------------
        scheduler.step()
        epoch_losses.append(np.mean(losses))
        epoch_accs.append(evaluate(dp_net, test_loader))

        minutes = (time.time() - epoch_start) / 60
        tag = "ProjSGD" if use_proj_this_epoch else "SGD"
        print(f"[{tag}] Epoch {epoch} done  "
              f"loss={epoch_losses[-1]:.3f}  "
              f"acc={epoch_accs[-1]:.2f}%  "
              f"({minutes:.1f} min)\n")

    # <<< NEW: return extra diagnostics -------------------------------------
    return {"loss": epoch_losses,
            "acc": epoch_accs,
            "diff_L2": local_diff_L2,
            "cos": local_cos}
    # -----------------------------------------------------------------------

###############################################################################
# 10-B. main_run
###############################################################################
def main_run(params: dict | None = None, output_path: str | None = None):
    """Run the virtual projection experiment.

    Parameters
    ----------
    params: dict, optional
        Hyperparameters to override at runtime.
    output_path: str, optional
        Where to save a JSON report of metrics. If ``None`` the report is
        returned but not written to disk.
    """
    apply_params(params)
    print_memory_usage("Start")

    # A) build virtual subsets
    v_subsets = build_virtual_subsets(
        train_ds, subsets_per_class, subset_size, virtual_augment_factor
    )
    print(f"Built {len(v_subsets)} virtual subsets.")

    # B) compute e_i for every virtual subset
    base_net = ResNet20().to(device).eval()
    e_list = [compute_fullmodel_grad_for_subset(base_net, idxs, train_ds)
              for idxs, _ in v_subsets]
    e_basis = gram_schmidt(e_list)
    print(f"Orthonormal basis size = {len(e_basis)}/{len(e_list)}")

    def project_onto(vec, basis):
        out = torch.zeros_like(vec)
        for b in basis:
            out += torch.dot(vec, b) * b
        return out

    # ProjSGD (after warm-start)
    dp_proj = build_model().to(device)
    opt_proj = optim.SGD(
        dp_proj.parameters(), lr=lr, momentum=outer_momentum, weight_decay=5e-4
    )
    proj_stats = run_training(
        dp_proj,
        opt_proj,
        True,
        e_basis,
        project_onto,
        train_loader,
        test_loader,
    )

    # Quick stats -----------------------------------------------------------
    mean_diff = float(np.mean(proj_stats["diff_L2"]))
    mean_cos = float(np.mean(proj_stats["cos"]))
    final_acc = float(proj_stats["acc"][-1])
    print("\nPopulation-level stats (Proj run):")
    print(f"  mean‖p − p′‖₂ = {mean_diff:.4f}")
    print(f"  mean cos(p,p′) = {mean_cos:.4f}")
    print("\nFinal accuracy:")
    print(f"  ProjSGD p′: {final_acc:.2f}%")

    history = [
        {"epoch": i + 1, "loss": float(l), "accuracy": float(a)}
        for i, (l, a) in enumerate(zip(proj_stats["loss"], proj_stats["acc"]))
    ]
    final_loss = float(proj_stats["loss"][-1]) if proj_stats["loss"] else None
    results = {
        "experiment_name": "dp_virtual_projection",
        "hyperparameters": params or {},
        "history": history,
        "final_metrics": {
            "accuracy": final_acc,
            "mean_diff_L2": mean_diff,
            "mean_cos": mean_cos,
            "final_loss": final_loss,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results

# -----------------------------------------------------------------------------
def train(
    cfg: ExperimentConfig | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Convenience wrapper accepting a dataclass config and output directory."""
    cfg = cfg or ExperimentConfig()
    params = asdict(cfg)
    path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = filename or f"dp_virtual_projection_{ts}.json"
        path = os.path.join(output_dir, fname)
    return main_run(params=params, output_path=path)

###############################################################################
if __name__ == "__main__":
    train(ExperimentConfig(), "outputs")

