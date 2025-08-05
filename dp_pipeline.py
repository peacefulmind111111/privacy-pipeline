"""DP training utilities."""
from dataclasses import dataclass
from typing import Optional

from opacus.accountants.utils import get_noise_multiplier


@dataclass
class DPCfg:
    sample_rate: float
    epochs: int
    delta: float
    accountant: str = "rdp"
    epsilon: Optional[float] = None
    noise_multiplier: Optional[float] = None


def prepare_sigma(cfg: DPCfg, steps: int) -> float:
    """Return a valid noise multiplier for DP-SGD.

    If ``cfg.noise_multiplier`` is provided it is returned directly. Otherwise
    we compute it from ``cfg.epsilon`` using :func:`get_noise_multiplier`.
    """
    if cfg.noise_multiplier is not None:
        return cfg.noise_multiplier

    if cfg.epsilon is None:
        raise ValueError(
            "cfg.epsilon must be set when noise_multiplier is None"
        )

    sigma = get_noise_multiplier(
        target_epsilon=cfg.epsilon,
        target_delta=cfg.delta,
        sample_rate=cfg.sample_rate,
        epochs=cfg.epochs,
        steps=steps,
        accountant=cfg.accountant,
    )
    return sigma
