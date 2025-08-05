from __future__ import annotations

import json
import os
import time
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any


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
    sample_rate: float | None = None
    mixup_alpha: float | None = None
    lasso_alpha: float = 0.01
    comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def current_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
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
