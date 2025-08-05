"""Utility for logging per-iteration training metrics to JSON."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List


class ExperimentLogger:
    """Collects metrics during training and writes a JSON report.

    The report schema is::

        {
            "experiment_name": str,
            "timestamp": str,
            "hyperparameters": {..},
            "iterations": [
                {
                    "iteration": int,
                    "epoch": int,
                    "loss": float,
                    "train_accuracy": float,
                    "epsilon": float,
                    "clip_value": float,
                    "grad_norm": float,
                    "lr": float,
                },
                ...
            ],
            "final_metrics": {"test_accuracy": float, "epsilon": float}
        }
    """

    def __init__(self, cfg: Any, output_dir: str) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.iterations: List[Dict[str, Any]] = []

    def log(self, iteration: int, metrics: Dict[str, Any]) -> None:
        record = {"iteration": iteration, **metrics}
        self.iterations.append(record)

    def save(self, final_metrics: Dict[str, Any]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        report = {
            "experiment_name": getattr(self.cfg, "experiment_name", "dp_experiment"),
            "timestamp": datetime.utcnow().isoformat(),
            "hyperparameters": asdict(self.cfg),
            "iterations": self.iterations,
            "final_metrics": final_metrics,
        }
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
