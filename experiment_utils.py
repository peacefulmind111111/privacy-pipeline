import os
import json
import time
import platform
import subprocess
import gc
import torch


def _git_commit() -> str | None:
    """Return current git commit hash if available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def save_json(
    experiment: str,
    params: dict | None,
    history: list[dict],
    final_metrics: dict,
    output_dir: str | None = None,
    filename: str | None = None,
    extra_metadata: dict | None = None,
):
    """Persist results to a standardized JSON file and return the record."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment}_{ts}"

    metadata = {
        "timestamp": ts,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    record = {
        "run_id": run_id,
        "experiment": experiment,
        "parameters": params or {},
        "history": history,
        "final_metrics": final_metrics,
        "metadata": metadata,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = filename or f"{run_id}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    return record


def clear_memory() -> None:
    """Best-effort memory cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
