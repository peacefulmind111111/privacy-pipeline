import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


@dataclass
class ExperimentResult:
    """Container for standardized experiment results."""
    experiment_name: str
    dataset: str
    model: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    epsilon: float
    delta: float
    timestamp: datetime = datetime.utcnow()

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


def save_result(result: ExperimentResult, output_path: str) -> Path:
    """Save an ExperimentResult to ``output_path`` as JSON.

    The directory is created if necessary. The function returns the
    :class:`pathlib.Path` to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_json(), f, indent=2, default=_default)
    return path


def load_results(pattern: str) -> Dict[str, ExperimentResult]:
    """Load multiple results matching ``pattern`` (glob).

    The ``pattern`` can be relative or absolute and may include wildcards.
    Returns a mapping from filename to :class:`ExperimentResult` objects."""
    import glob

    loaded: Dict[str, ExperimentResult] = {}
    for name in glob.glob(pattern):
        fp = Path(name)
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        loaded[fp.name] = ExperimentResult(
            experiment_name=data["experiment_name"],
            dataset=data["dataset"],
            model=data["model"],
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            epsilon=data.get("epsilon", 0.0),
            delta=data.get("delta", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    return loaded
