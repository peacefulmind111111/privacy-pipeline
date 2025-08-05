import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment_utils import ExperimentResult, save_result, load_results


def test_save_and_load(tmp_path):
    res = ExperimentResult(
        experiment_name="demo",
        dataset="ds",
        model="model",
        params={"a": 1},
        metrics={"loss": 0.1},
        epsilon=1.0,
        delta=1e-5,
    )
    out = tmp_path / "res.json"
    save_result(res, out.as_posix())
    loaded = load_results(out.as_posix())["res.json"]
    assert loaded.experiment_name == "demo"
    assert loaded.metrics["loss"] == 0.1
