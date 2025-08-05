import json
from dp_pipeline import ExperimentConfig, save_results_json, current_timestamp


def test_experiment_config_to_dict():
    config = ExperimentConfig(experiment_name="test", method="sgd")
    cfg_dict = config.to_dict()
    assert cfg_dict["experiment_name"] == "test"
    assert cfg_dict["method"] == "sgd"


def test_save_results_json(tmp_path):
    results = {"experiment_name": "exp", "value": 1}
    file_path = save_results_json(results, tmp_path)
    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == results


def test_current_timestamp_format():
    ts = current_timestamp()
    # ISO 8601 with Z suffix for UTC
    assert ts.endswith("Z")
    # Ensure string can be parsed
    import datetime as _dt
    _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
