import pytest
from dp_pipeline import ExperimentConfig, train_dp_sgd


def test_train_dp_sgd_requires_dependencies():
    config = ExperimentConfig(experiment_name="e", method="sgd")
    with pytest.raises(RuntimeError):
        train_dp_sgd(config, None, None)
