import pytest
from dp_pipeline import ExperimentConfig, train_dp_sgd, train_dp_mixup


def test_train_dp_sgd_requires_dependencies():
    config = ExperimentConfig(experiment_name="e", method="sgd")
    with pytest.raises(RuntimeError):
        train_dp_sgd(config, None, None)


def test_train_dp_mixup_checks_params_and_dependencies():
    config = ExperimentConfig(experiment_name="e", method="mixup")
    with pytest.raises(ValueError):
        train_dp_mixup(config, None, None)
    config.mixup_alpha = 0.1
    with pytest.raises(RuntimeError):
        train_dp_mixup(config, None, None)
