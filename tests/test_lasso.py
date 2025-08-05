import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import make_regression

from dp_pipeline import ExperimentConfig, dp_lasso


def test_dp_lasso_runs_and_returns_metrics():
    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=0)
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    config = ExperimentConfig(
        experiment_name="lasso", method="dp_lasso", noise_multiplier=0.1, lasso_alpha=0.01
    )
    results = dp_lasso(config, X_train, y_train, X_test, y_test)
    assert results["model"] == "Lasso"
    assert "train_mse" in results["metrics"]
    assert "test_mse" in results["metrics"]
