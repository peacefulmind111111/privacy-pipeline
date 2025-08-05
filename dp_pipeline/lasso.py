from __future__ import annotations

import time
from typing import Dict, Any

try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
except ImportError:  # pragma: no cover - handled at runtime
    Lasso = None  # type: ignore
    StandardScaler = None  # type: ignore
    mean_squared_error = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None  # type: ignore

from .config import ExperimentConfig, current_timestamp


def dp_lasso(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
) -> Dict[str, Any]:
    """Perform a differentially private LASSO regression."""
    if Lasso is None or np is None:
        raise RuntimeError("scikit-learn and NumPy must be installed to run DP LASSO")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sigma = config.noise_multiplier
    noise = sigma * np.random.randn(*X_train_scaled.shape)
    X_train_noisy = X_train_scaled + noise

    lasso = Lasso(alpha=config.lasso_alpha)
    start = time.time()
    lasso.fit(X_train_noisy, y_train)
    runtime = time.time() - start

    y_pred_train = lasso.predict(X_train_noisy)
    y_pred_test = lasso.predict(X_test_scaled)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    epsilon_est = config.noise_multiplier

    results = {
        "experiment_name": config.experiment_name,
        "dataset": "custom",
        "model": "Lasso",
        "method": config.method,
        "hyperparameters": config.to_dict(),
        "privacy": {
            "epsilon": epsilon_est,
            "delta": config.delta,
            "noise_multiplier": config.noise_multiplier,
            "sample_rate": None,
        },
        "metrics": {
            "train_mse": mse_train,
            "test_mse": mse_test,
            "runtime_sec": runtime,
        },
        "timestamp": current_timestamp(),
        "comments": config.comments,
    }
    return results
