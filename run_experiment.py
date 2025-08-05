import argparse
import json
from dataclasses import asdict
from typing import Any, Dict


def get_default_params(method: str) -> Dict[str, Any]:
    """Return default hyperparameters for an experiment."""
    if method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import ExperimentConfig

        return asdict(ExperimentConfig())
    elif method == "local_dpsgd":
        from local_dpsgd_experiment import ExperimentConfig

        return asdict(ExperimentConfig())
    else:
        raise ValueError("Unknown method")


def run_experiment(
    method: str,
    params: Dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Run the requested experiment with optional parameter overrides."""
    params = params or {}

    if method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import ExperimentConfig, train

        cfg = ExperimentConfig(**params)
        results = train(cfg, output_dir)
    elif method == "local_dpsgd":
        from local_dpsgd_experiment import ExperimentConfig, train

        cfg = ExperimentConfig(**params)
        results = train(cfg, output_dir)
    else:
        raise ValueError("Unknown method")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP experiments")
    parser.add_argument(
        "--method",
        choices=["dp_virtual_projection", "local_dpsgd"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="JSON string of hyperparameters to override",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save metrics JSON",
    )
    args = parser.parse_args()
    params = json.loads(args.params)

    results = run_experiment(args.method, params=params, output_dir=args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

