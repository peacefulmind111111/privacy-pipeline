import argparse
import json
import os
from typing import Any, Dict


def get_default_params(method: str) -> Dict[str, Any]:
    """Return a copy of the default hyperparameters for an experiment."""
    if method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import DEFAULT_PARAMS

        return DEFAULT_PARAMS.copy()
    elif method == "local_dpsgd":
        from local_dpsgd_experiment import DEFAULT_PARAMS

        return DEFAULT_PARAMS.copy()
    else:
        raise ValueError("Unknown method")


def run_experiment(method: str, params: Dict[str, Any] | None = None, output: str | None = None) -> Dict[str, Any]:
    """Run the requested experiment with optional parameter overrides."""
    params = params or {}

    if method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import main_run

        results = main_run(params=params)
    elif method == "local_dpsgd":
        from local_dpsgd_experiment import run_experiment as run_local

        results = run_local(params=params)
    else:
        raise ValueError("Unknown method")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
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
        "--output",
        type=str,
        default=os.path.join("outputs", "metrics.json"),
        help="Where to save metrics JSON",
    )
    args = parser.parse_args()
    params = json.loads(args.params)

    results = run_experiment(args.method, params=params, output=args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
