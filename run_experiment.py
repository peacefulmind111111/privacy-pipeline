import argparse
import json
import os


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

    if args.method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import main_run

        results = main_run(params=params)
    else:
        from local_dpsgd_experiment import run_experiment as run_local

        results = run_local(params=params)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
