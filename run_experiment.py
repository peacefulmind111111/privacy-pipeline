import argparse
import json
import os
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP experiments")
    parser.add_argument(
        "--method",
        choices=[
            "dp_virtual_projection",
            "local_dpsgd",
            "dp_mix_self",
            "dp_mix_diff",
            "embedding_cluster",
            "private_lasso",
            "local_dpsgd_momentum",
        ],
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
        default=os.path.join(
            "outputs", f"metrics_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        ),
        help="Where to save metrics JSON",
    )
    args = parser.parse_args()
    params = json.loads(args.params)

    if args.method == "dp_virtual_projection":
        from dp_virtual_projection_population_only import main_run

        results = main_run(params=params)
    elif args.method == "dp_mix_self":
        from local_dpsgd_experiment import train_mixup

        results = train_mixup(use_public=False, mode="self")
    elif args.method == "dp_mix_diff":
        from local_dpsgd_experiment import train_mixup

        results = train_mixup(use_public=True, mode="diff")
    elif args.method == "embedding_cluster":
        from embedding_clustering_experiment import run_experiment as run_embed

        results = run_embed(params=params)
    elif args.method == "private_lasso":
        from private_lasso_experiment import run_experiment as run_lasso

        results = run_lasso(params=params)
    elif args.method == "local_dpsgd_momentum":
        from local_dpsgd_experiment import train_momentum

        results = train_momentum(use_public=False)
    else:
        from local_dpsgd_experiment import run_experiment as run_local

        results = run_local(params=params)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
