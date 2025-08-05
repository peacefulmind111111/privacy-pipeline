"""Entry point for running DP experiments."""
import argparse

from privacy_pipeline.config import ExperimentConfig
from privacy_pipeline.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP experiment")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/experiment1", help="Where to store outputs"
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    train(cfg, args.output_dir)


if __name__ == "__main__":
    main()
