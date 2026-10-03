"""Run baselines and evaluation; fusion weights are selected on full val only."""

import argparse

import yaml

from rcr.methods.baselines.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/methods/baselines/clip.yaml")
    parser.add_argument("--modes", nargs="+", help="CLIP modes; default: all four")
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=["val", "test"]
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    run_experiment(cfg, modes=args.modes, splits=args.splits)


if __name__ == "__main__":
    main()
