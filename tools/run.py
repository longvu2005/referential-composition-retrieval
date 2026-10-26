"""Run one RCR research stage; experiment settings live in YAML."""

import argparse
from pathlib import Path

from rcr.common.config import (
    load_config,
    override_config,
    parse_overrides,
    validate_training_schedule,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True)
    shared.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "build-cache",
        "prepare-supervision",
        "train",
        "retrieve",
        "evaluate",
        "run",
        "calibrate",
    ):
        command = commands.add_parser(name, parents=[shared])
        if name == "build-cache":
            command.add_argument(
                "--cache-stage",
                default="all",
                choices=(
                    "all",
                    "detections",
                    "scene",
                    "fafa",
                    "fafa-image",
                    "fafa-query",
                ),
            )
        if name in ("retrieve", "evaluate", "run"):
            command.add_argument(
                "--split",
                choices=("val", "test") if name == "run" else ("train", "val", "test"),
                help="Override YAML split (default: val)",
            )
        if name in ("retrieve", "run"):
            command.add_argument(
                "--save-tuning",
                action="store_true",
                help="Proposed val only: export full-gallery scores for offline tuning",
            )
    args = parser.parse_args(argv)
    try:
        cfg = override_config(load_config(args.config), parse_overrides(args.set))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    command, method = args.command, cfg.get("method")
    if command == "calibrate":
        from rcr.proposed.calibration import calibrate

        return calibrate(cfg)
    if method not in ("proposed", "clip", "fafa"):
        parser.error("method must be proposed, clip or fafa")
    if (
        command in ("train", "build-cache", "prepare-supervision")
        and method != "proposed"
    ):
        parser.error(f"{command} is only available for proposed")
    if command in ("retrieve", "evaluate", "run"):
        cfg["split"] = args.split or cfg.get("split", "val")
        if cfg["split"] not in ("train", "val", "test") or (
            command == "run" and cfg["split"] == "train"
        ):
            parser.error(f"invalid split for {command}: {cfg['split']}")
    save_tuning = getattr(args, "save_tuning", False)
    if save_tuning and (method != "proposed" or cfg["split"] != "val"):
        parser.error("--save-tuning is only available for proposed with --split val")
    if command == "train":
        try:
            validate_training_schedule(cfg)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
    if method == "proposed" and command in ("train", "retrieve", "run"):
        if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
            parser.error("candidate_ks cannot exceed top_m")

    if command == "prepare":
        if method == "proposed":
            from rcr.proposed.cache.build import run_person_worker

            return run_person_worker(cfg, prepare=True)
        from rcr.baselines.prepare import prepare

        return prepare(cfg)
    if command == "build-cache":
        from rcr.proposed.cache.build import prepare_cache

        return prepare_cache(cfg, stage=args.cache_stage)
    if command == "prepare-supervision":
        from rcr.proposed.supervision import prepare_train_supervision

        return prepare_train_supervision(cfg)
    if command == "train":
        from rcr.proposed.train import train

        return train(cfg)
    if command == "run":
        if method == "proposed":
            from rcr.proposed.runner import run_experiment

            return run_experiment(
                cfg, Path(__file__).resolve(), save_tuning=save_tuning
            )
        from rcr.baselines.runner import run_experiment

        return run_experiment(cfg, splits=[cfg["split"]])
    if command == "retrieve":
        if method == "proposed":
            from rcr.proposed.retrieve import retrieve

            return retrieve(cfg, save_tuning=save_tuning)
        from rcr.baselines.runner import run_retrieval

        return run_retrieval(cfg)
    from rcr.evaluation.runner import evaluate_run

    return evaluate_run(cfg)


if __name__ == "__main__":
    main()
