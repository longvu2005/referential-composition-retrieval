"""Run RCR experiments from the repository root after pip install -e ."""

import argparse
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True)
    shared.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", parents=[shared])
    prepare.add_argument("--force", action="store_true")
    cache = commands.add_parser("build-cache", parents=[shared])
    cache.add_argument(
        "--cache-stage", choices=("all", "dino", "persons"), default="all"
    )
    commands.add_parser("train", parents=[shared])
    for name in ("retrieve", "evaluate", "run"):
        command = commands.add_parser(name, parents=[shared])
        command.add_argument(
            "--splits",
            nargs="+",
            choices=("val", "test") if name == "run" else ("train", "val", "test"),
        )
        command.add_argument(
            "--modes",
            nargs="+",
            choices=("clip_image", "clip_text", "early_fusion", "late_fusion"),
        )
        if name == "run":
            for flag in ("prepare", "build-cache", "train", "force"):
                command.add_argument(f"--{flag}", action="store_true")
        else:
            command.add_argument("--max-queries", type=int)
    ablate = commands.add_parser("ablate", parents=[shared])
    ablate.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    args = parser.parse_args(argv)

    from rcr.common.runtime import override_config

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    overrides = dict(item.split("=", 1) for item in args.set)
    cfg = override_config(
        cfg, {key: yaml.safe_load(value) for key, value in overrides.items()}
    )
    command = args.command
    if command == "ablate":
        from rcr.proposed.experiments import run_ablation

        return run_ablation(cfg, Path(__file__).resolve(), splits=args.splits)
    method = cfg["method"]
    if command in ("train", "build-cache") and method != "proposed":
        parser.error("train/build-cache are only available for proposed")
    if command == "run":
        if (args.train or args.build_cache) and method != "proposed":
            parser.error("--train/--build-cache are only available for proposed")
        if args.force and not args.prepare:
            parser.error("--force requires --prepare")
    if getattr(args, "modes", None) and method != "clip":
        parser.error("--modes is only available for CLIP")
    if getattr(args, "max_queries", None) is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    if method == "proposed" and command in ("train", "retrieve", "run"):
        if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
            parser.error("candidate_ks cannot exceed top_m")

    if command == "prepare" or getattr(args, "prepare", False):
        if method == "proposed":
            from rcr.proposed.cache.build import run_person_worker

            run_person_worker(cfg, prepare=True, force=args.force)
        else:
            from rcr.baselines.prepare import prepare

            prepare(cfg, force=args.force)
        if command == "prepare":
            return
    if command == "build-cache":
        from rcr.proposed.cache.build import prepare_cache

        return prepare_cache(cfg, stage=args.cache_stage)
    if command == "train":
        from rcr.proposed.train import train

        return train(cfg)
    splits = list(
        dict.fromkeys(
            args.splits
            or (["val", "test"] if command == "run" else [cfg.get("split", "val")])
        )
    )
    if command == "run":
        if method == "proposed":
            from rcr.proposed.experiments import run_experiment

            return run_experiment(
                cfg,
                Path(__file__).resolve(),
                splits=splits,
                build=args.build_cache,
                train=args.train,
            )
        from rcr.baselines.runner import run_experiment

        return run_experiment(cfg, modes=args.modes, splits=splits)

    from rcr.common.runtime import evaluate_run

    if command == "retrieve":
        if method == "proposed":
            from rcr.proposed.retrieve import retrieve
        else:
            from rcr.baselines.runner import run_retrieval as retrieve
    modes = (args.modes or [cfg["mode"]]) if method == "clip" else [None]
    for split in splits:
        for mode in modes:
            run_cfg = {**cfg, "split": split}
            if mode is not None:
                run_cfg["mode"] = mode
            if command == "retrieve":
                retrieve(run_cfg, max_queries=args.max_queries)
            else:
                evaluate_run(run_cfg, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
