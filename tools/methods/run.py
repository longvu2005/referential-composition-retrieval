"""One CLI for RCR experiments. Run from the repository root after pip install -e ."""

import argparse
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "build-cache",
            "train",
            "retrieve",
            "evaluate",
            "run",
            "ablate",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("clip_image", "clip_text", "early_fusion", "late_fusion"),
    )
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Override existing YAML keys, e.g. train.epochs=1 runtime.device=cpu",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare encoder/baseline assets before run",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Build proposed cache before run --train",
    )
    parser.add_argument(
        "--train", action="store_true", help="Train proposed before run"
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace prepared model assets"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Prefix subset for retrieve/evaluate smoke runs only",
    )
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    from rcr.methods.common.experiment import override_config

    try:
        overrides = dict(item.split("=", 1) for item in args.set)
        cfg = override_config(cfg, {k: yaml.safe_load(v) for k, v in overrides.items()})
    except (KeyError, ValueError) as error:
        parser.error(str(error))

    if args.command == "ablate":
        if any(
            (
                args.modes,
                args.prepare,
                args.build_cache,
                args.train,
                args.force,
                args.max_queries is not None,
            )
        ) or "train" in (args.splits or []):
            parser.error("ablate uses a suite YAML and complete val/test splits")
        from rcr.methods.proposed.ablation import run_ablation

        return run_ablation(
            cfg, Path(__file__).resolve(), splits=args.splits or ["val"]
        )

    method, command = cfg["method"], args.command
    if method not in ("proposed", "clip", "fafa"):
        parser.error(f"unknown method: {method}")
    if args.modes and method != "clip":
        parser.error("--modes is only for CLIP")
    if args.max_queries is not None and (
        args.max_queries < 1 or command not in ("retrieve", "evaluate")
    ):
        parser.error("--max-queries must be positive and is only for retrieve/evaluate")
    if command in ("train", "build-cache") or args.train or args.build_cache:
        if method != "proposed":
            parser.error("training and build-cache are only for proposed")
    if (args.train or args.build_cache or args.prepare) and command != "run":
        parser.error("--train, --build-cache and --prepare are run options")
    if args.force and command != "prepare" and not args.prepare:
        parser.error("--force requires prepare")
    if method == "proposed" and command in ("train", "retrieve", "run"):
        if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
            parser.error("candidate_ks cannot exceed retrieval.top_m")
        if command == "train" or args.train:
            if cfg["train"]["epochs"] < 1 or cfg["evaluation"]["every_epochs"] < 1:
                parser.error("epochs and evaluation.every_epochs must be positive")
    splits = list(
        dict.fromkeys(
            args.splits
            or (["val", "test"] if command == "run" else [cfg.get("split", "val")])
        )
    )
    if command == "run":
        if "train" in splits:
            parser.error(
                "run evaluates val/test; use retrieve/evaluate for train diagnostics"
            )
        splits = [s for s in ("val", "test") if s in splits]
    if command == "prepare" or args.prepare:
        if method == "proposed":
            from rcr.methods.proposed.person_encoder import run_person_worker

            run_person_worker(cfg, prepare=True, force=args.force)
        else:
            from rcr.methods.baselines.prepare import prepare

            prepare(cfg, force=args.force)
        if command == "prepare":
            return
    if command == "build-cache":
        from rcr.methods.proposed.build_cache import prepare_cache

        return prepare_cache(cfg)
    if command == "train":
        from rcr.methods.proposed.train import train

        return train(cfg)
    if command == "run":
        if method == "proposed":
            from rcr.methods.proposed.runner import run_experiment

            return run_experiment(
                cfg,
                Path(__file__).resolve(),
                splits=splits,
                build=args.build_cache,
                train=args.train,
            )
        from rcr.methods.baselines.runner import run_experiment

        return run_experiment(cfg, modes=args.modes, splits=splits)

    from rcr.methods.common.experiment import evaluate_run

    if command == "retrieve":
        if method == "proposed":
            from rcr.methods.proposed.inference import retrieve
        else:
            from rcr.methods.baselines.runner import run_retrieval as retrieve
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
