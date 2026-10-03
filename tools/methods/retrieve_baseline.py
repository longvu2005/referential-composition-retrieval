"""Run CLIP or FAFA on RCR and save complete rankings; no metric computation."""

import argparse

import yaml

from rcr.methods.baselines.runner import run_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "clip_image",
            "clip_text",
            "image",
            "text",
            "early_fusion",
            "late_fusion",
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Smoke run on a prefix; gallery remains complete",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.mode:
        if cfg["method"] != "clip":
            parser.error("--mode is only for CLIP")
        cfg["mode"] = args.mode
    if args.split:
        cfg["split"] = args.split
    run_retrieval(cfg, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
