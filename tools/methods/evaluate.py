"""Evaluate any method's saved rankings with the official RCR evaluator."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.methods.common.data import load_rcr_data, split_samples
from rcr.methods.common.results import output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rankings", help="Override the saved ranking path")
    parser.add_argument("--output", help="Override the metrics JSON path")
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
        "--max-queries", type=int, help="Match a prefix smoke retrieval run"
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.mode:
        if cfg.get("method") != "clip":
            parser.error("--mode is only for CLIP")
        cfg["mode"] = args.mode
    if args.split:
        cfg["split"] = args.split
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, cfg["split"])
    if args.max_queries is not None:
        if args.max_queries < 1:
            parser.error("--max-queries must be positive")
        samples = samples[: args.max_queries]
    configured_output = cfg["output"]
    if isinstance(configured_output, dict):
        directory = output_directory(cfg)
        ranking_path = directory / "rankings.pt"
        output_path = directory / "metrics.json"
    else:
        ranking_path = Path(cfg["rankings"].format(split=cfg["split"]))
        output_path = Path(configured_output.format(split=cfg["split"]))
    saved = torch.load(
        args.rankings or ranking_path, map_location="cpu", weights_only=True
    )
    result = evaluate_retrieval_output(
        data, samples, saved, cfg.get("candidate_ks", (500,)), split=cfg["split"]
    )
    output_path = Path(args.output or output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    print(json.dumps(result["by_case"], indent=2))


if __name__ == "__main__":
    main()
