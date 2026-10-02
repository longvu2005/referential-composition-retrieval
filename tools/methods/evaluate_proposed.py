"""Evaluate saved rankings from the proposed RCR model."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from rcr.methods.common.data import load_rcr_data, split_samples
from rcr.methods.proposed.retrieval import evaluate_retrieval_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/evaluate.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, cfg["split"])

    saved = torch.load(cfg["rankings"], map_location="cpu", weights_only=True)
    result = evaluate_retrieval_output(
        data,
        samples,
        saved,
        cfg["candidate_ks"],
        split=cfg["split"],
    )

    output = Path(cfg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["overall"], indent=2))
    print(json.dumps(result["by_case"], indent=2))


if __name__ == "__main__":
    main()
