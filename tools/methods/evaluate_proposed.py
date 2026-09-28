"""Evaluate saved rankings from the proposed RCR model."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from rcr.evaluation.evaluate import evaluate_rankings
from rcr.methods.common.data import load_rcr_data, split_samples


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
    gallery_ids = saved["gallery_ids"]
    sample_ids = saved["sample_ids"]

    if gallery_ids != data.gallery_ids:
        raise ValueError("saved gallery_ids do not match the current benchmark gallery")
    if sample_ids != [sample["sample_id"] for sample in samples]:
        raise ValueError("saved sample_ids do not match the requested split")

    rankings = {
        sample_id: [gallery_ids[i] for i in row.tolist()]
        for sample_id, row in zip(sample_ids, saved["rankings"], strict=True)
    }
    coarse = {
        sample_id: [gallery_ids[i] for i in row.tolist()]
        for sample_id, row in zip(sample_ids, saved["coarse_topm"], strict=True)
    }

    identities_by_image = {
        image_id: {
            box["identity_id"] for box in data.gt_head_boxes_by_image.get(image_id, [])
        }
        for image_id in gallery_ids
    }

    result = evaluate_rankings(
        samples,
        gallery_ids,
        identities_by_image,
        rankings,
        coarse_rankings=coarse,
        candidate_ks=tuple(cfg["candidate_ks"]),
    )

    output = Path(cfg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["overall"], indent=2))
    print(json.dumps(result["by_case"], indent=2))


if __name__ == "__main__":
    main()
