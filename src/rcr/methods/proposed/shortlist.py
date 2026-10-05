"""Read immutable reference coarse rankings/scores for controlled fine ablations."""

import json
from pathlib import Path

import torch

from rcr.methods.common.data import split_fingerprint
from rcr.methods.common.results import sha256_file

COARSE_KEYS = ("coarse_mode", "coarse_beta", "coarse_normalization", "top_m")


def load_fixed_coarse(cfg, data, cache_id, split):
    spec = cfg.get("evaluation", {}).get("fixed_coarse")
    # Training diagnostics retain their own train gallery; selection is val-only.
    if not spec or split == "train":
        return None
    if split not in spec:
        raise ValueError(f"missing fixed coarse artifact for {split}")
    record = spec[split]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("fixed coarse artifact changed; rerun the ablation suite")
    metadata = json.loads(path.with_name("run.json").read_text(encoding="utf-8"))
    if metadata["cache_id"] != cache_id:
        raise ValueError("fixed coarse cache differs from training/retrieval cache")
    if metadata["split_sha256"] != split_fingerprint(data, split):
        raise ValueError("fixed coarse split data changed")
    for key in COARSE_KEYS:
        value = cfg["retrieval"].get(key)
        if key == "coarse_beta" and value is None:
            value = cfg["model"].get("coarse_beta", 0.3)
        if metadata["retrieval"][key] != value:
            raise ValueError(f"fixed coarse settings differ: {key}")
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_fixed_coarse(output, samples, gallery_ids):
    """Validate full permutations and their score alignment before any scoring."""
    if output["gallery_ids"] != gallery_ids:
        raise ValueError("fixed coarse gallery order mismatch")
    ids = output["sample_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed coarse contains duplicate samples")
    by_sample = {sample_id: i for i, sample_id in enumerate(ids)}
    order, scores = output["coarse_rankings"], output["coarse_scores"]
    n = len(gallery_ids)
    if order.shape != (len(ids), n - 1) or scores.shape != (len(ids), n):
        raise ValueError("fixed coarse must contain complete rankings and scores")
    if order.dtype not in (torch.int32, torch.int64):
        raise ValueError("fixed coarse rankings must be integer indices")
    by_image = {name: i for i, name in enumerate(gallery_ids)}
    for sample in samples:
        index = by_sample.get(sample["sample_id"])
        if index is None:
            raise ValueError("sample missing from fixed coarse artifact")
        query = by_image[sample["query_image_id"]]
        expected = torch.arange(n)
        expected = expected[expected != query]
        if not torch.equal(order[index].long().sort().values, expected):
            raise ValueError("invalid fixed coarse permutation")
        values = scores[index]
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError("invalid fixed coarse scores")
        ranked = torch.argsort(values, descending=True, stable=True)
        ranked = ranked[ranked != query]
        if not torch.equal(order[index].long(), ranked):
            raise ValueError("fixed coarse ranking and scores disagree")
    return by_sample
