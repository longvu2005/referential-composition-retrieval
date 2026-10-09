"""Strict v2 checkpoint loading and shared-protocol retrieval outputs."""

from pathlib import Path

import torch

from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import output_directory, save_results, sha256_file
from rcr.common.runtime import resolve_device
from rcr.evaluation.provenance import benchmark_metadata, runtime_metadata
from rcr.proposed.cache.clip import CACHE_VERSION, FeatureCache, source_signature
from rcr.proposed.nn.model import ARCHITECTURE_VERSION, RCRModel
from rcr.proposed.ranking import retrieval_settings, retrieve_rankings


def load_checkpoint(path, cfg, cache, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
        or checkpoint.get("cache_version") != CACHE_VERSION
    ):
        raise ValueError(
            "incompatible architecture/cache version: "
            "retrain v2; partial loading is forbidden"
        )
    if (
        checkpoint["cache_id"] != cache.cache_id
        or checkpoint["dimensions"] != cache.dimensions
        or checkpoint["source"] != source_signature(cache.source)
        or checkpoint["encoder_metadata"] != cache.encoder_metadata
        or checkpoint["text_sha256"] != cache.index["text_sha256"]
    ):
        raise ValueError("checkpoint feature provenance differs from cache")
    if checkpoint["config"]["model"] != cfg["model"]:
        raise ValueError("model config differs from checkpoint")
    model = RCRModel(**cache.dimensions, **cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def retrieve(cfg, *, max_queries=None):
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    cache = FeatureCache(cfg, data)
    device = resolve_device(cfg)
    path = Path(cfg["checkpoint"])
    model, checkpoint = load_checkpoint(path, cfg, cache, device)
    if checkpoint["warmup"]:
        raise ValueError(
            "warmup checkpoint has untrained joint reasoning; finish main training"
        )
    samples = split_samples(data, cfg["split"])
    total = len(samples)
    if max_queries is not None:
        samples = samples[:max_queries]
    settings = retrieval_settings(cfg["retrieval"])
    output = retrieve_rankings(
        samples,
        cache,
        model,
        device,
        gallery_ids=split_image_ids(data, cfg["split"]),
        **settings,
    )
    metadata = {
        "method": "proposed",
        "config": cfg,
        "retrieval": settings,
        "architecture_version": ARCHITECTURE_VERSION,
        "cache_version": CACHE_VERSION,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "validation_sha256": split_fingerprint(data, "val"),
        "encoder_metadata": cache.encoder_metadata,
        "cache_id": cache.cache_id,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_seed": checkpoint["config"]["train"]["seed"],
        "best_epoch": checkpoint["best_epoch"],
        "best_val_full_map": checkpoint["best_full_map"],
        "query_subset": len(samples) != total,
        "oracle": False,
        **benchmark_metadata(data, cfg["split"], output),
        **runtime_metadata(device),
        **output["runtime"],
    }
    directory = output_directory(cfg)
    save_results(directory, output, metadata)
    print(f"Saved {directory}/rankings.pt")
    return output
