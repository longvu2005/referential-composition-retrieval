"""Retrieve gallery images with the proposed RCR model."""

from pathlib import Path
from time import perf_counter

import torch

from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.common.experiment import resolve_device
from rcr.methods.common.results import output_directory, save_results, sha256_file
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import retrieval_settings, retrieve_variants


def retrieve(cfg: dict, *, max_queries: int | None = None) -> dict:
    return retrieve_experiments({"run": cfg}, max_queries=max_queries)["run"]


def retrieve_experiments(
    configs: dict[str, dict], *, max_queries: int | None = None
) -> dict[str, dict]:
    """Load one checkpoint and share work across same-split inference variants."""
    started = perf_counter()
    cfg = next(iter(configs.values()))
    shared = ("checkpoint", "data", "runtime", "split")
    if any(
        other.get(key) != cfg.get(key) for other in configs.values() for key in shared
    ):
        raise ValueError(
            "inference variants must share checkpoint, data, device and split"
        )
    device = resolve_device(cfg)

    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if any(
        other.get("selected_checkpoint_sha256", checkpoint_sha256) != checkpoint_sha256
        for other in configs.values()
    ):
        raise ValueError("checkpoint changed since val selection; rerun the sweep")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_cfg = checkpoint["config"]
    model_cfg = train_cfg["model"]
    if "state_text_proj.weight" not in checkpoint["model"]:
        raise ValueError(
            "checkpoint predates identity+state coarse retrieval; "
            "train a new checkpoint with the updated train config"
        )
    dim = checkpoint["dim"]

    # Reject missing/incompatible caches before loading the text backbone.
    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, cfg["split"])
    if not samples:
        raise ValueError(f"{cfg['split']}: selected query split is empty")
    full_num_queries = len(samples)
    if max_queries is not None:
        samples = samples[:max_queries]
    cache = GalleryCache(data_cfg["cache"])
    cache.validate_gallery(data.gallery_ids)
    if "cache_id" in checkpoint and cache.cache_id != checkpoint["cache_id"]:
        raise ValueError("retrieval cache differs from the training cache")

    from transformers import AutoModel, AutoTokenizer

    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_backbone = AutoModel.from_pretrained(model_cfg["text_model"])
    text_backbone.resize_token_embeddings(len(tokenizer))
    text_encoder = TextEncoder(text_backbone, dim).to(device)
    text_encoder.load_state_dict(checkpoint["text_encoder"])

    max_subjects = checkpoint["model"]["composition.role.weight"].shape[0]
    model = RCRModel(
        dim=dim,
        identity_dim=model_cfg["identity_dim"],
        num_heads=model_cfg["num_heads"],
        max_subjects=max_subjects,
        mlp_ratio=model_cfg["mlp_ratio"],
        geo_dim=model_cfg["geo_dim"],
        state_dim=model_cfg.get("state_dim"),
        coarse_beta=model_cfg.get("coarse_beta", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    settings = {
        name: retrieval_settings(run_cfg["retrieval"], model.coarse_beta)
        for name, run_cfg in configs.items()
    }
    outputs = retrieve_variants(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=split_image_ids(data, cfg["split"]),
        variants=settings,
        description=cfg["split"],
    )
    elapsed = perf_counter() - started
    for name, output in outputs.items():
        directory = output_directory(configs[name])
        save_results(
            directory,
            output,
            {
                "method": "proposed",
                "config": configs[name],
                "retrieval": settings[name],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_seed": train_cfg.get("train", {}).get("seed"),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_val_full_map": checkpoint.get("best_full_map"),
                "cache_id": cache.cache_id,
                "num_queries": len(samples),
                "num_gallery": len(output["gallery_ids"]),
                "query_subset": len(samples) != full_num_queries,
                "shared_experiments": list(configs),
                "elapsed_seconds": elapsed,
            },
        )
        print(f"Saved {directory}/rankings.pt", flush=True)
    return outputs
