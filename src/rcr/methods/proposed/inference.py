"""Retrieve gallery images with the proposed RCR model."""

from pathlib import Path
from time import perf_counter

import torch

from rcr.methods.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.methods.common.experiment import resolve_device
from rcr.methods.common.results import (
    output_directory,
    save_results,
    sha256_file,
    write_json,
)
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.identity_balance import collect_branch_norms
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import (
    retrieval_settings,
    retrieve_variants,
    uses_state,
)
from rcr.methods.proposed.shortlist import COARSE_KEYS, load_fixed_coarse


def retrieve(cfg: dict, *, max_queries: int | None = None) -> dict:
    return retrieve_experiments({"run": cfg}, max_queries=max_queries)["run"]


def retrieve_experiments(
    configs: dict[str, dict],
    *,
    max_queries: int | None = None,
    save_coarse_scores: bool = False,
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
    if any(
        other.get("evaluation", {}).get("fixed_coarse")
        != cfg.get("evaluation", {}).get("fixed_coarse")
        for other in configs.values()
    ):
        raise ValueError("inference variants must share fixed coarse artifacts")
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
    state_untrained = (
        train_cfg.get("loss", {}).get("state_weight", 1.0) == 0
        or checkpoint.get("state_supervised_pairs", 1) == 0
    )
    if state_untrained and any(
        uses_state(run_cfg["retrieval"], model_cfg.get("coarse_beta", 0.3))
        for run_cfg in configs.values()
    ):
        raise ValueError(
            "checkpoint state branch is untrained; use identity_only or coarse_beta=0"
        )

    # Reject missing/incompatible caches before loading the text backbone.
    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    validation_sha256 = split_fingerprint(data, "val")
    if any(
        other.get("selected_validation_sha256", validation_sha256) != validation_sha256
        for other in configs.values()
    ):
        raise ValueError(
            "validation data changed since beta selection; rerun the sweep"
        )
    samples = split_samples(data, cfg["split"])
    if not samples:
        raise ValueError(f"{cfg['split']}: selected query split is empty")
    full_num_queries = len(samples)
    if max_queries is not None:
        samples = samples[:max_queries]
    cache = GalleryCache(
        data_cfg["cache"], lru_mib=cfg.get("cache", {}).get("lru_mib", 0)
    )
    cache.validate_gallery(data.gallery_ids)
    if "cache_id" in checkpoint and cache.cache_id != checkpoint["cache_id"]:
        raise ValueError("retrieval cache differs from the training cache")

    fixed_coarse = load_fixed_coarse(cfg, data, cache.cache_id, cfg["split"])

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
        identity_balance=model_cfg.get("identity_balance"),
        input_dim=checkpoint.get("input_dim", dim),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    settings = {
        name: retrieval_settings(run_cfg["retrieval"], model.coarse_beta)
        for name, run_cfg in configs.items()
    }
    if fixed_coarse is not None:
        first = next(iter(settings.values()))
        if any(
            other[key] != first[key]
            for other in settings.values()
            for key in COARSE_KEYS
        ):
            raise ValueError("fixed coarse variants must share their coarse settings")
    with collect_branch_norms(
        model, cfg.get("evaluation", {}).get("branch_norms", False)
    ) as norms:
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
            fixed_coarse=fixed_coarse,
            save_coarse_scores=save_coarse_scores,
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
                "validation_sha256": validation_sha256,
                "split_sha256": split_fingerprint(data, cfg["split"]),
                "initialization_sha256": checkpoint.get("initialization_sha256"),
                "identity_calibration": checkpoint.get("identity_calibration"),
                "fixed_coarse": cfg.get("evaluation", {}).get("fixed_coarse"),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_seed": train_cfg.get("train", {}).get("seed"),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_val_full_map": checkpoint.get("best_full_map"),
                "best_val_macro_full_map": checkpoint.get("best_macro_full_map"),
                "best_val_checkpoint_score": checkpoint.get("best_score"),
                "cache_id": cache.cache_id,
                "num_queries": len(samples),
                "num_gallery": len(output["gallery_ids"]),
                "query_subset": len(samples) != full_num_queries,
                "shared_experiments": list(configs),
                "elapsed_seconds": elapsed,
            },
        )
        if norms is not None:
            write_json(directory / "branch_norms.json", norms.report())
        else:
            (directory / "branch_norms.json").unlink(missing_ok=True)
        print(f"Saved {directory}/rankings.pt", flush=True)
    return outputs
