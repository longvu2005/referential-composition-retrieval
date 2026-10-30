"""Retrieve gallery images with the proposed RCR model."""

from pathlib import Path
from time import perf_counter

import torch

from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_inputs,
)
from rcr.common.io import (
    output_directory,
    save_results,
    sha256_file,
)
from rcr.common.runtime import resolve_device
from rcr.evaluation.provenance import benchmark_metadata, runtime_metadata
from rcr.proposed.cache.build import (
    cache_paths,
    check_cache_config,
    check_query_text_config,
)
from rcr.proposed.cache.native import checkpoint_native_metadata
from rcr.proposed.cache.store import GalleryCache, gallery_cache_options
from rcr.proposed.contracts import (
    COMPATIBLE_ARCHITECTURE_VERSIONS,
    SELECTED_RETRIEVAL_KEYS,
)
from rcr.proposed.nn.encoders import (
    check_clip_config,
    load_text_encoder_state,
    load_text_modules,
)
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.ranking import (
    retrieval_settings,
    retrieve_rankings,
    uses_state,
)


def retrieve(cfg: dict, *, max_queries: int | None = None) -> dict:
    """Save complete raw scores for val; fine-score only Top-M for other splits."""
    started = perf_counter()
    device = resolve_device(cfg)
    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    selection_keys = (
        "selected_checkpoint_sha256",
        "selected_validation_sha256",
        "selected_retrieval",
    )
    if cfg["split"] == "test":
        missing = [key for key in selection_keys if key not in cfg]
        if missing:
            raise ValueError(
                "test retrieval requires validation-selected configuration; "
                "run calibration and use its selected.yaml"
            )
    if cfg.get("selected_checkpoint_sha256", checkpoint_sha256) != checkpoint_sha256:
        raise ValueError("checkpoint changed since val selection; rerun the sweep")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_cfg = checkpoint["config"]
    model_cfg = train_cfg["model"]
    settings = retrieval_settings(
        cfg["retrieval"], model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    present = [key in cfg for key in selection_keys]
    if any(present) and not all(present):
        raise ValueError(
            "selected config predates the scoring contract; rerun val calibration"
        )
    selected = cfg.get("selected_retrieval")
    if isinstance(selected, dict):
        selected = {"pooling": "penalized", **selected}
    if selected is not None and (
        not isinstance(selected, dict) or set(selected) != set(SELECTED_RETRIEVAL_KEYS)
    ):
        raise ValueError(
            "selected retrieval contract is incomplete; rerun val calibration"
        )
    if selected is not None and any(
        settings[key] != selected[key] for key in SELECTED_RETRIEVAL_KEYS
    ):
        raise ValueError(
            "retrieval settings changed since val selection; rerun the sweep"
        )
    if checkpoint.get("architecture_version") not in COMPATIBLE_ARCHITECTURE_VERSIONS:
        raise ValueError(
            "checkpoint architecture is incompatible; rebuild CLIP scene/FAFA caches "
            "and train CLIP ViT-B/32 in a new output directory"
        )
    check_clip_config(train_cfg)
    check_clip_config(cfg)
    if checkpoint.get("representation", "shared_person") != "shared_person":
        raise ValueError("checkpoint representation differs from its architecture")
    if model_cfg.get("representation", "shared_person") != "shared_person":
        raise ValueError("checkpoint representation differs from its architecture")
    dim = checkpoint["dim"]
    state_untrained = (
        train_cfg.get("loss", {}).get("state_weight", 1.0) == 0
        or checkpoint.get("state_supervised_pairs", 1) == 0
    )
    if state_untrained and uses_state(
        settings, model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
    ):
        raise ValueError(
            "checkpoint state branch is untrained; use retrieval.coarse_beta=0"
        )

    # Reject missing/incompatible caches before loading the text backbone.
    cache_paths(cfg)
    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    validation_sha256 = split_fingerprint(data, "val")
    if cfg.get("selected_validation_sha256", validation_sha256) != validation_sha256:
        raise ValueError("validation data changed since val selection; rerun the sweep")
    samples = split_inputs(data, cfg["split"])
    if not samples:
        raise ValueError(f"{cfg['split']}: selected query split is empty")
    full_num_queries = len(samples)
    if max_queries is not None:
        samples = samples[:max_queries]
    cache = GalleryCache(**gallery_cache_options(cfg))
    check_cache_config(cache, cfg)
    cache.validate_gallery(data.gallery_ids)
    cache.validate_samples(samples)
    check_query_text_config(cache, cfg)
    expected_native = checkpoint_native_metadata(checkpoint.get("native_cache"))
    if cache.native.metadata != expected_native:
        if expected_native is not None and cache.native.metadata.get(
            "text_policy"
        ) != expected_native.get("text_policy"):
            raise ValueError(
                "native FAFA query text policy differs from training; "
                "use matching query features or train a new checkpoint"
            )
        raise ValueError("native FAFA cache differs from training")
    if "cache_id" in checkpoint and cache.cache_id != checkpoint["cache_id"]:
        raise ValueError("retrieval cache differs from the training cache")
    for key in ("scene_cache_id", "detection_cache_id"):
        if checkpoint.get(key) != getattr(cache, key, None):
            raise ValueError(f"retrieval {key} differs from training")
    if cache.global_dim != checkpoint.get("global_dim"):
        raise ValueError("retrieval CLIP global dimensions differ from training")
    if (cache.scene_dim, cache.person_dim) != (
        checkpoint.get("input_dim", dim),
        checkpoint["person_input_dim"],
    ):
        raise ValueError("retrieval feature dimensions differ from training")
    if cache.encoder_metadata != checkpoint.get("encoder_metadata"):
        raise ValueError("retrieval encoder metadata differs from training")

    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer, text_encoder = load_text_modules(
        model_cfg["text_model"],
        dim,
        device,
        tokenizer_source=tokenizer_path,
        revision=checkpoint.get("text_model_revision"),
        cache_mib=cfg.get("cache", {}).get("text_lru_mib", 0),
    )
    load_text_encoder_state(text_encoder, tokenizer, checkpoint["text_encoder"])

    max_subjects = checkpoint["model"]["composition.role.weight"].shape[0]
    model = RCRModel.from_config(
        model_cfg,
        input_dim=checkpoint.get("input_dim", dim),
        global_dim=checkpoint["global_dim"],
        person_input_dim=checkpoint["person_input_dim"],
        max_subjects=max_subjects,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    output = retrieve_rankings(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=split_image_ids(data, cfg["split"]),
        **settings,
        description=cfg["split"],
        full_gallery_scores=cfg["split"] == "val",
        state_trained=not state_untrained,
    )
    metadata = {
        "method": "proposed",
        "config": {**cfg, "retrieval": settings},
        "retrieval": settings,
        "score_schema_version": output["score_schema_version"],
        "schema_version": output["schema_version"],
        "score_config": output["score_config"],
        "clip_text_stats": output.get("clip_text_stats"),
        "fine_score_mode": settings["fine_score_mode"],
        "context_weight": settings["context_weight"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_sha256": validation_sha256,
        **benchmark_metadata(data, cfg["split"], output),
        **runtime_metadata(device),
        "architecture_version": model.architecture_version,
        "representation": "shared_person",
        "binding_scale": model.binding_scale.item(),
        "fine_matching": model.fine_matching,
        "encoder_metadata": checkpoint.get("encoder_metadata"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": train_cfg.get("train", {}).get("seed"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_full_map": checkpoint.get("best_full_map"),
        "best_val_macro_full_map": checkpoint.get("best_macro_full_map"),
        "best_val_checkpoint_score": checkpoint.get("best_score"),
        "cache_id": cache.cache_id,
        "scene_cache_id": getattr(cache, "scene_cache_id", None),
        "detection_cache_id": getattr(cache, "detection_cache_id", None),
        "native_cache": cache.native.metadata,
        "num_queries": len(samples),
        "num_gallery": len(output["gallery_ids"]),
        "query_subset": len(samples) != full_num_queries,
        "elapsed_seconds": perf_counter() - started,
    }
    # rankings.pt remains sufficient when copied without sibling logs/configs.
    output["provenance"] = metadata
    directory = output_directory(cfg)
    save_results(directory, output, metadata)
    print(f"Saved {directory}/rankings.pt", flush=True)
    return output
