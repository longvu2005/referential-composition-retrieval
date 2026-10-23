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
from rcr.proposed.cache.build import check_query_text_config
from rcr.proposed.cache.native import checkpoint_native_metadata
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import load_text_encoder_state, load_text_modules
from rcr.proposed.nn.model import ARCHITECTURE_VERSION, DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.ranking import (
    rank_weight_grid,
    retrieval_settings,
    uses_state,
)


def retrieve(
    cfg: dict, *, max_queries: int | None = None, save_tuning: bool = False
) -> dict:
    return _retrieve(cfg, max_queries=max_queries, save_tuning=save_tuning)[0]


def retrieve_weight_grid(cfg: dict, weights: list[tuple[float, float]]) -> list[dict]:
    """Score a validation weight grid with one checkpoint/backbone load."""
    if cfg["split"] != "val":
        raise ValueError("weight calibration requires the validation split")
    if not weights:
        raise ValueError("calibration requires at least one weight pair")
    return _retrieve(cfg, weights=weights)


def _retrieve(
    cfg: dict,
    *,
    max_queries: int | None = None,
    weights: list[tuple[float, float]] | None = None,
    save_tuning: bool = False,
) -> list[dict]:
    """Load the current model, rank, and save complete split results."""
    started = perf_counter()
    device = resolve_device(cfg)
    if save_tuning and cfg["split"] != "val":
        raise ValueError("tuning export requires the validation split")

    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    selection_keys = (
        "selected_checkpoint_sha256",
        "selected_validation_sha256",
        "selected_retrieval",
    )
    selected_retrieval_keys = (
        "fine_score_mode",
        "context_weight",
        "coarse_beta",
        "fine_coarse_weight",
        "top_m",
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
    if selected is not None and (
        not isinstance(selected, dict) or set(selected) != set(selected_retrieval_keys)
    ):
        raise ValueError(
            "selected retrieval contract is incomplete; rerun val calibration"
        )
    if selected is not None and any(
        settings[key] != selected[key] for key in selected_retrieval_keys
    ):
        raise ValueError(
            "retrieval settings changed since val selection; rerun the sweep"
        )
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "checkpoint architecture is incompatible; train FAFA-native V1 "
            "in a new output directory after building the FAFA image/query caches"
        )
    if checkpoint.get("representation", "shared_person") != "shared_person":
        raise ValueError("checkpoint representation differs from its architecture")
    if model_cfg.get("representation", "shared_person") != "shared_person":
        raise ValueError("checkpoint representation differs from its architecture")
    dim = checkpoint["dim"]
    state_untrained = (
        train_cfg.get("loss", {}).get("state_weight", 1.0) == 0
        or checkpoint.get("state_supervised_pairs", 1) == 0
    )
    if state_untrained and (
        any(beta != 0 for beta, _ in weights)
        if weights is not None
        else uses_state(
            cfg["retrieval"], model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
        )
    ):
        raise ValueError(
            "checkpoint state branch is untrained; use retrieval.coarse_beta=0"
        )

    # Reject missing/incompatible caches before loading the text backbone.
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
    cache = GalleryCache(
        data_cfg["fafa_image_cache"],
        scene_root=data_cfg.get("dino_cache"),
        query_root=data_cfg.get("fafa_query_cache"),
        lru_mib=cfg.get("cache", {}).get("lru_mib", 0),
        native_lru_mib=cfg.get("cache", {}).get("native_lru_mib", 0),
        mmap_shards=cfg.get("cache", {}).get("mmap_shards", False),
    )
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
        person_input_dim=checkpoint["person_input_dim"],
        max_subjects=max_subjects,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    is_grid = weights is not None
    weights = (
        weights
        if is_grid
        else [(settings["coarse_beta"], settings["fine_coarse_weight"])]
    )
    outputs = rank_weight_grid(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=split_image_ids(data, cfg["split"]),
        settings=settings,
        weights=weights,
        description=cfg["split"],
        save_tuning=save_tuning,
        state_trained=not state_untrained,
    )
    elapsed = perf_counter() - started
    for index, (output, (beta, fine_weight)) in enumerate(
        zip(outputs, weights, strict=True)
    ):
        run_cfg = {
            **cfg,
            "retrieval": {
                **settings,
                "coarse_beta": beta,
                "fine_coarse_weight": fine_weight,
            },
        }
        if is_grid:
            run_cfg["output"] = {
                "dir": str(Path(cfg["output"]["dir"]) / f"grid_{index:03d}")
            }
        directory = output_directory(run_cfg)
        benchmark = benchmark_metadata(data, cfg["split"], output)
        if save_tuning:
            # Keep the score file attributable when copied without run.json.
            # Only fingerprints/summary metadata are added after model scoring.
            output["tuning_scores"]["provenance"] = {
                "checkpoint_sha256": checkpoint_sha256,
                "validation_sha256": validation_sha256,
                "cache_id": cache.cache_id,
                "architecture_version": model.architecture_version,
                "query_subset": len(samples) != full_num_queries,
                **benchmark,
            }
        save_results(
            directory,
            output,
            {
                "method": "proposed",
                "config": run_cfg,
                "retrieval": run_cfg["retrieval"],
                "score_schema_version": output["score_schema_version"],
                "score_config": output["score_config"],
                "tuning_export": (
                    {
                        "schema_version": output["tuning_scores"]["schema_version"],
                        "scope": "full_gallery",
                        "state_trained": not state_untrained,
                    }
                    if save_tuning
                    else None
                ),
                "fine_score_mode": settings["fine_score_mode"],
                "context_weight": settings["context_weight"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "validation_sha256": validation_sha256,
                **benchmark,
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
                "native_cache": cache.native.metadata,
                "num_queries": len(samples),
                "num_gallery": len(output["gallery_ids"]),
                "query_subset": len(samples) != full_num_queries,
                "elapsed_seconds": elapsed,
            },
        )
        print(f"Saved {directory}/rankings.pt", flush=True)
    return outputs
