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
from rcr.common.runtime import configure_determinism, resolve_device
from rcr.evaluation.provenance import retrieval_metadata
from rcr.proposed.cache.build import (
    cache_paths,
    check_cache_config,
    check_query_text_config,
)
from rcr.proposed.cache.native import checkpoint_native_metadata
from rcr.proposed.cache.store import GalleryCache, gallery_cache_options
from rcr.proposed.contracts import (
    ARCHITECTURE_VERSION,
    DEFAULT_OWNER_MODE,
    FIXED_FINE_KEYS,
    SELECTED_RETRIEVAL_KEYS,
    SHARED_BINDING_ARCHITECTURE,
)
from rcr.proposed.matching import matching_settings
from rcr.proposed.nn.encoders import (
    check_clip_config,
    load_text_encoder_state,
    load_text_modules,
    text_lora_settings,
)
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.ranking import (
    retrieval_settings,
    retrieve_rankings,
)
from rcr.proposed.scores import FINE_SCORE_MODE


def retrieve(
    cfg: dict,
    *,
    max_queries: int | None = None,
    diagnostic_owner_mode: str | None = None,
) -> dict:
    """Rank one split and save its Top-M scores with complete provenance."""
    if diagnostic_owner_mode is not None and cfg["split"] != "val":
        raise ValueError("owner mode diagnostics require the validation split")
    started = perf_counter()
    configure_determinism(cfg["runtime"].get("deterministic", True))
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
    if train_cfg.get("retrieval", {}).get("fine_score_mode") != FINE_SCORE_MODE:
        raise ValueError("checkpoint uses an obsolete fine scorer; train a new run")
    # Architecture settings are checkpoint-authoritative for regular retrieval.
    # A retrieval YAML cannot silently change query encoding after val selection.
    model_cfg = {
        **train_cfg["model"],
        "fine_matching": matching_settings(train_cfg["model"].get("fine_matching")),
    }
    lora = text_lora_settings(model_cfg.get("text_lora"))
    background = model_cfg["fine_matching"]["background_logit"]
    trained_owner_mode = model_cfg["fine_matching"]["owner_mode"]
    settings = retrieval_settings(
        cfg["retrieval"], model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    trained_settings = retrieval_settings(
        train_cfg["retrieval"], model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    if any(settings[key] != trained_settings[key] for key in FIXED_FINE_KEYS):
        raise ValueError(
            "fine score mode and pooling must match training; train a new run"
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
    if selected is not None and cfg.get("selected_background_logit", 0.0) != background:
        raise ValueError("background changed since val selection; rerun val retrieval")
    if selected is not None:
        selected_matching = cfg.get("model", {}).get("fine_matching") or {}
        selected_owner_mode = selected_matching.get("owner_mode", DEFAULT_OWNER_MODE)
        if selected_owner_mode != trained_owner_mode:
            raise ValueError("owner mode differs from validation selection")
    if diagnostic_owner_mode is not None:
        model_cfg["fine_matching"] = matching_settings(
            {**model_cfg["fine_matching"], "owner_mode": diagnostic_owner_mode}
        )
    architecture = checkpoint.get("architecture_version")
    if architecture not in (ARCHITECTURE_VERSION, SHARED_BINDING_ARCHITECTURE):
        raise ValueError(
            "checkpoint architecture is incompatible; train in a new output directory"
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
    if state_untrained and settings["coarse_beta"] != 0:
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
        if checkpoint.get(key) != getattr(cache, key):
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
        lora=lora,
    )
    load_text_encoder_state(text_encoder, tokenizer, checkpoint["text_encoder"])

    max_subjects = checkpoint["model"]["composition.role.weight"].shape[0]
    model = RCRModel.from_config(
        model_cfg,
        input_dim=checkpoint.get("input_dim", dim),
        global_dim=checkpoint["global_dim"],
        person_input_dim=checkpoint["person_input_dim"],
        max_subjects=max_subjects,
        shared_binding=(
            architecture == SHARED_BINDING_ARCHITECTURE
            or checkpoint.get("binding_shared", False)
        ),
    ).to(device)
    state = checkpoint["model"]
    if architecture == SHARED_BINDING_ARCHITECTURE:
        # v4 used the grounding weights for both calls during inference.
        if any(key.startswith("reasoning_binding.") for key in state):
            raise ValueError(
                "shared-binding checkpoint contains reasoning_binding weights"
            )
        state.update(
            {
                key.replace("grounding.binding.", "reasoning_binding.", 1): value
                for key, value in state.items()
                if key.startswith("grounding.binding.")
            }
        )
    model.load_state_dict(state)

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
        state_trained=not state_untrained,
    )
    metadata = retrieval_metadata(
        data,
        {**cfg, "model": model_cfg, "retrieval": settings},
        output,
        checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        elapsed_seconds=perf_counter() - started,
    )
    if model_cfg["fine_matching"]["owner_mode"] != trained_owner_mode:
        metadata["fine_score_ablation"] = True
    # rankings.pt remains sufficient when copied without sibling logs/configs.
    output["provenance"] = metadata
    directory = output_directory(cfg)
    save_results(directory, output, metadata)
    print(f"Saved {directory}/rankings.pt", flush=True)
    return output
