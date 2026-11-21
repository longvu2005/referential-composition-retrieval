"""Train the proposed model from frozen CLIP and FAFA caches."""

import json
import math
import uuid
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from rcr.common.config import (
    checkpoint_settings,
    save_config,
    validate_training_schedule,
)
from rcr.common.data import (
    inference_samples,
    load_rcr_data,
    split_image_ids,
    split_samples,
)
from rcr.common.io import atomic_write_text, sha256_file, write_json
from rcr.common.runtime import derived_generator, resolve_device, seed_everything
from rcr.common.tensor_checks import TensorChecks, check_tensor
from rcr.common.torch_io import save_torch
from rcr.dataset.audit import negative_exclusions
from rcr.evaluation.evaluate import selection_metrics
from rcr.evaluation.provenance import retrieval_metadata
from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.cache.build import (
    cache_paths,
    check_cache_config,
    check_query_text_config,
)
from rcr.proposed.cache.store import GalleryCache, gallery_cache_options
from rcr.proposed.calibration import (
    calibrate_rankings,
    save_selection,
    write_calibration,
)
from rcr.proposed.contracts import CALIBRATED_RETRIEVAL_KEYS
from rcr.proposed.identity import IdentityCropSampler, identity_sampling_settings
from rcr.proposed.losses import compute_loss, loss_settings
from rcr.proposed.matching import matching_settings
from rcr.proposed.nn.encoders import (
    QueryTextCache,
    check_clip_config,
    load_text_encoder_state,
    load_text_modules,
    text_encoder_state,
)
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.prefetch import OrderedPrefetch, VisualStager
from rcr.proposed.ranking import (
    mine_hard_negatives,
    retrieval_settings,
    retrieve_rankings,
)
from rcr.proposed.resume import (
    TrainingControl,
    TrainingProgress,
    checkpoint_version,
    cleanup_validation_caches,
    load_resume,
    require_new_run,
    restore_history,
    restore_rng,
    resume_signature,
    training_lock,
    training_state,
    verify_best,
    verify_resume,
)
from rcr.proposed.sampling import (
    CandidateIndex,
    annotation_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)
from rcr.proposed.scores import MINING_CONTEXT_WEIGHT
from rcr.proposed.supervision import TrainSupervision


def _wandb_run(cfg, output, cache, num_train_samples, num_val_samples, progress):
    """Start optional experiment tracking without exposing local data paths."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    project = wandb_cfg.get("project", "referential-composition-retrieval")
    if progress.wandb_project is not None and progress.wandb_project != project:
        raise ValueError("resume must retain the W&B project or disable tracking")
    resuming = progress.wandb_id is not None
    progress.wandb_id = progress.wandb_id or uuid.uuid4().hex[:8]
    progress.wandb_project = project

    log_every = int(wandb_cfg.get("log_every_steps", 20))
    if log_every < 1:
        raise ValueError("wandb.log_every_steps must be at least 1")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed; "
            "install the proposed dependencies or set wandb.enabled=false"
        ) from error

    tracked_config = {
        key: cfg[key]
        for key in (
            "model",
            "train",
            "optimizer",
            "loss",
            "retrieval",
            "candidate_ks",
        )
    }
    tracked_config.update(
        optimizer={
            **cfg["optimizer"],
            "schedule": "cosine",
            "min_lr_ratio": cfg["optimizer"].get("min_lr_ratio", 0.1),
        },
        evaluation=cfg.get("evaluation", {"enabled": False}),
        cache={
            "id": cache.cache_id,
            "scene_id": cache.scene_cache_id,
            "detection_id": cache.detection_cache_id,
            "patch_hw": list(cache.patch_hw),
            "scene_dim": cache.scene_dim,
            "person_dim": cache.person_dim,
            "encoders": cache.encoder_metadata,
        },
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
    )
    run = wandb.init(
        project=project,
        id=progress.wandb_id,
        **(
            {"resume": "allow"}
            if resuming and wandb_cfg.get("mode", "online") == "online"
            else {}
        ),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode", "online"),
        dir=str(output),
        config=tracked_config,
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("epoch")
    run.define_metric("epoch/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    return run


def _gradient_norm(*modules):
    """Compute the global L2 gradient norm without changing gradients."""
    squared_norms = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norms.append(parameter.grad.detach().float().square().sum())
    if not squared_norms:
        parameter = next((p for module in modules for p in module.parameters()), None)
        return torch.zeros(()) if parameter is None else parameter.new_zeros(())
    return torch.stack(squared_norms).sum().sqrt()


def build_optimizer(model, text_encoder, config):
    """Keep distinct model and text-projection learning rates."""
    groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": config["lr"],
        },
        {"params": text_encoder.proj.parameters(), "lr": config["text_lr"]},
    ]
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def _evaluate(
    samples,
    data,
    cache,
    tokenizer,
    text_encoder,
    text_cache,
    model,
    device,
    cfg,
    epoch,
    output,
    *,
    state_trained,
):
    started = perf_counter()
    directory = output / "evaluation" / f"epoch_{epoch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / "pending_validation.pt"
    checkpoint_sha256 = sha256_file(output / "last.pt")
    if pending.is_file():
        saved = torch.load(pending, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("pending validation cache differs from last.pt")
        rankings = saved["rankings"]
    else:
        rankings = retrieve_rankings(
            samples,
            cache,
            tokenizer,
            text_encoder,
            model,
            device,
            gallery_ids=split_image_ids(data, "val"),
            text_cache=text_cache,
            description=f"val epoch {epoch}",
            state_trained=state_trained,
            **cfg["retrieval"],
        )
        save_torch(
            {"checkpoint_sha256": checkpoint_sha256, "rankings": rankings}, pending
        )
        print("RCR_VALIDATION_CACHE " + str(pending), flush=True)
    selection = calibrate_rankings(rankings, data, samples, cfg)
    selection.output["provenance"]["elapsed_seconds"] = perf_counter() - started
    write_json(
        directory / "val_metrics.json",
        {key: selection.metrics[key] for key in ("overall", "by_case")},
    )
    write_calibration(directory, selection, epoch=epoch)
    return selection


def _make_checkpoint(
    *,
    epoch,
    model,
    text_encoder,
    tokenizer,
    cfg,
    cache,
    dim,
    scene_dim,
    person_dim,
    best,
    state_pairs,
    validation=None,
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "text_encoder": text_encoder_state(text_encoder, tokenizer),
        "text_model_revision": getattr(
            text_encoder.backbone.config, "_commit_hash", None
        ),
        "config": {
            **cfg,
            "model": {
                **cfg["model"],
                "fine_matching": dict(model.fine_matching),
            },
        },
        "dim": dim,
        "input_dim": scene_dim,
        "person_input_dim": person_dim,
        "global_dim": model.global_dim,
        "encoder_metadata": cache.encoder_metadata,
        "cache_id": cache.cache_id,
        "scene_cache_id": cache.scene_cache_id,
        "detection_cache_id": cache.detection_cache_id,
        "native_cache": cache.native.metadata,
        "architecture_version": model.architecture_version,
        "representation": "shared_person",
        "best_epoch": best["epoch"],
        "best_full_map": best["full_map"],
        "best_macro_full_map": best["macro_full_map"],
        "best_score": best["score"],
        **{f"best_{key}": best.get(key) for key in CALIBRATED_RETRIEVAL_KEYS},
        "validation": validation,
        "state_supervised_pairs": state_pairs,
    }


def _training_config(cfg):
    check_clip_config(cfg)
    cfg = {
        **cfg,
        "loss": loss_settings(cfg["loss"]),
        "checkpointing": checkpoint_settings(cfg.get("checkpointing", {})),
        "model": {
            **cfg["model"],
            "fine_matching": matching_settings(cfg["model"].get("fine_matching")),
        },
        "train": {
            **cfg["train"],
            "context_batch_size": cfg["train"].get("context_batch_size", 64),
            "identity_sampling": identity_sampling_settings(
                cfg["train"].get("identity_sampling", {})
            ),
            "sampling": sampling_settings(cfg["train"].get("sampling", {})),
        },
        "checkpoint": str(Path(cfg["output"]["dir"]) / "best.pt"),
        "retrieval": retrieval_settings(
            cfg["retrieval"], cfg["model"].get("coarse_beta", DEFAULT_COARSE_BETA)
        ),
    }
    if not cfg["loss"]["state_weight"] and cfg["retrieval"]["coarse_beta"] != 0:
        raise ValueError("loss.state_weight=0 requires retrieval.coarse_beta=0")
    warmup_epochs = cfg["train"].get("context_warmup_epochs", 0)
    if warmup_epochs and not any(
        cfg["loss"][key]
        for key in (
            "grounding_weight",
            "identity_weight",
            "matching_weight",
            "coarse_weight",
            "state_weight",
        )
    ):
        raise ValueError(
            "context warmup requires a non-context loss; "
            "set train.context_warmup_epochs=0 for context-only training"
        )
    return cfg


def train(cfg: dict, *, resume: str | Path | None = None) -> Path:
    """Train on ``train`` and select ``best.pt`` by calibrated validation mAP."""
    validate_training_schedule(cfg)
    cfg = _training_config(cfg)
    output = Path(cfg["output"]["dir"])
    checkpoint = None
    if resume is not None:
        path = output / "last.pt" if resume == "auto" else Path(resume)
        loaded_version = checkpoint_version(path)
        checkpoint = load_resume(path, output)
    else:
        require_new_run(output)
    with training_lock(output):
        if checkpoint is None:
            require_new_run(output)
        elif checkpoint_version(path) != loaded_version:
            checkpoint = load_resume(path, output)
        return _train(cfg, output, checkpoint)


def _train(cfg, output, resumed):
    warmup_epochs = cfg["train"].get("context_warmup_epochs", 0)
    sampling = cfg["train"]["sampling"]
    max_negatives = cfg["train"]["candidates"] - 1
    matching_max_positives = min(sampling["positives_per_query"], max_negatives)
    semantic_hard = (
        sampling["semantic_hard_fraction"] > 0
        and int(max_negatives * sampling["identity_fraction"]) > 0
    )
    identity_rejection = (
        cfg["loss"]["context_weight"] > 0
        and cfg["loss"]["context_rejection_weight"] > 0
        and int(max_negatives * sampling["hard_fraction"]) > 0
    )
    fine_mining = semantic_hard or identity_rejection
    context_enabled = cfg["loss"]["context_weight"] > 0
    seed = cfg["train"]["seed"]
    deterministic = cfg["runtime"].get("deterministic", True)
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(cfg)

    cache_paths(cfg)
    data_cfg = cfg["data"]
    cache_cfg = cfg.get("cache", {})
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    train_samples = split_samples(data, "train")
    val_samples = split_samples(data, "val")
    train_gallery = split_image_ids(data, "train")

    cache = GalleryCache(**gallery_cache_options(cfg))
    check_cache_config(cache, cfg)
    cache.validate_gallery(data.gallery_ids)
    cache.validate_samples(inference_samples(val_samples))
    if fine_mining or context_enabled:
        # Fail before label preparation or epoch 1, not at the first refresh.
        cache.validate_samples(inference_samples(train_samples))
    check_query_text_config(cache, cfg)
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    signature = resume_signature(cfg, data, cache, device, amp)
    progress = TrainingProgress()
    if resumed is not None:
        progress = verify_resume(resumed, signature)
        verify_best(output, progress)
        cleanup_validation_caches(output, progress)
        if progress.phase == "epoch_end" and progress.epoch == cfg["train"]["epochs"]:
            restore_rng(resumed["training_state"]["rng"])
            print("Training is already complete", flush=True)
            return output / (
                "best.pt" if progress.best["epoch"] is not None else "last.pt"
            )

    supervision = TrainSupervision(
        data,
        cache,
        cache_file=cache_cfg.get("supervision_cache"),
    )
    identity_settings = cfg["train"]["identity_sampling"]
    use_extra_identities = (
        identity_settings["max_crops"] > 0 and cfg["loss"]["identity_weight"] != 0
    )
    supervision.prepare(
        train_samples,
        all_train=use_extra_identities,
        supplemental=use_extra_identities
        and identity_settings["hard_negative_fraction"] > 0,
        include_direction_swaps=context_enabled
        and bool(cfg["loss"]["context_swap_weight"]),
    )
    if context_enabled and cfg["loss"]["context_swap_weight"]:
        print(
            f"Verified swaps: {len(supervision.direction_swaps)} train queries",
            flush=True,
        )
    identity_sampler = (
        IdentityCropSampler(supervision, identity_settings)
        if use_extra_identities
        else None
    )
    identity_pools, partial_pools = annotation_candidate_pools(
        data,
        train_samples,
        train_gallery,
        include_partial=sampling["partial_id_quota"] > 0,
    )
    excluded = negative_exclusions(train_samples)
    candidate_index = CandidateIndex(
        train_samples,
        train_gallery,
        identity_pools,
        excluded,
        partial_pools=partial_pools,
    )
    del partial_pools
    identity_images = {
        sample_id: set(pool) for sample_id, pool in identity_pools.items()
    }

    scene, people, *_ = cache.load(torch.tensor([0]))
    scene_dim, person_dim = scene.shape[-1], people.shape[-1]
    model_cfg = cfg["model"]
    dim = model_cfg.get("dim", scene_dim)
    tokenizer, text_encoder = load_text_modules(
        model_cfg["text_model"],
        dim,
        device,
        revision=cfg["image_encoder"]["revision"],
        cache_mib=cache_cfg.get("text_lru_mib", 0),
        **({"tokenizer_source": output / "tokenizer"} if resumed is not None else {}),
    )
    if text_encoder.global_dim != cache.global_dim:
        raise ValueError("CLIP image/text global dimensions differ")
    text_cache = QueryTextCache(tokenizer)
    text_cache.prepare([*train_samples, *val_samples])
    cfg = {**cfg, "clip_text_stats": dict(text_cache.stats)}
    print(f"CLIP text policy (77 tokens): {text_cache.stats}", flush=True)
    model = RCRModel.from_config(
        model_cfg,
        input_dim=scene_dim,
        global_dim=cache.global_dim,
        person_input_dim=person_dim,
    ).to(device)

    optimizer = build_optimizer(model, text_encoder, cfg["optimizer"])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    min_lr_ratio = cfg["optimizer"].get("min_lr_ratio", 0.1)
    if hasattr(getattr(torch, "amp", None), "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if resumed is not None:
        model.load_state_dict(resumed["model"])
        load_text_encoder_state(text_encoder, tokenizer, resumed["text_encoder"])
        optimizer.load_state_dict(resumed["training_state"]["optimizer"])
        scaler.load_state_dict(resumed["training_state"]["scaler"])
    else:
        save_config(output / "config.yaml", cfg)
        atomic_write_text(output / "history.jsonl", "")
        tokenizer.save_pretrained(output / "tokenizer")
    write_json(
        output / "runtime.json",
        {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp": amp,
            "seed": seed,
            "deterministic": deterministic,
            "scene_dim": scene_dim,
            "person_dim": person_dim,
            "patch_hw": list(cache.patch_hw),
        },
    )
    write_json(
        output / "training_data.json",
        {
            "queries": len(train_samples),
            "gallery": len(train_gallery),
            "sampling": sampling,
            "identity_sampling": identity_settings,
            "excluded_negative_pairs": sum(map(len, excluded.values())),
        },
    )
    print(
        f"Training {len(train_samples)} queries on {device}; "
        f"batch={cfg['train']['batch_size']}, candidates={cfg['train']['candidates']}, "
        f"AMP={amp}",
        flush=True,
    )

    run = _wandb_run(cfg, output, cache, len(train_samples), len(val_samples), progress)
    log_every = (
        int(cfg.get("wandb", {}).get("log_every_steps", 20)) if run is not None else 0
    )
    epochs = cfg["train"]["epochs"]
    evaluation = cfg.get("evaluation", {"enabled": False})
    every_steps = cfg["checkpointing"]["every_steps"]
    prefetch = bool(cfg["train"].get("prefetch_batches", 0))

    def make_checkpoint(validation=None):
        return _make_checkpoint(
            epoch=progress.epoch,
            model=model,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            cfg=cfg,
            cache=cache,
            dim=dim,
            scene_dim=scene_dim,
            person_dim=person_dim,
            best=progress.best,
            state_pairs=progress.state_pairs_seen + progress.state_pairs_epoch,
            validation=validation,
        )

    def save_last(validation=None, *, reason="running"):
        if device.type == "cuda":
            # Commit consumed model/optimizer work; do not fence the producer's
            # independent copies for the following input batch.
            torch.cuda.current_stream(device).synchronize()
        checkpoint = make_checkpoint(validation)
        checkpoint["training_state"] = training_state(
            progress, optimizer, scaler, signature
        )
        save_torch(checkpoint, output / "last.pt")
        status = {
            "epoch": progress.epoch,
            "phase": progress.phase,
            "next_batch": progress.next_batch,
            "global_step": progress.global_step,
            "complete": progress.phase == "epoch_end" and progress.epoch == epochs,
            "reason": "interrupted" if control.stopping else reason,
        }
        write_json(output / "training_status.json", status)
        print("RCR_CHECKPOINT " + json.dumps(status), flush=True)

    with TrainingControl(run) as control, ExitStack() as resources:
        stager = resources.enter_context(VisualStager(device, enabled=prefetch))
        if resumed is not None:
            restore_history(output, progress)
            restore_rng(resumed["training_state"]["rng"])
            print(
                f"Resuming epoch {progress.epoch}, phase={progress.phase}, "
                f"next_batch={progress.next_batch}, global_step={progress.global_step}",
                flush=True,
            )
        else:
            save_last()
        first_epoch = progress.epoch + (progress.phase == "epoch_end")

        for epoch in range(first_epoch, epochs + 1):
            resuming_epoch = progress.epoch == epoch and progress.phase != "epoch_end"
            if not resuming_epoch:
                progress.epoch, progress.phase, progress.next_batch = epoch, "train", 0
                progress.state_pairs_epoch = 0
            # Apply one cosine factor to every group without resetting AdamW state.
            epoch_fraction = (epoch - 1) / max(1, epochs - 1)
            lr_factor = (
                min_lr_ratio
                + (1 - min_lr_ratio) * (1 + math.cos(math.pi * epoch_fraction)) / 2
            )
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr * lr_factor
            in_warmup = epoch <= warmup_epochs and cfg["loss"]["context_weight"] > 0
            phase = "warmup" if in_warmup else "joint"
            epoch_loss = {
                **cfg["loss"],
                "context_weight": 0.0 if in_warmup else cfg["loss"]["context_weight"],
                "context_swap_weight": (
                    cfg["loss"]["context_swap_weight"]
                    if context_enabled and not in_warmup
                    else 0.0
                ),
                "context_rejection_weight": (
                    cfg["loss"]["context_rejection_weight"]
                    if identity_rejection and epoch - 1 >= sampling["warmup_epochs"]
                    else 0.0
                ),
            }
            # Warmup skips context and its native tokens.
            fafa_pooling = (
                cfg["retrieval"]["pooling"]
                if context_enabled and not in_warmup
                else None
            )
            # One independent stream per (seed, purpose, epoch); see derive_seed.
            generator = derived_generator(seed, "sampling", epoch)
            identity_generator = derived_generator(seed, "identity", epoch)
            epoch_index = epoch - 1
            crop_mining = (
                use_extra_identities
                and identity_settings["hard_negative_fraction"] > 0
                and epoch_index >= sampling["warmup_epochs"]
            )
            hard_quota = int(
                (cfg["train"]["candidates"] - 1) * sampling["hard_fraction"]
            )
            rejection_mining_weight = (
                sampling["rejection_mining_context_weight"]
                if epoch_loss["context_weight"]
                and epoch_loss["context_rejection_weight"]
                else 0.0
            )
            if (
                not resuming_epoch
                and (hard_quota or semantic_hard)
                and epoch_index >= sampling["warmup_epochs"]
                and (epoch_index - sampling["warmup_epochs"])
                % sampling["refresh_every_epochs"]
                == 0
            ):
                progress.hard_pools = mine_hard_negatives(
                    train_samples,
                    cache,
                    tokenizer,
                    text_encoder,
                    model,
                    device,
                    gallery_ids=train_gallery,
                    retrieval=cfg["retrieval"],
                    pool_size=sampling["pool_size"],
                    excluded=excluded,
                    query_batch_size=cfg["train"].get("mining_batch_size", 1),
                    text_cache=text_cache,
                    identity_image_ids=identity_images,
                    semantic_hard=semantic_hard,
                    identity_rejection=identity_rejection,
                    rejection_mining_context_weight=rejection_mining_weight,
                )
                write_json(
                    output / "hard_negatives.json",
                    {
                        "epoch": epoch,
                        "mining": "coarse_then_fine" if fine_mining else "coarse",
                        "normalization": (
                            "per_query_screened_union_population"
                            if fine_mining
                            else "split_gallery_population"
                        ),
                        "retrieval": {
                            **cfg["retrieval"],
                            "context_weight": MINING_CONTEXT_WEIGHT,
                        },
                        "screen_pool_size": sampling["pool_size"],
                        "rejection_mining_context_weight": rejection_mining_weight,
                        "rejection_normalization": (
                            "per_query_screened_wrong_id_population"
                        ),
                        "pools": progress.hard_pools,
                    },
                )
            batches = training_batches(
                train_samples,
                cfg["train"]["batch_size"],
                generator,
                case_balanced=cfg["train"].get("case_balanced", True),
            )
            if resuming_epoch:
                if progress.batches != [
                    [row["sample_id"] for row in batch] for batch in batches
                ]:
                    raise ValueError(
                        "resume batch plan differs from the seeded training order"
                    )
                batches = [
                    [data.samples_by_id[key] for key in row] for row in progress.batches
                ]
                generator.set_state(progress.sampling_rng)
                identity_generator.set_state(progress.identity_rng)
            else:
                progress.batches = [
                    [row["sample_id"] for row in batch] for batch in batches
                ]
            model.train()
            text_encoder.train()
            totals = dict.fromkeys(
                (
                    "loss",
                    "grounding",
                    "identity",
                    "matching",
                    "matching_active_fraction",
                    "coarse",
                    "context",
                    "context_semantic",
                    "context_rejection",
                    "context_swap",
                    "context_fda_error",
                    "state",
                    "context_active_fraction",
                    "context_rejection_active_fraction",
                    "context_swap_active_fraction",
                ),
                0.0,
            )
            crop_totals = dict.fromkeys(
                (
                    "identity_hard_crops",
                    "identity_random_crops",
                    "identity_supplemental_crops",
                ),
                0,
            )
            state_pairs_epoch = (
                torch.tensor(
                    progress.state_pairs_epoch, dtype=torch.long, device=device
                )
                if cfg["loss"]["state_weight"]
                else None
            )
            candidate_totals = {
                "candidate_semantic_hard": 0,
                "candidate_identity_hard": 0,
                "candidate_partial_id": 0,
                "candidate_rejection_context": 0,
            }
            if resuming_epoch:
                totals, crop_totals, candidate_totals = (
                    progress.totals,
                    progress.crop_totals,
                    progress.candidate_totals,
                )
            else:
                progress.totals, progress.crop_totals, progress.candidate_totals = (
                    totals,
                    crop_totals,
                    candidate_totals,
                )
            if control.stopping:
                progress.sampling_rng, progress.identity_rng = (
                    generator.get_state(),
                    identity_generator.get_state(),
                )
                save_last(reason="interrupted")
                return output / (
                    "best.pt" if progress.best["epoch"] is not None else "last.pt"
                )
            # Worker RNGs can advance ahead; checkpoints commit only consumed states.
            sampling_worker = torch.Generator().set_state(generator.get_state())
            identity_worker = torch.Generator().set_state(
                identity_generator.get_state()
            )
            visual_options = {
                "supervision": supervision,
                "identity_sampling": identity_settings if crop_mining else None,
                "fafa_pooling": fafa_pooling,
                "defer_native_targets": True,
                "load_target_context": fafa_pooling is not None,
                "defer_target_scenes": fafa_pooling is not None,
                "preserve_storage": device.type == "cuda",
                "pack_target_persons": True,
            }

            def prepare_training_batch(
                index_rows,
                sampling_worker=sampling_worker,
                identity_worker=identity_worker,
                visual_options=visual_options,
                hard_pools=progress.hard_pools,
                rejection_enabled=bool(epoch_loss["context_rejection_weight"]),
                rejection_mining_weight=rejection_mining_weight,
            ):
                batch_index, rows = index_rows
                sampled = sample_candidates(
                    rows,
                    cfg["train"]["candidates"],
                    sampling_worker,
                    index=candidate_index,
                    positives_per_query=sampling["positives_per_query"],
                    identity_fraction=sampling["identity_fraction"],
                    hard_pools=hard_pools,
                    hard_fraction=sampling["hard_fraction"],
                    semantic_hard_fraction=sampling["semantic_hard_fraction"],
                    identity_rejection=identity_rejection,
                    partial_id_quota=sampling["partial_id_quota"],
                    rejection_mining_context_weight=rejection_mining_weight,
                )
                identity_crops = (
                    identity_sampler.sample(rows, sampled.images, identity_worker)
                    if identity_sampler is not None
                    else []
                )
                visual = prepare_visual_batch(
                    rows,
                    sampled.images,
                    cache,
                    identity_image_ids=[
                        identity_images[row["sample_id"]] for row in rows
                    ],
                    identity_crops=identity_crops,
                    hard_identity_image_ids=sampled.hard_identity_ids,
                    **visual_options,
                )
                if prefetch and "context_layout" in visual:
                    selected = visual["context_layout"][
                        "rejection_rows" if rejection_enabled else "identity_rows"
                    ]
                    visual["target_scene"].prepare(selected)
                    visual["fafa_targets"].prepare(
                        selected,
                        min(cache_cfg.get("native_lru_mib", 0), 128) * 1024**2,
                    )
                return (
                    batch_index,
                    rows,
                    sampled,
                    identity_crops,
                    stager.stage(visual),
                    sampling_worker.get_state(),
                    identity_worker.get_state(),
                )

            prepared_batches = resources.enter_context(
                OrderedPrefetch(
                    enumerate(batches[progress.next_batch :], progress.next_batch),
                    prepare_training_batch,
                    enabled=prefetch,
                )
            )
            batch_progress = tqdm(
                prepared_batches,
                initial=progress.next_batch,
                total=len(batches),
                desc=f"epoch {epoch}/{epochs} ({phase})",
            )
            for (
                batch_index,
                rows,
                sampled,
                identity_crops,
                prepared_visual,
                sampling_state,
                identity_state,
            ) in batch_progress:
                generator.set_state(sampling_state)
                identity_generator.set_state(identity_state)
                optimizer.zero_grad(set_to_none=True)
                candidates = sampled.images
                candidate_totals["candidate_semantic_hard"] += (
                    sampled.semantic_hard_count
                )
                candidate_totals["candidate_identity_hard"] += (
                    sampled.identity_hard_count
                )
                candidate_totals["candidate_partial_id"] += sampled.partial_id_count
                candidate_totals["candidate_rejection_context"] += (
                    sampled.rejection_context_count
                )
                batch = build_batch(
                    rows,
                    candidates,
                    cache,
                    tokenizer,
                    text_encoder,
                    device,
                    identity_image_ids=[
                        identity_images[row["sample_id"]] for row in rows
                    ],
                    identity_crops=identity_crops,
                    hard_identity_image_ids=sampled.hard_identity_ids,
                    text_cache=text_cache,
                    prepared_visual=prepared_visual,
                    include_direction_swaps=bool(
                        epoch_loss["context_weight"]
                        and epoch_loss["context_swap_weight"]
                    ),
                    **visual_options,
                )
                if state_pairs_epoch is not None:
                    positives = (
                        batch["positive_mask"] & batch["identity_positive_mask"]
                    ).sum(1)
                    negatives = (
                        ~batch["positive_mask"] & batch["identity_positive_mask"]
                    ).sum(1)
                    state_pairs_epoch += (positives * negatives).sum()

                with TensorChecks() as checks:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        loss, parts = compute_loss(
                            model,
                            batch,
                            cache.patch_hw,
                            matching_max_positives=matching_max_positives,
                            context_batch_size=cfg["train"]["context_batch_size"],
                            context_score_weight=cfg["retrieval"]["context_weight"],
                            coverage_score_weight=cfg["retrieval"]["coverage_weight"],
                            **epoch_loss,
                        )
                    check_tensor(
                        torch.isfinite(loss),
                        f"non-finite loss at epoch {epoch}: "
                        f"{[row['sample_id'] for row in rows]}",
                        FloatingPointError,
                    )
                    metrics = {"loss": loss.detach()}
                    metrics.update(
                        (key, value)
                        for key, value in parts.items()
                        if key not in crop_totals
                    )
                    host_metrics = checks.cpu_values(metrics)
                    values = dict(
                        zip(
                            metrics,
                            torch.stack(list(host_metrics.values())).tolist(),
                            strict=True,
                        )
                    )
                # Values and all numerical checks share one device-to-host copy.
                # Reject invalid loss before backward or an optimizer side effect.
                scaler.scale(loss).backward()
                progress.global_step += 1
                should_log = run is not None and (
                    progress.global_step == 1 or progress.global_step % log_every == 0
                )
                if should_log:
                    scaler.unscale_(optimizer)
                    norms = checks.cpu_values(
                        {
                            "model": _gradient_norm(model),
                            "text": _gradient_norm(text_encoder.proj),
                        }
                    )
                    model_gradient_norm, text_gradient_norm = torch.stack(
                        list(norms.values())
                    ).tolist()
                    gradient_norm = math.hypot(model_gradient_norm, text_gradient_norm)
                scaler.step(optimizer)
                scaler.update()

                for key in crop_totals:
                    crop_totals[key] += int(parts[key])
                for name, value in values.items():
                    totals[name] += value * len(rows)
                batch_progress.set_postfix(loss=f"{values['loss']:.4f}")
                if should_log:
                    run.log(
                        {
                            "global_step": progress.global_step,
                            **{
                                f"train/{name}": value for name, value in values.items()
                            },
                            "train/gradient_norm": gradient_norm,
                            "train/model_gradient_norm": model_gradient_norm,
                            "train/text_gradient_norm": text_gradient_norm,
                            "train/model_lr": optimizer.param_groups[0]["lr"],
                            "train/text_lr": optimizer.param_groups[1]["lr"],
                            "train/context_weight": epoch_loss["context_weight"],
                            "train/context_rejection_weight": epoch_loss[
                                "context_rejection_weight"
                            ],
                            "train/context_swap_weight": epoch_loss[
                                "context_swap_weight"
                            ],
                            "train/warmup": in_warmup,
                        }
                    )
                del batch, loss, parts, prepared_visual
                progress.next_batch = batch_index + 1
                last_batch = progress.next_batch == len(batches)
                if (
                    last_batch
                    or progress.global_step % every_steps == 0
                    or control.stopping
                ):
                    progress.sampling_rng = generator.get_state()
                    progress.identity_rng = identity_generator.get_state()
                    progress.state_pairs_epoch = (
                        int(state_pairs_epoch) if state_pairs_epoch is not None else 0
                    )
                    progress.phase = "validation" if last_batch else "train"
                    save_last(reason="interrupted" if control.stopping else "running")
                if control.stopping:
                    return output / (
                        "best.pt" if progress.best["epoch"] is not None else "last.pt"
                    )

            progress.state_pairs_epoch = (
                int(state_pairs_epoch) if state_pairs_epoch is not None else 0
            )
            epoch_log = {
                "epoch": epoch,
                "phase": phase,
                "context_weight": epoch_loss["context_weight"],
                "context_rejection_weight": epoch_loss["context_rejection_weight"],
                "context_swap_weight": epoch_loss["context_swap_weight"],
                "model_lr": optimizer.param_groups[0]["lr"],
                "text_lr": optimizer.param_groups[1]["lr"],
                **{
                    f"train_{name}": value / len(train_samples)
                    for name, value in totals.items()
                },
                **{f"train_{name}": value for name, value in crop_totals.items()},
                **{f"train_{name}": value for name, value in candidate_totals.items()},
            }
            optimizer.zero_grad(set_to_none=True)
            is_best = False
            validation = None
            should_evaluate = (
                not in_warmup
                and evaluation.get("enabled", False)
                and (epoch % evaluation["every_epochs"] == 0 or epoch == epochs)
            )
            if should_evaluate:
                validation = _evaluate(
                    val_samples,
                    data,
                    cache,
                    tokenizer,
                    text_encoder,
                    text_cache,
                    model,
                    device,
                    cfg,
                    epoch,
                    output,
                    state_trained=bool(
                        cfg["loss"]["state_weight"]
                        and (progress.state_pairs_seen + progress.state_pairs_epoch)
                    ),
                )
                result = validation.metrics
                metrics = {
                    **selection_metrics(result),
                    **{key: validation.best[key] for key in CALIBRATED_RETRIEVAL_KEYS},
                    "calibration_reference_score": validation.rows[0][
                        "checkpoint_score"
                    ],
                    "id_map": float(result["overall"]["id_map"]),
                    "candidate_recall": float(
                        result["overall"][
                            f"candidate_recall_{cfg['retrieval']['top_m']}"
                        ]
                    ),
                    **{
                        f"full_map_{case.lower()}": float(row["full_map"])
                        for case, row in result["by_case"].items()
                        if row["num_queries"]
                    },
                }
                epoch_log.update(
                    {f"val_{key}": value for key, value in metrics.items()}
                )
                if (
                    progress.best["score"] is None
                    or metrics["checkpoint_score"] > progress.best["score"]
                ):
                    progress.best = {
                        "epoch": epoch,
                        "full_map": metrics["full_map"],
                        "macro_full_map": metrics["macro_full_map"],
                        "score": metrics["checkpoint_score"],
                        **{key: metrics[key] for key in CALIBRATED_RETRIEVAL_KEYS},
                    }
                    is_best = True
                print(
                    f"epoch {epoch}: calibrated val "
                    f"Full-mAP={metrics['full_map']:.4f}, "
                    f"macro={metrics['macro_full_map']:.4f}, "
                    f"a={metrics['coverage_weight']:g}, "
                    f"b={metrics['context_weight']:g}, "
                    f"eta={metrics['fine_coarse_weight']:g}, "
                    f"score={metrics['checkpoint_score']:.4f}",
                    flush=True,
                )

            progress.state_pairs_seen += progress.state_pairs_epoch
            progress.state_pairs_epoch = 0
            progress.phase = "epoch_end"
            progress.history.append(epoch_log)
            if is_best:
                best_path = output / "best.pt"
                checkpoint = make_checkpoint(validation.best)
                save_torch(checkpoint, best_path)
                selected_cfg = {
                    **cfg,
                    "split": "val",
                    "retrieval": {
                        **cfg["retrieval"],
                        **{
                            key: progress.best[key] for key in CALIBRATED_RETRIEVAL_KEYS
                        },
                    },
                }
                validation.output["provenance"] = retrieval_metadata(
                    data,
                    selected_cfg,
                    validation.output,
                    checkpoint,
                    checkpoint_sha256=sha256_file(best_path),
                    device=device,
                    elapsed_seconds=validation.output["provenance"]["elapsed_seconds"],
                )
                save_selection(
                    output,
                    validation,
                    selected_cfg,
                    epoch=epoch,
                )
            restore_history(output, progress)

            if run is not None:
                tracked_epoch = {
                    "epoch": epoch,
                    "epoch/context_weight": epoch_loss["context_weight"],
                    "epoch/context_rejection_weight": epoch_loss[
                        "context_rejection_weight"
                    ],
                    "epoch/context_swap_weight": epoch_loss["context_swap_weight"],
                    "epoch/warmup": in_warmup,
                    "epoch/model_lr": epoch_log["model_lr"],
                    "epoch/text_lr": epoch_log["text_lr"],
                    **{f"epoch/{name}": epoch_log[f"train_{name}"] for name in totals},
                    **{f"epoch/{name}": value for name, value in crop_totals.items()},
                    **{
                        f"epoch/{name}": value
                        for name, value in candidate_totals.items()
                    },
                }
                if should_evaluate:
                    tracked_epoch.update(
                        {f"val/{name}": value for name, value in metrics.items()}
                    )
                run.log(tracked_epoch)
                if is_best:
                    run.summary["best_epoch"] = progress.best["epoch"]
                    run.summary["best_val_full_map"] = progress.best["full_map"]
                    run.summary["best_val_macro_full_map"] = progress.best[
                        "macro_full_map"
                    ]
                    run.summary["best_val_checkpoint_score"] = progress.best["score"]
                    for key in CALIBRATED_RETRIEVAL_KEYS:
                        run.summary[f"best_{key}"] = progress.best[key]
            save_last(
                None if validation is None else validation.best,
                reason="interrupted" if control.stopping else "running",
            )
            cleanup_validation_caches(output, progress)
            if control.stopping:
                break

        return output / ("best.pt" if progress.best["epoch"] is not None else "last.pt")
