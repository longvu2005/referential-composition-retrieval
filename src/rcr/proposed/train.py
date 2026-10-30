"""Train the proposed model from frozen CLIP and FAFA caches."""

import json
import math
from pathlib import Path
from statistics import fmean

import torch
from tqdm import tqdm

from rcr.common.config import save_config, validate_training_schedule
from rcr.common.data import (
    inference_samples,
    load_rcr_data,
    split_image_ids,
    split_samples,
)
from rcr.common.io import atomic_write_text, write_json
from rcr.common.runtime import resolve_device
from rcr.common.torch_io import link_checkpoint, save_torch
from rcr.dataset.audit import negative_exclusions
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed.batch import build_batch
from rcr.proposed.cache.build import (
    cache_paths,
    check_cache_config,
    check_query_text_config,
)
from rcr.proposed.cache.store import GalleryCache, gallery_cache_options
from rcr.proposed.identity import IdentityCropSampler, identity_sampling_settings
from rcr.proposed.losses import compute_loss, loss_settings
from rcr.proposed.nn.encoders import (
    QueryTextCache,
    check_clip_config,
    load_text_modules,
    text_encoder_state,
)
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.ranking import (
    mine_hard_negatives,
    retrieval_settings,
    retrieve_rankings,
)
from rcr.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)
from rcr.proposed.supervision import TrainSupervision


def _wandb_run(cfg, output, cache, num_train_samples, num_val_samples):
    """Start optional experiment tracking without exposing local data paths."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

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
            "scene_id": getattr(cache, "scene_cache_id", None),
            "detection_id": getattr(cache, "detection_cache_id", None),
            "patch_hw": list(cache.patch_hw),
            "scene_dim": cache.scene_dim,
            "person_dim": cache.person_dim,
            "encoders": cache.encoder_metadata,
        },
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
    )
    run = wandb.init(
        project=wandb_cfg.get("project", "referential-composition-retrieval"),
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
        return 0.0
    return torch.stack(squared_norms).sum().sqrt().item()


def _selection_metrics(result):
    full_map = float(result["overall"]["full_map"])
    cases = [
        float(row["full_map"])
        for row in result["by_case"].values()
        if row["num_queries"]
    ]
    macro = fmean(cases)
    return {
        "full_map": full_map,
        "macro_full_map": macro,
        "checkpoint_score": 0.5 * (full_map + macro),
    }


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
    top_m = cfg["retrieval"]["top_m"]
    result = evaluate_retrieval_output(
        data,
        samples,
        rankings,
        sorted({*cfg["candidate_ks"], top_m}),
        split="val",
    )
    selected = _selection_metrics(result)
    result["overall"].update(selected)
    directory = output / "evaluation" / f"epoch_{epoch:03d}"
    write_json(
        directory / "val_metrics.json",
        {"overall": result["overall"], "by_case": result["by_case"]},
    )
    return {
        **selected,
        "id_map": float(result["overall"]["id_map"]),
        "candidate_recall": float(result["overall"][f"candidate_recall_{top_m}"]),
        **{
            f"full_map_{case.lower()}": float(row["full_map"])
            for case, row in result["by_case"].items()
            if row["num_queries"]
        },
    }


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
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "text_encoder": text_encoder_state(text_encoder, tokenizer),
        "text_model_revision": getattr(
            text_encoder.backbone.config, "_commit_hash", None
        ),
        "config": cfg,
        "dim": dim,
        "input_dim": scene_dim,
        "person_input_dim": person_dim,
        "global_dim": model.global_dim,
        "encoder_metadata": cache.encoder_metadata,
        "cache_id": cache.cache_id,
        "scene_cache_id": getattr(cache, "scene_cache_id", None),
        "detection_cache_id": getattr(cache, "detection_cache_id", None),
        "native_cache": cache.native.metadata,
        "architecture_version": model.architecture_version,
        "representation": "shared_person",
        "best_epoch": best["epoch"],
        "best_full_map": best["full_map"],
        "best_macro_full_map": best["macro_full_map"],
        "best_score": best["score"],
        "state_supervised_pairs": state_pairs,
    }


def train(cfg: dict) -> Path:
    """Train on ``train`` and select ``best.pt`` only from validation mAP."""
    validate_training_schedule(cfg)
    output = Path(cfg["output"]["dir"])
    if any((output / name).exists() for name in ("config.yaml", "best.pt", "last.pt")):
        raise FileExistsError(
            f"training run already exists at {output}; use a new output.dir"
        )
    check_clip_config(cfg)
    cfg = {
        **cfg,
        "loss": loss_settings(cfg["loss"]),
        "checkpoint": str(Path(cfg["output"]["dir"]) / "best.pt"),
        "retrieval": retrieval_settings(
            cfg["retrieval"], cfg["model"].get("coarse_beta", DEFAULT_COARSE_BETA)
        ),
    }
    if not cfg["loss"]["state_weight"] and cfg["retrieval"]["coarse_beta"] != 0:
        raise ValueError("loss.state_weight=0 requires retrieval.coarse_beta=0")
    warmup_epochs = cfg["train"].get("context_warmup_epochs", 0)
    if warmup_epochs and not any(
        value
        for key, value in cfg["loss"].items()
        if key.endswith("weight") and key != "context_weight"
    ):
        raise ValueError(
            "context warmup requires a non-context loss; "
            "set train.context_warmup_epochs=0 for context-only training"
        )
    torch.manual_seed(cfg["train"]["seed"])
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
    check_query_text_config(cache, cfg)

    supervision = TrainSupervision(
        data,
        cache,
        cache_file=cache_cfg.get("supervision_cache"),
    )
    identity_settings = identity_sampling_settings(
        cfg["train"].get("identity_sampling", {})
    )
    use_extra_identities = (
        identity_settings["max_crops"] > 0
        and cfg["loss"].get("identity_weight", 0.1) != 0
    )
    supervision.prepare(train_samples, all_train=use_extra_identities)
    identity_sampler = (
        IdentityCropSampler(supervision, identity_settings)
        if use_extra_identities
        else None
    )
    identity_pools = identity_candidate_pools(data, train_samples, train_gallery)
    excluded = negative_exclusions(train_samples)
    candidate_index = CandidateIndex(
        train_samples, train_gallery, identity_pools, excluded
    )
    identity_images = {
        sample_id: set(pool) for sample_id, pool in identity_pools.items()
    }
    sampling = sampling_settings(cfg["train"].get("sampling", {}))

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

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": cfg["optimizer"]["lr"],
            },
            {
                "params": text_encoder.proj.parameters(),
                "lr": cfg["optimizer"]["text_lr"],
            },
        ],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    min_lr_ratio = cfg["optimizer"].get("min_lr_ratio", 0.1)
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    if hasattr(getattr(torch, "amp", None), "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    output.mkdir(parents=True, exist_ok=True)
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

    run = _wandb_run(cfg, output, cache, len(train_samples), len(val_samples))
    log_every = (
        int(cfg.get("wandb", {}).get("log_every_steps", 20)) if run is not None else 0
    )
    global_step = 0
    best = {
        "epoch": None,
        "full_map": None,
        "macro_full_map": None,
        "score": None,
    }
    state_pairs_seen = 0
    epochs = cfg["train"]["epochs"]
    evaluation = cfg.get("evaluation", {"enabled": False})
    hard_pools = None

    for epoch in range(1, epochs + 1):
        # Use the base rates first and reach each group's floor in the final epoch.
        progress = (epoch - 1) / max(1, epochs - 1)
        lr_factor = (
            min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * lr_factor
        in_warmup = epoch <= warmup_epochs and cfg["loss"]["context_weight"] > 0
        phase = "warmup" if in_warmup else "joint"
        epoch_loss = {
            **cfg["loss"],
            "context_weight": 0.0 if in_warmup else cfg["loss"]["context_weight"],
        }
        generator = torch.Generator().manual_seed(cfg["train"]["seed"] + epoch - 1)
        identity_generator = torch.Generator().manual_seed(
            cfg["train"]["seed"] + epoch + 1_000_002
        )
        epoch_index = epoch - 1
        hard_quota = int((cfg["train"]["candidates"] - 1) * sampling["hard_fraction"])
        if (
            hard_quota
            and epoch_index >= sampling["warmup_epochs"]
            and (epoch_index - sampling["warmup_epochs"])
            % sampling["refresh_every_epochs"]
            == 0
        ):
            hard_pools = mine_hard_negatives(
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
            )
            write_json(output / "hard_negatives.json", hard_pools)
        batches = training_batches(
            train_samples,
            cfg["train"]["batch_size"],
            generator,
            case_balanced=cfg["train"].get("case_balanced", True),
        )
        model.train()
        text_encoder.train()
        totals = dict.fromkeys(
            (
                "loss",
                "grounding",
                "identity",
                "coarse",
                "context",
                "state",
                "context_active_fraction",
            ),
            0.0,
        )
        progress = tqdm(batches, desc=f"epoch {epoch}/{epochs} ({phase})")
        for rows in progress:
            candidates = sample_candidates(
                rows,
                cfg["train"]["candidates"],
                generator,
                index=candidate_index,
                positives_per_query=sampling["positives_per_query"],
                identity_fraction=sampling["identity_fraction"],
                hard_pools=hard_pools,
                hard_fraction=sampling["hard_fraction"],
            )
            identity_crops = (
                identity_sampler.sample(rows, candidates, identity_generator)
                if identity_sampler is not None
                else []
            )
            batch = build_batch(
                rows,
                candidates,
                cache,
                tokenizer,
                text_encoder,
                device,
                supervision=supervision,
                identity_image_ids=[identity_images[row["sample_id"]] for row in rows],
                identity_crops=identity_crops,
                text_cache=text_cache,
            )
            positives = (batch["positive_mask"] & batch["identity_positive_mask"]).sum(
                1
            )
            negatives = (~batch["positive_mask"] & batch["identity_positive_mask"]).sum(
                1
            )
            if cfg["loss"]["state_weight"]:
                state_pairs_seen += int((positives * negatives).sum())

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, parts = compute_loss(model, batch, cache.patch_hw, **epoch_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}: "
                    f"{[row['sample_id'] for row in rows]}"
                )
            scaler.scale(loss).backward()
            global_step += 1
            should_log = run is not None and (
                global_step == 1 or global_step % log_every == 0
            )
            if should_log:
                scaler.unscale_(optimizer)
                model_gradient_norm = _gradient_norm(model)
                text_gradient_norm = _gradient_norm(text_encoder.proj)
                gradient_norm = math.hypot(model_gradient_norm, text_gradient_norm)
            scaler.step(optimizer)
            scaler.update()

            values = {
                "loss": float(loss.detach()),
                **{key: float(value) for key, value in parts.items()},
            }
            for name, value in values.items():
                totals[name] += value * len(rows)
            progress.set_postfix(loss=f"{values['loss']:.4f}")
            if should_log:
                run.log(
                    {
                        "global_step": global_step,
                        **{f"train/{name}": value for name, value in values.items()},
                        "train/gradient_norm": gradient_norm,
                        "train/model_gradient_norm": model_gradient_norm,
                        "train/text_gradient_norm": text_gradient_norm,
                        "train/model_lr": optimizer.param_groups[0]["lr"],
                        "train/text_lr": optimizer.param_groups[1]["lr"],
                        "train/context_weight": epoch_loss["context_weight"],
                        "train/warmup": in_warmup,
                    }
                )

        epoch_log = {
            "epoch": epoch,
            "phase": phase,
            "context_weight": epoch_loss["context_weight"],
            "model_lr": optimizer.param_groups[0]["lr"],
            "text_lr": optimizer.param_groups[1]["lr"],
            **{
                f"train_{name}": value / len(train_samples)
                for name, value in totals.items()
            },
        }
        optimizer.zero_grad(set_to_none=True)
        is_best = False
        should_evaluate = (
            not in_warmup
            and evaluation.get("enabled", False)
            and (epoch % evaluation["every_epochs"] == 0 or epoch == epochs)
        )
        if should_evaluate:
            metrics = _evaluate(
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
                state_trained=bool(cfg["loss"]["state_weight"] and state_pairs_seen),
            )
            epoch_log.update({f"val_{key}": value for key, value in metrics.items()})
            if best["score"] is None or metrics["checkpoint_score"] > best["score"]:
                best = {
                    "epoch": epoch,
                    "full_map": metrics["full_map"],
                    "macro_full_map": metrics["macro_full_map"],
                    "score": metrics["checkpoint_score"],
                }
                is_best = True
            print(
                f"epoch {epoch}: val Full-mAP={metrics['full_map']:.4f}, "
                f"macro={metrics['macro_full_map']:.4f}",
                flush=True,
            )

        checkpoint = _make_checkpoint(
            epoch=epoch,
            model=model,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            cfg=cfg,
            cache=cache,
            dim=dim,
            scene_dim=scene_dim,
            person_dim=person_dim,
            best=best,
            state_pairs=state_pairs_seen,
        )
        last = output / "last.pt"
        save_torch(checkpoint, last)
        if is_best:
            link_checkpoint(last, output / "best.pt")
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_log) + "\n")

        if run is not None:
            tracked_epoch = {
                "epoch": epoch,
                "epoch/context_weight": epoch_loss["context_weight"],
                "epoch/warmup": in_warmup,
                "epoch/model_lr": epoch_log["model_lr"],
                "epoch/text_lr": epoch_log["text_lr"],
                **{f"epoch/{name}": epoch_log[f"train_{name}"] for name in totals},
            }
            if should_evaluate:
                tracked_epoch.update(
                    {f"val/{name}": value for name, value in metrics.items()}
                )
            run.log(tracked_epoch)
            if is_best:
                run.summary["best_epoch"] = best["epoch"]
                run.summary["best_val_full_map"] = best["full_map"]
                run.summary["best_val_macro_full_map"] = best["macro_full_map"]
                run.summary["best_val_checkpoint_score"] = best["score"]

    if run is not None:
        run.finish()
    return output / ("best.pt" if best["epoch"] is not None else "last.pt")
