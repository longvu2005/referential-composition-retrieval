"""Train the proposed RCR model from cached visual features."""

import json
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import torch
import yaml
from torch import nn
from tqdm import tqdm

from rcr.common.config import validate_training_schedule
from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.common.io import write_json
from rcr.common.runtime import resolve_device
from rcr.common.torch_io import (
    link_checkpoint,
    require_disk_space,
    save_torch,
    tensor_bytes,
)
from rcr.dataset.audit import negative_exclusions, positive_conflicts
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed.batch import (
    finish_batch,
    prefetch_batches,
    prepare_visual_batch,
)
from rcr.proposed.cache.build import check_query_text_config
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.losses import compute_loss
from rcr.proposed.nn.encoders import QueryTextCache, load_text_modules
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA, RCRModel
from rcr.proposed.ranking import (
    mine_hard_negatives,
    retrieve_rankings,
    uses_state,
)
from rcr.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)
from rcr.proposed.supervision import TrainSupervision


def _wandb_run(
    cfg: dict,
    output: Path,
    cache: GalleryCache,
    num_train_samples: int,
    num_val_samples: int,
) -> Any | None:
    """Initialize optional W&B tracking with only reproducibility essentials."""

    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    log_every = int(wandb_cfg.get("log_every_steps", 20))
    if log_every < 1:
        raise ValueError("wandb.log_every_steps must be at least 1")

    import wandb

    tracked_config = {
        key: cfg[key]
        for key in ("model", "train", "optimizer", "loss", "retrieval", "candidate_ks")
    }
    tracked_config.update(
        evaluation=cfg.get("evaluation", {"enabled": False}),
        cache={
            "id": cache.cache_id,
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
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("epoch")
    for prefix in ("epoch", "train_eval", "val"):
        run.define_metric(f"{prefix}/*", step_metric="epoch")
    return run


def _gradient_norm(modules: tuple[nn.Module, ...]) -> float:
    """Return the global L2 gradient norm without modifying gradients."""

    total = None
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            squared = parameter.grad.detach().float().square().sum()
            total = squared if total is None else total + squared
    return 0.0 if total is None else total.sqrt().item()


def _fixed_subset(samples: list[dict], maximum: int, seed: int) -> list[dict]:
    """Select a deterministic random subset and retain dataset order."""

    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if maximum == 0:
        return []
    if maximum >= len(samples):
        return list(samples)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=generator)[:maximum].tolist()
    return [samples[index] for index in sorted(indices)]


def _diagnostics(counts: dict) -> dict:
    """Ratios from summed counts, so short batches do not get extra weight."""
    tp, fp, fn = (counts.get(f"grounding_{key}", 0) for key in ("tp", "fp", "fn"))
    subjects = max(counts.get("grounding_subjects", 0), 1)
    negatives = max(
        sum(counts.get(f"sampled_{k}", 0) for k in ("identity", "hard", "random")), 1
    )
    return {
        "grounding_precision": tp / max(tp + fp, 1),
        "grounding_recall": tp / max(tp + fn, 1),
        "grounding_supervised_rate": counts.get("grounding_supervised_subjects", 0)
        / subjects,
        "grounding_complete_rate": counts.get("grounding_complete_subjects", 0)
        / subjects,
        "state_pairs": counts.get("state_pairs", 0),
        "state_active_query_rate": counts.get("state_active_queries", 0)
        / max(counts.get("queries", 0), 1),
        "fine_coverage": counts.get("fine_coverage_sum", 0)
        / max(counts.get("fine_pairs", 0), 1),
        "fine_members_per_pair": counts.get("fine_members", 0)
        / max(counts.get("fine_pairs", 0), 1),
        "fine_unmatched_member_rate": (
            counts.get("fine_members", 0) - counts.get("fine_matched_members", 0)
        )
        / max(counts.get("fine_members", 0), 1),
        "fine_empty_selection_rate": counts.get("fine_empty_pairs", 0)
        / max(counts.get("fine_pairs", 0), 1),
        "sampled_positives_per_query": counts.get("sampled_positive", 0)
        / max(counts.get("queries", 0), 1),
        **{
            f"sampled_{key}_fraction": counts.get(f"sampled_{key}", 0) / negatives
            for key in ("identity", "hard", "random")
        },
    }


def _checkpoint_metrics(result: dict) -> dict[str, float]:
    """Equal-weight average over nonempty cases, combined with overall mAP."""
    full_map = float(result["overall"]["full_map"])
    cases = [
        float(row["full_map"])
        for row in result["by_case"].values()
        if row["num_queries"] > 0
    ]
    if not cases or not all(math.isfinite(x) for x in [full_map, *cases]):
        raise ValueError("checkpoint selection requires finite, nonempty metrics")
    macro_full_map = fmean(cases)
    return {
        "macro_full_map": macro_full_map,
        "checkpoint_score": 0.5 * full_map + 0.5 * macro_full_map,
    }


def _require_finite_loss(values, rows, epoch, step, amp_enabled, output):
    """Abort before backward/step so failed AMP runs cannot select a checkpoint."""
    if all(math.isfinite(value) for value in values.values()):
        return
    path = output / "nonfinite_batch.json"
    write_json(
        path,
        {
            "epoch": epoch,
            "step": step,
            "amp_enabled": amp_enabled,
            "sample_ids": [row["sample_id"] for row in rows],
            "query_image_ids": [row["query_image_id"] for row in rows],
            "losses": {
                key: value if math.isfinite(value) else str(value)
                for key, value in values.items()
            },
        },
    )
    raise FloatingPointError(
        f"non-finite training loss before backward at epoch={epoch}, step={step}; "
        f"see {path}. Fix the forward path and train from scratch."
    )


def _scalar_values(values: dict[str, torch.Tensor]) -> dict[str, float | int]:
    """Read scalar diagnostics with one device-to-host synchronization."""
    numbers = torch.stack([value.detach().float() for value in values.values()])
    return {
        key: number if values[key].is_floating_point() else int(number)
        for key, number in zip(values, numbers.cpu().tolist(), strict=True)
    }


def train(cfg: dict) -> Path:
    """Select best.pt by 0.5 * overall + 0.5 * macro-case validation Full-mAP."""
    validate_training_schedule(cfg)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optimizer"]
    loss_cfg = cfg["loss"]
    evaluation_cfg = cfg.get("evaluation", {"enabled": False})
    output_cfg = cfg["output"]
    sampling_cfg = sampling_settings(train_cfg.get("sampling", {}))
    cache_cfg = cfg.get("cache", {})
    prefetch = cache_cfg.get("prefetch_batches", 0)
    if prefetch not in (0, 1):
        raise ValueError("cache.prefetch_batches must be 0 or 1")
    mining_batch_size = train_cfg.get("mining_batch_size", 1)
    if not isinstance(mining_batch_size, int) or mining_batch_size < 1:
        raise ValueError("train.mining_batch_size must be a positive integer")
    use_state = uses_state(
        cfg["retrieval"], model_cfg.get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    if loss_cfg.get("state_weight", 1.0) == 0 and use_state:
        raise ValueError("state_weight=0 requires retrieval.coarse_beta=0")

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    device = resolve_device(cfg)

    # Data, cache, and the candidate pool.
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, "train")
    if not samples:
        raise ValueError("training requires a non-empty train split")
    val_samples = split_samples(data, "val")
    if evaluation_cfg.get("enabled", False) and not val_samples:
        raise ValueError("periodic evaluation requires a non-empty val split")
    train_eval_samples = _fixed_subset(
        samples,
        int(evaluation_cfg.get("train_max_queries", 0)),
        seed,
    )
    cache = GalleryCache(
        data_cfg["fafa_image_cache"],
        scene_root=data_cfg.get("dino_cache"),
        query_root=data_cfg.get("fafa_query_cache"),
        lru_mib=cache_cfg.get("lru_mib", 0),
        native_lru_mib=cache_cfg.get("native_lru_mib", 0),
        mmap_shards=cache_cfg.get("mmap_shards", False),
    )
    cache.validate_gallery(data.gallery_ids)
    cache.validate_samples([*samples, *val_samples])
    check_query_text_config(cache, cfg)
    expected_backend = "fafa" if "person_encoder" in cfg else None
    actual_backend = (cache.encoder_metadata.get("person_encoder") or {}).get(
        "backend", "dino"
    )
    if expected_backend is not None and expected_backend != actual_backend:
        raise ValueError("cache person encoder differs from the training config")
    candidate_ids = split_image_ids(data, "train")
    supervision = TrainSupervision(
        data,
        cache,
        scene_size=cfg.get("image_encoder", {}).get("scene_size"),
        cache_file=cache_cfg.get("supervision_cache"),
    )
    identity_pools = identity_candidate_pools(data, samples, candidate_ids)
    state_images = {key: set(ids) for key, ids in identity_pools.items()}
    excluded = negative_exclusions(samples)
    candidate_index = CandidateIndex(samples, candidate_ids, identity_pools, excluded)
    hard_pools = None
    print(f"Training gallery: train ({len(candidate_ids)} images)")

    first_scene, first_persons, *_ = cache.load(torch.tensor([0]))
    person_input_dim = first_persons.shape[-1]
    input_dim = first_scene.shape[-1]
    dim = model_cfg.get("dim", input_dim)
    if dim % model_cfg["num_heads"]:
        raise ValueError("model.dim must be divisible by num_heads")

    # BERT stays frozen, including the resized Subject-token embeddings.
    tokenizer, text_encoder = load_text_modules(
        model_cfg["text_model"],
        dim,
        device,
        add_subject_markers=True,
        cache_mib=cache_cfg.get("text_lru_mib", 0),
    )
    text_cache = QueryTextCache(tokenizer)
    text_cache.prepare([*samples, *val_samples])

    model = RCRModel.from_config(
        model_cfg,
        person_input_dim=person_input_dim,
        input_dim=input_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": optim_cfg["lr"]},
            {"params": text_encoder.proj.parameters(), "lr": optim_cfg["text_lr"]},
        ],
        weight_decay=optim_cfg["weight_decay"],
    )

    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def prepare(job):
        rows, candidates, stats = job
        cpu_times = {}
        prepare_start = perf_counter()
        visual = prepare_visual_batch(
            rows,
            candidates,
            cache,
            supervision=supervision,
            state_image_ids=[state_images[row["sample_id"]] for row in rows],
            timings=cpu_times,
        )
        start = perf_counter()
        supervision.save()
        cpu_times["save"] = perf_counter() - start
        start = perf_counter()
        tokens = text_cache.batch(rows)
        cpu_times["tokens"] = perf_counter() - start
        start = perf_counter()
        if device.type == "cuda":
            visual = {key: value.pin_memory() for key, value in visual.items()}
            tokens = {key: value.pin_memory() for key, value in tokens.items()}
        cpu_times["pin"] = perf_counter() - start
        cpu_times["total"] = perf_counter() - prepare_start
        return rows, stats, visual, tokens, cpu_times

    output = Path(output_cfg["dir"])
    output.mkdir(parents=True, exist_ok=True)
    runtime = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "amp_enabled": amp_enabled,
        "cpu_threads": torch.get_num_threads(),
        "visible_gpus": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "patch_hw": list(cache.patch_hw),
        "scene_dim": input_dim,
        "person_dim": person_input_dim,
        "batch_size": train_cfg["batch_size"],
        "candidates": train_cfg["candidates"],
        "fafa_text_policy": getattr(cache.native, "text_policy", None),
        "scene_lru_mib": cache_cfg.get("lru_mib", 0),
        "native_lru_mib": cache_cfg.get("native_lru_mib", 0),
        "supervision_cache": cache_cfg.get("supervision_cache"),
        "supervision_cached_images": len(supervision._ids),
    }
    write_json(output / "runtime.json", runtime)
    print(
        f"Train runtime: device={device} ({runtime['gpu'] or 'CPU'}), "
        f"AMP={amp_enabled}, threads={runtime['cpu_threads']}, "
        f"patches={cache.patch_hw}, "
        f"up to {train_cfg['batch_size'] * train_cfg['candidates']} pairs/step",
        flush=True,
    )
    # Reserve latest + best + an atomic temporary, including Adam's two moments.
    weights_bytes = tensor_bytes(
        {"model": model.state_dict(), "text": text_encoder.state_dict()}
    )
    adam_bytes = 2 * sum(
        p.numel() * p.element_size()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    )
    require_disk_space(output, 3 * (weights_bytes + adam_bytes))
    (output / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (output / "best.pt").unlink(missing_ok=True)
    (output / "history.jsonl").write_text("", encoding="utf-8")
    (output / "hard_negatives.json").unlink(missing_ok=True)
    (output / "nonfinite_batch.json").unlink(missing_ok=True)
    write_json(
        output / "training_data.json",
        {
            "num_queries": len(samples),
            "sampling": sampling_cfg,
            "case_balanced": train_cfg.get("case_balanced", True),
            "excluded_negative_pairs": sum(map(len, excluded.values())),
            "conflicts": positive_conflicts(samples),
            "policy": "retain reviewed positives; ignore disputed negative pairs",
        },
    )
    tokenizer.save_pretrained(output / "tokenizer")
    wandb_cfg = cfg.get("wandb", {})
    log_every = int(wandb_cfg.get("log_every_steps", 20))
    run = _wandb_run(
        cfg,
        output,
        cache,
        len(samples),
        len(val_samples),
    )
    global_step = 0
    best_full_map: float | None = None
    best_macro_full_map: float | None = None
    best_score: float | None = None
    best_epoch: int | None = None
    state_pairs_seen = 0

    try:
        for epoch in range(train_cfg["epochs"]):
            epoch_start = perf_counter()
            mining_seconds = 0.0
            if (
                int((train_cfg["candidates"] - 1) * sampling_cfg["hard_fraction"]) > 0
                and epoch >= sampling_cfg["warmup_epochs"]
                and (epoch - sampling_cfg["warmup_epochs"])
                % sampling_cfg["refresh_every_epochs"]
                == 0
            ):
                print(f"epoch {epoch + 1}: mining train negatives", flush=True)
                mining_start = perf_counter()
                hard_pools = mine_hard_negatives(
                    samples,
                    cache,
                    tokenizer,
                    text_encoder,
                    model,
                    device,
                    gallery_ids=candidate_ids,
                    retrieval=cfg["retrieval"],
                    pool_size=sampling_cfg["pool_size"],
                    query_batch_size=mining_batch_size,
                    text_cache=text_cache,
                    excluded=excluded,
                )
                write_json(
                    output / "hard_negatives.json",
                    {
                        "model_epoch": epoch,
                        "cache_id": cache.cache_id,
                        "split": "train",
                        "pools": hard_pools,
                    },
                )
                mining_seconds = perf_counter() - mining_start
            generator = torch.Generator().manual_seed(seed + epoch)
            batches = training_batches(
                samples,
                train_cfg["batch_size"],
                generator,
                case_balanced=train_cfg.get("case_balanced", True),
            )

            model.train()
            text_encoder.train()
            totals = {
                "loss": 0.0,
                "grounding": 0.0,
                "identity": 0.0,
                "retrieval": 0.0,
                "state": 0.0,
                "identity_active": 0.0,
            }
            counts = {}
            timings = {"input_wait": 0.0, "forward": 0.0, "backward_step": 0.0}
            timings.update({
                f"prepare_{name}": 0.0
                for name in (
                    "scene", "labels", "fafa", "save", "tokens", "pin", "total",
                )
            })
            steps = 0

            def jobs(batches=batches, generator=generator, hard_pools=hard_pools):
                for rows in batches:
                    stats = {}
                    candidates = sample_candidates(
                        rows,
                        candidate_ids,
                        train_cfg["candidates"],
                        generator,
                        identity_pools=identity_pools,
                        hard_pools=hard_pools,
                        excluded=excluded,
                        identity_fraction=sampling_cfg["identity_fraction"],
                        hard_fraction=sampling_cfg["hard_fraction"],
                        stats=stats,
                        index=candidate_index,
                        positives_per_query=sampling_cfg["positives_per_query"],
                    )
                    yield rows, candidates, stats

            train_start = perf_counter()
            with prefetch_batches(jobs(), prepare, prefetch) as prepared:
                progress = tqdm(
                    prepared,
                    total=len(batches),
                    desc=f"epoch {epoch + 1}/{train_cfg['epochs']}",
                )
                waiting_since = perf_counter()
                for rows, sampling_counts, visual, tokens, cpu_times in progress:
                    forward_start = perf_counter()
                    timings["input_wait"] += forward_start - waiting_since
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.float16, enabled=amp_enabled
                    ):
                        batch = finish_batch(visual, tokens, text_encoder, device)
                        loss, parts = compute_loss(
                            model, batch, cache.patch_hw, **loss_cfg
                        )
                    values = _scalar_values(
                        {
                            "loss": loss,
                            **{
                                k: parts[k]
                                for k in ("grounding", "identity", "retrieval", "state")
                            },
                        }
                    )
                    _require_finite_loss(
                        values, rows, epoch + 1, global_step + 1, amp_enabled, output
                    )
                    backward_start = perf_counter()
                    timings["forward"] += backward_start - forward_start
                    scaler.scale(loss).backward()
                    global_step += 1
                    should_log = run is not None and (
                        global_step == 1 or global_step % log_every == 0
                    )
                    if should_log:
                        scaler.unscale_(optimizer)
                    grad_norm = (
                        _gradient_norm((model, text_encoder)) if should_log else None
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    values["identity_active"] = float(values["identity"] > 0)
                    for name, value in values.items():
                        totals[name] += value * len(rows)
                    batch_counts = {
                        **_scalar_values(
                            {k: v for k, v in parts.items() if k not in values}
                        ),
                        **sampling_counts,
                    }
                    timings["backward_step"] += perf_counter() - backward_start
                    steps += 1
                    for name, seconds in cpu_times.items():
                        timings[f"prepare_{name}"] += seconds
                    if steps == 1 or steps % 20 == 0:
                        detail = " ".join(
                            f"{name}={seconds:.2f}s"
                            for name, seconds in cpu_times.items()
                        )
                        tqdm.write(f"CPU prep step {global_step}: {detail}")
                    state_pairs_seen += batch_counts["state_pairs"]
                    for name, value in batch_counts.items():
                        counts[name] = counts.get(name, 0) + value
                    progress.set_postfix(
                        loss=f"{values['loss']:.4f}",
                        wait=f"{timings['input_wait'] / steps:.2f}s",
                        fwd=f"{timings['forward'] / steps:.2f}s",
                        bwd=f"{timings['backward_step'] / steps:.2f}s",
                    )

                    if should_log:
                        run.log(
                            {
                                "global_step": global_step,
                                "train/loss": values["loss"],
                                "train/grounding_loss": values["grounding"],
                                "train/identity_loss": values["identity"],
                                "train/identity_active": values["identity_active"],
                                "train/retrieval_loss": values["retrieval"],
                                "train/state_loss": values["state"],
                                "train/binding_scale": model.binding_scale.item(),
                                **{
                                    f"train/{k}": v
                                    for k, v in _diagnostics(batch_counts).items()
                                },
                                "train/gradient_norm": grad_norm,
                                "train/model_lr": optimizer.param_groups[0]["lr"],
                                "train/text_lr": optimizer.param_groups[1]["lr"],
                            }
                        )

                    # Do not retain the last batch/text graph through validation or
                    # while constructing the next batch.
                    del batch, loss, parts, visual, tokens
                    waiting_since = perf_counter()

            train_seconds = perf_counter() - train_start
            optimizer.zero_grad(set_to_none=True)
            count = len(samples)
            summary = {name: value / count for name, value in totals.items()}
            print(
                f"epoch {epoch + 1}: "
                f"loss={summary['loss']:.4f} "
                f"ground={summary['grounding']:.4f} "
                f"id={summary['identity']:.4f} "
                f"retrieval={summary['retrieval']:.4f} "
                f"state={summary['state']:.4f}"
            )
            epoch_number = epoch + 1
            epoch_log = {
                "epoch": epoch_number,
                "epoch/loss": summary["loss"],
                "epoch/grounding_loss": summary["grounding"],
                "epoch/identity_loss": summary["identity"],
                "epoch/identity_active_rate": summary["identity_active"],
                "epoch/retrieval_loss": summary["retrieval"],
                "epoch/state_loss": summary["state"],
                "epoch/binding_scale": model.binding_scale.item(),
                **{f"epoch/{k}": v for k, v in _diagnostics(counts).items()},
                "epoch/mining_seconds": mining_seconds,
                "epoch/train_seconds": train_seconds,
                **{f"epoch/{key}_seconds": value for key, value in timings.items()},
            }
            is_best = False
            evaluation_seconds = 0.0

            # Train diagnostics and validation use their own complete image galleries.
            if evaluation_cfg.get("enabled", False) and (
                epoch_number % int(evaluation_cfg["every_epochs"]) == 0
                or epoch_number == train_cfg["epochs"]
            ):
                evaluation_start = perf_counter()
                print(
                    f"epoch {epoch_number}: train/val retrieval evaluation", flush=True
                )
                if use_state and state_pairs_seen == 0:
                    raise ValueError(
                        "no supervised state pairs; increase identity sampling or "
                        "evaluate with retrieval.coarse_beta=0"
                    )
                metrics_dir = output / "evaluation" / f"epoch_{epoch_number:03d}"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                for split, rows in (
                    ("train", train_eval_samples),
                    ("val", val_samples),
                ):
                    if not rows:
                        continue
                    split_output = retrieve_rankings(
                        rows,
                        cache,
                        tokenizer,
                        text_encoder,
                        model,
                        device,
                        gallery_ids=split_image_ids(data, split),
                        **cfg["retrieval"],
                        text_cache=text_cache,
                        description=f"{split} epoch {epoch_number}",
                    )
                    result = evaluate_retrieval_output(
                        data,
                        rows,
                        split_output,
                        cfg["candidate_ks"],
                        split=split,
                    )
                    result["overall"].update(_checkpoint_metrics(result))
                    compact = {
                        "overall": result["overall"],
                        "by_case": result["by_case"],
                    }
                    write_json(metrics_dir / f"{split}_metrics.json", compact)
                    prefix = "train_eval" if split == "train" else "val"
                    epoch_log.update(
                        {
                            f"{prefix}/{name}": float(value)
                            for name, value in result["overall"].items()
                            if name != "num_queries"
                        }
                    )
                    if split == "val":
                        val_metrics = result["overall"]

                val_full_map = float(val_metrics["full_map"])
                val_score = float(val_metrics["checkpoint_score"])
                if best_score is None or val_score > best_score:
                    best_score = val_score
                    best_full_map = val_full_map
                    best_macro_full_map = float(val_metrics["macro_full_map"])
                    best_epoch = epoch_number
                    is_best = True
                print(
                    f"validation epoch {epoch_number}: "
                    f"Full-mAP={val_full_map:.4f} "
                    f"macro={val_metrics['macro_full_map']:.4f} "
                    f"selection={val_score:.4f} "
                    f"Full-R@1={val_metrics['full_r1']:.4f} "
                    f"ID-mAP={val_metrics['id_map']:.4f}"
                )
                evaluation_seconds = perf_counter() - evaluation_start

            epoch_log["epoch/evaluation_seconds"] = evaluation_seconds
            # Always save the latest state; validation alone selects best.pt.
            checkpoint_start = perf_counter()
            checkpoint = {
                "epoch": epoch_number,
                "model": model.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "dim": dim,
                "input_dim": input_dim,
                "person_input_dim": person_input_dim,
                "encoder_metadata": getattr(cache, "encoder_metadata", None),
                "cache_id": cache.cache_id,
                "native_cache": cache.native.metadata,
                "architecture_version": model.architecture_version,
                "representation": "shared_person",
                "best_full_map": best_full_map,
                "best_macro_full_map": best_macro_full_map,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "state_supervised_pairs": state_pairs_seen,
            }
            last_path = output / "last.pt"
            save_torch(checkpoint, last_path)
            if is_best:
                link_checkpoint(last_path, output / "best.pt")
            epoch_log["epoch/checkpoint_seconds"] = perf_counter() - checkpoint_start
            epoch_log["epoch/total_seconds"] = perf_counter() - epoch_start
            print(
                f"epoch {epoch_number} time: train={train_seconds:.1f}s, "
                f"mining={mining_seconds:.1f}s, evaluation={evaluation_seconds:.1f}s, "
                f"checkpoint={epoch_log['epoch/checkpoint_seconds']:.1f}s",
                flush=True,
            )
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_log) + "\n")
            if run is not None:
                run.log(epoch_log)
                if is_best:
                    run.summary["best_val_full_map"] = best_full_map
                    run.summary["best_val_macro_full_map"] = best_macro_full_map
                    run.summary["best_val_checkpoint_score"] = best_score
                    run.summary["best_epoch"] = best_epoch

    finally:
        if run is not None:
            run.finish()
    return output / ("best.pt" if best_epoch is not None else "last.pt")
