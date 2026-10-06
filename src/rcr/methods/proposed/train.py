"""Train the proposed RCR model from cached visual features."""

import json
import math
import shutil
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import yaml
from torch import nn
from tqdm import tqdm

from rcr.dataset.audit import negative_exclusions, positive_conflicts
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.common.experiment import resolve_device
from rcr.methods.common.results import write_json
from rcr.methods.proposed.batch import (
    finish_batch,
    prefetch_batches,
    prepare_visual_batch,
)
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import SUBJECT_MARKERS, QueryTextCache, TextEncoder
from rcr.methods.proposed.identity_balance import (
    calibrate_identity,
    collect_branch_norms,
    parameter_fingerprint,
)
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.objective import compute_loss
from rcr.methods.proposed.retrieval import (
    mine_hard_negatives,
    retrieve_rankings,
    uses_state,
)
from rcr.methods.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)
from rcr.methods.proposed.shortlist import load_fixed_coarse


def _wandb_run(
    cfg: dict,
    output: Path,
    cache: GalleryCache,
    feature_dim: int,
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
            "feature_dim": feature_dim,
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


def train(cfg: dict) -> Path:
    """Select best.pt by 0.5 * overall + 0.5 * macro-case validation Full-mAP."""
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
    use_state = uses_state(cfg["retrieval"], model_cfg.get("coarse_beta", 0.3))
    if loss_cfg.get("state_weight", 1.0) == 0 and use_state:
        raise ValueError("state_weight=0 requires identity_only or coarse_beta=0")

    from transformers import AutoModel, AutoTokenizer

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
    cache = GalleryCache(data_cfg["cache"], lru_mib=cache_cfg.get("lru_mib", 0))
    cache.validate_gallery(data.gallery_ids)
    candidate_ids = split_image_ids(data, "train")
    identity_pools = identity_candidate_pools(data, samples, candidate_ids)
    state_images = {key: set(ids) for key, ids in identity_pools.items()}
    excluded = negative_exclusions(samples)
    candidate_index = CandidateIndex(samples, candidate_ids, identity_pools, excluded)
    hard_pools = None
    print(f"Training gallery: train ({len(candidate_ids)} images)")

    first_scene, *_ = cache.load(torch.tensor([0]))
    input_dim = first_scene.shape[-1]
    dim = model_cfg.get("dim", input_dim)
    if dim % model_cfg["num_heads"]:
        raise ValueError("model.dim must be divisible by num_heads")

    # BERT stays frozen, including the resized Subject-token embeddings.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model"])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SUBJECT_MARKERS.values())}
    )
    text_cache = QueryTextCache(tokenizer)
    text_cache.prepare([*samples, *val_samples])
    text_backbone = AutoModel.from_pretrained(model_cfg["text_model"])
    text_backbone.resize_token_embeddings(len(tokenizer))
    text_encoder = TextEncoder(text_backbone, dim).to(device)

    max_subjects = len(SUBJECT_MARKERS)
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
        input_dim=input_dim,
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    initialization_sha256 = parameter_fingerprint(model, text_encoder)
    balance_cfg = model_cfg.get("identity_balance")
    calibration = None
    if balance_cfg and (
        balance_cfg["mode"] != "none" or evaluation_cfg.get("branch_norms", False)
    ):
        calibration = calibrate_identity(
            model,
            samples,
            cache,
            tokenizer,
            text_encoder,
            device,
            max_queries=balance_cfg.get("calibration_queries", 128),
            text_cache=text_cache,
        )
    fixed_val = load_fixed_coarse(cfg, data, cache.cache_id, "val")

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
        visual = prepare_visual_batch(
            rows,
            candidates,
            cache,
            state_image_ids=[state_images[row["sample_id"]] for row in rows],
        )
        tokens = text_cache.batch(rows)
        if device.type == "cuda":
            visual = {key: value.pin_memory() for key, value in visual.items()}
            tokens = {key: value.pin_memory() for key, value in tokens.items()}
        return rows, stats, visual, tokens

    output = Path(output_cfg["dir"])
    output.mkdir(parents=True, exist_ok=True)
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
    write_json(
        output / "initialization.json",
        {
            "seed": seed,
            "initialization_sha256": initialization_sha256,
            "identity_calibration": calibration,
        },
    )
    tokenizer.save_pretrained(output / "tokenizer")
    wandb_cfg = cfg.get("wandb", {})
    log_every = int(wandb_cfg.get("log_every_steps", 20))
    run = _wandb_run(
        cfg,
        output,
        cache,
        input_dim,
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
            if (
                int((train_cfg["candidates"] - 1) * sampling_cfg["hard_fraction"]) > 0
                and epoch >= sampling_cfg["warmup_epochs"]
                and (epoch - sampling_cfg["warmup_epochs"])
                % sampling_cfg["refresh_every_epochs"]
                == 0
            ):
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

            with prefetch_batches(jobs(), prepare, prefetch) as prepared:
                progress = tqdm(
                    prepared,
                    total=len(batches),
                    desc=f"epoch {epoch + 1}/{train_cfg['epochs']}",
                )
                for rows, sampling_counts, visual, tokens in progress:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.float16, enabled=amp_enabled
                    ):
                        batch = finish_batch(visual, tokens, text_encoder, device)
                        loss, parts = compute_loss(
                            model, batch, cache.patch_hw, **loss_cfg
                        )
                    values = {
                        "loss": loss.item(),
                        **{
                            k: parts[k].item()
                            for k in ("grounding", "identity", "retrieval", "state")
                        },
                    }
                    _require_finite_loss(
                        values, rows, epoch + 1, global_step + 1, amp_enabled, output
                    )
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
                        **{k: v.item() for k, v in parts.items() if k not in values},
                        **sampling_counts,
                    }
                    state_pairs_seen += batch_counts["state_pairs"]
                    for name, value in batch_counts.items():
                        counts[name] = counts.get(name, 0) + value
                    progress.set_postfix(loss=f"{values['loss']:.4f}")

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
                **{f"epoch/{k}": v for k, v in _diagnostics(counts).items()},
            }
            is_best = False

            # Train diagnostics and validation use their own complete image galleries.
            if evaluation_cfg.get("enabled", False) and (
                epoch_number % int(evaluation_cfg["every_epochs"]) == 0
                or epoch_number == train_cfg["epochs"]
            ):
                if use_state and state_pairs_seen == 0:
                    raise ValueError(
                        "no supervised state pairs; increase identity sampling or "
                        "evaluate with coarse_mode=identity_only"
                    )
                metrics_dir = output / "evaluation" / f"epoch_{epoch_number:03d}"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                for split, rows in (
                    ("train", train_eval_samples),
                    ("val", val_samples),
                ):
                    if not rows:
                        continue
                    with collect_branch_norms(
                        model, evaluation_cfg.get("branch_norms", False)
                    ) as norms:
                        split_output = retrieve_rankings(
                            rows,
                            cache,
                            tokenizer,
                            text_encoder,
                            model,
                            device,
                            gallery_ids=split_image_ids(data, split),
                            fixed_coarse=fixed_val if split == "val" else None,
                            **cfg["retrieval"],
                            text_cache=text_cache,
                            description=f"{split} epoch {epoch_number}",
                        )
                    if norms is not None:
                        write_json(
                            metrics_dir / f"{split}_branch_norms.json", norms.report()
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

            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_log) + "\n")

            if run is not None:
                run.log(epoch_log)
                if is_best:
                    run.summary["best_val_full_map"] = best_full_map
                    run.summary["best_val_macro_full_map"] = best_macro_full_map
                    run.summary["best_val_checkpoint_score"] = best_score
                    run.summary["best_epoch"] = best_epoch

            # Always save the latest state; validation alone selects best.pt.
            checkpoint = {
                "epoch": epoch_number,
                "model": model.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "dim": dim,
                "input_dim": input_dim,
                "cache_id": cache.cache_id,
                "initialization_sha256": initialization_sha256,
                "identity_calibration": calibration,
                "best_full_map": best_full_map,
                "best_macro_full_map": best_macro_full_map,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "state_supervised_pairs": state_pairs_seen,
            }
            last_path = output / "last.pt"
            temporary = output / "last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(last_path)
            if is_best:
                temporary = output / "best.pt.tmp"
                shutil.copyfile(last_path, temporary)
                temporary.replace(output / "best.pt")

    finally:
        if run is not None:
            run.finish()
    return output / ("best.pt" if best_epoch is not None else "last.pt")
