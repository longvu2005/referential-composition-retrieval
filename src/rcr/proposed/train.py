"""Warm up G/P_id, then train matching and joint reasoning on cached features."""

import json
import math
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from rcr.common.config import validate_training_schedule
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import write_json
from rcr.common.runtime import resolve_device
from rcr.common.storage import atomic_copy, atomic_torch_save
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed.batch import build_batch, prefetch_batches, to_device
from rcr.proposed.cache.clip import CACHE_VERSION, FeatureCache, source_signature
from rcr.proposed.losses import compute_loss
from rcr.proposed.nn.model import ARCHITECTURE_VERSION, RCRModel
from rcr.proposed.ranking import mine_hard_negatives, retrieve_rankings
from rcr.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    negative_exclusions,
    sample_candidates,
    sampling_settings,
    training_batches,
)


def checkpoint_score(result):
    score = float(result["overall"]["full_map"])
    if not math.isfinite(score):
        raise ValueError(
            "checkpoint selection requires finite overall validation Full-mAP"
        )
    return score


def train(cfg):
    validate_training_schedule(cfg)
    train_cfg = cfg["train"]
    if not 0 <= train_cfg["warmup_epochs"] <= train_cfg["epochs"]:
        raise ValueError("warmup_epochs must be between zero and total epochs")
    if "case_balanced" in train_cfg:
        raise ValueError("remove obsolete train.case_balanced; batches are uniform")
    seed, device = train_cfg["seed"], resolve_device(cfg)
    torch.manual_seed(seed)
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples, val_samples = split_samples(data, "train"), split_samples(data, "val")
    if not samples or (cfg["evaluation"]["enabled"] and not val_samples):
        raise ValueError("train/validation samples are required")
    cache = FeatureCache(cfg, data)
    labels = cache.supervision()
    gallery_ids = split_image_ids(data, "train")
    train_vocab = {
        identity: i
        for i, identity in enumerate(
            sorted(
                {
                    str(x["identity_id"])
                    for image_id in gallery_ids
                    for x in data.gt_head_boxes_by_image.get(image_id, [])
                }
            )
        )
    }
    identity_pools = identity_candidate_pools(data, samples, gallery_ids)
    excluded = negative_exclusions(samples)
    index = CandidateIndex(samples, gallery_ids, identity_pools, excluded)
    sampling = sampling_settings(train_cfg["sampling"])
    model = RCRModel(**cache.dimensions, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    amp = train_cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists():
        raise ValueError(
            "output already contains a training run; select a new output.dir"
        )
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(
        output / "training_data.json",
        {
            "split": "train",
            "queries": len(samples),
            "identities": len(train_vocab),
            "excluded_negative_pairs": sum(map(len, excluded.values())),
            "sampling": sampling,
            "null_policy": "confirmed unusable counterpart only; "
            "unknown detections/absence ignored",
        },
    )
    run = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb

        run = wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("name"),
            mode=cfg["wandb"].get("mode", "online"),
            dir=str(output),
            config=cfg,
        )
    best_score, best_epoch, hard_pools = None, None, None
    try:
        for epoch in range(train_cfg["epochs"]):
            warmup = epoch < train_cfg["warmup_epochs"]
            if (
                epoch > train_cfg["warmup_epochs"]
                and sampling["hard_fraction"] > 0
                and (epoch - train_cfg["warmup_epochs"] - 1)
                % sampling["refresh_every_epochs"]
                == 0
            ):
                hard_pools = mine_hard_negatives(
                    samples,
                    cache,
                    model,
                    device,
                    gallery_ids=gallery_ids,
                    retrieval=cfg["retrieval"],
                    pool_size=sampling["pool_size"],
                    excluded=excluded,
                )
                write_json(
                    output / "hard_negatives.json",
                    {"split": "train", "model_epoch": epoch, "pools": hard_pools},
                )
            generator = torch.Generator().manual_seed(seed + epoch)
            batches = training_batches(samples, train_cfg["batch_size"], generator)
            totals, counts = {}, {}
            model.train()

            def jobs(
                batches=batches,
                generator=generator,
                hard_pools=hard_pools,
                warmup=warmup,
            ):
                for rows in batches:
                    stats = {}
                    candidates = sample_candidates(
                        rows,
                        gallery_ids,
                        train_cfg["candidates"],
                        generator,
                        identity_pools=identity_pools,
                        hard_pools=hard_pools,
                        excluded=excluded,
                        identity_fraction=sampling["identity_fraction"],
                        hard_fraction=0 if warmup else sampling["hard_fraction"],
                        stats=stats,
                        index=index,
                        positives_per_query=sampling["positives_per_query"],
                    )
                    yield rows, candidates, stats

            def prepare(job):
                rows, candidates, stats = job
                return (
                    rows,
                    stats,
                    build_batch(rows, candidates, cache, labels, train_vocab),
                )

            with prefetch_batches(
                jobs(), prepare, cfg["cache"]["prefetch_batches"]
            ) as ready:
                progress = tqdm(
                    ready,
                    total=len(batches),
                    desc=(
                        f"{'warmup' if warmup else 'train'} "
                        f"{epoch + 1}/{train_cfg['epochs']}"
                    ),
                )
                for rows, stats, batch in progress:
                    optimizer.zero_grad(set_to_none=True)
                    batch = to_device(batch, device)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        loss, parts = compute_loss(
                            model,
                            batch,
                            cache.patch_hw,
                            warmup=warmup,
                            pair_batch_size=train_cfg["pair_batch_size"],
                            **cfg["loss"],
                        )
                    if not torch.isfinite(loss):
                        write_json(
                            output / "nonfinite_batch.json",
                            {
                                "epoch": epoch + 1,
                                "sample_ids": [s["sample_id"] for s in rows],
                                "loss": str(loss.item()),
                            },
                        )
                        raise FloatingPointError(
                            "nonfinite loss; optimizer not stepped"
                        )
                    if loss.requires_grad:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_cfg["max_grad_norm"],
                            error_if_nonfinite=True,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    for key, value in {
                        "loss": loss.item(),
                        **{
                            k: parts[k]
                            for k in ("grounding", "identity", "matching", "retrieval")
                        },
                    }.items():
                        totals[key] = totals.get(key, 0) + value * len(rows)
                    for key, value in {
                        **stats,
                        **{k: v for k, v in parts.items() if k not in totals},
                    }.items():
                        counts[key] = (
                            max(counts.get(key, 0), value)
                            if key.startswith("transport_")
                            else counts.get(key, 0) + value
                        )
                    progress.set_postfix(loss=f"{loss.item():.4f}")
                    del batch, loss
            log = {
                "epoch": epoch + 1,
                "warmup": warmup,
                **{k: v / len(samples) for k, v in totals.items()},
                **counts,
            }
            log["identity_active_rate"] = counts.get(
                "identity_active_anchors", 0
            ) / max(counts.get("identity_anchors", 0), 1)
            is_best = False
            # Warmup checkpoints are saved separately and never selected as primary.
            if (
                not warmup
                and cfg["evaluation"]["enabled"]
                and (
                    (epoch + 1) % cfg["evaluation"]["every_epochs"] == 0
                    or epoch + 1 == train_cfg["epochs"]
                )
            ):
                rankings = retrieve_rankings(
                    val_samples,
                    cache,
                    model,
                    device,
                    gallery_ids=split_image_ids(data, "val"),
                    **cfg["retrieval"],
                )
                result = evaluate_retrieval_output(
                    data, val_samples, rankings, cfg["candidate_ks"], split="val"
                )
                directory = output / "evaluation" / f"epoch_{epoch + 1:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "val_metrics.json", result)
                log.update(
                    {f"val/{key}": value for key, value in result["overall"].items()}
                )
                score = checkpoint_score(result)
                if best_score is None or score > best_score:
                    best_score, best_epoch, is_best = score, epoch + 1, True
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
            if run is not None:
                run.log(log)
            checkpoint = {
                "architecture_version": ARCHITECTURE_VERSION,
                "cache_version": CACHE_VERSION,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "dimensions": cache.dimensions,
                "cache_id": cache.cache_id,
                "source": source_signature(cache.source),
                "encoder_metadata": cache.encoder_metadata,
                "text_sha256": cache.index["text_sha256"],
                "supervision_sha256": cache.index["supervision_sha256"],
                "train_sha256": split_fingerprint(data, "train"),
                "validation_sha256": split_fingerprint(data, "val"),
                "epoch": epoch + 1,
                "warmup": warmup,
                "best_full_map": best_score,
                "best_epoch": best_epoch,
            }
            atomic_torch_save(checkpoint, output / "last.pt")
            if warmup and epoch + 1 == train_cfg["warmup_epochs"]:
                atomic_copy(output / "last.pt", output / "warmup.pt")
            if is_best:
                atomic_copy(output / "last.pt", output / "best.pt")
    finally:
        if run is not None:
            run.finish()
    return output / ("best.pt" if best_epoch is not None else "last.pt")
