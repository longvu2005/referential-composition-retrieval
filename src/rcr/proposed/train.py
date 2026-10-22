"""Train the proposed model from frozen DINO and FAFA caches."""

import json
from pathlib import Path
from statistics import fmean

import torch
import yaml
from tqdm import tqdm

from rcr.common.config import validate_training_schedule
from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.common.io import write_json
from rcr.common.runtime import resolve_device
from rcr.common.torch_io import link_checkpoint, save_torch
from rcr.dataset.audit import negative_exclusions
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed.batch import build_batch
from rcr.proposed.cache.build import check_query_text_config
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.identity import IdentityCropSampler, identity_sampling_settings
from rcr.proposed.losses import compute_loss
from rcr.proposed.nn.encoders import (
    QueryTextCache,
    load_text_modules,
    text_encoder_state,
)
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import mine_hard_negatives, retrieve_rankings
from rcr.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)
from rcr.proposed.supervision import TrainSupervision


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
        **cfg["retrieval"],
    )
    result = evaluate_retrieval_output(
        data,
        samples,
        rankings,
        cfg["candidate_ks"],
        split="val",
    )
    selected = _selection_metrics(result)
    result["overall"].update(selected)
    directory = output / "evaluation" / f"epoch_{epoch:03d}"
    write_json(
        directory / "val_metrics.json",
        {"overall": result["overall"], "by_case": result["by_case"]},
    )
    return selected


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
        "encoder_metadata": cache.encoder_metadata,
        "cache_id": cache.cache_id,
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
    torch.manual_seed(cfg["train"]["seed"])
    device = resolve_device(cfg)

    data_cfg = cfg["data"]
    cache_cfg = cfg.get("cache", {})
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    train_samples = split_samples(data, "train")
    val_samples = split_samples(data, "val")
    train_gallery = split_image_ids(data, "train")

    cache = GalleryCache(
        data_cfg["fafa_image_cache"],
        scene_root=data_cfg.get("dino_cache"),
        query_root=data_cfg.get("fafa_query_cache"),
        lru_mib=cache_cfg.get("lru_mib", 0),
        native_lru_mib=cache_cfg.get("native_lru_mib", 0),
        mmap_shards=cache_cfg.get("mmap_shards", False),
    )
    cache.validate_gallery(data.gallery_ids)
    cache.validate_samples([*train_samples, *val_samples])
    check_query_text_config(cache, cfg)

    supervision = TrainSupervision(
        data,
        cache,
        scene_size=cfg.get("image_encoder", {}).get("scene_size"),
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
    state_images = {sample_id: set(pool) for sample_id, pool in identity_pools.items()}
    sampling = sampling_settings(cfg["train"].get("sampling", {}))

    scene, people, *_ = cache.load(torch.tensor([0]))
    scene_dim, person_dim = scene.shape[-1], people.shape[-1]
    model_cfg = cfg["model"]
    dim = model_cfg.get("dim", scene_dim)
    tokenizer, text_encoder = load_text_modules(
        model_cfg["text_model"],
        dim,
        device,
        add_subject_markers=True,
        cache_mib=cache_cfg.get("text_lru_mib", 0),
    )
    text_cache = QueryTextCache(tokenizer)
    text_cache.prepare([*train_samples, *val_samples])
    model = RCRModel.from_config(
        model_cfg,
        input_dim=scene_dim,
        person_input_dim=person_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg["optimizer"]["lr"]},
            {
                "params": text_encoder.proj.parameters(),
                "lr": cfg["optimizer"]["text_lr"],
            },
        ],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    if hasattr(getattr(torch, "amp", None), "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (output / "history.jsonl").write_text("", encoding="utf-8")
    (output / "best.pt").unlink(missing_ok=True)
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
            ("loss", "grounding", "identity", "retrieval", "state"), 0.0
        )
        progress = tqdm(batches, desc=f"epoch {epoch}/{epochs}")
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
                state_image_ids=[state_images[row["sample_id"]] for row in rows],
                identity_crops=identity_crops,
                text_cache=text_cache,
            )
            positives = (batch["positive_mask"] & batch["state_mask"]).sum(1)
            negatives = (~batch["positive_mask"] & batch["state_mask"]).sum(1)
            state_pairs_seen += int((positives * negatives).sum())

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, parts = compute_loss(model, batch, cache.patch_hw, **cfg["loss"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}: "
                    f"{[row['sample_id'] for row in rows]}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            values = {
                "loss": float(loss.detach()),
                **{key: float(value) for key, value in parts.items()},
            }
            for name, value in values.items():
                totals[name] += value * len(rows)
            progress.set_postfix(loss=f"{values['loss']:.4f}")

        epoch_log = {
            "epoch": epoch,
            **{
                f"train_{name}": value / len(train_samples)
                for name, value in totals.items()
            },
        }
        optimizer.zero_grad(set_to_none=True)
        is_best = False
        should_evaluate = evaluation.get("enabled", False) and (
            epoch % evaluation["every_epochs"] == 0 or epoch == epochs
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

    return output / ("best.pt" if best["epoch"] is not None else "last.pt")
