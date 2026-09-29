"""Train the proposed RCR model from cached visual features."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from tqdm import tqdm

from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.proposed.batch import build_batch
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import (
    evaluate_retrieval_output,
    retrieve_rankings,
    slice_retrieval_output,
)
from rcr.methods.proposed.sampling import sample_candidates
from rcr.methods.proposed.training import compute_loss


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

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled; install the project dependencies first"
        ) from exc

    tracked_config = {
        "model": cfg["model"],
        "train": cfg["train"],
        "optimizer": cfg["optimizer"],
        "loss": cfg["loss"],
        "evaluation": cfg.get("evaluation", {"enabled": False}),
        "cache": {
            "id": cache.cache_id,
            "patch_hw": list(cache.patch_hw),
            "feature_dim": feature_dim,
        },
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
    }
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
    run.define_metric("epoch/*", step_metric="epoch")
    run.define_metric("train_eval/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
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


def _validate_evaluation_config(cfg: dict) -> None:
    """Reject invalid periodic-evaluation settings before training starts."""

    if not cfg.get("enabled", False):
        return
    positive_keys = (
        "every_epochs",
        "top_m",
        "fine_batch_size",
        "identity_batch_size",
    )
    for key in positive_keys:
        if int(cfg[key]) < 1:
            raise ValueError(f"evaluation.{key} must be at least 1")
    if int(cfg["train_max_queries"]) < 0:
        raise ValueError("evaluation.train_max_queries must be non-negative")
    candidate_ks = tuple(int(value) for value in cfg["candidate_ks"])
    if not candidate_ks or any(value < 1 for value in candidate_ks):
        raise ValueError("evaluation.candidate_ks must contain positive integers")
    if max(candidate_ks) > int(cfg["top_m"]):
        raise ValueError("evaluation.candidate_ks cannot exceed evaluation.top_m")


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


def _evaluation_due(epoch: int, total_epochs: int, every_epochs: int) -> bool:
    """Evaluate at the requested interval and always at the final epoch."""

    return epoch % every_epochs == 0 or epoch == total_epochs


def _tracked_metrics(prefix: str, result: dict) -> dict[str, float]:
    """Flatten official aggregate metrics for W&B without per-query noise."""

    return {
        f"{prefix}/{name}": float(value)
        for name, value in result["overall"].items()
        if name != "num_queries"
    }


def _save_evaluation(path: Path, result: dict) -> None:
    """Persist compact aggregate metrics; rankings remain in memory only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {"overall": result["overall"], "by_case": result["by_case"]}
    path.write_text(json.dumps(compact, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/train.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optimizer"]
    loss_cfg = cfg["loss"]
    evaluation_cfg = cfg.get("evaluation", {"enabled": False})
    output_cfg = cfg["output"]
    _validate_evaluation_config(evaluation_cfg)

    from transformers import AutoModel, AutoTokenizer

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    device_name = train_cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, "train")
    val_samples = split_samples(data, "val")
    if evaluation_cfg.get("enabled", False) and not val_samples:
        raise ValueError("periodic evaluation requires a non-empty val split")
    train_eval_samples = _fixed_subset(
        samples,
        int(evaluation_cfg.get("train_max_queries", 0)),
        seed,
    )
    cache = GalleryCache(data_cfg["cache"])
    cache.validate_gallery(data.gallery_ids)
    negative_pool = train_cfg.get("negative_pool", "full_gallery")
    if negative_pool == "train_gallery":
        candidate_ids = split_image_ids(data, "train")
    elif negative_pool == "full_gallery":
        candidate_ids = cache.image_ids
    else:
        raise ValueError("negative_pool must be full_gallery or train_gallery")
    print(f"Training negative pool: {negative_pool} ({len(candidate_ids)} images)")

    first_scene, *_ = cache.load(torch.tensor([0]))
    dim = first_scene.shape[-1]
    if dim % model_cfg["num_heads"]:
        raise ValueError("feature dim must be divisible by num_heads")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S1]", "[S2]"]})
    text_backbone = AutoModel.from_pretrained(model_cfg["text_model"])
    text_backbone.resize_token_embeddings(len(tokenizer))
    text_encoder = TextEncoder(text_backbone, dim).to(device)

    max_subjects = max(len(sample["subjects"]) for sample in data.samples)
    model = RCRModel(
        dim=dim,
        identity_dim=model_cfg["identity_dim"],
        num_heads=model_cfg["num_heads"],
        max_subjects=max_subjects,
        mlp_ratio=model_cfg["mlp_ratio"],
        geo_dim=model_cfg["geo_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": optim_cfg["lr"]},
            {"params": text_encoder.parameters(), "lr": optim_cfg["text_lr"]},
        ],
        weight_decay=optim_cfg["weight_decay"],
    )

    groups: dict[int, list[dict]] = {}
    for sample in samples:
        groups.setdefault(len(sample["subjects"]), []).append(sample)

    output = Path(output_cfg["dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "config.yaml")
    tokenizer.save_pretrained(output / "tokenizer")
    wandb_cfg = cfg.get("wandb", {})
    log_every = int(wandb_cfg.get("log_every_steps", 20))
    run = _wandb_run(
        cfg,
        output,
        cache,
        dim,
        len(samples),
        len(val_samples),
    )
    global_step = 0
    best_full_map: float | None = None
    best_epoch: int | None = None

    for epoch in range(train_cfg["epochs"]):
        generator = torch.Generator().manual_seed(seed + epoch)
        batches = []
        for rows in groups.values():
            order = torch.randperm(len(rows), generator=generator).tolist()
            rows = [rows[i] for i in order]
            batches.extend(
                rows[i : i + train_cfg["batch_size"]]
                for i in range(0, len(rows), train_cfg["batch_size"])
            )
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]

        model.train()
        text_encoder.train()
        totals = {
            "loss": 0.0,
            "grounding": 0.0,
            "identity": 0.0,
            "retrieval": 0.0,
            "identity_active": 0.0,
        }

        progress = tqdm(
            batches,
            desc=f"epoch {epoch + 1}/{train_cfg['epochs']}",
        )
        for rows in progress:
            candidates = sample_candidates(
                rows,
                candidate_ids,
                train_cfg["candidates"],
                generator,
            )

            optimizer.zero_grad(set_to_none=True)
            batch = build_batch(
                rows,
                candidates,
                cache,
                tokenizer,
                text_encoder,
                device,
            )
            loss, parts = compute_loss(
                model,
                batch,
                cache.patch_hw,
                grounding_weight=loss_cfg["grounding_weight"],
                identity_weight=loss_cfg["identity_weight"],
                retrieval_weight=loss_cfg["retrieval_weight"],
                identity_temperature=loss_cfg["identity_temperature"],
            )
            loss.backward()
            global_step += 1
            should_log = run is not None and (
                global_step == 1 or global_step % log_every == 0
            )
            grad_norm = _gradient_norm((model, text_encoder)) if should_log else None
            optimizer.step()

            totals["loss"] += loss.item()
            for name, value in parts.items():
                totals[name] += value.item()
            identity_active = float(parts["identity"].item() > 0)
            totals["identity_active"] += identity_active

            progress.set_postfix(loss=f"{loss.item():.4f}")

            if should_log:
                supervised = batch["grounding_targets"].bool().any(dim=-1)
                if "subject_mask" in batch:
                    valid_subjects = batch["subject_mask"].bool()
                    denominator = valid_subjects.sum().clamp_min(1)
                    supervised_rate = (
                        (supervised & valid_subjects).sum() / denominator
                    ).item()
                else:
                    supervised_rate = supervised.float().mean().item()
                run.log(
                    {
                        "global_step": global_step,
                        "train/loss": loss.item(),
                        "train/grounding_loss": parts["grounding"].item(),
                        "train/identity_loss": parts["identity"].item(),
                        "train/identity_active": identity_active,
                        "train/retrieval_loss": parts["retrieval"].item(),
                        "train/grounding_supervised_rate": supervised_rate,
                        "train/gradient_norm": grad_norm,
                        "train/model_lr": optimizer.param_groups[0]["lr"],
                        "train/text_lr": optimizer.param_groups[1]["lr"],
                    }
                )

        count = len(batches)
        summary = {name: value / count for name, value in totals.items()}
        print(
            f"epoch {epoch + 1}: "
            f"loss={summary['loss']:.4f} "
            f"ground={summary['grounding']:.4f} "
            f"id={summary['identity']:.4f} "
            f"retrieval={summary['retrieval']:.4f}"
        )
        epoch_number = epoch + 1
        epoch_log = {
            "epoch": epoch_number,
            "epoch/loss": summary["loss"],
            "epoch/grounding_loss": summary["grounding"],
            "epoch/identity_loss": summary["identity"],
            "epoch/identity_active_rate": summary["identity_active"],
            "epoch/retrieval_loss": summary["retrieval"],
        }
        is_best = False

        if evaluation_cfg.get("enabled", False) and _evaluation_due(
            epoch_number,
            train_cfg["epochs"],
            int(evaluation_cfg["every_epochs"]),
        ):
            combined_samples = [*train_eval_samples, *val_samples]
            retrieved = retrieve_rankings(
                combined_samples,
                cache,
                tokenizer,
                text_encoder,
                model,
                device,
                top_m=int(evaluation_cfg["top_m"]),
                fine_batch_size=int(evaluation_cfg["fine_batch_size"]),
                identity_batch_size=int(evaluation_cfg["identity_batch_size"]),
                description=f"evaluate epoch {epoch_number}",
            )
            metrics_dir = output / "evaluation" / f"epoch_{epoch_number:03d}"
            train_count = len(train_eval_samples)

            if train_count:
                train_output = slice_retrieval_output(retrieved, 0, train_count)
                train_result = evaluate_retrieval_output(
                    data,
                    train_eval_samples,
                    train_output,
                    evaluation_cfg["candidate_ks"],
                )
                _save_evaluation(metrics_dir / "train_metrics.json", train_result)
                epoch_log.update(_tracked_metrics("train_eval", train_result))

            val_output = slice_retrieval_output(
                retrieved,
                train_count,
                len(combined_samples),
            )
            val_result = evaluate_retrieval_output(
                data,
                val_samples,
                val_output,
                evaluation_cfg["candidate_ks"],
            )
            _save_evaluation(metrics_dir / "val_metrics.json", val_result)
            epoch_log.update(_tracked_metrics("val", val_result))

            val_full_map = float(val_result["overall"]["full_map"])
            if best_full_map is None or val_full_map > best_full_map:
                best_full_map = val_full_map
                best_epoch = epoch_number
                is_best = True
            print(
                f"validation epoch {epoch_number}: "
                f"Full-mAP={val_full_map:.4f} "
                f"Full-R@1={val_result['overall']['full_r1']:.4f} "
                f"ID-mAP={val_result['overall']['id_map']:.4f}"
            )

        if run is not None:
            run.log(epoch_log)
            if is_best:
                run.summary["best_val_full_map"] = best_full_map
                run.summary["best_epoch"] = best_epoch

        checkpoint = {
            "epoch": epoch_number,
            "model": model.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "dim": dim,
            "cache_id": cache.cache_id,
            "best_full_map": best_full_map,
            "best_epoch": best_epoch,
        }
        last_path = output / "last.pt"
        torch.save(checkpoint, last_path)
        if is_best:
            shutil.copyfile(last_path, output / "best.pt")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
