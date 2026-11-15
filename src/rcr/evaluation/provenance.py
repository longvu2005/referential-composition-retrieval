"""Small, shared provenance records; never used to choose hyperparameters."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import machine, python_version, release, system
from subprocess import DEVNULL, PIPE, Popen

from rcr.common.data import split_fingerprint
from rcr.dataset.cases import CASE_TYPES
from rcr.proposed.contracts import CALIBRATED_RETRIEVAL_KEYS

PROTOCOL = "rcr-id-full-v1"
BENCHMARK_FIELDS = (
    "dataset_version",
    "dataset_sha256",
    "split",
    "split_sha256",
    "sample_ids_sha256",
    "gallery_ids_sha256",
    "num_queries",
    "num_gallery",
    "case_counts",
    "evaluation_protocol",
    "self_exclusion",
    "ranking_scope",
)


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def benchmark_metadata(data, split, output) -> dict:
    """Fingerprint annotations and ordered inputs, independently of file paths.

    The dataset digest is for cross-run checks only. Model and weight selection
    use the validation split fingerprint.
    """
    counts = Counter(
        data.samples_by_id[sample_id]["case_type"] for sample_id in output["sample_ids"]
    )
    manifest = getattr(data, "manifest", {})
    return {
        "dataset_version": manifest.get("version"),
        "dataset_sha256": digest_json(
            {
                "manifest": manifest,
                "samples": data.samples,
                "images": data.images_by_id,
                "gallery": data.gallery_ids,
                "heads": data.gt_head_boxes_by_image,
                "splits": data.splits,
            }
        ),
        "split": split,
        "split_sha256": split_fingerprint(data, split),
        "sample_ids_sha256": digest_json(output["sample_ids"]),
        "gallery_ids_sha256": digest_json(output["gallery_ids"]),
        "num_queries": len(output["sample_ids"]),
        "num_gallery": len(output["gallery_ids"]),
        "case_counts": {case: counts[case] for case in CASE_TYPES},
        "evaluation_protocol": PROTOCOL,
        "self_exclusion": "query_image",
        "ranking_scope": "complete_split_gallery",
    }


def retrieval_metadata(
    data,
    cfg: dict,
    output: dict,
    checkpoint: dict,
    *,
    checkpoint_sha256: str,
    device,
    elapsed_seconds: float,
) -> dict:
    """Describe rankings from training validation or a loaded checkpoint alike."""
    settings = cfg["retrieval"]
    train_cfg = checkpoint["config"]
    return {
        "method": "proposed",
        "config": {
            **cfg,
            "model": {**cfg["model"], "fine_matching": output["fine_matching"]},
        },
        "retrieval": settings,
        "score_schema_version": output["score_schema_version"],
        "schema_version": output["schema_version"],
        "score_config": output["score_config"],
        "clip_text_stats": output.get("clip_text_stats"),
        **{
            key: settings[key]
            for key in (
                "fine_score_mode",
                "context_weight",
                "coverage_weight",
                "fine_coarse_weight",
            )
        },
        "checkpoint": str(cfg["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_sha256": split_fingerprint(data, "val"),
        **benchmark_metadata(data, cfg["split"], output),
        **runtime_metadata(device),
        "architecture_version": checkpoint["architecture_version"],
        "representation": checkpoint.get("representation", "shared_person"),
        "binding_scale": output["binding_scale"],
        "fine_matching": output["fine_matching"],
        "encoder_metadata": checkpoint.get("encoder_metadata"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": train_cfg.get("train", {}).get("seed"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_full_map": checkpoint.get("best_full_map"),
        "best_val_macro_full_map": checkpoint.get("best_macro_full_map"),
        "best_val_checkpoint_score": checkpoint.get("best_score"),
        **{
            f"best_{key}": checkpoint.get(f"best_{key}")
            for key in CALIBRATED_RETRIEVAL_KEYS
        },
        **{
            key: checkpoint.get(key)
            for key in (
                "cache_id",
                "scene_cache_id",
                "detection_cache_id",
                "native_cache",
            )
        },
        "query_subset": len(output["sample_ids"]) != len(data.splits[cfg["split"]]),
        "elapsed_seconds": elapsed_seconds,
    }


def runtime_metadata(device) -> dict:
    """Record the checkout and installed versions without importing models."""
    import torch

    root = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            with Popen(
                ["git", "-C", str(root), *args], stdout=PIPE, stderr=DEVNULL
            ) as process:
                stdout, _ = process.communicate()
                return stdout.decode().strip() if process.returncode == 0 else None
        except OSError:
            return None

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "transformers",
        "scipy",
        "PyYAML",
    ):
        with suppress(PackageNotFoundError):
            packages[name] = version(name)
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "runtime": {
            "python": python_version(),
            "platform": f"{system()} {release()} {machine()}",
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "packages": packages,
        },
    }
