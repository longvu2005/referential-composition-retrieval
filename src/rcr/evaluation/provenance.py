"""Small, shared provenance records; never used to choose hyperparameters."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import machine, python_version, release, system
from subprocess import DEVNULL, PIPE, Popen

from rcr.common.data import split_fingerprint
from rcr.dataset.cases import CASE_TYPES

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

    The dataset digest covers all labels for cross-run checks only. Selection
    continues to use the existing validation-only fingerprint/context.
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
        "clip",
        "transformers",
        "scipy",
        "PyYAML",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
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
