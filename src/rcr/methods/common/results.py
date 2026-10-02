"""Small helpers for baseline caches and the shared saved-ranking format."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch


def output_directory(cfg: dict) -> Path:
    """Use one path rule in retrieval and evaluation, including CLI overrides."""
    return Path(
        cfg["output"]["dir"].format(
            method=cfg["method"], mode=cfg.get("mode", ""), split=cfg["split"]
        )
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_directory(root: str | Path, signature: dict) -> Path:
    """Separate incompatible caches; publish completed files with atomic rename."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(payload.encode()).hexdigest()[:24]
    directory = Path(root) / key
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "signature.json").write_text(payload + "\n", encoding="utf-8")
    return directory


def image_signature(data, gallery_ids: Sequence[str]) -> list[list]:
    """Fingerprint ordered image inputs without using identity/positive labels."""
    signature = []
    for image_id in gallery_ids:
        path = data.image_path(image_id)
        stat = path.stat()  # Fail before loading a model if an image is missing.
        signature.append(
            [image_id, str(path.resolve()), stat.st_size, stat.st_mtime_ns]
        )
    return signature


def scores_to_rankings(
    samples: Sequence[dict], gallery_ids: Sequence[str], scores: np.ndarray
) -> dict:
    """Rank all split images, excluding self; ties follow canonical gallery order."""
    gallery_ids = list(gallery_ids)
    sample_ids = [sample["sample_id"] for sample in samples]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("samples must be non-empty with unique sample_ids")
    if len(gallery_ids) < 2 or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("gallery_ids must contain at least two unique images")
    if scores.shape != (len(samples), len(gallery_ids)):
        raise ValueError("score shape does not match samples and gallery_ids")
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("scores must have a floating-point dtype")
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    rankings = np.empty((len(samples), len(gallery_ids) - 1), dtype=np.int32)
    for index, sample in enumerate(samples):
        row = np.asarray(scores[index])
        if not np.isfinite(row).all():
            raise ValueError(f"{sample['sample_id']}: scores contain NaN/Inf")
        query_index = gallery_index[sample["query_image_id"]]
        order = np.argsort(-row, kind="stable")
        rankings[index] = order[order != query_index]
    return {
        "sample_ids": sample_ids,
        "gallery_ids": gallery_ids,
        "rankings": torch.from_numpy(rankings),
    }


def save_results(directory: str | Path, output: dict, metadata: Mapping) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "rankings.pt.tmp"
    metadata_temporary = directory / "run.json.tmp"
    torch.save(output, temporary)
    metadata_temporary.write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Reusing a run directory must not leave metrics for the previous rankings.
    (directory / "metrics.json").unlink(missing_ok=True)
    temporary.replace(directory / "rankings.pt")
    metadata_temporary.replace(directory / "run.json")
