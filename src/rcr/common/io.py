"""Dataset IO, saved rankings and experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import numpy as np
import torch

JsonObject = dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[JsonObject]:
    """Iterate over non-empty JSONL records."""

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[JsonObject]:
    """Load all records from a JSONL file."""

    return list(iter_jsonl(path))


def write_jsonl(
    path: str | Path,
    records: Iterable[JsonObject],
) -> None:
    """Write records to a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def image_relative_path(url: str) -> Path:
    """Map PIPA/local-files URLs to safe split-relative image paths."""

    parsed = urlparse(url)
    value = parse_qs(parsed.query).get("d", [parsed.path])[0]
    value = unquote(value).replace("\\", "/")
    for marker in ("/PIPA/images/", "/images/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.lstrip("/")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image path {url!r}")
    return path


def output_directory(cfg: dict) -> Path:
    """Each run root owns separate mode/split directories; no path templates."""
    root = Path(cfg["output"]["dir"])
    if cfg["method"] == "clip":
        root /= cfg["mode"]
    return root / cfg["split"]


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_summary(root: str | Path, rows: list[dict]) -> None:
    path = Path(root) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    preserve_file(path)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        # Ablations can report different candidate K values (e.g. different Top-M).
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    print(f"Saved summary: {path}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_directory(directory: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    history = directory / ".history" / f"{stamp}-{uuid4().hex[:8]}"
    history.mkdir(parents=True)
    return history


def preserve_file(path: str | Path) -> None:
    """Keep the previous file before publishing a replacement."""
    path = Path(path)
    if path.is_file():
        shutil.copy2(path, _history_directory(path.parent) / path.name)


def preserve_run(directory: str | Path) -> None:
    """Move a previous run together, before writing any new scores/rankings."""
    directory = Path(directory)
    previous = [
        directory / name
        for name in ("scores.npy", "rankings.pt", "run.json", "metrics.json")
        if (directory / name).is_file()
    ]
    if previous:
        history = _history_directory(directory)
        for path in previous:
            path.replace(history / path.name)


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
    # Baselines archive before writing scores; Proposed publishes rankings here.
    if any((directory / name).is_file() for name in ("rankings.pt", "run.json")):
        preserve_run(directory)
    else:
        preserve_file(directory / "metrics.json")
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
