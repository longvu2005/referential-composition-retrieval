"""Dataset IO, saved rankings and experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

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


def index_unique(records: Iterable[JsonObject], key: str, label: str) -> dict:
    """Index records and reject duplicate keys at data-loading boundaries."""
    indexed = {}
    for row in records:
        value = row[key]
        if value in indexed:
            raise ValueError(f"{label}: duplicate {key} {value}")
        indexed[value] = row
    return indexed


def atomic_write_text(path: str | Path, content: str) -> None:
    """Publish a text file only after the complete content is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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
    """Keep each split's results under its experiment root."""
    return Path(cfg["output"]["dir"]) / cfg["split"]


def write_json(path: str | Path, value) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_summary(root: str | Path, rows: list[dict]) -> None:
    path = Path(root) / "summary.csv"
    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    print(f"Saved summary: {path}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_results(directory: str | Path, output: dict, metadata: Mapping) -> None:
    """Replace one result; old metrics must never describe new rankings."""
    from rcr.common.torch_io import save_torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").unlink(missing_ok=True)
    # Invalidate metadata first so an interrupted replacement cannot look complete.
    (directory / "run.json").unlink(missing_ok=True)
    save_torch(output, directory / "rankings.pt")
    write_json(directory / "run.json", dict(metadata))
