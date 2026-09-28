"""Canonical gallery construction from the PIPA index and local image tree."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def indexed_image_ids(index_lines: Iterable[str]) -> list[str]:
    """Return unique image IDs in deterministic first-seen index order."""

    image_ids: dict[str, None] = {}
    for line in index_lines:
        album_id, photo_id, *_ = line.split()
        image_ids.setdefault(f"{album_id}_{photo_id}", None)
    return list(image_ids)


def build_indexed_gallery(
    index_lines: Iterable[str],
    image_root: Path,
) -> list[JsonObject]:
    """Resolve every indexed image to its split-relative local image path."""

    image_ids = indexed_image_ids(index_lines)
    required = set(image_ids)
    paths_by_id: dict[str, str] = {}

    for directory, _, filenames in os.walk(image_root, followlinks=True):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            image_id = path.stem
            if image_id not in required:
                continue
            if image_id in paths_by_id:
                raise ValueError(f"duplicate image file for indexed image {image_id}")

            paths_by_id[image_id] = path.relative_to(image_root).as_posix()

    missing = [image_id for image_id in image_ids if image_id not in paths_by_id]
    if missing:
        raise ValueError(
            f"indexed gallery image is missing from {image_root}: {missing[0]}"
        )

    return [
        {
            "image_id": image_id,
            "path": paths_by_id[image_id],
            "url": f"/images/{paths_by_id[image_id]}",
        }
        for image_id in image_ids
    ]
