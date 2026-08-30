"""Shared metadata helpers for the repo-local human annotation UIs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")
INDEX_DATA = Path("dataset/data/raw/metadata/index.txt")


def _axis_size(
    start: int,
    size: int,
    norm_start: float,
    norm_size: float,
) -> int | None:
    candidates = []

    if norm_start > 0:
        candidates.append(start / norm_start)
    if norm_start + norm_size < 1:
        candidates.append((start + size) / (norm_start + norm_size))

    if not candidates:
        return None

    rounded = {round(value) for value in candidates}
    if len(rounded) != 1:
        raise ValueError("Inconsistent box metadata.")
    return rounded.pop()


def _normalize_box(box: dict, width: int, height: int) -> dict:
    left = max(box["x"], 0)
    top = max(box["y"], 0)
    right = min(box["x"] + box["width"], width)
    bottom = min(box["y"] + box["height"], height)

    return {
        "image_id": box["image_id"],
        "label": box["label"],
        "x": left / width,
        "y": top / height,
        "width": max(0, right - left) / width,
        "height": max(0, bottom - top) / height,
    }


def load_box_index(
    pair_path: Path = PAIR_DATA,
    index_path: Path = INDEX_DATA,
) -> dict[str, list[dict]]:
    """Load complete normalized head boxes using index.txt as the source."""

    pair_data = json.loads(pair_path.read_text(encoding="utf-8"))
    gallery_ids = {image["image_id"] for image in pair_data["images"]}
    pair_boxes = {(box["image_id"], box["label"]): box for box in pair_data["boxes"]}

    index_boxes = []
    image_sizes: dict[str, list[set[int]]] = defaultdict(lambda: [set(), set()])

    for line in index_path.read_text(encoding="utf-8").splitlines():
        album_id, photo_id, x, y, width, height, identity_id, _ = line.split()
        image_id = f"{album_id}_{photo_id}"
        if image_id not in gallery_ids:
            continue

        box = {
            "image_id": image_id,
            "label": identity_id,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }
        index_boxes.append(box)

        pair_box = pair_boxes.get((image_id, identity_id))
        if pair_box is None:
            continue

        image_width = _axis_size(
            box["x"], box["width"], pair_box["x"], pair_box["width"]
        )
        image_height = _axis_size(
            box["y"], box["height"], pair_box["y"], pair_box["height"]
        )
        if image_width is not None:
            image_sizes[box["image_id"]][0].add(image_width)
        if image_height is not None:
            image_sizes[box["image_id"]][1].add(image_height)

    boxes = defaultdict(list)
    for box in index_boxes:
        widths, heights = image_sizes[box["image_id"]]
        if len(widths) > 1 or len(heights) > 1:
            raise ValueError(f"{box['image_id']}: inconsistent image dimensions")

        if widths and heights:
            boxes[box["image_id"]].append(
                _normalize_box(box, next(iter(widths)), next(iter(heights)))
            )
            continue

        pair_box = pair_boxes.get((box["image_id"], box["label"]))
        if pair_box is None:
            raise ValueError(f"{box['image_id']}: cannot normalize head box")
        boxes[box["image_id"]].append(pair_box)

    return boxes
