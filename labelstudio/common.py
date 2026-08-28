"""Shared helpers for offline Label Studio handoffs."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from rcr.utils.jsonl import load_jsonl, write_jsonl

PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")
INDEX_DATA = Path("dataset/data/raw/metadata/index.txt")
COLORS = ("#e53935", "#1e88e5", "#43a047", "#f9a825", "#8e24aa")


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


def write_tasks(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON task list: {path}")
    return data


def local_image_url(url: str) -> str:
    value = parse_qs(urlparse(url).query).get("d", [""])[0]
    marker = "/images/"
    if marker in value:
        value = value.split(marker, 1)[1]
    return f"/data/local-files/?d={quote(value, safe='/')}"


def _color(subject_id: int) -> str:
    return COLORS[(subject_id - 1) % len(COLORS)]


def _subject_by_identity(subjects: list[dict]) -> dict[str, int]:
    return {
        identity_id: subject["subject_id"]
        for subject in subjects
        for identity_id in subject["identity_ids"]
    }


def image_html(
    image_id: str,
    image_url: str,
    subjects: list[dict],
    boxes_by_image: dict[str, list[dict]],
    title: str,
    candidate_identity_ids: set[str] | None = None,
) -> str:
    subject_by_identity = _subject_by_identity(subjects)
    overlays = []

    for box in boxes_by_image.get(image_id, []):
        identity_id = box["label"]
        subject_id = subject_by_identity.get(identity_id)
        if subject_id is None and (
            candidate_identity_ids is None or identity_id not in candidate_identity_ids
        ):
            continue

        if subject_id is None:
            border = "2px dashed #757575"
            label = (
                "position:absolute;left:0;top:0;padding:1px 4px;"
                "background:#757575;color:white;font-size:12px"
            )
            text = f"ID {html.escape(identity_id)}"
        else:
            color = _color(subject_id)
            border = f"3px solid {color}"
            label = (
                "position:absolute;left:0;top:0;padding:1px 4px;"
                f"background:{color};color:white;font-size:12px"
            )
            text = f"S{subject_id} · ID {html.escape(identity_id)}"

        style = (
            f"left:{100 * box['x']:.3f}%;top:{100 * box['y']:.3f}%;"
            f"width:{100 * box['width']:.3f}%;"
            f"height:{100 * box['height']:.3f}%;"
            f"border:{border};"
            "position:absolute;box-sizing:border-box;pointer-events:none"
        )
        overlays.append(
            f'<div style="{style}"><span style="{label}">{text}</span></div>'
        )

    url = html.escape(local_image_url(image_url), quote=True)
    return (
        '<div style="flex:1;min-width:280px">'
        f"<b>{html.escape(title)}</b>"
        '<div style="position:relative;margin-top:6px">'
        f'<a href="{url}" target="_blank"><img src="{url}" '
        'style="width:100%;display:block"></a>'
        f"{''.join(overlays)}"
        "</div></div>"
    )


def subjects_html(subjects: list[dict]) -> str:
    rows = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        ids = ", ".join(subject["identity_ids"])
        rows.append(
            f'<div><b style="color:{_color(subject_id)}">Subject {subject_id}</b>'
            f" &nbsp; IDs: {html.escape(ids)}</div>"
        )
    return "".join(rows)


def original_html(row: dict) -> str:
    rows = []

    for subject in row["subjects"]:
        rows.append(
            f"<b>Subject {subject['subject_id']}</b><br>"
            f"Description: {html.escape(subject.get('description') or '')}<br>"
            f"Change: {html.escape(subject.get('change') or '')}"
        )

    pair_change = row.get("pair_change")
    if pair_change:
        rows.append(
            f"<b>Relation</b><br>{html.escape(pair_change.get('relation') or '')}"
        )

    return "<br><br>".join(rows)


def textarea(annotation: dict, name: str) -> str | None:
    for result in annotation.get("result", []):
        if result.get("from_name") == name:
            values = result.get("value", {}).get("text", [])
            if values and values[0].strip():
                return values[0].strip()
    return None


def choices(annotation: dict, name: str) -> list[str]:
    selected = optional_choices(annotation, name)
    return selected if selected is not None else []


def optional_choices(annotation: dict, name: str) -> list[str] | None:
    for result in annotation.get("result", []):
        if result.get("from_name") == name:
            return result.get("value", {}).get("choices", [])
    return None


def latest_annotation(task: dict) -> dict | None:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled")
    ]
    return annotations[-1] if annotations else None


def merge_output(path: Path, additions: list[dict]) -> None:
    existing = load_jsonl(path) if path.exists() else []
    rows = {row["submission_id"]: row for row in existing}
    rows.update({row["submission_id"]: row for row in additions})
    write_jsonl(path, rows.values())
