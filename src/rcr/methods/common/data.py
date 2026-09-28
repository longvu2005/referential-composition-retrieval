"""Load finalized RCR data; sample validation belongs to dataset construction."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rcr.dataset.rewrite import parse_selection_texts
from rcr.utils.jsonl import load_jsonl

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class RCRData:
    """In-memory indexes over `dataset/data/final`."""

    final_dir: Path
    image_root: Path
    samples: list[dict]
    samples_by_id: dict[str, dict]
    images_by_id: dict[str, dict]
    gallery_ids: list[str]
    gt_head_boxes_by_image: dict[str, list[dict]]
    splits: dict[str, list[str]]
    manifest: dict

    def image_path(self, image_id: str) -> Path:
        return self.image_root / self.images_by_id[image_id]["path"]


def _unique_index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    output = {row[key]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label}: duplicate {key}")
    return output


def _load_split_ids(final_dir: Path) -> dict[str, list[str]]:
    splits = {}
    for name in SPLIT_NAMES:
        path = final_dir / "splits" / f"{name}.txt"
        splits[name] = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return splits


def _check_indexes(
    samples_by_id: dict[str, dict],
    images_by_id: dict[str, dict],
    gallery_ids: list[str],
    splits: dict[str, list[str]],
) -> None:
    gallery_set = set(gallery_ids)
    image_ids = set(images_by_id)

    if len(gallery_ids) != len(gallery_set):
        raise ValueError("gallery.jsonl contains duplicate image_id")
    if gallery_set != image_ids:
        raise ValueError("gallery.jsonl must contain exactly the registered images")

    split_ids = [sample_id for name in SPLIT_NAMES for sample_id in splits[name]]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("a sample_id appears in more than one split")
    if set(split_ids) != set(samples_by_id):
        raise ValueError("splits must contain every sample exactly once")


def load_rcr_data(
    final_dir: str | Path = "dataset/data/final",
    image_root: str | Path = "dataset/data/raw/images",
) -> RCRData:
    """Load final benchmark files and build small deterministic lookup indexes."""

    final_dir = Path(final_dir)
    image_root = Path(image_root)

    samples = load_jsonl(final_dir / "samples.jsonl")
    images = load_jsonl(final_dir / "images.jsonl")
    gallery = load_jsonl(final_dir / "gallery.jsonl")
    head_boxes = load_jsonl(final_dir / "head_boxes.jsonl")
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))

    samples_by_id = _unique_index(samples, "sample_id", "samples.jsonl")
    images_by_id = _unique_index(images, "image_id", "images.jsonl")
    gallery_ids = [row["image_id"] for row in gallery]
    splits = _load_split_ids(final_dir)
    _check_indexes(samples_by_id, images_by_id, gallery_ids, splits)

    gt_head_boxes_by_image = defaultdict(list)
    for box in head_boxes:
        gt_head_boxes_by_image[box["image_id"]].append(box)

    return RCRData(
        final_dir=final_dir,
        image_root=image_root,
        samples=samples,
        samples_by_id=samples_by_id,
        images_by_id=images_by_id,
        gallery_ids=gallery_ids,
        gt_head_boxes_by_image=dict(gt_head_boxes_by_image),
        splits=splits,
        manifest=manifest,
    )


def split_samples(data: RCRData, split: str) -> list[dict]:
    """Return samples in the saved split order."""

    return [data.samples_by_id[sample_id] for sample_id in data.splits[split]]


def sample_selection_texts(sample: dict) -> list[str]:
    """Return the referential text aligned with `sample['subjects']`."""

    subject_ids = [subject["subject_id"] for subject in sample["subjects"]]
    return parse_selection_texts(sample["final_desc"], subject_ids)


def required_identity_ids(sample: dict) -> list[str]:
    """Flatten required identities in deterministic Subject order."""

    return [
        identity_id
        for subject in sample["subjects"]
        for identity_id in subject["identity_ids"]
    ]


def split_image_ids(data: RCRData, split: str) -> list[str]:
    """Images supervised by a split: queries plus reviewed Full Positives."""

    seen = set()
    image_ids = []
    for sample in split_samples(data, split):
        for image_id in [sample["query_image_id"], *sample["positive_image_ids"]]:
            if image_id not in seen:
                seen.add(image_id)
                image_ids.append(image_id)
    return image_ids
