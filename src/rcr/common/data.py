"""Load finalized RCR data; sample validation belongs to dataset construction."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rcr.common.io import load_jsonl
from rcr.dataset.text import parse_selection_texts

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

    samples_by_id = {row["sample_id"]: row for row in samples}
    images_by_id = {row["image_id"]: row for row in images}
    if len(samples_by_id) != len(samples):
        raise ValueError("duplicate sample_id in samples.jsonl")
    if len(images_by_id) != len(images):
        raise ValueError("duplicate image_id in images.jsonl")
    gallery_ids = [row["image_id"] for row in gallery]
    if len(set(gallery_ids)) != len(gallery_ids) or set(gallery_ids) != set(
        images_by_id
    ):
        raise ValueError("gallery must contain every registered image exactly once")
    splits = _load_split_ids(final_dir)
    seen = set()
    for name, ids in splits.items():
        if seen.intersection(ids):
            raise ValueError(f"sample belongs to more than one split: {name}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate sample_id in {name}.txt")
        seen.update(ids)
    if seen != set(samples_by_id):
        raise ValueError("splits must contain every sample exactly once")

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
    """Return split queries with Full Positives restricted to the same gallery.

    Reviewed annotations may include valid positives outside the query split.
    Project their labels in memory without changing the stored annotations.
    """
    gallery = set(split_image_ids(data, split))
    samples = []
    for sample_id in data.splits[split]:
        sample = data.samples_by_id[sample_id]
        if not {sample["query_image_id"], sample["target_image_id"]} <= gallery:
            raise ValueError(f"{sample_id}: query and seed must belong to {split}")
        samples.append(
            {
                **sample,
                "positive_image_ids": [
                    image_id
                    for image_id in sample["positive_image_ids"]
                    if image_id in gallery
                ],
            }
        )
    return samples


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
    """All registered images in a split, retaining the canonical gallery order."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid query split {split!r}")
    return [
        image_id
        for image_id in data.gallery_ids
        if PurePosixPath(data.images_by_id[image_id]["path"]).parts[0] == split
    ]


def split_fingerprint(data: RCRData, split: str) -> str:
    """Fingerprint the exact split text/labels/gallery used for model selection."""
    images = split_image_ids(data, split)
    payload = {
        "samples": split_samples(data, split),
        "images": [data.images_by_id[image_id] for image_id in images],
        "heads": [data.gt_head_boxes_by_image.get(image_id, []) for image_id in images],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
