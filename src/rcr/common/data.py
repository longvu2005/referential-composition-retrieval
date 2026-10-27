"""Load finalized RCR data; sample validation belongs to dataset construction."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rcr.common.io import index_unique, load_jsonl
from rcr.dataset.text import parse_selection_texts

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class RCRInputs:
    """Image/text/Subject-slot inputs, without supervision or case labels."""

    final_dir: Path
    image_root: Path
    samples: list[dict]
    samples_by_id: dict[str, dict]
    images_by_id: dict[str, dict]
    gallery_ids: list[str]
    splits: dict[str, list[str]]
    manifest: dict

    def image_path(self, image_id: str) -> Path:
        return self.image_root / self.images_by_id[image_id]["path"]


@dataclass
class RCRData(RCRInputs):
    """Annotated data for training, evaluation and experiment provenance."""

    gt_head_boxes_by_image: dict[str, list[dict]]


def inference_sample(sample: dict) -> dict:
    """Allowlist public fields; never forward the annotated Subject objects."""
    return {
        **{
            key: sample[key]
            for key in (
                "sample_id",
                "query_image_id",
                "final_desc",
                "final_change",
                "final_instruction",
            )
            if key in sample
        },
        "subjects": [{"subject_id": s["subject_id"]} for s in sample["subjects"]],
    }


def inference_samples(samples) -> list[dict]:
    return [inference_sample(sample) for sample in samples]


def inference_data(data: RCRInputs) -> RCRInputs:
    """Public view for scorers; the runner retains annotations separately."""
    samples = inference_samples(data.samples)
    return RCRInputs(
        final_dir=data.final_dir,
        image_root=data.image_root,
        samples=samples,
        samples_by_id={s["sample_id"]: s for s in samples},
        images_by_id={
            i: {"image_id": i, "path": row["path"]}
            for i, row in data.images_by_id.items()
        },
        gallery_ids=list(data.gallery_ids),
        splits={name: list(ids) for name, ids in data.splits.items()},
        manifest={"version": data.manifest.get("version")},
    )


def load_rcr_inputs(
    final_dir: str | Path = "dataset/data/final",
    image_root: str | Path = "dataset/data/raw/images",
) -> RCRInputs:
    """Load public fields without retaining labels or opening head annotations."""
    return _load_data(final_dir, image_root, inputs_only=True)


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
    """Load annotations for supervision, evaluation and provenance."""
    return _load_data(final_dir, image_root, inputs_only=False)


def _load_data(final_dir, image_root, *, inputs_only):

    final_dir = Path(final_dir)
    image_root = Path(image_root)

    samples = load_jsonl(final_dir / "samples.jsonl")
    images = load_jsonl(final_dir / "images.jsonl")
    gallery = load_jsonl(final_dir / "gallery.jsonl")
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))

    samples_by_id = index_unique(samples, "sample_id", "samples.jsonl")
    images_by_id = index_unique(images, "image_id", "images.jsonl")
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

    inputs = RCRInputs(
        final_dir=final_dir,
        image_root=image_root,
        samples=samples,
        samples_by_id=samples_by_id,
        images_by_id=images_by_id,
        gallery_ids=gallery_ids,
        splits=splits,
        manifest=manifest,
    )
    if inputs_only:
        return inference_data(inputs)
    gt_head_boxes_by_image = defaultdict(list)
    for box in load_jsonl(final_dir / "head_boxes.jsonl"):
        gt_head_boxes_by_image[box["image_id"]].append(box)
    return RCRData(**vars(inputs), gt_head_boxes_by_image=dict(gt_head_boxes_by_image))


def split_inputs(data: RCRInputs, split: str) -> list[dict]:
    """Select public queries independently of seed targets and positive labels."""
    gallery = set(split_image_ids(data, split))
    rows = [inference_sample(data.samples_by_id[i]) for i in data.splits[split]]
    if any(row["query_image_id"] not in gallery for row in rows):
        raise ValueError(f"query must belong to {split}")
    return rows


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


def subject_identity_ids(subjects: list[dict]) -> list[str]:
    return [
        identity_id for subject in subjects for identity_id in subject["identity_ids"]
    ]


def required_identity_ids(sample: dict) -> list[str]:
    """Flatten required identities in deterministic Subject order."""
    return subject_identity_ids(sample["subjects"])


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
