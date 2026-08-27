"""Final RCR dataset construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rcr.dataset.rewrite import (
    CASE_TYPES,
    prepare_rewrite_inputs,
    validate_rewrite_output,
)

JsonObject = dict[str, Any]


def _index_by_submission_id(
    records: Iterable[JsonObject],
    label: str,
) -> dict[str, JsonObject]:
    """Index handoff records and reject missing or duplicate IDs."""

    indexed = {}

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{label}: record {position} must be a JSON object")

        submission_id = record.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError(f"{label}: record {position} has an invalid submission_id")
        if submission_id in indexed:
            raise ValueError(f"{label}: duplicate submission_id {submission_id}")
        indexed[submission_id] = record

    return indexed


def _validate_review(record: JsonObject) -> None:
    required_fields = {"submission_id", "final_desc", "final_change"}
    allowed_fields = required_fields | {"case_type"}
    if not required_fields <= set(record) or not set(record) <= allowed_fields:
        raise ValueError(
            f"{record.get('submission_id')}: reviewed record must contain "
            "submission_id, final_desc, final_change, and optional case_type"
        )

    case_type = record.get("case_type")
    if case_type is not None and case_type not in CASE_TYPES:
        raise ValueError(
            f"{record['submission_id']}: invalid case_type {case_type!r}"
        )

    for field in ("final_desc", "final_change"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{record['submission_id']}: {field} must be a non-empty string"
            )


def _validate_positive_set(record: JsonObject) -> None:
    expected_fields = {"submission_id", "target_image_ids"}
    if set(record) != expected_fields:
        raise ValueError(
            f"{record.get('submission_id')}: positive-set record must contain "
            "exactly submission_id and target_image_ids"
        )

    targets = record["target_image_ids"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or len(targets) != len(set(targets))
    ):
        raise ValueError(
            f"{record['submission_id']}: target_image_ids must be a non-empty "
            "list of unique strings"
        )


def build_final_dataset(
    selected: list[JsonObject],
    reviewed: list[JsonObject],
    positive_sets: list[JsonObject],
    pair_data: JsonObject,
    index_lines: Iterable[str],
    version: str,
) -> JsonObject:
    """Build the final RCR dataset."""

    selected_by_id = _index_by_submission_id(selected, "selected")
    reviewed_by_id = _index_by_submission_id(reviewed, "reviewed")
    positives_by_id = _index_by_submission_id(positive_sets, "positive_sets")
    sample_ids = list(selected_by_id)

    if set(reviewed_by_id) != set(sample_ids):
        raise ValueError("reviewed samples do not match selected samples")

    if set(positives_by_id) != set(sample_ids):
        raise ValueError("positive sets do not match selected samples")

    for submission_id, record in reviewed_by_id.items():
        _validate_review(record)
        source = prepare_rewrite_inputs([selected_by_id[submission_id]])[0]
        validate_rewrite_output(
            source,
            {
                "final_desc": record["final_desc"],
                "final_change": record["final_change"],
            },
        )
    for record in positives_by_id.values():
        _validate_positive_set(record)

    images = [
        {
            "image_id": image["image_id"],
            "path": image["url"].split("/PIPA/images/", 1)[1],
        }
        for image in pair_data["images"]
    ]
    gallery_ids = {image["image_id"] for image in images}

    head_boxes = []
    identities_by_image = defaultdict(set)

    for line in index_lines:
        album_id, photo_id, x, y, width, height, identity_id, _ = line.split()
        image_id = f"{album_id}_{photo_id}"

        if image_id not in gallery_ids:
            continue

        identities_by_image[image_id].add(identity_id)
        head_boxes.append(
            {
                "box_id": f"{image_id}::pid{identity_id}",
                "image_id": image_id,
                "identity_id": identity_id,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
        )

    samples = []
    splits = {"train": [], "val": [], "test": []}

    for source in selected:
        submission_id = source["submission_id"]
        review = reviewed_by_id[submission_id]
        targets = positives_by_id[submission_id]["target_image_ids"]

        if source["target_image_id"] not in targets:
            raise ValueError(f"{submission_id}: missing seed positive")

        if source["query_image_id"] in targets:
            raise ValueError(f"{submission_id}: query image is positive")

        subjects = [
            {
                "subject_id": subject["subjectId"],
                "identity_ids": subject["desc"]["queryGroupIds"],
            }
            for subject in source["annotation"]["subjects"]
        ]
        required_ids = {
            identity_id
            for subject in subjects
            for identity_id in subject["identity_ids"]
        }

        for target in targets:
            if (
                target not in gallery_ids
                or not required_ids <= identities_by_image[target]
            ):
                raise ValueError(f"{submission_id}: invalid positive {target}")

        final_desc = review["final_desc"].strip()
        final_change = review["final_change"].strip()

        samples.append(
            {
                "sample_id": submission_id,
                "case_type": review.get(
                    "case_type", source["annotation"]["caseType"]
                ),
                "query_image_id": source["query_image_id"],
                "target_image_ids": targets,
                "subjects": subjects,
                "final_desc": final_desc,
                "final_change": final_change,
                "final_instruction": f"{final_desc}; {final_change}.",
            }
        )
        splits[source["split"].lower()].append(submission_id)

    return {
        "samples": samples,
        "images": images,
        "gallery": [{"image_id": image["image_id"]} for image in images],
        "head_boxes": head_boxes,
        "splits": splits,
        "manifest": {
            "version": version,
            "num_samples": len(samples),
            "num_images": len(images),
            "num_head_boxes": len(head_boxes),
            "splits": {split: len(ids) for split, ids in splits.items()},
        },
    }
