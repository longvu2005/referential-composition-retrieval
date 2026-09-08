"""Final RCR dataset construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rcr.dataset.review import normalize_review_subjects
from rcr.dataset.rewrite import CASE_TYPES, validate_review_output

JsonObject = dict[str, Any]


def _index_by_submission_id(
    records: Iterable[JsonObject],
    label: str,
) -> dict[str, JsonObject]:
    indexed = {}

    for record in records:
        submission_id = record["submission_id"]
        if submission_id in indexed:
            raise ValueError(f"{label}: duplicate submission_id {submission_id}")
        indexed[submission_id] = record

    return indexed




def _validate_handoffs(
    selected_by_id: dict[str, JsonObject],
    reviewed_by_id: dict[str, JsonObject],
    positives_by_id: dict[str, JsonObject],
) -> None:
    expected_ids = set(selected_by_id)

    if set(reviewed_by_id) != expected_ids:
        raise ValueError("reviewed samples do not match selected samples")

    if set(positives_by_id) != expected_ids:
        raise ValueError("positive sets do not match selected samples")

    for submission_id, source in selected_by_id.items():
        review = reviewed_by_id[submission_id]

        case_type = review.get("case_type")
        if case_type is not None and case_type not in CASE_TYPES:
            raise ValueError(
                f"{submission_id}: invalid case_type {case_type!r}"
            )

        reviewed_case = case_type or source["annotation"]["caseType"]
        normalize_review_subjects(
            submission_id,
            reviewed_case,
            review.get("subjects"),
        )
        validate_review_output(
            reviewed_case,
            {
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
            },
        )

        positive_ids = positives_by_id[submission_id]["positive_image_ids"]
        if not positive_ids or len(positive_ids) != len(set(positive_ids)):
            raise ValueError(
                f"{submission_id}: positive_image_ids must be non-empty and unique"
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

    _validate_handoffs(
        selected_by_id,
        reviewed_by_id,
        positives_by_id,
    )

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
        positive_ids = positives_by_id[submission_id]["positive_image_ids"]

        if source["target_image_id"] not in positive_ids:
            raise ValueError(f"{submission_id}: missing seed positive")

        if source["query_image_id"] in positive_ids:
            raise ValueError(f"{submission_id}: query image is positive")

        case_type = review.get(
            "case_type",
            source["annotation"]["caseType"],
        )
        subjects = normalize_review_subjects(
            submission_id,
            case_type,
            review.get("subjects"),
        )

        required_ids = {
            identity_id
            for subject in subjects
            for identity_id in subject["identity_ids"]
        }

        if not required_ids <= identities_by_image[source["query_image_id"]]:
            raise ValueError(
                f"{submission_id}: reviewed identities must appear in the query image"
            )

        for positive_id in positive_ids:
            if (
                positive_id not in gallery_ids
                or not required_ids <= identities_by_image[positive_id]
            ):
                raise ValueError(f"{submission_id}: invalid positive {positive_id}")

        final_desc = review["final_desc"].strip()
        final_change = review["final_change"].strip()

        samples.append(
            {
                "sample_id": submission_id,
                "case_type": case_type,
                "query_image_id": source["query_image_id"],
                "target_image_id": source["target_image_id"],
                "positive_image_ids": positive_ids,
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
        "gallery": [
            {"image_id": image["image_id"]}
            for image in images
        ],
        "head_boxes": head_boxes,
        "splits": splits,
        "manifest": {
            "version": version,
            "num_samples": len(samples),
            "num_images": len(images),
            "num_head_boxes": len(head_boxes),
            "splits": {
                split: len(sample_ids)
                for split, sample_ids in splits.items()
            },
        },
    }