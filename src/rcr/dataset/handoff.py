"""Prepare inputs for manual review and Full Positive labeling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rcr.dataset.review import normalize_review_subjects
from rcr.dataset.rewrite import CASE_TYPES, validate_review_output

JsonObject = dict[str, Any]


def _build_metadata(
    pair_data: JsonObject,
    index_lines: Iterable[str],
) -> tuple[
    dict[str, JsonObject],
    dict[str, list[JsonObject]],
    dict[str, set[str]],
]:
    """Build image, box, and identity lookup tables used by both handoffs."""

    image_by_id = {image["image_id"]: image for image in pair_data["images"]}
    gallery_ids = set(image_by_id)
    boxes_by_image = defaultdict(list)
    images_by_identity = defaultdict(set)

    for line in index_lines:
        album_id, photo_id, x, y, width, height, identity_id, _ = line.split()
        image_id = f"{album_id}_{photo_id}"
        if image_id not in gallery_ids:
            continue

        boxes_by_image[image_id].append(
            {
                "identity_id": identity_id,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
        )
        images_by_identity[identity_id].add(image_id)

    return image_by_id, boxes_by_image, images_by_identity


def select_unfinished_records(
    records: Iterable[JsonObject],
    completed_ids: set[str],
) -> list[JsonObject]:
    """Keep records that have not completed the target handoff."""

    return [
        record for record in records if record["submission_id"] not in completed_ids
    ]


def prepare_review_inputs(
    selected: list[JsonObject],
    rewrite_outputs: list[JsonObject],
    pair_data: JsonObject,
    index_lines: Iterable[str],
) -> list[JsonObject]:
    """Attach source images and subject boxes to Gemini rewrite outputs."""

    selected_by_id = {row["submission_id"]: row for row in selected}
    image_by_id, boxes_by_image, _ = _build_metadata(pair_data, index_lines)
    outputs = []

    for rewrite in rewrite_outputs:
        source = selected_by_id[rewrite["submission_id"]]
        annotation = source["annotation"]
        query_id = source["query_image_id"]
        target_id = source["target_image_id"]

        subjects = [
            {
                "subject_id": subject["subjectId"],
                "identity_ids": subject["desc"]["queryGroupIds"],
                "description": subject["desc"]["final"],
                "change": subject["change"]["final"],
            }
            for subject in sorted(
                annotation["subjects"],
                key=lambda subject: subject["subjectId"],
            )
        ]
        subject_by_identity = {}
        for subject in subjects:
            for identity_id in subject["identity_ids"]:
                if identity_id in subject_by_identity:
                    raise ValueError(
                        f"{rewrite['submission_id']}: identity_id {identity_id} is "
                        "assigned to multiple subjects"
                    )
                subject_by_identity[identity_id] = subject["subject_id"]

        query_identity_ids = {
            box["identity_id"] for box in boxes_by_image.get(query_id, [])
        }
        target_identity_ids = {
            box["identity_id"] for box in boxes_by_image.get(target_id, [])
        }
        candidate_identity_ids = sorted(
            query_identity_ids & target_identity_ids,
            key=int,
        )
        missing_ids = set(subject_by_identity) - set(candidate_identity_ids)
        if missing_ids:
            missing = ", ".join(sorted(missing_ids))
            raise ValueError(
                f"{rewrite['submission_id']}: subject identities are not present "
                f"in both query and seed target: {missing}"
            )

        query_boxes = [
            {
                "subject_id": subject_by_identity[box["identity_id"]],
                **box,
            }
            for box in boxes_by_image.get(query_id, [])
            if box["identity_id"] in subject_by_identity
        ]
        target_boxes = [
            {
                "subject_id": subject_by_identity[box["identity_id"]],
                **box,
            }
            for box in boxes_by_image.get(target_id, [])
            if box["identity_id"] in subject_by_identity
        ]

        pair_change = annotation.get("pairChange")
        if pair_change is not None:
            pair_change = {
                "subject_1_id": subjects[0]["subject_id"],
                "subject_2_id": subjects[1]["subject_id"],
                "relation": pair_change["final"],
            }

        legacy_error = set(rewrite) == {"submission_id", "error"}

        outputs.append(
            {
                "submission_id": rewrite["submission_id"],
                "case_type": annotation["caseType"],
                "query_image_id": query_id,
                "query_image_url": image_by_id[query_id]["url"],
                "target_image_id": target_id,
                "target_image_url": image_by_id[target_id]["url"],
                "subjects": subjects,
                "candidate_identity_ids": candidate_identity_ids,
                "pair_change": pair_change,
                "query_boxes": query_boxes,
                "target_boxes": target_boxes,
                "final_desc": None if legacy_error else rewrite["final_desc"],
                "final_change": None if legacy_error else rewrite["final_change"],
            }
        )

    return outputs


def prepare_positive_set_inputs(
    selected: list[JsonObject],
    reviewed: list[JsonObject],
    pair_data: JsonObject,
    index_lines: Iterable[str],
) -> list[JsonObject]:
    """Build identity-compatible candidates for Full Positive labeling."""

    selected_by_id = {row["submission_id"]: row for row in selected}
    image_by_id, boxes_by_image, images_by_identity = _build_metadata(
        pair_data,
        index_lines,
    )
    outputs = []

    for review in reviewed:
        submission_id = review["submission_id"]
        for field in ("final_desc", "final_change"):
            value = review.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{submission_id}: {field} must be completed before "
                    "positive-set handoff"
                )

        source = selected_by_id[submission_id]
        annotation = source["annotation"]
        case_type = review.get("case_type", annotation["caseType"])
        if case_type not in CASE_TYPES:
            raise ValueError(f"{submission_id}: invalid case_type {case_type!r}")
        query_id = source["query_image_id"]
        seed_id = source["target_image_id"]

        subjects = normalize_review_subjects(
            submission_id,
            case_type,
            review.get("subjects"),
        )
        validate_review_output(
            case_type,
            {
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
            },
        )

        subject_by_identity = {
            identity_id: subject["subject_id"]
            for subject in subjects
            for identity_id in subject["identity_ids"]
        }

        for identity_id in subject_by_identity:
            identity_images = images_by_identity.get(identity_id, set())
            if query_id not in identity_images or seed_id not in identity_images:
                raise ValueError(
                    f"{submission_id}: reviewed identity_id {identity_id} must appear "
                    "in both query and seed target"
                )
        required_ids = list(subject_by_identity)

        candidate_sets = [
            images_by_identity.get(identity_id, set()) for identity_id in required_ids
        ]
        candidate_ids = set.intersection(*candidate_sets) if candidate_sets else set()
        candidate_ids.discard(query_id)

        if seed_id not in candidate_ids:
            raise ValueError(f"seed target {seed_id} is not identity-compatible")

        ordered_ids = [seed_id, *sorted(candidate_ids - {seed_id})]
        candidates = []
        for image_id in ordered_ids:
            subject_boxes = [
                {
                    "subject_id": subject_by_identity[box["identity_id"]],
                    **box,
                }
                for box in boxes_by_image.get(image_id, [])
                if box["identity_id"] in subject_by_identity
            ]
            candidates.append(
                {
                    "image_id": image_id,
                    "image_url": image_by_id[image_id]["url"],
                    "is_seed": image_id == seed_id,
                    "subject_boxes": subject_boxes,
                }
            )

        outputs.append(
            {
                "submission_id": review["submission_id"],
                "case_type": case_type,
                "query_image_id": query_id,
                "query_image_url": image_by_id[query_id]["url"],
                "seed_target_image_id": seed_id,
                "subjects": subjects,
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
                "candidates": candidates,
                "target_image_ids": [seed_id],
            }
        )

    return outputs
