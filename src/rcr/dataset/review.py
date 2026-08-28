"""Canonical validation for human-reviewed subject assignments."""

from __future__ import annotations

from typing import Any

from rcr.dataset.rewrite import subject_ids_for_case

JsonObject = dict[str, Any]


def normalize_review_subjects(
    submission_id: str,
    case_type: str,
    subjects: Any,
    candidate_identity_ids: list[str] | None = None,
) -> list[JsonObject]:
    """Validate and normalize Subject-to-identity assignments for one review."""

    expected_subject_ids = subject_ids_for_case(case_type)
    if not isinstance(subjects, list):
        raise ValueError(f"{submission_id}: reviewed subjects are required")
    if any(not isinstance(subject, dict) for subject in subjects):
        raise ValueError(f"{submission_id}: each reviewed subject must be an object")

    subjects = sorted(subjects, key=lambda subject: subject.get("subject_id", -1))
    actual_subject_ids = [subject.get("subject_id") for subject in subjects]
    if actual_subject_ids != expected_subject_ids:
        expected = ", ".join(f"S{subject_id}" for subject_id in expected_subject_ids)
        raise ValueError(f"{submission_id}: {case_type} requires exactly {expected}")

    candidate_order = None
    candidate_set = None
    if candidate_identity_ids is not None:
        if len(candidate_identity_ids) != len(set(candidate_identity_ids)):
            raise ValueError(f"{submission_id}: duplicate candidate identity_id")
        candidate_order = {
            identity_id: index
            for index, identity_id in enumerate(candidate_identity_ids)
        }
        candidate_set = set(candidate_identity_ids)

    assigned_ids: set[str] = set()
    normalized = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        identity_ids = subject.get("identity_ids")
        if not isinstance(identity_ids, list) or not identity_ids:
            raise ValueError(
                f"{submission_id}: Subject {subject_id} must have at least one "
                "identity_id"
            )
        if any(
            not isinstance(identity_id, str) or not identity_id
            for identity_id in identity_ids
        ):
            raise ValueError(
                f"{submission_id}: Subject {subject_id} has an invalid identity_id"
            )
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError(
                f"{submission_id}: duplicate identity_id within Subject {subject_id}"
            )

        identity_set = set(identity_ids)
        if candidate_set is not None:
            invalid_ids = identity_set - candidate_set
            if invalid_ids:
                invalid = ", ".join(sorted(invalid_ids))
                raise ValueError(
                    f"{submission_id}: invalid identity_id for Subject "
                    f"{subject_id}: {invalid}"
                )

        overlap = assigned_ids & identity_set
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise ValueError(
                f"{submission_id}: identity_id assigned to multiple subjects: "
                f"{duplicate}"
            )
        assigned_ids.update(identity_set)

        if candidate_order is not None:
            identity_ids = sorted(identity_ids, key=candidate_order.__getitem__)

        normalized.append(
            {
                "subject_id": subject_id,
                "identity_ids": list(identity_ids),
            }
        )

    return normalized
