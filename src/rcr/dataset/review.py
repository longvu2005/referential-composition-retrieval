"""Normalize human-reviewed Subject → identity assignments."""

from rcr.dataset.cases import (
    ONE_SUBJECT_CASES,
    infer_single_subject_case,
    subject_ids_for_case,
)


def normalize_review_assignment(
    submission_id: str,
    case_type: str,
    subjects,
    candidate_identity_ids: list[str] | None = None,
) -> tuple[str, list[dict]]:
    expected = subject_ids_for_case(case_type)
    if not isinstance(subjects, list) or any(not isinstance(x, dict) for x in subjects):
        raise ValueError(f"{submission_id}: reviewed subjects are required")

    subjects = sorted(subjects, key=lambda x: x.get("subject_id", -1))
    if [x.get("subject_id") for x in subjects] != expected:
        names = ", ".join(f"S{x}" for x in expected)
        raise ValueError(f"{submission_id}: {case_type} requires exactly {names}")

    candidate_order = None
    if candidate_identity_ids is not None:
        if len(candidate_identity_ids) != len(set(candidate_identity_ids)):
            raise ValueError(f"{submission_id}: duplicate candidate identity_id")
        candidate_order = {x: i for i, x in enumerate(candidate_identity_ids)}

    assigned = set()
    normalized = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        identity_ids = subject.get("identity_ids")

        if not isinstance(identity_ids, list) or not identity_ids:
            raise ValueError(
                f"{submission_id}: Subject {subject_id} "
                "must have at least one identity_id"
            )
        if any(not isinstance(x, str) or not x for x in identity_ids):
            raise ValueError(
                f"{submission_id}: Subject {subject_id} has an invalid identity_id"
            )
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError(
                f"{submission_id}: duplicate identity_id within Subject {subject_id}"
            )

        identity_set = set(identity_ids)
        if candidate_order is not None:
            invalid = identity_set - set(candidate_order)
            if invalid:
                raise ValueError(
                    f"{submission_id}: invalid identity_id for Subject {subject_id}: "
                    + ", ".join(sorted(invalid))
                )
            identity_ids = sorted(identity_ids, key=candidate_order.__getitem__)

        overlap = assigned & identity_set
        if overlap:
            raise ValueError(
                f"{submission_id}: identity_id assigned to multiple subjects: "
                + ", ".join(sorted(overlap))
            )
        assigned |= identity_set

        normalized.append({"subject_id": subject_id, "identity_ids": identity_ids})

    if case_type in ONE_SUBJECT_CASES:
        case_type = infer_single_subject_case(len(normalized[0]["identity_ids"]))

    return case_type, normalized
