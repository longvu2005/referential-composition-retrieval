"""Validate and deterministically order Full Positive decisions."""


def validate_positive_set(source: dict, payload: dict) -> dict:
    """Validate one browser submission and return the canonical positive row."""

    sample_id = source["sample_id"]
    if payload.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: sample_id mismatch")

    candidate_ids = [candidate["image_id"] for candidate in source["candidates"]]
    if (
        not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or candidate_ids[0] != source["seed_target_image_id"]
    ):
        raise ValueError(f"{sample_id}: candidates must be unique with seed first")
    positive_ids = payload.get("positive_image_ids")
    if not isinstance(positive_ids, list) or any(
        not isinstance(image_id, str) or not image_id for image_id in positive_ids
    ):
        raise ValueError(f"{sample_id}: positive_image_ids must be a string list")
    if len(set(positive_ids)) != len(positive_ids):
        raise ValueError(f"{sample_id}: duplicate positive_image_id")

    invalid = set(positive_ids) - set(candidate_ids)
    if invalid:
        raise ValueError(f"{sample_id}: invalid positive_image_id {sorted(invalid)[0]}")

    seed_id = source["seed_target_image_id"]
    if seed_id not in positive_ids:
        raise ValueError(f"{sample_id}: seed target must remain selected")

    selected = set(positive_ids)
    return {
        "sample_id": sample_id,
        "positive_image_ids": [
            image_id for image_id in candidate_ids if image_id in selected
        ],
    }


def normalize_positive_sets(catalog: list[dict], decisions: list[dict]) -> list[dict]:
    """Validate all decisions before returning them in catalog/candidate order."""

    sources = {row["sample_id"]: row for row in catalog}
    saved = {row["sample_id"]: row for row in decisions}
    if len(sources) != len(catalog) or len(saved) != len(decisions):
        raise ValueError("duplicate sample_id in catalog or decisions")
    unknown = set(saved) - set(sources)
    if unknown:
        raise ValueError(f"positive decision outside catalog: {min(unknown)}")
    return [
        validate_positive_set(source, saved[source["sample_id"]])
        for source in catalog
        if source["sample_id"] in saved
    ]


def check_positive_context(
    catalog: list[dict],
    decisions: list[dict],
    selected: list[dict],
    reviewed: list[dict],
) -> None:
    """Reject labels whose catalog no longer matches the current pair/review."""

    catalog_by_id = {row["sample_id"]: row for row in catalog}
    selected_by_id = {row["sample_id"]: row for row in selected}
    reviewed_by_id = {row["sample_id"]: row for row in reviewed}
    for decision in decisions:
        sample_id = decision["sample_id"]
        if (
            sample_id not in selected_by_id
            or sample_id not in reviewed_by_id
            or sample_id not in catalog_by_id
        ):
            raise ValueError(
                f"{sample_id}: positive decision needs selected/review/catalog"
            )
        source = selected_by_id[sample_id]
        review = reviewed_by_id[sample_id]
        task = catalog_by_id[sample_id]
        tracked = ("case_type", "subjects", "final_desc", "final_change")
        if (
            any(task[k] != review[k] for k in tracked)
            or task["query_image_id"] != source["query_image_id"]
            or task["seed_target_image_id"] != source["target_image_id"]
        ):
            raise ValueError(
                f"{sample_id}: positive catalog is stale; inspect its labels "
                "before rebuilding final data"
            )
