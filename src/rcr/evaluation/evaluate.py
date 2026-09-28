"""Shared evaluation protocol for saved RCR rankings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from rcr.dataset.cases import CASE_TYPES
from rcr.evaluation.metrics import (
    average_precision,
    candidate_recall_at_k,
    recall_at_k,
)

RECALL_KS = (1, 5, 10)


def required_identity_ids(sample: dict) -> set[str]:
    """All identities required by all Subjects in one query."""

    return {
        identity_id
        for subject in sample["subjects"]
        for identity_id in subject["identity_ids"]
    }


def identity_positive_ids(
    sample: dict,
    gallery_ids: Sequence[str],
    identities_by_image: Mapping[str, set[str]],
) -> list[str]:
    """Gallery images containing every identity required by the query.

    Identity positives ignore the requested change/relation. Subject grouping
    and ordering are therefore irrelevant for this diagnostic target set.
    """

    required_ids = required_identity_ids(sample)
    if not required_ids:
        raise ValueError(f"{sample['sample_id']}: no required identities")

    query_id = sample["query_image_id"]
    return [
        image_id
        for image_id in gallery_ids
        if image_id != query_id
        and required_ids <= identities_by_image.get(image_id, set())
    ]


def _mean(rows: Sequence[dict], key: str) -> float:
    return fmean(row[key] for row in rows)


def _aggregate_overall(rows: Sequence[dict], candidate_ks: Sequence[int]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")

    output = {
        "num_queries": len(rows),
        "id_map": _mean(rows, "id_ap"),
        "id_r1": _mean(rows, "id_r1"),
        "id_r5": _mean(rows, "id_r5"),
        "id_r10": _mean(rows, "id_r10"),
        "full_map": _mean(rows, "full_ap"),
        "full_r1": _mean(rows, "full_r1"),
        "full_r5": _mean(rows, "full_r5"),
        "full_r10": _mean(rows, "full_r10"),
    }
    for k in candidate_ks:
        key = f"candidate_recall_{k}"
        if key in rows[0]:
            output[key] = _mean(rows, key)
    return output


def _aggregate_cases(rows: Sequence[dict]) -> dict[str, dict]:
    output = {}
    for case_type in CASE_TYPES:
        case_rows = [row for row in rows if row["case_type"] == case_type]
        if not case_rows:
            continue
        output[case_type] = {
            "num_queries": len(case_rows),
            "full_map": _mean(case_rows, "full_ap"),
            "full_r1": _mean(case_rows, "full_r1"),
        }
    return output


def evaluate_rankings(
    samples: Sequence[dict],
    gallery_ids: Sequence[str],
    identities_by_image: Mapping[str, set[str]],
    rankings: Mapping[str, Sequence[str]],
    coarse_rankings: Mapping[str, Sequence[str]] | None = None,
    candidate_ks: Sequence[int] = (500,),
) -> dict:
    """Evaluate final rankings and optional coarse rankings.

    The same final ranking is scored against two target sets:
    identity positives and reviewed Full Positives. CandidateRecall@K is a
    proposed-method diagnostic and is computed from the coarse ranking against
    Full Positives.
    """

    gallery_set = set(gallery_ids)
    if len(gallery_ids) != len(gallery_set):
        raise ValueError("gallery_ids must be unique")
    if any(k <= 0 for k in candidate_ks):
        raise ValueError("candidate_ks must be positive")

    rows = []
    seen_sample_ids = set()

    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id}")
        seen_sample_ids.add(sample_id)

        if sample["case_type"] not in CASE_TYPES:
            raise ValueError(f"{sample_id}: invalid case_type {sample['case_type']!r}")
        query_id = sample["query_image_id"]
        valid_gallery = gallery_set - {query_id}
        ranking = rankings[sample_id]
        if len(ranking) != len(valid_gallery) or set(ranking) != valid_gallery:
            raise ValueError(
                f"{sample_id}: ranking must contain the full gallery "
                "except the query image, with each image exactly once"
            )

        full_positives = set(sample["positive_image_ids"])
        if not full_positives:
            raise ValueError(f"{sample_id}: no Full Positives")

        id_positives = set(
            identity_positive_ids(sample, gallery_ids, identities_by_image)
        )
        if not full_positives <= id_positives:
            raise ValueError(
                f"{sample_id}: every Full Positive must also be an ID Positive"
            )

        row = {
            "sample_id": sample_id,
            "case_type": sample["case_type"],
            "num_id_positives": len(id_positives),
            "num_full_positives": len(full_positives),
            "id_ap": average_precision(ranking, id_positives),
            "full_ap": average_precision(ranking, full_positives),
        }
        for k in RECALL_KS:
            row[f"id_r{k}"] = recall_at_k(ranking, id_positives, k)
            row[f"full_r{k}"] = recall_at_k(ranking, full_positives, k)

        if coarse_rankings is not None:
            coarse = coarse_rankings[sample_id]
            coarse_set = set(coarse)
            if len(coarse) != len(coarse_set):
                raise ValueError(f"{sample_id}: coarse ranking contains duplicates")
            if not coarse_set <= valid_gallery:
                raise ValueError(
                    f"{sample_id}: coarse ranking contains query or unknown image"
                )
            for k in candidate_ks:
                row[f"candidate_recall_{k}"] = candidate_recall_at_k(
                    coarse, full_positives, k
                )

        rows.append(row)

    return {
        "overall": _aggregate_overall(rows, candidate_ks),
        "by_case": _aggregate_cases(rows),
        "per_query": rows,
    }
