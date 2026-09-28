"""Retrieval metrics for unique rankings, checked once by evaluate_rankings.

Standalone callers must also pass rankings without duplicate image IDs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _positive_set(positive_ids: Iterable[str]) -> set[str]:
    positives = set(positive_ids)
    if not positives:
        raise ValueError("positive_ids must be non-empty")
    return positives


def average_precision(ranking: Sequence[str], positive_ids: Iterable[str]) -> float:
    """Average precision with the complete positive set as denominator.

    A positive missing from ``ranking`` contributes zero. Official evaluation
    passes a full gallery ranking, but this definition also makes shortlist
    diagnostics safe.
    """

    positives = _positive_set(positive_ids)

    hits = 0
    precision_sum = 0.0
    for rank, image_id in enumerate(ranking, start=1):
        if image_id in positives:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(positives)


def recall_at_k(ranking: Sequence[str], positive_ids: Iterable[str], k: int) -> float:
    """Official R@K: whether at least one positive occurs in the top K.

    RCR reports this query-level hit/success measure as R@K. It is not the
    fraction of all positives recovered in the top K.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    positives = _positive_set(positive_ids)
    return float(any(image_id in positives for image_id in ranking[:k]))


def candidate_recall_at_k(
    coarse_ranking: Sequence[str], full_positive_ids: Iterable[str], k: int
) -> float:
    """Fraction of Full Positives retained by the coarse top-K shortlist."""

    if k <= 0:
        raise ValueError("k must be positive")

    positives = _positive_set(full_positive_ids)
    hits = sum(image_id in positives for image_id in coarse_ranking[:k])
    return hits / len(positives)
