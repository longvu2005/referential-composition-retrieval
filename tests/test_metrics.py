"""Metric definitions used in every baseline and proposed run."""

import pytest

from rcr.evaluation.metrics import (
    average_precision,
    candidate_recall_at_k,
    recall_at_k,
)


def test_average_precision_uses_all_full_positives():
    ranking = ["n", "p1", "p2"]
    assert average_precision(ranking, {"p1", "p2", "missing"}) == pytest.approx(
        (1 / 2 + 2 / 3) / 3
    )


def test_recall_at_k_is_query_level_success():
    ranking = ["n", "p1", "p2"]
    assert recall_at_k(ranking, {"p1", "p2"}, 1) == 0
    assert recall_at_k(ranking, {"p1", "p2"}, 2) == 1


def test_candidate_recall_measures_shortlist_coverage():
    assert candidate_recall_at_k(["p1", "n", "p2"], {"p1", "p2"}, 2) == 0.5
