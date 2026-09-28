"""Tests for the shared RCR evaluation protocol."""

import pytest

from rcr.evaluation.evaluate import evaluate_rankings
from rcr.evaluation.metrics import (
    average_precision,
    candidate_recall_at_k,
    recall_at_k,
)


def test_average_precision_uses_complete_positive_set() -> None:
    ranking = ["a", "b", "c", "d", "e"]
    positives = {"b", "e", "missing"}

    assert average_precision(ranking, positives) == pytest.approx(
        ((1 / 2) + (2 / 5)) / 3
    )


def test_official_recall_is_query_level_success() -> None:
    ranking = ["a", "b", "c", "d"]
    positives = {"b", "d", "missing"}

    assert recall_at_k(ranking, positives, 1) == 0.0
    assert recall_at_k(ranking, positives, 2) == 1.0


def test_candidate_recall_uses_fraction_of_full_positives() -> None:
    coarse_ranking = ["a", "b", "c", "d"]
    full_positives = {"b", "d", "missing"}

    assert candidate_recall_at_k(coarse_ranking, full_positives, 2) == pytest.approx(
        1 / 3
    )


def test_evaluator_uses_same_ranking_for_id_and_full_targets() -> None:
    gallery = ["q", "a", "b", "c", "d"]
    samples = [
        {
            "sample_id": "s1",
            "case_type": "GROUP",
            "query_image_id": "q",
            "positive_image_ids": ["b"],
            "subjects": [{"subject_id": 1, "identity_ids": ["p1", "p2"]}],
        }
    ]
    identities = {
        "q": {"p1", "p2"},
        "a": {"p1", "p2"},
        "b": {"p1", "p2"},
        "c": {"p1"},
        "d": {"p2"},
    }
    rankings = {"s1": ["a", "b", "c", "d"]}
    coarse = {"s1": ["b", "a", "c", "d"]}

    result = evaluate_rankings(
        samples,
        gallery,
        identities,
        rankings,
        coarse_rankings=coarse,
        candidate_ks=(1, 2),
    )

    row = result["per_query"][0]
    assert row["num_id_positives"] == 2
    assert row["num_full_positives"] == 1
    assert row["id_ap"] == pytest.approx(1.0)
    assert row["full_ap"] == pytest.approx(0.5)
    assert row["id_r1"] == 1.0
    assert row["full_r1"] == 0.0
    assert row["candidate_recall_1"] == 1.0
    assert result["by_case"]["GROUP"]["full_map"] == pytest.approx(0.5)


def test_evaluator_requires_full_gallery_ranking() -> None:
    sample = {
        "sample_id": "s1",
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": ["a"],
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
    }

    with pytest.raises(ValueError, match="full gallery"):
        evaluate_rankings(
            [sample],
            ["q", "a", "b"],
            {"q": {"p1"}, "a": {"p1"}, "b": set()},
            {"s1": ["a"]},
        )


def test_evaluator_uses_four_case_taxonomy() -> None:
    gallery = ["q1", "q2", "q3", "q4", "a", "b", "c", "d"]
    case_types = ["INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL"]
    samples = []
    rankings = {}
    identities = {image_id: {"p"} for image_id in gallery}

    for index, case_type in enumerate(case_types, start=1):
        sample_id = f"s{index}"
        query_id = f"q{index}"
        positive_id = chr(ord("a") + index - 1)
        samples.append(
            {
                "sample_id": sample_id,
                "case_type": case_type,
                "query_image_id": query_id,
                "positive_image_ids": [positive_id],
                "subjects": [{"subject_id": 1, "identity_ids": ["p"]}],
            }
        )
        rankings[sample_id] = [image_id for image_id in gallery if image_id != query_id]

    result = evaluate_rankings(samples, gallery, identities, rankings)
    assert list(result["by_case"]) == case_types


@pytest.mark.parametrize("ranking", [["a", "a"], ["a", "q"], ["a", "unknown"]])
def test_evaluator_rejects_invalid_final_ranking(ranking) -> None:
    sample = {
        "sample_id": "s1",
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": ["a"],
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
    }
    with pytest.raises(ValueError, match="full gallery"):
        evaluate_rankings(
            [sample],
            ["q", "a", "b"],
            {"q": {"p1"}, "a": {"p1"}, "b": set()},
            {"s1": ranking},
        )


@pytest.mark.parametrize("coarse", [["a", "a"], ["q"], ["unknown"]])
@pytest.mark.parametrize("candidate_ks", [(1, 2), ()])
def test_evaluator_checks_coarse_ranking_before_metrics(coarse, candidate_ks) -> None:
    sample = {
        "sample_id": "s1",
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": ["a"],
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
    }
    with pytest.raises(ValueError, match="coarse ranking"):
        evaluate_rankings(
            [sample],
            ["q", "a", "b"],
            {"q": {"p1"}, "a": {"p1"}, "b": set()},
            {"s1": ["a", "b"]},
            coarse_rankings={"s1": coarse},
            candidate_ks=candidate_ks,
        )


@pytest.mark.parametrize("positives", [[], ["q"], ["unknown"], ["b"]])
def test_evaluator_rejects_invalid_full_positives(positives) -> None:
    sample = {
        "sample_id": "s1",
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": positives,
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
    }
    with pytest.raises(ValueError, match="Full Positive"):
        evaluate_rankings(
            [sample],
            ["q", "a", "b"],
            {"q": {"p1"}, "a": {"p1"}, "b": set()},
            {"s1": ["a", "b"]},
        )
