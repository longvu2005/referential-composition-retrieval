"""Multi-positive retrieval metrics through the official evaluator."""

import pytest

from rcr.evaluation.evaluate import evaluate_rankings


def test_full_and_identity_metrics_use_their_own_positive_sets():
    gallery = ["q", "p1", "p2", "wrong_id", "wrong_state"]
    identities = {image: {"A", "B"} for image in gallery}
    identities["wrong_id"] = {"A"}
    sample = {
        "sample_id": "query",
        "case_type": "DUAL",
        "query_image_id": "q",
        "subjects": [
            {"subject_id": 1, "identity_ids": ["A"]},
            {"subject_id": 2, "identity_ids": ["B"]},
        ],
        "positive_image_ids": ["p1", "p2"],
    }
    result = evaluate_rankings(
        [sample],
        gallery,
        identities,
        {"query": ["wrong_id", "p1", "wrong_state", "p2"]},
    )
    assert result["overall"]["full_map"] == pytest.approx(0.5)
    assert result["overall"]["id_map"] == pytest.approx((1 / 2 + 2 / 3 + 3 / 4) / 3)
    assert result["overall"]["full_r1"] == 0
    assert result["overall"]["full_r5"] == 1
    assert result["by_case"]["DUAL"]["full_map"] == pytest.approx(0.5)
