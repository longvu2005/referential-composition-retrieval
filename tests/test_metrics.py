"""Metric definitions used in every baseline and proposed run."""

import random

import pytest

from rcr.evaluation.evaluate import evaluate_rankings, identity_positive_ids
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


@pytest.mark.parametrize("required_count", [1, 2, 3])
def test_identity_index_matches_gallery_scan_and_observes_changed_labels(
    required_count,
):
    rng = random.Random(37)
    gallery = [f"image-{i}" for i in range(80)]
    identities = [f"person-{i}" for i in range(12)]
    labels = {image: set(rng.sample(identities, rng.randrange(5))) for image in gallery}
    required = set(identities[:required_count])
    labels[gallery[0]] |= required
    labels[gallery[1]] |= required
    labels["outside-gallery"] = required
    sample = {
        "sample_id": "query",
        "case_type": "INDIVIDUAL" if required_count == 1 else "GROUP",
        "query_image_id": gallery[0],
        "subjects": [{"subject_id": 1, "identity_ids": sorted(required)}],
        "positive_image_ids": [gallery[1]],
    }
    ranking = gallery[1:]
    rng.shuffle(ranking)
    for _ in range(2):
        expected = [
            image for image in gallery[1:] if required <= labels.get(image, set())
        ]
        assert identity_positive_ids(sample, gallery, labels) == expected
        result = evaluate_rankings([sample], gallery, labels, {"query": ranking})
        assert result["overall"]["id_map"] == average_precision(ranking, expected)
        labels[gallery[2]] |= required
        labels[gallery[3]] -= required
