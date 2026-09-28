import pytest

from rcr.dataset.positives import normalize_positive_sets


def _catalog():
    return [
        {
            "sample_id": sid,
            "seed_target_image_id": "seed",
            "candidates": [{"image_id": image} for image in ("seed", "c", "a", "b")],
        }
        for sid in ("s1", "s2", "s3")
    ]


def test_normalize_preserves_sets_and_supports_pending_tasks():
    decisions = [
        {"sample_id": "s2", "positive_image_ids": ["b", "seed", "a"]},
        {"sample_id": "s1", "positive_image_ids": ["c", "seed"]},
    ]
    normalized = normalize_positive_sets(_catalog(), decisions)
    assert normalized == [
        {"sample_id": "s1", "positive_image_ids": ["seed", "c"]},
        {"sample_id": "s2", "positive_image_ids": ["seed", "a", "b"]},
    ]
    assert normalized == normalize_positive_sets(_catalog(), normalized)
    for before in decisions:
        after = next(x for x in normalized if x["sample_id"] == before["sample_id"])
        assert set(before["positive_image_ids"]) == set(after["positive_image_ids"])


@pytest.mark.parametrize(
    "decisions,message",
    [
        ([{"sample_id": "other", "positive_image_ids": ["seed"]}], "outside catalog"),
        ([{"sample_id": "s1", "positive_image_ids": ["a"]}], "seed target"),
        ([{"sample_id": "s1", "positive_image_ids": ["seed", "unknown"]}], "invalid"),
        ([{"sample_id": "s1", "positive_image_ids": ["seed", "seed"]}], "duplicate"),
        ([{"sample_id": "s1", "positive_image_ids": ["seed"]}] * 2, "duplicate sample"),
    ],
)
def test_normalize_rejects_bad_labels(decisions, message):
    with pytest.raises(ValueError, match=message):
        normalize_positive_sets(_catalog(), decisions)
