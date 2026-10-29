"""Candidate quotas, exclusions and sparse same-ID hard pools."""

import pytest
import torch

from rcr.proposed.sampling import CandidateIndex, sample_candidates, sampling_settings


def candidates(identity, hard, *, fraction=1.0, seed=3):
    sample = {
        "sample_id": "q",
        "query_image_id": "self",
        "positive_image_ids": ["p", "unsampled-positive"],
    }
    gallery = ["self", "p", "unsampled-positive", "excluded", *identity]
    gallery += [f"other-{i}" for i in range(30)]
    index = CandidateIndex([sample], gallery, {"q": identity}, {"q": {"excluded"}})
    stats = {}
    rows = sample_candidates(
        [sample],
        11,
        torch.Generator().manual_seed(seed),
        index=index,
        positives_per_query=1,
        identity_fraction=0.5,
        identity_hard_fraction=fraction,
        hard_pools={"q": hard},
        hard_fraction=0.2,
        stats=stats,
    )
    return rows[0], stats


def test_same_identity_hard_quota_precedes_random_and_excludes_labels():
    identity = [f"id-{i}" for i in range(12)]
    hard = ["self", "p", "unsampled-positive", "excluded", "id-11", "other-0"]
    for seed in range(12):
        row, stats = candidates(identity, hard, seed=seed)
        assert "id-11" in row
        assert stats["identity_hard"] == 1
        assert stats["identity_random"] == 4
        assert stats["same_identity_negative"] >= 5
        assert stats["same_identity_negative"] == sum(i in identity for i in row)
        assert len(row) == len(set(row)) == 11
        assert not {"self", "excluded"} & set(row)
        assert len({"p", "unsampled-positive"} & set(row)) == 1
        assert sum(v for k, v in stats.items() if k != "same_identity_negative") == 11


def test_sparse_identity_pool_falls_back_without_growing_candidate_set():
    row, stats = candidates(["id-0", "id-1"], ["id-1"])
    assert {"id-0", "id-1"} <= set(row)
    assert stats["identity_hard"] == 1
    assert stats["identity_random"] == 1
    assert stats["uniform"] == 8
    assert len(row) == len(set(row)) == 11


def test_no_mined_identity_falls_back_to_full_random_identity_quota():
    identity = [f"id-{i}" for i in range(12)]
    first = candidates(identity, [], seed=7)
    assert first == candidates(identity, [], seed=7)
    row, stats = first
    assert stats["identity_hard"] == 0
    assert stats["identity_random"] == stats["same_identity_negative"] == 5
    assert len(row) == 11


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_identity_hard_fraction(fraction):
    with pytest.raises(ValueError, match="identity_hard_fraction"):
        sampling_settings({"identity_hard_fraction": fraction})
