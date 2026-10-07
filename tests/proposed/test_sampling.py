from types import SimpleNamespace

import pytest
import torch

from rcr.proposed.sampling import (
    case_sampling_weights,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)


def _samples():
    return [
        {
            "sample_id": "s1",
            "query_image_id": "q1",
            "positive_image_ids": ["p1", "p2"],
        },
        {
            "sample_id": "s2",
            "query_image_id": "q2",
            "target_image_id": "p3",
        },
    ]


def test_sample_candidates() -> None:
    gallery = ["q1", "q2", "p1", "p2", "p3", "n1", "n2", "n3", "n4"]
    rows = sample_candidates(
        _samples(),
        gallery,
        num_candidates=4,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(rows) == 2
    assert all(len(row) == 4 for row in rows)

    assert len(set(rows[0]) & {"p1", "p2"}) == 2
    assert "q1" not in rows[0]
    assert len(set(rows[0])) == 4

    assert "p3" in rows[1]
    assert "q2" not in rows[1]


def test_positive_images_are_never_sampled_as_negatives() -> None:
    gallery = ["q1", "p1", "p2", "n1", "n2", "n3"]
    row = sample_candidates(
        [_samples()[0]],
        gallery,
        num_candidates=3,
        generator=torch.Generator().manual_seed(1),
        positives_per_query=1,
    )[0]

    positives = {"p1", "p2"}
    assert len(set(row) & positives) == 1


def test_sampling_is_reproducible() -> None:
    gallery = ["q1", "q2", "p1", "p2", "p3", "n1", "n2", "n3", "n4"]

    a = sample_candidates(
        _samples(),
        gallery,
        4,
        torch.Generator().manual_seed(7),
    )
    b = sample_candidates(
        _samples(),
        gallery,
        4,
        torch.Generator().manual_seed(7),
    )

    assert a == b


def test_mixed_sampler_quotas_exclusions_and_reproducibility():
    sample = _samples()[0]
    ids = [f"i{k}" for k in range(12)]
    hard = [f"h{k}" for k in range(12)]
    random = [f"r{k}" for k in range(20)]
    gallery = ["q1", "p1", "p2", "disputed", *ids, *hard, *random]
    options = dict(
        identity_pools={"s1": ["q1", "p1", "p2", "outside", "disputed", *ids]},
        hard_pools={"s1": ["p1", "disputed", "outside", *hard]},
        excluded={"s1": {"disputed"}},
        identity_fraction=0.5,
        hard_fraction=0.3,
    )
    stats = {}
    first = sample_candidates(
        [sample], gallery, 24, torch.Generator().manual_seed(3), stats=stats, **options
    )[0]
    second = sample_candidates(
        [sample], gallery, 24, torch.Generator().manual_seed(3), **options
    )[0]
    assert first == second and len(set(first)) == 24
    assert len(set(first) & {"p1", "p2"}) == 2
    assert not set(first) & {"q1", "outside", "disputed"}
    assert stats == {
        "sampled_positive": 2,
        "sampled_identity": 11,
        "sampled_hard": 6,
        "sampled_random": 5,
    }


def test_exhausted_identity_and_warmup_hard_slots_fall_back_to_random():
    stats = {}
    row = sample_candidates(
        [_samples()[0]],
        ["q1", "p1", "p2", "i", "n1", "n2", "n3", "n4"],
        6,
        torch.Generator().manual_seed(5),
        identity_pools={"s1": ["i"]},
        identity_fraction=0.5,
        hard_fraction=0.3,
        stats=stats,
    )[0]
    assert len(row) == len(set(row)) == 6
    assert stats == {
        "sampled_positive": 2,
        "sampled_identity": 1,
        "sampled_hard": 0,
        "sampled_random": 3,
    }


def test_identity_pool_requires_every_subject_identity_and_only_given_gallery():
    data = SimpleNamespace(
        gt_head_boxes_by_image={
            "q": [{"identity_id": "a"}, {"identity_id": "b"}],
            "p": [{"identity_id": "a"}, {"identity_id": "b"}],
            "partial": [{"identity_id": "a"}],
            "n": [{"identity_id": "a"}, {"identity_id": "b"}],
            "val": [{"identity_id": "a"}, {"identity_id": "b"}],
        }
    )
    sample = {
        "sample_id": "s",
        "query_image_id": "q",
        "positive_image_ids": ["p"],
        "subjects": [{"subject_id": 1, "identity_ids": ["a", "b"]}],
    }
    assert identity_candidate_pools(data, [sample], ["n", "q", "partial", "p"]) == {
        "s": ["n", "p"]
    }


@pytest.mark.parametrize(
    "options",
    [
        {"identity_fraction": 0.8, "hard_fraction": 0.3},
        {"hard_fraction": -0.1},
        {"pool_size": 0},
        {"refresh_every_epochs": 0},
        {"positives_per_query": 0},
        {"positives_per_query": 1.5},
    ],
)
def test_invalid_sampling_settings_fail(options):
    with pytest.raises(ValueError):
        sampling_settings(options)


@pytest.mark.parametrize("gallery_size,excluded_count", [(1000, 10), (80, 61)])
def test_reusable_sampler_index_preserves_exclusions_and_fills_dense_gallery(
    gallery_size, excluded_count
):
    from rcr.proposed.sampling import CandidateIndex

    sample = _samples()[0]
    gallery = ["q1", "p1", "p2", *map(str, range(gallery_size))]
    excluded = {"s1": set(map(str, range(excluded_count))) | {"outside"}}
    # Include duplicates and ineligible IDs in both specialized pools.
    options = dict(
        identity_pools={"s1": ["q1", "p1", "0", str(excluded_count)] * 2},
        hard_pools={"s1": [str(excluded_count), str(excluded_count + 1), "outside"]},
        excluded=excluded,
        identity_fraction=0.5,
        hard_fraction=0.3,
    )
    index = CandidateIndex([sample], gallery, options["identity_pools"], excluded)
    saved_forbidden = set(index.rows["s1"][1])
    for seed in range(5):
        first = sample_candidates(
            [sample], gallery, 16, torch.Generator().manual_seed(seed), **options
        )[0]
        stats = {}
        second = sample_candidates(
            [sample],
            gallery,
            16,
            torch.Generator().manual_seed(seed),
            index=index,
            stats=stats,
            **options,
        )[0]
        assert first == second and len(set(second)) == 16
        assert len(set(second) & {"p1", "p2"}) == 2
        assert not set(second) & (excluded["s1"] | {"q1", "outside"})
        assert stats == {
            "sampled_positive": 2,
            "sampled_identity": 1,
            "sampled_hard": 1,
            "sampled_random": 12,
        }
        assert index.rows["s1"][1] == saved_forbidden


@pytest.mark.parametrize(
    "positive_ids,candidates,expected",
    [
        (["p1", "p1"], 4, 1),
        (["p1", "p2", "p3"], 4, 2),
        (["p1", "p2"], 2, 1),
    ],
)
def test_positive_count_adapts_without_duplicates_or_false_negatives(
    positive_ids, candidates, expected
):
    sample = {**_samples()[0], "positive_image_ids": positive_ids}
    gallery = ["q1", "p1", "p2", "p3", "n1", "n2", "n3"]
    row = sample_candidates(
        [sample], gallery, candidates, torch.Generator().manual_seed(0)
    )[0]
    assert len(row) == len(set(row)) == candidates
    assert len(set(row) & set(positive_ids)) == expected
    assert "q1" not in row


def _case_samples():
    return [
        {"sample_id": str(i), "case_type": case, "subjects": [{}] * subjects}
        for i, (case, subjects) in enumerate(
            [("INDIVIDUAL", 1)] * 81 + [("GROUP", 1)] * 9 + [("DUAL", 2)]
        )
    ]


def test_case_weights_give_sqrt_case_probabilities():
    samples = _case_samples()
    weights = case_sampling_weights(samples)
    probability = weights / weights.sum()
    # 81:9:1 sample counts become 9:3:1 case probabilities.
    torch.testing.assert_close(
        torch.stack(
            (probability[:81].sum(), probability[81:90].sum(), probability[90])
        ),
        torch.tensor([9 / 13, 3 / 13, 1 / 13], dtype=torch.double),
    )


@pytest.mark.parametrize("balanced", [True, False])
def test_epoch_batches_keep_every_draw_and_equal_subject_counts(balanced):
    samples = _case_samples()
    first = training_batches(
        samples, 7, torch.Generator().manual_seed(2), case_balanced=balanced
    )
    second = training_batches(
        samples, 7, torch.Generator().manual_seed(2), case_balanced=balanced
    )
    assert first == second
    assert sum(map(len, first)) == len(samples)
    assert all(1 <= len(batch) <= 7 for batch in first)
    assert all(len({len(row["subjects"]) for row in batch}) == 1 for batch in first)
    ids = [row["sample_id"] for batch in first for row in batch]
    if balanced:
        assert len(set(ids)) < len(samples)  # Sampling with replacement.
    else:
        assert set(ids) == {row["sample_id"] for row in samples}
