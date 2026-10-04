from types import SimpleNamespace

import pytest
import torch

from rcr.methods.proposed.sampling import (
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
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

    assert len(set(rows[0]) & {"p1", "p2"}) == 1
    assert "q1" not in rows[0]
    assert not (
        (set(rows[0]) & {"p1", "p2"}) - {next(x for x in rows[0] if x in {"p1", "p2"})}
    )

    assert "p3" in rows[1]
    assert "q2" not in rows[1]


def test_positive_images_are_never_sampled_as_negatives() -> None:
    gallery = ["q1", "p1", "p2", "n1", "n2", "n3"]
    row = sample_candidates(
        [_samples()[0]],
        gallery,
        num_candidates=3,
        generator=torch.Generator().manual_seed(1),
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
        [sample], gallery, 16, torch.Generator().manual_seed(3), stats=stats, **options
    )[0]
    second = sample_candidates(
        [sample], gallery, 16, torch.Generator().manual_seed(3), **options
    )[0]
    assert first == second and len(set(first)) == 16
    assert len(set(first) & {"p1", "p2"}) == 1
    assert not set(first) & {"q1", "outside", "disputed"}
    assert stats == {"sampled_identity": 7, "sampled_hard": 4, "sampled_random": 4}


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
    assert stats == {"sampled_identity": 1, "sampled_hard": 0, "sampled_random": 4}


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
    ],
)
def test_invalid_sampling_settings_fail(options):
    with pytest.raises(ValueError):
        sampling_settings(options)
