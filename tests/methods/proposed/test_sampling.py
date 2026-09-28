import torch

from rcr.methods.proposed.sampling import sample_candidates


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
