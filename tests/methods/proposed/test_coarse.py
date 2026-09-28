import torch

from rcr.methods.proposed.coarse import coarse_scores


def test_coarse_scores_match_formula() -> None:
    query_identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grounding_logits = torch.tensor([[4.0, -2.0], [-2.0, 4.0]])
    gallery_identity = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    score = coarse_scores(query_identity, grounding_logits, gallery_identity)

    weight = grounding_logits.sigmoid()
    expected = torch.tensor(
        [
            1.0,
            (weight[0, 0] / weight[0].sum() + weight[1, 0] / weight[1].sum()) / 2,
        ]
    )
    torch.testing.assert_close(score, expected)


def test_low_membership_person_has_small_effect() -> None:
    query_identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grounding_logits = torch.tensor([[6.0, -6.0]])
    gallery_identity = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )

    score = coarse_scores(query_identity, grounding_logits, gallery_identity)
    assert score[0] > score[1]


def test_gallery_padding_is_ignored() -> None:
    query_identity = torch.tensor([[1.0, 0.0]])
    grounding_logits = torch.tensor([[6.0]])
    gallery_identity = torch.tensor(
        [
            [[-1.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.0]],
        ]
    )
    gallery_mask = torch.tensor([[True, False], [True, False]])

    score = coarse_scores(
        query_identity,
        grounding_logits,
        gallery_identity,
        gallery_mask,
    )
    torch.testing.assert_close(score, torch.tensor([-1.0, 0.5]))


def test_subjects_are_averaged_even_when_group_has_more_people() -> None:
    query = torch.eye(3)
    gallery = torch.tensor([[[1.0, 1.0, 0.0]]])
    logits = torch.tensor([[20.0, 20.0, -20.0], [-20.0, -20.0, 20.0]])
    result = coarse_scores(query, logits, gallery)
    torch.testing.assert_close(result, torch.tensor([0.5]), atol=1e-7, rtol=1e-6)


def test_empty_query_has_deterministic_zero_coarse_score() -> None:
    assert coarse_scores(
        torch.empty(0, 2), torch.empty(1, 0), torch.randn(2, 1, 2)
    ).tolist() == [0.0, 0.0]


def test_empty_gallery_person_sets_score_minus_infinity() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logits = torch.tensor([[6.0, -torch.inf]])
    gallery = torch.ones(2, 1, 2)
    mask = torch.tensor([[False], [True]])
    scores = coarse_scores(query, logits, gallery, mask)
    assert scores[0] == -torch.inf
    assert torch.isfinite(scores[1])
    assert not scores.isnan().any()
    empty_scores = coarse_scores(query, logits, torch.empty(2, 0, 2))
    assert (empty_scores == -torch.inf).all()
