import pytest
import torch

from rcr.proposed.scores import combine_scores, identity_scores


def test_identity_scores_match_formula() -> None:
    query_identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grounding_logits = torch.tensor([[4.0, -2.0], [-2.0, 4.0]])
    gallery_identity = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    score = identity_scores(query_identity, grounding_logits, gallery_identity)

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

    score = identity_scores(query_identity, grounding_logits, gallery_identity)
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

    score = identity_scores(
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
    result = identity_scores(query, logits, gallery)
    torch.testing.assert_close(result, torch.tensor([0.5]), atol=1e-7, rtol=1e-6)


def test_empty_query_has_deterministic_zero_coarse_score() -> None:
    assert identity_scores(
        torch.empty(0, 2), torch.empty(1, 0), torch.randn(2, 1, 2)
    ).tolist() == [0.0, 0.0]


def test_empty_gallery_person_sets_score_minus_infinity() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logits = torch.tensor([[6.0, -torch.inf]])
    gallery = torch.ones(2, 1, 2)
    mask = torch.tensor([[False], [True]])
    scores = identity_scores(query, logits, gallery, mask)
    assert scores[0] == -torch.inf
    assert torch.isfinite(scores[1])
    assert not scores.isnan().any()
    empty_scores = identity_scores(query, logits, torch.empty(2, 0, 2))
    assert (empty_scores == -torch.inf).all()


def test_zscore_uses_common_gallery_support_and_excludes_self():
    identity = torch.tensor([10000.0, 2, 4, 6, -torch.inf])
    state = torch.tensor([-10000.0, 0.1, 0.2, 0.7, 10000.0])
    result = combine_scores(identity, state, beta=0.5, exclude_index=0)
    i, s = identity[1:4], state[1:4]
    expected = (i - i.mean()) / i.std(unbiased=False)
    expected += 0.5 * (s - s.mean()) / s.std(unbiased=False)
    torch.testing.assert_close(result[1:4], expected)
    assert result[[0, 4]].isneginf().all()
    # Shifting or positively rescaling either branch must not change fusion.
    shifted = combine_scores(
        3 * identity + 10, 7 * state - 3, beta=0.5, exclude_index=0
    )
    torch.testing.assert_close(result, shifted)


@pytest.mark.parametrize("identity", [[-torch.inf] * 3, [2, 2, 2], [2, -torch.inf, 2]])
def test_zscore_empty_constant_and_singleton_support(identity):
    values = torch.tensor(identity, dtype=torch.float32)
    result = combine_scores(values, None, beta=0, exclude_index=0)
    assert not result.isnan().any()
    assert (result[result.isfinite()] == 0).all()
    assert result[0].isneginf()
