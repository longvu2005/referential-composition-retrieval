import pytest
import torch

from rcr.proposed.scores import coarse_scores, combine_scores


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


def test_state_breaks_identity_tie_without_changing_identity_policy() -> None:
    query = torch.tensor([[1.0, 0.0]])
    logits = torch.tensor([[6.0]])
    gallery = torch.tensor([[[1.0, 0.0]]] * 3)
    mask = torch.tensor([[True], [True], [False]])
    kwargs = dict(
        query_state=torch.tensor([1.0, 0.0]),
        gallery_state=torch.tensor([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    identity = coarse_scores(query, logits, gallery, mask)
    zero = coarse_scores(query, logits, gallery, mask, beta=0, **kwargs)
    assert torch.equal(identity, zero)
    result = coarse_scores(query, logits, gallery, mask, beta=0.3, **kwargs)
    torch.testing.assert_close(result[:2], torch.tensor([0.7, 1.3]))
    assert result[2] == -torch.inf
    assert result.argsort(descending=True).tolist() == [1, 0, 2]


def test_empty_query_uses_state_and_missing_embeddings_fail() -> None:
    args = (torch.empty(0, 2), torch.empty(1, 0), torch.randn(2, 1, 2))
    score = coarse_scores(
        *args,
        beta=0.5,
        query_state=torch.tensor([1.0, 0.0]),
        gallery_state=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
    )
    torch.testing.assert_close(score, torch.tensor([-0.5, 0.5]))
    with pytest.raises(ValueError, match="requires both"):
        coarse_scores(*args, beta=0.3)


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


def test_state_only_ignores_identity_mask_and_beta():
    identity = torch.full((4,), -torch.inf)
    state = torch.tensor([100.0, -0.2, 0.4, 0.1])
    result = combine_scores(identity, state, mode="state_only", beta=0, exclude_index=0)
    torch.testing.assert_close(
        result, combine_scores(None, state, mode="state_only", beta=99, exclude_index=0)
    )
    assert result[1:].isfinite().all()
    assert result.argsort(descending=True).tolist() == [2, 3, 1, 0]


@pytest.mark.parametrize("identity", [[-torch.inf] * 3, [2, 2, 2], [2, -torch.inf, 2]])
def test_zscore_empty_constant_and_singleton_support(identity):
    values = torch.tensor(identity, dtype=torch.float32)
    result = combine_scores(values, None, beta=0, exclude_index=0)
    assert not result.isnan().any()
    assert (result[result.isfinite()] == 0).all()
    assert result[0].isneginf()


def test_raw_fusion_and_zero_beta_are_backward_compatible():
    identity = torch.tensor([0.1, 0.5, -torch.inf])
    state = torch.tensor([0.4, -0.2, 1.0])
    result = combine_scores(identity, state, beta=0.4, normalization="none")
    torch.testing.assert_close(result, identity + 0.4 * state)
    result = combine_scores(identity, None, beta=0, normalization="none")
    assert torch.equal(result, identity)
