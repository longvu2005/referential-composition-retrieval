"""Small contracts for the equations that define the proposed method."""

import pytest
import torch

from rcr.proposed.losses import identity_loss, retrieval_loss
from rcr.proposed.matching import exclusive_membership, partial_identity_matching
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import fuse_fine_coarse_scores


def _boxes(batch, people):
    boxes = torch.rand(batch, people, 4)
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:] * (1 - boxes[..., :2])
    return boxes


def test_subject_membership_is_variable_sized_and_exclusive():
    logits = torch.tensor([[[5.0, 4.0, -5.0], [-5.0, -4.0, 5.0]]])
    membership = exclusive_membership(logits, torch.ones(1, 2, dtype=torch.bool))
    assert (membership > 0).tolist() == [[[True, True, False], [False, False, True]]]
    assert ((membership > 0).sum(1) <= 1).all()
    assert not membership.requires_grad


def test_partial_matching_is_global_and_allows_null():
    query = torch.tensor([[[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.6, -0.8]]])
    match = partial_identity_matching(
        query,
        target,
        torch.ones(1, 1, 3),
        torch.tensor([[True]]),
        identity_threshold=0.1,
    )
    assert match.indices.tolist() == [[1, 0, -1]]
    assert match.matched.sum() == 2


def test_losses_rank_positives_and_same_identity_together():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    bad = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    assert identity_loss(good, labels) < identity_loss(bad, labels)

    scores = torch.tensor([[2.0, 0.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[True, False, False]])
    loss = identity_loss(good, labels) + retrieval_loss(scores, positive)
    loss.backward()
    assert torch.isfinite(good.grad).all()
    assert torch.isfinite(scores.grad).all()


def test_model_forward_has_the_expected_gradient_boundary():
    torch.manual_seed(0)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    query_people = torch.randn(2, 3, 8, requires_grad=True)
    target_people = torch.randn(2, 4, 8, requires_grad=True)
    logits, score = model(
        query_scene=torch.randn(2, 6, 8),
        query_persons=query_people,
        query_boxes=_boxes(2, 3),
        selections=torch.randn(2, 2, 4, 8),
        change=torch.randn(2, 5, 8),
        subject_pos=torch.tensor([[0, 3], [1, 4]]),
        target_scene=torch.randn(2, 6, 8),
        target_persons=target_people,
        target_boxes=_boxes(2, 4),
        patch_hw=(2, 3),
        fafa_query=torch.randn(2, 2, 3, 5),
        fafa_targets=torch.randn(2, 4, 7, 5),
    )
    assert logits.shape == (2, 2, 3)
    assert score.shape == (2,)
    (logits.mean() + score.mean()).backward()
    assert query_people.grad is not None
    assert target_people.grad is not None
    # Fine assignment is detached; identity learns from its explicit loss.
    assert model.identity_head.proj.weight.grad is None


def test_reasoner_has_no_unused_cross_attention_and_loads_legacy_state():
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    state = model.state_dict()
    assert not any(key.startswith("reasoner.cross_attn.") for key in state)
    state["reasoner.cross_attn.in_proj_weight"] = torch.zeros(24, 8)
    model.load_state_dict(state)


def test_fine_coarse_fusion_uses_common_shortlist_statistics():
    fine = torch.tensor([2.0, 1.0, 0.0])
    coarse = torch.tensor([0.0, 1.0, 2.0])
    fused = fuse_fine_coarse_scores(fine, coarse, weight=0.5)
    assert fused.shape == fine.shape
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fuse_fine_coarse_scores(fine, coarse, weight=0), fine)


def test_grounding_balances_subject_foreground_and_background():
    from rcr.proposed.losses import grounding_loss

    logits = torch.zeros(1, 1, 4, requires_grad=True)
    targets = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    loss = grounding_loss(logits, targets, torch.ones_like(targets, dtype=torch.bool))
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 0, 0].abs(), logits.grad[0, 0, 1:].sum())
    # More background observations do not change the relative class weight.
    first = grounding_loss(
        torch.tensor([[[-1.0, -2.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.ones(1, 1, 2, dtype=torch.bool),
    )
    second = grounding_loss(
        torch.tensor([[[-1.0, *([-2.0] * 8)]]]),
        torch.tensor([[[1.0, *([0.0] * 8)]]]),
        torch.ones(1, 1, 9, dtype=torch.bool),
    )
    torch.testing.assert_close(first, second)


def test_grounding_ignores_unknown_crops_subjects_and_queries():
    from rcr.proposed.losses import grounding_loss

    logits = torch.tensor(
        [
            [[0.0, float("nan")], [float("nan"), float("nan")]],
            [[float("nan"), float("nan")], [float("nan"), float("nan")]],
        ],
        requires_grad=True,
    )
    targets = torch.ones_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 0, 0] = True
    value = grounding_loss(logits, targets, mask)
    torch.testing.assert_close(value, torch.tensor(2.0).log())
    value.backward()
    assert logits.grad[~mask].count_nonzero() == 0
    assert torch.isfinite(logits.grad).all()


def test_ranking_averages_queries_instead_of_number_of_pairs():
    import torch.nn.functional as F

    scores = torch.tensor(
        [[2.0, 0.0, 0.0, float("nan")], [-1.0, -1.0, 1.0, 1.0]], requires_grad=True
    )
    positives = torch.tensor([[True, False, False, False], [True, True, False, False]])
    valid = torch.ones_like(positives)
    valid[0, -1] = False
    loss = retrieval_loss(scores, positives, valid)
    expected = (F.softplus(torch.tensor(-2.0)) + F.softplus(torch.tensor(2.0))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, -1] == 0
    assert (scores.grad[positives] < 0).all()
    assert (scores.grad[~positives & valid] > 0).all()


def test_ranking_empty_supervision_is_finite_differentiable_zero():
    scores = torch.tensor([[float("nan"), float("inf")]], requires_grad=True)
    positives = torch.tensor([[True, False]])
    loss = retrieval_loss(scores, positives, torch.zeros_like(positives))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))
    all_positive = torch.tensor([[2.0, 3.0]], requires_grad=True)
    retrieval_loss(all_positive, torch.ones(1, 2, dtype=torch.bool)).backward()
    assert all_positive.grad.count_nonzero() == 0


@pytest.mark.parametrize(
    "config",
    [
        {"retrieval_weight": 1},
        {"coarse_temperature": 0},
        {"identity_temperature": float("nan")},
        {"context_weight": -1},
        {"state_weight": True},
        {"grounding_weight": float("inf")},
        {
            "grounding_weight": 0,
            "identity_weight": 0,
            "coarse_weight": 0,
            "context_weight": 0,
            "state_weight": 0,
        },
    ],
)
def test_invalid_or_stale_loss_settings_fail_explicitly(config):
    from rcr.proposed.losses import loss_settings

    with pytest.raises(ValueError, match="loss|weight"):
        loss_settings(config)
