"""Small contracts for the equations that define the proposed method."""

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
    assert (membership > 0).tolist() == [
        [[True, True, False], [False, False, True]]
    ]
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
    bad = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
    )
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


def test_fine_coarse_fusion_uses_common_shortlist_statistics():
    fine = torch.tensor([2.0, 1.0, 0.0])
    coarse = torch.tensor([0.0, 1.0, 2.0])
    fused = fuse_fine_coarse_scores(fine, coarse, weight=0.5)
    assert fused.shape == fine.shape
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(
        fuse_fine_coarse_scores(fine, coarse, weight=0), fine
    )
