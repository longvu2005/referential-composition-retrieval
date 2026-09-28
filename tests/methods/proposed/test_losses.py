import torch
import torch.nn.functional as F

from rcr.methods.proposed.losses import grounding_loss, identity_loss, retrieval_loss


def test_grounding_loss_matches_bce_and_mask() -> None:
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    targets = torch.tensor([[[1.0, 0.0, 1.0]]])
    mask = torch.tensor([[[True, True, False]]])

    expected = F.binary_cross_entropy_with_logits(
        logits[..., :2],
        targets[..., :2],
    )
    torch.testing.assert_close(grounding_loss(logits, targets, mask), expected)


def test_identity_loss_prefers_same_identity() -> None:
    labels = torch.tensor([0, 0, 1, 1])

    good = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        requires_grad=True,
    )
    bad = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    good_loss = identity_loss(good, labels)
    bad_loss = identity_loss(bad, labels)
    assert good_loss < bad_loss

    good_loss.backward()
    assert good.grad is not None


def test_identity_loss_ignores_unknown_identity() -> None:
    identity = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [100.0, -100.0],
        ]
    )
    labels = torch.tensor([0, 0, -1])

    expected = identity_loss(identity[:2], labels[:2])
    actual = identity_loss(identity, labels)
    torch.testing.assert_close(actual, expected)


def test_retrieval_loss_rewards_positive_scores() -> None:
    positive = torch.tensor([[True, False, False]])
    good = torch.tensor([[3.0, 0.0, -1.0]], requires_grad=True)
    bad = torch.tensor([[0.0, 3.0, -1.0]])

    good_loss = retrieval_loss(good, positive)
    bad_loss = retrieval_loss(bad, positive)
    assert good_loss < bad_loss

    good_loss.backward()
    assert good.grad is not None


def test_retrieval_loss_supports_multiple_positives_and_padding() -> None:
    scores = torch.tensor([[2.0, 1.0, 100.0]])
    positive = torch.tensor([[True, True, False]])
    valid = torch.tensor([[True, True, False]])

    loss = retrieval_loss(scores, positive, valid_mask=valid)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_retrieval_is_pairwise_softplus() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False]])
    expected = (
        F.softplus(scores[0, 2] - scores[0, 0])
        + F.softplus(scores[0, 2] - scores[0, 1])
    ) / 2
    torch.testing.assert_close(retrieval_loss(scores, positives), expected)


def test_grounding_masked_infinity_never_enters_bce() -> None:
    logits = torch.tensor([[[1.0, -torch.inf]]], requires_grad=True)
    value = grounding_loss(
        logits, torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[True, False]]])
    )
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()
