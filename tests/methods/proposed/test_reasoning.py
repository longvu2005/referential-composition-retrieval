import torch

from rcr.methods.proposed.reasoning import FineReasoner, TargetPersonBuilder


def test_target_person_builder_shape_and_grad() -> None:
    torch.manual_seed(0)
    model = TargetPersonBuilder(dim=8, identity_dim=6)

    evidence = torch.randn(2, 3, 8, requires_grad=True)
    identity = torch.randn(2, 3, 6, requires_grad=True)
    boxes = torch.rand(2, 3, 4)
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:] * (1 - boxes[..., :2])

    out = model(evidence, boxes, identity)
    assert out.shape == (2, 3, 8)

    out.mean().backward()
    assert evidence.grad is not None
    assert identity.grad is not None


def test_target_person_builder_uses_box_geometry() -> None:
    torch.manual_seed(1)
    model = TargetPersonBuilder(dim=8, identity_dim=6).eval()

    evidence = torch.randn(1, 1, 8)
    identity = torch.randn(1, 1, 6)
    left = torch.tensor([[[0.0, 0.1, 0.4, 0.9]]])
    right = torch.tensor([[[0.6, 0.1, 1.0, 0.9]]])

    assert not torch.allclose(
        model(evidence, left, identity),
        model(evidence, right, identity),
    )


def test_fine_reasoner_shape_and_grad() -> None:
    torch.manual_seed(2)
    model = FineReasoner(dim=8, num_heads=2)

    query = torch.randn(2, 7, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, requires_grad=True)

    score = model(query, target)
    assert score.shape == (2,)

    score.mean().backward()
    assert query.grad is not None
    assert target.grad is not None


def test_fine_reasoner_ignores_padded_targets() -> None:
    torch.manual_seed(3)
    model = FineReasoner(dim=8, num_heads=2).eval()

    query = torch.randn(1, 5, 8)
    target = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, False]])

    expected = model(query, target, target_mask=mask)
    target[:, 2] = 1000.0
    actual = model(query, target, target_mask=mask)

    torch.testing.assert_close(actual, expected)


def test_fine_reasoner_ignores_padded_query_tokens() -> None:
    torch.manual_seed(4)
    model = FineReasoner(dim=8, num_heads=2).eval()

    query = torch.randn(1, 5, 8)
    target = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, True, False, False]])

    expected = model(query, target, query_mask=mask)
    query[:, 3:] = 1000.0
    actual = model(query, target, query_mask=mask)

    torch.testing.assert_close(actual, expected)
