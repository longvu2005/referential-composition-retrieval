import torch

from rcr.proposed.nn.binding import EvidenceBinding


def test_binding_shape_and_grad() -> None:
    torch.manual_seed(0)
    model = EvidenceBinding(dim=8, num_heads=2)

    scene = torch.randn(2, 6, 8, requires_grad=True)
    persons = torch.randn(2, 3, 8, requires_grad=True)
    reference = torch.randn(2, 4, 8, requires_grad=True)
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.4, 1.0], [0.3, 0.1, 0.7, 0.9], [0.6, 0.0, 1.0, 1.0]]]
    ).expand(2, -1, -1)

    out = model(scene, persons, boxes, reference, patch_hw=(2, 3))
    assert out.shape == persons.shape

    out.mean().backward()
    assert scene.grad is not None
    assert persons.grad is not None
    assert reference.grad is not None


def test_reference_padding_is_ignored() -> None:
    torch.manual_seed(1)
    model = EvidenceBinding(dim=8, num_heads=2).eval()

    scene = torch.randn(1, 6, 8)
    persons = torch.randn(1, 1, 8)
    boxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
    reference = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])

    expected = model(scene, persons, boxes, reference, (2, 3), mask)
    reference[:, 2:] = 1000.0
    actual = model(scene, persons, boxes, reference, (2, 3), mask)

    torch.testing.assert_close(actual, expected)


def test_geometry_depends_on_box() -> None:
    torch.manual_seed(2)
    model = EvidenceBinding(dim=8, num_heads=2)

    left = torch.tensor([[[0.0, 0.0, 0.4, 1.0]]])
    right = torch.tensor([[[0.6, 0.0, 1.0, 1.0]]])

    assert not torch.allclose(
        model._geo_bias(left, (2, 3)),
        model._geo_bias(right, (2, 3)),
    )
