import torch

from rcr.methods.proposed.binding import EvidenceBinding
from rcr.methods.proposed.grounding import SubjectGrounding


def test_grounding_shape_and_grad() -> None:
    torch.manual_seed(0)
    binding = EvidenceBinding(dim=8, num_heads=2)
    model = SubjectGrounding(binding, dim=8)

    scene = torch.randn(2, 6, 8, requires_grad=True)
    persons = torch.randn(2, 3, 8, requires_grad=True)
    selections = torch.randn(2, 2, 4, 8, requires_grad=True)
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.4, 1.0], [0.3, 0.1, 0.7, 0.9], [0.6, 0.0, 1.0, 1.0]]]
    ).expand(2, -1, -1)

    logits = model(scene, persons, boxes, selections, patch_hw=(2, 3))
    assert logits.shape == (2, 2, 3)

    logits.mean().backward()
    assert scene.grad is not None
    assert persons.grad is not None
    assert selections.grad is not None


def test_same_selection_gives_same_logits() -> None:
    torch.manual_seed(1)
    binding = EvidenceBinding(dim=8, num_heads=2)
    model = SubjectGrounding(binding, dim=8).eval()

    scene = torch.randn(1, 6, 8)
    persons = torch.randn(1, 2, 8)
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]])
    selection = torch.randn(1, 1, 4, 8)
    selections = selection.expand(-1, 2, -1, -1)

    logits = model(scene, persons, boxes, selections, patch_hw=(2, 3))
    torch.testing.assert_close(logits[:, 0], logits[:, 1])
