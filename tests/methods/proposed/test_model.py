import torch

from rcr.methods.proposed.binding import EvidenceBinding
from rcr.methods.proposed.model import RCRModel


def _boxes(batch: int, people: int) -> torch.Tensor:
    boxes = torch.rand(batch, people, 4)
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:] * (1 - boxes[..., :2])
    return boxes


def test_model_forward_shape_and_grad() -> None:
    torch.manual_seed(0)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)

    query_scene = torch.randn(2, 6, 8, requires_grad=True)
    query_persons = torch.randn(2, 3, 8, requires_grad=True)
    selections = torch.randn(2, 2, 4, 8, requires_grad=True)
    change = torch.randn(2, 5, 8, requires_grad=True)

    target_scene = torch.randn(2, 6, 8, requires_grad=True)
    target_persons = torch.randn(2, 4, 8, requires_grad=True)

    logits, score = model(
        query_scene=query_scene,
        query_persons=query_persons,
        query_boxes=_boxes(2, 3),
        selections=selections,
        change=change,
        subject_pos=torch.tensor([[0, 3], [1, 4]]),
        target_scene=target_scene,
        target_persons=target_persons,
        target_boxes=_boxes(2, 4),
        patch_hw=(2, 3),
    )

    assert logits.shape == (2, 2, 3)
    assert score.shape == (2,)

    (logits.mean() + score.mean()).backward()
    for tensor in (
        query_scene,
        query_persons,
        selections,
        change,
        target_scene,
        target_persons,
    ):
        assert tensor.grad is not None
    assert model.identity_head.proj.weight.grad is not None


def test_model_uses_one_shared_binding() -> None:
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    bindings = [
        module for module in model.modules() if isinstance(module, EvidenceBinding)
    ]
    assert len(bindings) == 1


def test_binding_order_and_composed_reference_are_shared() -> None:
    torch.manual_seed(3)
    model = RCRModel(8, 6, 2).eval()
    binding = model.grounding.binding
    events = []
    captured = {}

    def record_ref(_module, args):
        events.append("scene reads reference")
        captured["reference"] = args[1].detach().clone()

    def record_person(_module, args):
        events.append("person reads conditioned scene")

    a = binding.ref_attn.register_forward_pre_hook(record_ref)
    b = binding.person_attn.register_forward_pre_hook(record_person)
    change = torch.randn(1, 4, 8)
    model(
        torch.randn(1, 4, 8),
        torch.randn(1, 2, 8),
        _boxes(1, 2),
        torch.randn(1, 1, 3, 8),
        change,
        torch.tensor([[1]]),
        torch.randn(1, 4, 8),
        torch.randn(1, 2, 8),
        _boxes(1, 2),
        (2, 2),
    )
    a.remove()
    b.remove()
    assert events == ["scene reads reference", "person reads conditioned scene"] * 2
    assert captured["reference"].shape[1] == 1 + change.shape[1] + 2


def test_target_score_changes_with_membership_prior() -> None:
    torch.manual_seed(4)
    model = RCRModel(8, 6, 2).eval()
    ref = torch.randn(1, 5, 8)
    scene = torch.randn(1, 4, 8)
    persons = torch.randn(1, 2, 8)
    boxes = _boxes(1, 2)
    mask = torch.ones(1, 5, dtype=torch.bool)
    a = model.score_target(ref, mask, torch.zeros(1, 5), scene, persons, boxes, (2, 2))
    b = model.score_target(
        ref,
        mask,
        torch.tensor([[0.0, 0.0, 0.0, -15.0, 0.0]]),
        scene,
        persons,
        boxes,
        (2, 2),
    )
    assert not torch.allclose(a, b)


def test_empty_query_and_target_are_finite() -> None:
    model = RCRModel(8, 6, 2)
    logits, score = model(
        torch.randn(1, 4, 8),
        torch.empty(1, 0, 8),
        torch.empty(1, 0, 4),
        torch.randn(1, 1, 3, 8),
        torch.randn(1, 4, 8),
        torch.tensor([[1]]),
        torch.randn(1, 4, 8),
        torch.empty(1, 0, 8),
        torch.empty(1, 0, 4),
        (2, 2),
    )
    assert logits.shape == (1, 1, 0)
    assert torch.isfinite(score).all()


def test_batched_scores_equal_independent_pairs() -> None:
    torch.manual_seed(5)
    model = RCRModel(8, 6, 2).eval()
    q_scene = torch.randn(2, 4, 8)
    q_people = torch.randn(2, 2, 8)
    q_box = _boxes(2, 2)
    selection = torch.randn(2, 1, 3, 8)
    change = torch.randn(2, 4, 8)
    t_scene = torch.randn(2, 4, 8)
    t_people = torch.randn(2, 3, 8)
    t_box = _boxes(2, 3)
    args = (
        q_scene,
        q_people,
        q_box,
        selection,
        change,
        torch.tensor([[1], [1]]),
        t_scene,
        t_people,
        t_box,
        (2, 2),
    )
    both = model(*args)[1]
    one = torch.stack(
        [
            model(*[x[i : i + 1] if isinstance(x, torch.Tensor) else x for x in args])[
                1
            ][0]
            for i in range(2)
        ]
    )
    torch.testing.assert_close(both, one, atol=1e-5, rtol=1e-5)


def test_model_ignores_padded_change_and_people() -> None:
    torch.manual_seed(1)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2).eval()

    query_scene = torch.randn(1, 6, 8)
    query_persons = torch.randn(1, 3, 8)
    selections = torch.randn(1, 2, 4, 8)
    change = torch.randn(1, 5, 8)

    target_scene = torch.randn(1, 6, 8)
    target_persons = torch.randn(1, 3, 8)

    kwargs = dict(
        query_scene=query_scene,
        query_persons=query_persons,
        query_boxes=_boxes(1, 3),
        selections=selections,
        change=change,
        subject_pos=torch.tensor([[0, 2]]),
        target_scene=target_scene,
        target_persons=target_persons,
        target_boxes=_boxes(1, 3),
        patch_hw=(2, 3),
        selection_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        change_mask=torch.tensor([[True, True, True, False, False]]),
        query_person_mask=torch.tensor([[True, True, False]]),
        target_mask=torch.tensor([[True, True, False]]),
    )

    _, expected = model(**kwargs)

    changed = dict(kwargs)
    changed["change"] = change.clone()
    changed["change"][:, 3:] = 1000.0
    changed["query_persons"] = query_persons.clone()
    changed["query_persons"][:, 2] = 1000.0
    changed["target_persons"] = target_persons.clone()
    changed["target_persons"][:, 2] = 1000.0

    _, actual = model(**changed)
    torch.testing.assert_close(actual, expected)


def test_inference_ignores_padded_subject_and_supports_repeated_mentions() -> None:
    torch.manual_seed(10)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2).eval()
    kwargs = dict(
        query_scene=torch.randn(1, 6, 8),
        query_persons=torch.randn(1, 3, 8),
        query_boxes=_boxes(1, 3),
        selections=torch.randn(1, 2, 4, 8),
        change=torch.randn(1, 5, 8),
        subject_pos=torch.tensor([[0, 0]]),
        target_scene=torch.randn(1, 6, 8),
        target_persons=torch.randn(1, 3, 8),
        target_boxes=_boxes(1, 3),
        patch_hw=(2, 3),
        subject_mask=torch.tensor([[True, False]]),
        subject_token_mask=torch.tensor(
            [[[True, False, False, True, False], [False] * 5]]
        ),
    )
    _, expected = model(**kwargs)
    changed = dict(kwargs)
    changed["selections"] = kwargs["selections"].clone()
    changed["selections"][:, 1] = 1000
    _, actual = model(**changed)
    torch.testing.assert_close(actual, expected)
    # Removing the padded Subject entirely must produce the same score.
    trimmed = dict(kwargs)
    for key in ("selections", "subject_pos", "subject_mask", "subject_token_mask"):
        trimmed[key] = kwargs[key][:, :1]
    _, actual = model(**trimmed)
    torch.testing.assert_close(actual, expected)
