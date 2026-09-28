import torch

from rcr.methods.proposed.composition import StructuredComposition


def test_role_mapping_uses_subject_id_not_list_position() -> None:
    torch.manual_seed(12)
    model = StructuredComposition(8, 6, 2).eval()
    change = torch.randn(1, 3, 8)
    ids = torch.randn(1, 2, 6)
    logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    a = model(
        change, ids, logits, torch.tensor([[0, 2]]), subject_ids=torch.tensor([[1, 2]])
    )
    b = model(
        change, ids, logits, torch.tensor([[0, 2]]), subject_ids=torch.tensor([[2, 1]])
    )
    assert not torch.allclose(a[:, 0], b[:, 0])


def test_composition_shape_and_grad() -> None:
    torch.manual_seed(0)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2)

    change = torch.randn(2, 5, 8, requires_grad=True)
    identity = torch.randn(2, 3, 6, requires_grad=True)
    logits = torch.randn(2, 2, 3, requires_grad=True)
    subject_pos = torch.tensor([[0, 3], [1, 4]])

    out = model(change, identity, logits, subject_pos)
    assert out.shape == (2, 12, 8)

    out.square().mean().backward()
    assert change.grad is not None
    assert identity.grad is not None
    assert logits.grad is not None


def test_grounding_pool_matches_normalized_sigmoid_membership() -> None:
    torch.manual_seed(1)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2).eval()

    change = torch.randn(1, 5, 8)
    identity = torch.randn(1, 3, 6)
    logits = torch.tensor([[[2.0, 0.0, -1.0], [-2.0, 1.0, 0.5]]])
    subject_pos = torch.tensor([[0, 3]])

    captured = {}

    def capture_bind_input(_module, args) -> None:
        captured["input"] = args[0].detach().clone()

    handle = model.bind_norm.register_forward_pre_hook(capture_bind_input)
    model(change, identity, logits, subject_pos)
    handle.remove()

    role = model.role(torch.arange(2))
    ids = model.identity_proj(identity)[:, None] + role[None, :, None]
    membership = logits.sigmoid()
    p = membership / membership.sum(dim=-1, keepdim=True)

    batch = torch.arange(1)[:, None]
    subjects = change[batch, subject_pos]
    expected = subjects + torch.sum(p[..., None] * ids, dim=2)

    torch.testing.assert_close(captured["input"][batch, subject_pos], expected)


def test_repeated_mentions_receive_the_same_grounded_identity() -> None:
    torch.manual_seed(7)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2)
    change = torch.randn(1, 5, 8)
    identity = torch.randn(1, 3, 6)
    logits = torch.randn(1, 2, 3)
    mentions = torch.tensor(
        [[[True, False, True, False, False], [False, True, False, False, True]]]
    )
    captured = {}

    def capture(_module, args):
        captured["input"] = args[0].detach().clone()

    handle = model.bind_norm.register_forward_pre_hook(capture)
    model(change, identity, logits, torch.tensor([[0, 1]]), subject_token_mask=mentions)
    handle.remove()
    ids = model.identity_proj(identity)[:, None] + model.role.weight[None, :, None]
    membership = logits.sigmoid()
    weight = membership / membership.sum(-1, keepdim=True)
    grounded = (weight[..., None] * ids).sum(2)
    delta = captured["input"] - change
    for subject, positions in enumerate(([0, 2], [1, 4])):
        for position in positions:
            torch.testing.assert_close(delta[:, position], grounded[:, subject])
    torch.testing.assert_close(delta[:, 3], torch.zeros_like(delta[:, 3]))


def test_padded_subject_has_no_effect_on_real_composition_tokens() -> None:
    torch.manual_seed(8)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2)
    change = torch.randn(1, 5, 8)
    identity = torch.randn(1, 3, 6)
    logits = torch.randn(1, 1, 3)
    expected = model(change, identity, logits, torch.tensor([[1]]))
    padded_logits = torch.cat((logits, torch.full_like(logits, -torch.inf)), 1)
    actual = model(
        change,
        identity,
        padded_logits,
        torch.tensor([[1, 0]]),
        subject_mask=torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(actual[:, : expected.shape[1]], expected)
    assert torch.isfinite(actual).all()


def test_empty_query_composition_is_finite() -> None:
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2)
    out = model(
        torch.randn(1, 5, 8),
        torch.empty(1, 0, 6),
        torch.empty(1, 1, 0),
        torch.tensor([[0]]),
    )
    assert out.shape == (1, 6, 8)
    assert torch.isfinite(out).all()


def test_identity_order_is_permutation_invariant() -> None:
    torch.manual_seed(2)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2).eval()

    change = torch.randn(1, 5, 8)
    identity = torch.randn(1, 3, 6)
    logits = torch.randn(1, 2, 3)
    subject_pos = torch.tensor([[0, 3]])

    out = model(change, identity, logits, subject_pos)

    perm = torch.tensor([2, 0, 1])
    out_perm = model(change, identity[:, perm], logits[:, :, perm], subject_pos)

    text_end = 1 + change.shape[1]
    torch.testing.assert_close(out[:, :text_end], out_perm[:, :text_end])

    ids = out[:, text_end:].reshape(1, 2, 3, 8)
    ids_perm = out_perm[:, text_end:].reshape(1, 2, 3, 8)
    torch.testing.assert_close(ids, ids_perm[:, :, torch.argsort(perm)])


def test_subject_marker_uses_its_grounding_distribution() -> None:
    torch.manual_seed(3)
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2).eval()

    change = torch.randn(1, 5, 8)
    identity = torch.randn(1, 2, 6)
    logits = torch.zeros(1, 2, 2)
    subject_pos = torch.tensor([[0, 3]])

    logits_a = logits.clone()
    logits_a[0, 0] = torch.tensor([20.0, -20.0])
    out_a = model(change, identity, logits_a, subject_pos)

    logits_b = logits.clone()
    logits_b[0, 0] = torch.tensor([-20.0, 20.0])
    out_b = model(change, identity, logits_b, subject_pos)

    s1 = 1 + subject_pos[0, 0]
    assert not torch.allclose(out_a[:, s1], out_b[:, s1])


def test_binding_has_no_content_attention() -> None:
    model = StructuredComposition(dim=8, identity_dim=6, num_heads=2)
    assert not hasattr(model, "bind_attn")
