"""Contracts for the jointly learned person representation."""

import copy

import pytest
import torch

from rcr.proposed import retrieve as inference
from rcr.proposed.losses import (
    compute_loss,
    grounding_loss,
    identity_loss,
    retrieval_loss,
)
from rcr.proposed.nn.model import RCRModel
from tests.proposed.test_training import _batch


def _model(*, selected=True):
    model = RCRModel(
        8,
        6,
        2,
        input_dim=10,
        person_input_dim=12,
        fine_matching={"identity_threshold": -1.0, "duplicate_iou": 1.0},
    )
    if selected:
        with torch.no_grad():
            model.grounding.score.weight.zero_()
            model.grounding.score.bias.fill_(5)
    return model


def _inputs():
    batch = _batch()
    for key in ("query_scene", "target_scene"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 10)
    for key in ("query_persons", "target_persons"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 12)
    return batch


def _query(model, batch):
    return model.encode_query(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        batch["change"],
        batch["subject_pos"],
        (2, 3),
        selection_mask=batch["selection_mask"],
        change_mask=batch["change_mask"],
        query_person_mask=batch["query_person_mask"],
        fafa_query=batch["fafa_query"],
    )


def _fine(model, batch, query):
    return model.score_target(
        query.repeat_candidates(batch["positive_mask"].shape[1]),
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        (2, 3),
        batch["target_mask"].flatten(0, 1),
        return_details=True,
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
    )


def _assert_person_gradient(model):
    for parameter in model.person_proj.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_identity_loss_updates_shared_person_and_identity_head_only():
    torch.manual_seed(90)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    identity_loss(
        model.encode_identity(torch.randn(6, 12)), torch.tensor([0, 1, 2, 0, 1, 2])
    ).backward()
    _assert_person_gradient(model)
    assert all(p.grad.abs().sum() > 0 for p in model.identity_head.parameters())
    assert all(p.grad is None for p in model.visual_proj.parameters())
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith(("person_proj.", "identity_head.")):
            assert not torch.equal(before[name], parameter)
        else:
            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)


def test_grounding_loss_updates_shared_person_without_identity_head():
    torch.manual_seed(91)
    model, batch = _model(selected=False), _inputs()
    query = _query(model, batch)
    mask = batch["query_person_mask"][:, None].expand_as(query.logits)
    grounding_loss(query.logits, batch["grounding_targets"], mask).backward()
    _assert_person_gradient(model)
    assert model.grounding.score.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.identity_head.parameters())


@pytest.mark.parametrize("branch", ["context", "retrieval"])
def test_fine_gradients_reach_query_and_target_person_but_not_identity(branch):
    torch.manual_seed(92)
    model, batch = _model(), _inputs()
    batch["query_persons"].requires_grad_()
    batch["target_persons"].requires_grad_()
    query = _query(model, batch)
    query.identity.retain_grad()
    details = _fine(model, batch, query)
    assert details["coverage"].gt(0).all()
    assert not query.membership.requires_grad
    assert not query.prior.requires_grad
    assert not details["membership"].requires_grad
    assert not details["matched_indices"].requires_grad
    assert not details["coverage"].requires_grad
    if branch == "retrieval":
        loss = retrieval_loss(
            details["score"].reshape_as(batch["positive_mask"]),
            batch["positive_mask"],
            batch["candidate_mask"],
        )
    else:
        loss = details["context"].sum()
    loss.backward()
    _assert_person_gradient(model)
    assert query.identity.grad is None
    assert all(p.grad is None for p in model.identity_head.parameters())
    assert model.grounding.score.weight.grad is None
    assert batch["query_persons"].grad[:, :2].abs().sum() > 0
    assert batch["query_persons"].grad[:, 2].count_nonzero() == 0
    assert batch["target_persons"].grad.abs().sum() > 0


def test_retrieval_only_optimizer_step_leaves_identity_head_untouched():
    torch.manual_seed(93)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    person_before = model.person_proj[0].weight.detach().clone()
    identity_before = copy.deepcopy(model.identity_head.state_dict())
    loss, parts = compute_loss(
        model,
        _inputs(),
        (2, 3),
        grounding_weight=0,
        identity_weight=0,
        state_weight=0,
    )
    torch.testing.assert_close(loss.detach(), parts["retrieval"])
    loss.backward()
    _assert_person_gradient(model)
    assert all(p.grad is None for p in model.identity_head.parameters())
    optimizer.step()
    assert not torch.equal(person_before, model.person_proj[0].weight)
    for name, actual in model.identity_head.state_dict().items():
        torch.testing.assert_close(identity_before[name], actual, rtol=0, atol=0)


def test_grounding_composition_identity_and_target_evidence_read_same_u():
    torch.manual_seed(94)
    model, batch = _model().eval(), _inputs()
    projected, identity, grounding, composition, evidence = [], [], [], [], []
    hooks = [
        model.person_proj.register_forward_hook(
            lambda m, a, out: projected.append(out)
        ),
        model.identity_head.register_forward_pre_hook(
            lambda m, a: identity.append(a[0])
        ),
        model.grounding.register_forward_pre_hook(lambda m, a: grounding.append(a[1])),
        model.composition.register_forward_pre_hook(
            lambda m, a: composition.append(a[1])
        ),
        model.grounding.binding.register_forward_pre_hook(
            lambda m, a: evidence.append(a[1])
        ),
    ]
    query = _query(model, batch)
    details = _fine(model, batch, query)
    for hook in hooks:
        hook.remove()
    assert len(projected) == len(identity) == 2  # one projection per side
    assert identity[0] is grounding[0] is composition[0] is projected[0]
    assert identity[1] is projected[1]
    # LayerNorm u is not L2-normalized before evidence/composition.
    assert not torch.allclose(projected[0].norm(dim=-1), torch.ones(2, 3))
    assert model.binding_log_scale is not None
    assert model.person_proj[0].in_features == 12
    assert model.visual_proj.in_features == 10
    indices = details["matched_indices"].clamp_min(0)
    aligned = projected[1].gather(1, indices[..., None].expand(-1, -1, 8))
    aligned = aligned[:, None].expand(-1, 2, -1, -1)
    matched = (details["membership"] > 0) & (details["matched_indices"] >= 0)[:, None]
    expected = aligned.masked_fill(~matched[..., None], 0).flatten(1, 2)
    torch.testing.assert_close(evidence[-1], expected)


def test_masked_nonfinite_people_and_boxes_do_not_change_output_or_gradient():
    torch.manual_seed(95)
    model, batch = _model().eval(), _inputs()
    batch["target_mask"][..., -1] = False
    expected = _fine(model, batch, _query(model, batch))
    bad = copy.deepcopy(batch)
    for prefix, mask_key in (("query", "query_person_mask"), ("target", "target_mask")):
        for suffix in ("persons", "boxes"):
            value = bad[f"{prefix}_{suffix}"]
            value[~bad[mask_key]] = float("nan" if suffix == "persons" else "inf")
        bad[f"{prefix}_persons"].requires_grad_()
    actual = _fine(model, bad, _query(model, bad))
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])
    actual["score"].sum().backward()
    for prefix, mask_key in (("query", "query_person_mask"), ("target", "target_mask")):
        gradient = bad[f"{prefix}_persons"].grad
        assert torch.isfinite(gradient).all()
        assert gradient[~bad[mask_key]].count_nonzero() == 0


@pytest.mark.parametrize("query_people,target_people", [(0, 0), (0, 2), (2, 0)])
def test_zero_person_images_keep_null_penalties_and_finite_backward(
    query_people,
    target_people,
):
    torch.manual_seed(96)
    model, batch = _model(), _inputs()
    for key in ("query_persons", "query_boxes", "query_person_mask"):
        batch[key] = batch[key][:, :query_people]
    batch["fafa_query"] = batch["fafa_query"][:, :, :query_people]
    for key in ("target_persons", "target_boxes", "target_mask"):
        batch[key] = batch[key][:, :, :target_people]
    batch["fafa_targets"] = batch["fafa_targets"][:, :, :target_people]
    query = _query(model, batch)
    details = _fine(model, batch, query)
    assert details["coverage"].count_nonzero() == 0
    torch.testing.assert_close(details["binding"], torch.full((6,), -1.1))
    torch.testing.assert_close(details["gated_context"], torch.zeros(6))
    torch.testing.assert_close(details["score"], -1.1 * model.binding_scale.expand(6))
    details["score"].sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_shared_person_amp_has_finite_losses_and_person_update(device):
    torch.manual_seed(97)
    model = _model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.person_proj[0].weight.detach().clone()
    batch = {key: value.to(device) for key, value in _inputs().items()}
    batch["target_mask"][..., -1] = False
    batch["target_persons"][..., -1, :] = float("nan")
    batch["target_boxes"][:, :, 0] = torch.tensor(
        [0.4, 0.3, 0.400001, 0.300001],
        device=device,
    )
    # Match train.py's FP16 AMP; existing SciPy matching does not support BF16.
    with torch.autocast(device, dtype=torch.float16):
        loss, parts = compute_loss(model, batch, (2, 3))
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(parts[key])
        for key in ("identity", "grounding", "retrieval", "state")
    )
    loss.backward()
    _assert_person_gradient(model)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert not torch.equal(before, model.person_proj[0].weight)


def test_checkpoint_roundtrip_reproduces_query_and_all_fine_details(tmp_path):
    torch.manual_seed(98)
    model, batch = _model().eval(), _inputs()
    query = _query(model, batch)
    expected = _fine(model, batch, query)
    path = tmp_path / "shared_person.pt"
    torch.save(model.state_dict(), path)
    restored = _model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual_query = _query(restored, batch)
    for name in vars(query):
        torch.testing.assert_close(
            getattr(query, name), getattr(actual_query, name), rtol=0, atol=0
        )
    actual = _fine(restored, batch, actual_query)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_checkpoint_representation_metadata_cannot_disagree(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {"model": {"representation": "shared_person"}},
            "representation": "dual",
            "architecture_version": _model().architecture_version,
        },
        path,
    )
    with pytest.raises(ValueError, match="differs from its architecture"):
        inference.retrieve({"checkpoint": str(path), "runtime": {"device": "cpu"}})
