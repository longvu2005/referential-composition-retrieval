"""Regressions for geometry-driven FP16 NaNs and stopping failed training runs."""

import json
from pathlib import Path

import pytest
import torch

from rcr.methods.proposed import train as training
from rcr.methods.proposed.binding import EvidenceBinding
from rcr.methods.proposed.losses import retrieval_loss
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.objective import compute_loss
from rcr.methods.proposed.reasoning import FineReasoner
from tests.methods.proposed.test_runner import experiment as experiment
from tests.methods.proposed.test_runner import local_stages as local_stages
from tests.methods.proposed.test_training import _batch
from tools.methods import run as cli

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA required for T4/FP16 kernel checks",
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_large_geometry_bias_stays_finite_in_fp16_forward_and_backward(device, seed):
    torch.manual_seed(seed)
    model = EvidenceBinding(64, 8).to(device).train()
    boxes = torch.tensor(
        [[[0.1, 0.1, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0], [0.4, 0.3, 0.400001, 0.300001]]],
        device=device,
    )
    scene = torch.randn(1, 196, 64, device=device, requires_grad=True)
    people = torch.randn(1, 3, 64, device=device, requires_grad=True)
    reference = torch.randn(1, 10, 64, device=device, requires_grad=True)
    bias = model._geo_bias(boxes, (14, 14))
    assert bias.dtype == torch.float32
    # Padding is neutral, while genuine tiny boxes keep their large geometry.
    assert bias[:, 1].count_nonzero() == 0
    assert bias[:, 2].abs().max() > torch.finfo(torch.float16).max
    observed = []
    hook = model.person_attn.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(
            (args[0].dtype, kwargs["attn_mask"].dtype)
        ),
        with_kwargs=True,
    )
    with torch.autocast(device, dtype=torch.float16):
        output = model(scene, people, boxes, reference, (14, 14))
        loss = output[..., 0].float().square().mean()
    hook.remove()
    assert observed == [(torch.float32, torch.float32)]
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert scene.grad.abs().sum() > 0 and reference.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_padded_target_batch_has_finite_retrieval_loss_and_real_update(device, seed):
    torch.manual_seed(seed)
    model = RCRModel(8, 6, 2).to(device).train()
    batch = {name: value.to(device) for name, value in _batch().items()}
    batch["target_boxes"][:, :, -1] = 0
    batch["target_persons"][:, :, -1] = 0
    batch["target_mask"][:, :, -1] = False
    batch["target_boxes"][:, :, 0] = torch.tensor(
        [0.4, 0.3, 0.400001, 0.300001], device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device, init_scale=16)
    before = model.reasoner.score[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16):
            loss, parts = compute_loss(model, batch, (2, 3))
        assert torch.isfinite(loss) and torch.isfinite(parts["retrieval"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
    assert not torch.equal(before, model.reasoner.score[-1].weight)


def test_valid_box_geometry_is_unaffected_by_padding():
    model = EvidenceBinding(8, 2)
    valid = torch.tensor([[[0.1, 0.2, 0.9, 0.8]]])
    padded = torch.cat((valid, torch.zeros(1, 2, 4)), dim=1)
    torch.testing.assert_close(
        model._geo_bias(padded, (2, 3))[:, :1], model._geo_bias(valid, (2, 3))
    )


def test_masked_nonfinite_target_tokens_cannot_contaminate_fine_attention():
    torch.manual_seed(18)
    model = FineReasoner(8, 2).eval()
    query = torch.randn(2, 4, 8)
    target = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    expected = model(query, target, mask)
    poisoned = target.clone()
    poisoned[~mask] = float("nan")
    poisoned[0, 2] = float("inf")
    poisoned.requires_grad_()
    actual = model(query, poisoned, mask)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(poisoned.grad).all()
    assert poisoned.grad[~mask].count_nonzero() == 0


@pytest.mark.parametrize("has_pair", [False, True])
def test_retrieval_loss_ignores_nonfinite_invalid_candidates(has_pair):
    scores = torch.tensor([[0.2, 0.5, float("nan"), float("inf")]], requires_grad=True)
    positive = torch.tensor([[True, not has_pair, False, False]])
    valid = torch.tensor([[True, True, False, False]])
    expected = retrieval_loss(scores[:, :2], positive[:, :2])
    actual = retrieval_loss(scores, positive, valid)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[:, 2:].count_nonzero() == 0


def test_nonfinite_loss_aborts_before_backward_update_or_checkpoint(
    experiment,
    local_stages,
    monkeypatch,
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    original = training.compute_loss
    calls = []

    def fail(model, *args, **kwargs):
        loss, parts = original(model, *args, **kwargs)
        # Valid inference scores may remain finite while the loss fails; the
        # train loop must still abort instead of selecting a random checkpoint.
        parts["retrieval"] = loss.detach() * float("nan")
        calls.append(1)
        return loss * float("nan"), parts

    monkeypatch.setattr(training, "compute_loss", fail)
    with pytest.raises(FloatingPointError, match="before backward"):
        cli.main(["run", "--config", str(path), "--train", "--splits", "val"])
    root = Path(cfg["output"]["dir"])
    detail = json.loads((root / "nonfinite_batch.json").read_text())
    assert len(calls) == detail["epoch"] == detail["step"] == 1
    assert detail["sample_ids"] == ["train"]
    assert detail["losses"]["retrieval"] == "nan"
    assert not (root / "best.pt").exists()
    assert not (root / "last.pt").exists()
    assert not (root / "val/rankings.pt").exists()
