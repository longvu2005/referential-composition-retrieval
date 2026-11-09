"""CPU contracts for the residual context objective (context_residual_scale)."""

import json
from types import SimpleNamespace

import pytest
import torch

from rcr.proposed import train as training
from tests.test_train import (
    TextBackbone,
    TextEncoder,
    TinyCache,
    Tokenizer,
    _only_loss,
    _sample,
    tiny_training_run,  # noqa: F401  (fixture)
)

_MODEL_ARGS = dict(
    dim=8,
    identity_dim=6,
    state_dim=4,
    num_heads=2,
    fine_matching={"selection_threshold": 0.0001, "identity_threshold": -1.0},
)


def _build(fafa_pooling=None):
    from rcr.proposed.batch import build_batch

    return build_batch(
        [_sample("train", "q", "a")],
        [["a", "n", "va"]],
        TinyCache(),
        Tokenizer(),
        TextEncoder(TextBackbone(), dim=8),
        "cpu",
        supervision=SimpleNamespace(
            load_ids=lambda ids, width: [["p1"] * width for _ in ids]
        ),
        identity_image_ids=[{"a", "n"}],
        fafa_pooling=fafa_pooling,
    )


@pytest.fixture
def residual_batch():
    torch.manual_seed(23)
    model = training.RCRModel(**_MODEL_ARGS).eval()
    batch = _build(fafa_pooling="subject_mean")
    batch = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["extra_identity_persons"] = torch.randn(2, 8)
    batch["extra_identity_labels"] = torch.tensor([1, 1])
    return model, batch


def _context_settings(scale):
    settings = _only_loss("context")
    settings["context_residual_scale"] = scale
    return settings


def _fake_binding(monkeypatch, values):
    """Replace native FDA with a fixed offset per candidate (one query, 3)."""

    def binding(query, targets, **kwargs):
        return torch.tensor(values, dtype=torch.float32)[: targets.shape[0]]

    monkeypatch.setattr("rcr.proposed.losses.fafa_binding_scores", binding)


def _grad_norm(model):
    return torch.sqrt(
        sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
    ).item()


def test_constant_fda_offset_leaves_the_loss_unchanged(residual_batch, monkeypatch):
    model, batch = residual_batch
    base, _ = training.compute_loss(model, batch, (2, 2), **_context_settings(0.0))
    _fake_binding(monkeypatch, [3.0, 3.0, 3.0])
    shifted, parts = training.compute_loss(
        model, batch, (2, 2), **_context_settings(5.0)
    )
    torch.testing.assert_close(base, shifted, rtol=0, atol=1e-5)
    assert parts["context_fda_error"].item() == 1  # ties count as wrong


def test_context_only_learns_pairs_that_fda_ranks_wrongly(residual_batch, monkeypatch):
    model, batch = residual_batch
    _fake_binding(monkeypatch, [1.0, 0.0, 0.0])  # FDA already ranks correctly
    right, parts = training.compute_loss(
        model, batch, (2, 2), **_context_settings(50.0)
    )
    right.backward()
    assert right.item() < 1e-6
    assert _grad_norm(model) < 1e-6
    assert parts["context_fda_error"].item() == 0

    model.zero_grad(set_to_none=True)
    _fake_binding(monkeypatch, [0.0, 1.0, 0.0])  # FDA ranks the negative first
    wrong, parts = training.compute_loss(
        model, batch, (2, 2), **_context_settings(50.0)
    )
    wrong.backward()
    assert wrong.item() > 30
    assert _grad_norm(model) > 0
    assert parts["context_fda_error"].item() == 1


def test_residual_scale_requires_native_tokens_in_batch(monkeypatch):
    model = training.RCRModel(**_MODEL_ARGS).eval()
    batch = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in _build().items()
    }
    assert "fafa_query" not in batch
    with pytest.raises(ValueError, match="FAFA"):
        training.compute_loss(model, batch, (2, 2), **_context_settings(1.0))
    # Disabled scale (default) works on the same batch and without native tokens.
    total, parts = training.compute_loss(model, batch, (2, 2), **_context_settings(0.0))
    assert torch.isfinite(total)
    assert parts["context_fda_error"].item() == 0


def test_residual_inactive_when_context_loss_is_off(residual_batch):
    model, batch = residual_batch
    settings = _only_loss("identity")
    settings["context_residual_scale"] = 5.0
    total, parts = training.compute_loss(model, batch, (2, 2), **settings)
    assert torch.isfinite(total)
    assert parts["context_fda_error"].item() == 0


def test_batch_carries_native_tokens_only_when_requested():
    plain = _build()
    assert not {"fafa_query", "fafa_targets", "fafa_pooling"} & plain.keys()
    batch = _build(fafa_pooling="penalized")
    assert batch["fafa_pooling"] == "penalized"
    assert batch["fafa_query"].shape == (1, 1, batch["query_persons"].shape[1], 5)
    assert batch["fafa_targets"].shape == (
        1,
        3,
        batch["target_persons"].shape[2],
        3,
        5,
    )
    with pytest.raises(ValueError, match="pooling"):
        _build(fafa_pooling="nope")


def test_loss_settings_validate_residual_scale():
    assert training.loss_settings({})["context_residual_scale"] == 0
    assert (
        training.loss_settings({"context_residual_scale": 2.5})[
            "context_residual_scale"
        ]
        == 2.5
    )
    with pytest.raises(ValueError):
        training.loss_settings({"context_residual_scale": -1.0})
    with pytest.raises(ValueError, match="at least one"):
        training.loss_settings(
            {
                "grounding_weight": 0,
                "identity_weight": 0,
                "coarse_weight": 0,
                "context_weight": 0,
                "state_weight": 0,
                "context_residual_scale": 1.0,
            }
        )


def test_training_logs_fda_error_with_residual_objective(tiny_training_run):  # noqa: F811
    cfg, _, _, _ = tiny_training_run
    cfg["loss"]["context_residual_scale"] = 1.0
    checkpoint = training.train(cfg)
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert history
    for row in history:
        assert 0.0 <= row["train_context_fda_error"] <= 1.0
