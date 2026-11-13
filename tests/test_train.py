"""CPU workflow tests for train."""

import json

import pytest
import torch

from rcr.proposed import train as training
from tests.helpers import _has_gradient, _only_loss


@pytest.mark.parametrize("warmup", [0, 1])
def test_training_warmup_and_validation_selected_checkpoint(
    tiny_training_run, monkeypatch, warmup
):
    cfg, cache, wandb_run, _ = tiny_training_run
    cfg["train"].update(epochs=3, context_warmup_epochs=warmup)
    cfg["retrieval"]["context_weight"] = 0.23
    weights, native_tokens = [], []
    compute = training.compute_loss

    def observe(model, batch, *args, **settings):
        weights.append(settings)
        native_tokens.append("fafa_targets" in batch)
        return compute(model, batch, *args, **settings)

    monkeypatch.setattr(training, "compute_loss", observe)
    checkpoint = training.train(cfg)
    expected = [0.0 if epoch <= warmup else 1.0 for epoch in range(1, 4)]
    assert [row["context_weight"] for row in weights] == expected
    assert native_tokens == [bool(value) for value in expected]
    assert all(row["context_score_weight"] == 0.23 for row in weights)
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in history] == [
        "warmup" if value == 0 else "joint" for value in expected
    ]
    assert all("train_context_fda_error" in row for row in history)
    assert all("val_checkpoint_score" not in row for row in history[:warmup])
    best = max(
        (row for row in history if "val_checkpoint_score" in row),
        key=lambda row: row["val_checkpoint_score"],
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    last = torch.load(
        checkpoint.parent / "last.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint.name == "best.pt"
    assert saved["epoch"] == saved["best_epoch"] == best["epoch"]
    assert last["epoch"] == 3
    assert saved["config"]["retrieval"]["fine_score_mode"] == "bounded_residual"
    assert saved["config"]["retrieval"]["context_weight"] == 0.23
    assert saved["scene_cache_id"] == cache.scene_cache_id
    assert saved["detection_cache_id"] == cache.detection_cache_id
    assert saved["state_supervised_pairs"] > 0
    assert set(saved["text_encoder"]) == {"projection", "text_policy"}
    assert "optimizer" not in saved and "scaler" not in saved
    assert wandb_run.finished


@pytest.mark.parametrize(
    "component", ["grounding", "identity", "coarse", "context", "state"]
)
def test_each_objective_trains_its_intended_modules(objective_batch, component):
    model, batch = objective_batch
    total, parts = training.compute_loss(model, batch, (2, 2), **_only_loss(component))
    assert torch.isfinite(total)
    total.backward()
    assert _has_gradient(model.identity_head) == (component in ("identity", "coarse"))
    if component not in ("identity", "coarse"):
        assert all(p.grad is None for p in model.identity_head.parameters())
    assert _has_gradient(model.reasoner) == (component == "context")
    assert _has_gradient(model.composition) == (component == "context")
    assert _has_gradient(model.state_text_proj) == (component == "state")
    assert _has_gradient(model.state_image_proj) == (component == "state")
    assert model.binding_log_scale.grad is None
    assert not model.binding_log_scale.requires_grad
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if component == "context":
        assert parts["context_active_fraction"].item() == 1
        assert batch["target_persons"].grad[0, 2].count_nonzero() == 0
        assert batch["target_persons"].grad[0, :2].abs().sum() > 0
    if component == "state":
        assert batch["target_global"].grad[0, 2].count_nonzero() == 0
        assert batch["target_global"].grad[0, :2].abs().sum() > 0


def test_joint_objective_learns_on_a_fixed_supervised_batch(objective_batch):
    model, batch = objective_batch
    settings = training.loss_settings({})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        value, parts = training.compute_loss(model, batch, (2, 2), **settings)
        assert torch.isfinite(value)
        assert parts["context_active_fraction"].item() == 1
        losses.append(value.item())
        value.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.binding_scale.item() == 1.0


def test_training_enables_identity_crop_mining_after_warmup(
    tiny_training_run, monkeypatch
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"].update(epochs=2)
    cfg["retrieval"]["coarse_beta"] = 0  # This fixture has no same-ID state pair.
    cfg["train"]["sampling"].update(warmup_epochs=1, hard_fraction=0)
    cfg["train"]["identity_sampling"].update(max_crops=4, hard_negative_fraction=0.5)
    data = training.load_rcr_data(None)
    data.gt_head_boxes_by_image["n"][0]["identity_id"] = "p2"
    build, evaluate = training.build_batch, training._evaluate
    mining, backgrounds = [], []

    def observe_batch(*args, **kwargs):
        mining.append(kwargs["identity_sampling"] is not None)
        return build(*args, **kwargs)

    def observe_evaluate(*args, **kwargs):
        backgrounds.append(args[6].fine_matching["background_logit"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "build_batch", observe_batch)
    monkeypatch.setattr(training, "_evaluate", observe_evaluate)
    checkpoint = training.train(cfg)
    assert mining == [False, True]
    assert backgrounds == [-3.5, -3.5]
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["model"]["fine_matching"]["background_logit"] == -3.5
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["train_identity_hard_crops"] for row in history] == [
        0,
        1,
    ]
    for row in history:
        assert row["train_identity_supplemental_crops"] <= 4
        assert row["train_identity_supplemental_crops"] == (
            row["train_identity_random_crops"] + row["train_identity_hard_crops"]
        )
