"""Slot-conditioned evidence must preserve the baseline and resume trajectory."""

import copy
import json
import math
import random
import signal
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rcr.common.config import load_config, override_config
from rcr.common.runtime import derived_generator
from rcr.proposed import calibration, retrieve
from rcr.proposed import train as training
from rcr.proposed.diagnostics import slot_query_diagnostics
from rcr.proposed.nn.binding import EvidenceBinding, person_attention_query
from rcr.proposed.nn.encoders import QueryTextCache
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.resume import capture_rng
from tests.helpers import Tokenizer, _only_loss, _sample


def _model(slot_query, *, dropout=0.0):
    torch.manual_seed(23)
    return RCRModel(
        dim=8,
        identity_dim=6,
        state_dim=4,
        num_heads=2,
        dropout=dropout,
        fine_matching={"selection_threshold": 0.0001, "identity_threshold": -1.0},
        slot_query=slot_query,
    ).eval()


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _equal(left, right)
    else:
        assert actual == expected


def _encode(model, batch):
    return model.encode_query(
        *(batch[name] for name in ("query_scene", "query_persons", "query_boxes")),
        batch["selections"],
        batch["change"],
        batch["subject_pos"],
        (2, 2),
        selection_mask=batch["selection_mask"],
        change_mask=batch["change_mask"],
        query_person_mask=batch["query_person_mask"],
        subject_token_mask=batch["subject_token_mask"],
        subject_ids=batch["subject_ids"],
        fafa_query=batch["fafa_query"],
    )


def test_zero_slot_gate_preserves_initialization_context_loss_and_query(
    objective_batch,
):
    _, batch = objective_batch
    baseline = _model(False)
    rng = torch.get_rng_state()
    enabled = _model(True)
    _equal(torch.get_rng_state(), rng)
    assert not hasattr(baseline, "slot_gate")
    assert "slot_gate" not in baseline.state_dict()
    state = {
        key: value for key, value in enabled.state_dict().items() if key != "slot_gate"
    }
    _equal(state, baseline.state_dict())
    baseline.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in enabled.parameters()) == (
        sum(p.numel() for p in baseline.parameters()) + 1
    )
    assert enabled.slot_gate.shape == () and enabled.slot_gate.item() == 0
    with torch.no_grad():
        queries = [_encode(model, batch) for model in (baseline, enabled)]
        for field in fields(queries[0]):
            _equal(getattr(queries[0], field.name), getattr(queries[1], field.name))
        results = []
        for model, query in zip((baseline, enabled), queries, strict=True):
            results.append(
                model.score_context(
                    query.repeat_candidates(3),
                    *(
                        batch[name].flatten(0, 1)
                        for name in ("target_scene", "target_persons", "target_boxes")
                    ),
                    (2, 2),
                    batch["target_mask"].flatten(0, 1),
                )
            )
        _equal(results[0][0], results[1][0])
        _equal(results[0][1].indices, results[1][1].indices)
        losses = [
            training.compute_loss(model, batch, (2, 2), **training.loss_settings({}))
            for model in (baseline, enabled)
        ]
        _equal(losses[0], losses[1])
        enabled.slot_gate.fill_(0.8)
        after = _encode(enabled, batch)
        for field in fields(after):
            _equal(getattr(after, field.name), getattr(queries[0], field.name))


@pytest.mark.parametrize("gate", [-0.4, 0.0, 0.7, 2.0])
def test_slot_attention_query_preserves_norm_and_zero_tokens(gate):
    generator = torch.Generator().manual_seed(7)
    persons = torch.randn(2, 3, 8, generator=generator)
    slots = torch.randn(2, 3, 8, generator=generator)
    persons[0, 1] = 0
    query = person_attention_query(persons, slots, torch.tensor(gate))
    assert torch.isfinite(query).all()
    assert query[0, 1].count_nonzero() == 0
    torch.testing.assert_close(query.norm(dim=-1), persons.norm(dim=-1))
    if gate == 0:
        _equal(query, persons)


def test_slot_gate_has_gradient_on_first_context_step(objective_batch):
    _, batch = objective_batch
    model = _model(True)
    loss, parts = training.compute_loss(model, batch, (2, 2), **_only_loss("context"))
    assert parts["context_active_fraction"] == 1
    loss.backward()
    assert model.slot_gate.item() == 0
    assert model.slot_gate.grad.isfinite() and model.slot_gate.grad.abs() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.composition.parameters()
    )


def test_binding_changes_only_person_attention_query_and_keeps_slot_gradients():
    torch.manual_seed(8)
    binding = EvidenceBinding(8, 2).eval()
    scene = torch.randn(1, 4, 8)
    persons = torch.randn(1, 2, 8)
    boxes = torch.tensor([[[0.0, 0.0, 0.4, 0.8], [0.5, 0.0, 1.0, 1.0]]])
    reference = torch.randn(1, 3, 8)
    slots = torch.randn_like(persons).requires_grad_()
    gate = torch.tensor(0.6, requires_grad=True)
    observed, residuals = [], []

    def attention(_module, args, kwargs):
        observed.append((args, kwargs))

    def residual(_module, args):
        residuals.append(args[0])

    with (
        binding.person_attn.register_forward_pre_hook(attention, with_kwargs=True),
        binding.norm_p.register_forward_pre_hook(residual),
    ):
        output = binding(
            scene, persons, boxes, reference, (2, 2), person_slots=slots, slot_gate=gate
        )
        baseline = binding(scene, persons, boxes, reference, (2, 2))
    args, kwargs = observed[0]
    assert kwargs["need_weights"] is False
    _equal(args[0], person_attention_query(persons, slots, gate))
    _equal(args[1], observed[1][0][1])
    _equal(args[2], observed[1][0][2])
    update, _ = binding.person_attn(*args, **kwargs)
    _equal(residuals[0], persons + update)
    assert not torch.equal(output, baseline)
    output[..., 0].sum().backward()
    assert slots.grad.isfinite().all() and slots.grad.abs().sum() > 0


@pytest.mark.parametrize("mask_first", [False, True])
def test_score_context_masks_unmatched_and_inactive_slots(objective_batch, mask_first):
    _, batch = objective_batch
    model = _model(True)
    query = _encode(model, batch)
    target_identity = model.encode_identity(batch["target_persons"][:, 0])
    # Two physical query people with disjoint Subject owners; one target is absent.
    query = replace(
        query,
        identity=torch.cat((target_identity, -target_identity), dim=1),
        membership=torch.tensor([[[0.9, 0.0], [0.0, 0.8]]]),
        subjects=torch.ones(1, 2, dtype=torch.bool),
        reference=torch.randn(1, 5, 8, requires_grad=True),
        mask=torch.tensor([[True, not mask_first, True, True, False]]),
        prior=torch.zeros(1, 5),
    )
    captured = []

    def observe(_module, args, kwargs):
        captured.append(kwargs["person_slots"])

    with model.grounding.binding.register_forward_pre_hook(observe, with_kwargs=True):
        _, matching = model.score_context(
            query,
            batch["target_scene"][:, 0],
            batch["target_persons"][:, 0],
            batch["target_boxes"][:, 0],
            (2, 2),
            batch["target_mask"][:, 0],
        )
    matched = ((matching.membership > 0) & matching.matched[:, None]).flatten(1)
    valid = matched & query.mask[:, 1:]
    assert matched.sum() == 1
    assert valid.sum() == (not mask_first)
    _equal(captured[0], query.reference[:, 1:].masked_fill(~valid[..., None], 0))
    assert captured[0].requires_grad


def test_slot_optimizer_group_preserves_model_text_order(tiny_ranker):
    _, _, text, _, _ = tiny_ranker
    settings = {"lr": 2e-4, "text_lr": 3e-5, "slot_gate_lr": 7e-3, "weight_decay": 0.2}
    for flag in (False, True):
        model = _model(flag)
        optimizer = training.build_optimizer(model, text, settings)
        assert len(optimizer.param_groups) == 2 + flag
        gate = model.slot_gate if flag else None
        expected = [p for p in model.parameters() if p.requires_grad and p is not gate]
        assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
            id(p) for p in expected
        ]
        assert [id(p) for p in optimizer.param_groups[1]["params"]] == [
            id(p) for p in text.proj.parameters()
        ]
        if flag:
            assert all(p is not gate for p in optimizer.param_groups[0]["params"])
            assert optimizer.param_groups[2]["params"] == [gate]
            assert optimizer.param_groups[2]["weight_decay"] == 0
            assert optimizer.param_groups[2]["lr"] == settings["slot_gate_lr"]


def test_slot_warmup_matches_baseline_and_cosine_covers_three_groups(
    tiny_training_run, monkeypatch
):
    cfg, _, wandb, _ = tiny_training_run
    cfg["train"].update(epochs=3, context_warmup_epochs=2)
    cfg["evaluation"]["enabled"] = False
    cfg["model"]["fine_matching"].update(
        selection_threshold=0.0001, identity_threshold=-1.0
    )
    cfg["optimizer"]["slot_gate_lr"] = 0.007
    rates = []
    build = training.build_optimizer

    def observe(model, text, settings):
        optimizer = build(model, text, settings)
        if model.slot_query:
            optimizer.register_step_pre_hook(
                lambda opt, _args, _kwargs: rates.append(
                    [group["lr"] for group in opt.param_groups]
                )
            )
        return optimizer

    monkeypatch.setattr(training, "build_optimizer", observe)
    histories, states = [], []
    for flag in (False, True):
        run = copy.deepcopy(cfg)
        run["model"]["slot_query"] = flag
        run["output"]["dir"] += f"-{flag}"
        checkpoint = training.train(run)
        states.append(torch.load(checkpoint, weights_only=True))
        histories.append(
            [
                json.loads(line)
                for line in (checkpoint.parent / "history.jsonl")
                .read_text()
                .splitlines()
            ]
        )
    for baseline, enabled in zip(*[history[:2] for history in histories], strict=True):
        _equal(
            baseline,
            {
                key: value
                for key, value in enabled.items()
                if not key.startswith("slot_")
            },
        )
    assert all(not any(key.startswith("slot_") for key in row) for row in histories[0])
    base_rates = [cfg["optimizer"][key] for key in ("lr", "text_lr", "slot_gate_lr")]
    for epoch, actual in enumerate(rates):
        floor = cfg["optimizer"]["min_lr_ratio"]
        factor = floor + (1 - floor) * (1 + math.cos(math.pi * epoch / 2)) / 2
        assert actual == pytest.approx([rate * factor for rate in base_rates])
    assert len(rates) == 3
    for row in histories[1]:
        for key in ("slot_gate", "slot_attn_kl", "slot_energy_ratio", "slot_gate_lr"):
            assert math.isfinite(row[key])
            assert any(log.get(f"epoch/{key}") == row[key] for log in wandb.logs)
    assert len(states[0]["training_state"]["optimizer"]["param_groups"]) == 2
    assert len(states[1]["training_state"]["optimizer"]["param_groups"]) == 3


def test_slot_resume_matches_uninterrupted_training(tiny_training_run, monkeypatch):
    cfg, _, _, _ = tiny_training_run
    cfg["model"]["slot_query"] = True
    cfg["model"]["fine_matching"].update(
        selection_threshold=0.0001, identity_threshold=-1.0
    )
    cfg["train"].update(epochs=3, context_warmup_epochs=1)
    cfg["evaluation"]["calibration"] = {"fine_coarse_weights": [0.15]}
    data = training.load_rcr_data(None)
    for key in ("train2", "train3"):
        row = _sample(key, "q", "a")
        data.samples.append(row)
        data.samples_by_id[key] = row
        data.splits["train"].append(key)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["output"]["dir"] += "-reference"
    reference_path = training.train(reference_cfg)
    reference = torch.load(reference_path.parent / "last.pt", weights_only=True)
    compute = training.compute_loss
    steps = 0

    def interrupt(*args, **kwargs):
        nonlocal steps
        result = compute(*args, **kwargs)
        steps += 1
        if steps == 5:
            signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as paused:
        paused.setattr(training, "compute_loss", interrupt)
        training.train(cfg)
    output = Path(cfg["output"]["dir"])
    pending = torch.load(output / "last.pt", weights_only=True)
    assert pending["training_state"]["progress"]["global_step"] == 5
    assert pending["model"]["slot_gate"].abs() > 0
    for key, value in (("slot_query", False), ("slot_gate_lr", 0.02)):
        changed = copy.deepcopy(cfg)
        changed["model" if key == "slot_query" else "optimizer"][key] = value
        with pytest.raises(ValueError, match="resume checkpoint differs in config"):
            training.train(changed, resume="auto")
    resumed_path = training.train(cfg, resume="auto")
    resumed = torch.load(output / "last.pt", weights_only=True)
    for key in (
        "model",
        "text_encoder",
        "best_epoch",
        "best_score",
        "state_supervised_pairs",
    ):
        _equal(resumed[key], reference[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(resumed["training_state"][key], reference["training_state"][key])
    _equal(
        torch.load(resumed_path, weights_only=True)["model"],
        torch.load(reference_path, weights_only=True)["model"],
    )
    assert (output / "history.jsonl").read_text() == (
        reference_path.parent / "history.jsonl"
    ).read_text()


@pytest.mark.parametrize("failure", [False, True])
def test_slot_diagnostics_restore_rng_modes_and_gradients(
    tiny_ranker, monkeypatch, failure
):
    cache, _, text, samples, _ = tiny_ranker
    model = _model(True, dropout=0.4).train()
    model.fine_matching["background_logit"] = -3.5
    text.train()
    text_cache = QueryTextCache(Tokenizer())
    streams = [
        derived_generator(17, name) for name in ("sampling", "identity", "hard_pool")
    ]
    states = [generator.get_state() for generator in streams]
    load_groups = cache.load_groups

    def noisy_load(*args):
        random.random()
        np.random.random()
        torch.rand(4)
        assert not model.training and not text.training and not torch.is_grad_enabled()
        if failure:
            raise RuntimeError("diagnostic failure")
        return load_groups(*args)

    monkeypatch.setattr(cache, "load_groups", noisy_load)
    model.slot_gate.grad = torch.tensor(0.7)
    rng = capture_rng()
    if failure:
        with pytest.raises(RuntimeError, match="diagnostic failure"):
            slot_query_diagnostics(model, text, cache, text_cache, samples, "cpu")
    else:
        metrics = slot_query_diagnostics(model, text, cache, text_cache, samples, "cpu")
        assert metrics == {
            "slot_gate": 0.0,
            "slot_attn_kl": 0.0,
            "slot_energy_ratio": 0.0,
        }
        with torch.no_grad():
            model.slot_gate.fill_(0.8)
        metrics = slot_query_diagnostics(model, text, cache, text_cache, samples, "cpu")
        assert metrics["slot_attn_kl"] > 0 and metrics["slot_energy_ratio"] > 0
    _equal(capture_rng(), rng)
    assert model.training and text.training
    _equal(model.slot_gate.grad, torch.tensor(0.7))
    for generator, state in zip(streams, states, strict=True):
        _equal(generator.get_state(), state)
    assert not model.grounding.binding._forward_pre_hooks
    assert not model.grounding.binding.person_attn._forward_pre_hooks
    baseline = _model(False).train()
    assert slot_query_diagnostics(baseline, text, None, None, [], "cpu") == {}


@pytest.mark.parametrize("checkpoint_flag", [True, False, None])
def test_retrieval_and_calibration_restore_checkpoint_slot_flag(
    saved_pipeline, tmp_path, checkpoint_flag
):
    cfg, _, _ = saved_pipeline
    path = Path(cfg["checkpoint"])
    checkpoint = torch.load(path, weights_only=True)
    if checkpoint_flag is None:
        checkpoint["config"]["model"].pop("slot_query")
    else:
        checkpoint["config"]["model"]["slot_query"] = checkpoint_flag
    if not checkpoint_flag:
        checkpoint["model"].pop("slot_gate")
    torch.save(checkpoint, path)
    cfg["model"]["slot_query"] = not checkpoint_flag
    output = retrieve.retrieve(cfg)
    assert output["provenance"]["config"]["model"]["slot_query"] is bool(
        checkpoint_flag
    )
    root = tmp_path / "calibrated"
    calibration.calibrate(
        {
            "rankings": [str(Path(cfg["output"]["dir"]) / "val/rankings.pt")],
            "output_dir": str(root),
        }
    )
    selected = load_config(root / "selected.yaml")
    assert selected["model"]["slot_query"] is bool(checkpoint_flag)


def test_slot_config_keys_support_overrides_and_legacy_defaults():
    cfg = load_config("configs/proposed.yaml")
    assert cfg["model"]["slot_query"] is True
    assert cfg["optimizer"]["slot_gate_lr"] == 1e-3
    cfg = override_config(
        cfg, {"model.slot_query": False, "optimizer.slot_gate_lr": 0.005}
    )
    assert cfg["model"]["slot_query"] is False
    cfg["model"].pop("slot_query")
    cfg["optimizer"].pop("slot_gate_lr")
    resolved = training._training_config(cfg)
    assert resolved["model"]["slot_query"] is False
    assert resolved["optimizer"]["slot_gate_lr"] == 1e-3
