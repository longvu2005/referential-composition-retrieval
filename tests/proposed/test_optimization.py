"""Real FP16 overflow and optimizer atomicity; CPU tests need no GPU/weights."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rcr.proposed.cache import build as builder
from rcr.proposed.optimization import NonfiniteStepError, train_step
from tests.proposed.test_text_and_pipeline import experiment as _experiment
from tests.proposed.test_text_and_pipeline import tiny_clip as _tiny_clip

experiment = _experiment
tiny_clip = _tiny_clip


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_real_fp16_overflow_retries_same_batch_once(device):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Dropout(0.25), nn.Linear(2, 1)).to(device).train()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    expected = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
    scaler = torch.amp.GradScaler(device)
    inputs = torch.ones(4, 2, device=device)
    modes = []
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None

    def loss_fn():
        modes.append(torch.is_autocast_enabled(device))
        return model(inputs).float().sum(), {}

    loss, parts = train_step(
        model, optimizer, scaler, loss_fn, device, amp=True, max_grad_norm=5
    )
    assert modes == [True, False]
    assert parts["fp32_retries"] == 1 and torch.isfinite(loss)
    assert scaler.get_scale() == 32768
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    reference(inputs).sum().backward()
    nn.utils.clip_grad_norm_(reference.parameters(), 5, error_if_nonfinite=True)
    expected.step()
    # Exact single FP32 update: failed AMP must not apply weight decay or moments.
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted)
        for key, value in optimizer.state[actual].items():
            torch.testing.assert_close(value, expected.state[wanted][key])


@pytest.mark.parametrize("amp", [True, False])
def test_finite_step_updates_once_without_retry(amp):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=1)
    loss, parts = train_step(
        model,
        optimizer,
        scaler,
        lambda: (model(torch.ones(1, 2)).float().sum(), {}),
        "cpu",
        amp=amp,
        max_grad_norm=5,
    )
    assert torch.isfinite(loss) and parts["fp32_retries"] == 0
    assert all(state["step"] == 1 for state in optimizer.state.values())
    assert scaler.get_scale() == (256 if amp else 128)


def test_forward_overflow_retries_without_scaler_initialization():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu")
    loss, parts = train_step(
        model,
        optimizer,
        scaler,
        lambda: (model(torch.tensor([[40000.0]])).float().sum(), {}),
        "cpu",
        amp=True,
        max_grad_norm=5,
    )
    assert loss == 80000 and parts["fp32_retries"] == 1
    assert all(state["step"] == 1 for state in optimizer.state.values())


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("failure", ["loss", "gradient", "norm"])
def test_persistent_nonfinite_does_not_mutate_optimizer(amp, failure):
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu")

    def loss_fn():
        weight = model.weight
        if failure == "loss":
            return weight.sum() * float("nan"), {}
        if failure == "gradient":
            return weight.sqrt().sum(), {}  # Finite loss, infinite derivative.
        return weight.sum() * 1e30, {}  # Finite elements, overflowing L2 norm.

    with pytest.raises(NonfiniteStepError) as error:
        train_step(model, optimizer, scaler, loss_fn, "cpu", amp=amp, max_grad_norm=5)
    attempts = error.value.details["attempts"]
    assert len(attempts) == (2 if amp else 1)
    assert attempts[-1]["precision"] == "float32"
    if failure == "gradient":
        assert attempts[-1]["parameters"] == ["weight"]
    torch.testing.assert_close(model.weight, before)
    assert not optimizer.state and model.weight.grad is None


def test_no_supervision_does_not_step_or_initialize_scaler():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    loss, parts = train_step(
        model,
        optimizer,
        torch.amp.GradScaler("cpu"),
        lambda: (torch.tensor(0.0), {}),
        "cpu",
        amp=True,
        max_grad_norm=5,
    )
    assert loss == 0 and parts["fp32_retries"] == 0 and not optimizer.state


def test_real_rcr_warmup_and_joint_training_recover_amp_overflow(
    experiment, monkeypatch
):
    cfg, _ = experiment
    builder.prepare_cache(cfg)
    training = importlib.import_module("rcr.proposed.train")
    # Exercise FP16 kernels and real GradScaler on CPU, including both RCR phases.
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**32)

    def cpu_amp_step(model, optimizer, ignored_scaler, loss_fn, device, **kwargs):
        return train_step(
            model,
            optimizer,
            scaler,
            loss_fn,
            device,
            **{**kwargs, "amp": True},
        )

    monkeypatch.setattr(training, "train_step", cpu_amp_step)
    best = training.train(cfg)
    assert best.name == "best.pt" and best.is_file()
    history = [
        json.loads(line)
        for line in (best.parent / "history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 3
    assert all(row["fp32_retries"] == 1 for row in history)
    checkpoint = torch.load(best, weights_only=True)
    assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
    assert not (best.parent / "nonfinite_batch.json").exists()


def test_training_records_bad_parameter_and_sample_without_checkpoint(
    experiment, monkeypatch
):
    cfg, _ = experiment
    builder.prepare_cache(cfg)
    training = importlib.import_module("rcr.proposed.train")

    def invalid_loss(model, *args, **kwargs):
        parameter = next(model.parameters())
        return (parameter - parameter.detach()).sqrt().sum(), {}

    monkeypatch.setattr(training, "compute_loss", invalid_loss)
    with pytest.raises(NonfiniteStepError):
        training.train(cfg)
    output = Path(cfg["output"]["dir"])
    report = json.loads((output / "nonfinite_batch.json").read_text())
    assert report["epoch"] == 1 and report["sample_ids"] == ["train"]
    assert report["attempts"][0]["parameters"] == ["identity_head.proj.weight"]
    assert not (output / "last.pt").exists() and not (output / "best.pt").exists()
