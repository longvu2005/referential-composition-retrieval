"""Bounded context training retains losses, gradients and dropout semantics."""

import weakref
from copy import deepcopy

import pytest
import torch
from torch import nn

from rcr.proposed import losses
from rcr.proposed import train as training
from tests.test_train import _only_loss
from tests.test_train import objective_batch as objective_batch
from tests.test_train import tiny_training_run as tiny_training_run


def _expanded_batch(batch):
    rows = {}
    for key, value in batch.items():
        if value.ndim and value.shape[0] == 1:
            value = value.repeat(3, *([1] * (value.ndim - 1)))
        rows[key] = value.detach().clone().requires_grad_(value.requires_grad)
    rows["query_scene"][1] += 0.3
    rows["target_mask"][1, 1] = False
    rows["hard_identity_negative_mask"][:, 2] = True
    rows["change"].requires_grad_()
    rows["selections"].requires_grad_()
    return rows


def _backward(model, batch, size, settings, *, amp=False):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        total, parts = losses.compute_loss(
            model, batch, (2, 2), context_batch_size=size, **settings
        )
    total.backward()
    gradients = {
        key: None if p.grad is None else p.grad.clone()
        for key, p in model.named_parameters()
    }
    gradients.update(
        (f"batch.{key}", value.grad.clone())
        for key, value in batch.items()
        if value.requires_grad and value.grad is not None
    )
    return total.detach(), parts, gradients


def _assert_same(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    assert actual[1].keys() == expected[1].keys()
    for key in actual[1]:
        torch.testing.assert_close(
            actual[1][key], expected[1][key], rtol=2e-5, atol=2e-6
        )
    assert actual[2].keys() == expected[2].keys()
    for key, gradient in actual[2].items():
        other = expected[2][key]
        if gradient is None:
            assert other is None, key
        else:
            torch.testing.assert_close(gradient, other, rtol=3e-4, atol=2e-6, msg=key)


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_chunked_context_matches_full_loss_and_all_gradients(
    objective_batch, chunk_size
):
    model, batch = objective_batch
    settings = losses.loss_settings({"context_rejection_weight": 0.2})
    expected = _backward(deepcopy(model), _expanded_batch(batch), 64, settings)
    actual = _backward(deepcopy(model), _expanded_batch(batch), chunk_size, settings)
    _assert_same(actual, expected)
    assert actual[2]["batch.change"].abs().sum() > 0
    assert actual[2]["batch.selections"].abs().sum() > 0


@pytest.mark.parametrize("amp", [False, True])
def test_checkpoint_preserves_chunked_dropout_gradients_and_rng(
    objective_batch, monkeypatch, amp
):
    model, batch = objective_batch
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.1
    settings = _only_loss("context")
    torch.manual_seed(100)
    actual = _backward(deepcopy(model), _expanded_batch(batch), 4, settings, amp=amp)
    actual_rng = torch.rand(5)
    monkeypatch.setattr(
        losses, "checkpoint", lambda function, *args, **kwargs: function(*args)
    )
    torch.manual_seed(100)
    expected = _backward(deepcopy(model), _expanded_batch(batch), 4, settings, amp=amp)
    _assert_same(actual, expected)
    torch.testing.assert_close(actual_rng, torch.rand(5), rtol=0, atol=0)


def test_checkpoint_reduces_saved_context_activations(objective_batch):
    model, batch = objective_batch
    settings = _only_loss("context")

    def saved_bytes(size):
        storages = {}

        def pack(value):
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            total, _ = losses.compute_loss(
                model,
                _expanded_batch(batch),
                (2, 2),
                context_batch_size=size,
                **settings,
            )
        total.backward()
        model.zero_grad(set_to_none=True)
        return sum(storages.values())

    assert saved_bytes(4) < saved_bytes(64) / 2


def test_training_releases_inputs_before_next_batch_and_validation(
    tiny_training_run, monkeypatch
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"].update(epochs=2, context_batch_size=1)
    build, evaluate = training.build_batch, training._evaluate
    references = []

    def check_released():
        assert all(ref() is None for ref in references)

    def observe_batch(*args, **kwargs):
        check_released()
        batch = build(*args, **kwargs)
        references[:] = [weakref.ref(batch[key]) for key in ("target_scene", "change")]
        return batch

    def observe_validation(*args, **kwargs):
        check_released()
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "build_batch", observe_batch)
    monkeypatch.setattr(training, "_evaluate", observe_validation)
    training.train(cfg)
    check_released()
