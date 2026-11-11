"""Seeding contracts: one seed gives one run, neighbouring seeds stay independent."""

import os
import random

import numpy as np
import pytest
import torch

from rcr.common.config import validate_training_schedule
from rcr.common.runtime import derive_seed, derived_generator, seed_everything


def test_derived_seeds_are_pinned_to_the_documented_hash():
    # Changing this derivation changes every sampled batch of every future run.
    assert derive_seed(0, "sampling", 1) == 6571336432712843325
    assert derive_seed(0, "identity", 1) == 5332491773792835603
    assert derive_seed(1, "sampling", 1) == 5886473527773478530


def test_derived_seeds_never_repeat_across_seeds_streams_or_epochs():
    seeds = {
        derive_seed(seed, stream, epoch)
        for seed in range(8)
        for stream in ("sampling", "identity")
        for epoch in range(1, 41)
    }
    assert len(seeds) == 8 * 2 * 40
    assert all(0 <= value < 2**63 for value in seeds)


def test_neighbouring_run_seeds_do_not_replay_each_others_epochs():
    # The former ``seed + epoch`` offsets made seed 1 epoch 1 equal seed 0 epoch 2.
    first = torch.randperm(1000, generator=derived_generator(0, "sampling", 2))
    second = torch.randperm(1000, generator=derived_generator(1, "sampling", 1))
    assert not torch.equal(first, second)


def test_derived_generator_repeats_exactly_and_streams_differ():
    def draw(*key):
        return torch.randperm(1000, generator=derived_generator(*key))

    assert torch.equal(draw(3, "sampling", 5), draw(3, "sampling", 5))
    assert not torch.equal(draw(3, "sampling", 5), draw(3, "identity", 5))
    assert not torch.equal(draw(3, "sampling", 5), draw(3, "sampling", 6))


def test_seed_everything_seeds_python_numpy_and_torch():
    def draw():
        return random.random(), float(np.random.rand()), float(torch.rand(()))

    seed_everything(7, deterministic=False)
    first = draw()
    seed_everything(7, deterministic=False)
    assert draw() == first
    seed_everything(8, deterministic=False)
    assert draw() != first


def test_seed_everything_accepts_seeds_wider_than_numpy_allows():
    seed_everything(2**40 + 3, deterministic=False)
    first = float(np.random.rand())
    seed_everything(2**40 + 3, deterministic=False)
    assert float(np.random.rand()) == first


def test_deterministic_mode_requests_reproducible_kernels(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    # Operations without a deterministic kernel must warn, never abort a run.
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_mode_keeps_an_explicit_cublas_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(0, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_disabling_deterministic_mode_leaves_backend_settings_alone():
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    seed_everything(0, deterministic=False)
    assert before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


@pytest.mark.parametrize("seed", [-1, 1.5, True, "0"])
def test_invalid_seed_is_rejected_before_any_write(seed):
    cfg = {"train": {"epochs": 2, "batch_size": 1, "seed": seed}}
    with pytest.raises(ValueError, match="train.seed"):
        validate_training_schedule(cfg)


def test_deterministic_flag_must_be_boolean():
    cfg = {
        "train": {"epochs": 2, "batch_size": 1, "seed": 0},
        "runtime": {"deterministic": "yes"},
    }
    with pytest.raises(ValueError, match="runtime.deterministic"):
        validate_training_schedule(cfg)
    cfg["runtime"]["deterministic"] = False
    validate_training_schedule(cfg)
