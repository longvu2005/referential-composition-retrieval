from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.methods.train_proposed import (
    _fixed_subset,
    _gradient_norm,
    _validate_evaluation_config,
    _wandb_run,
)


class _Run:
    def __init__(self) -> None:
        self.metrics = []

    def define_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))


def _config() -> dict:
    return {
        "data": {"cache": "private/path"},
        "model": {"identity_dim": 16},
        "train": {"epochs": 2},
        "optimizer": {"lr": 1e-4},
        "loss": {"identity_weight": 0.1},
        "evaluation": {
            "enabled": True,
            "every_epochs": 2,
            "train_max_queries": 10,
            "top_m": 20,
            "fine_batch_size": 4,
            "identity_batch_size": 8,
            "candidate_ks": [5, 20],
        },
        "wandb": {
            "enabled": True,
            "project": "rcr-test",
            "name": "smoke",
            "mode": "disabled",
            "log_every_steps": 2,
        },
    }


def test_wandb_tracks_only_training_essentials(monkeypatch, tmp_path) -> None:
    captured = {}
    run = _Run()

    def init(**kwargs):
        captured.update(kwargs)
        return run

    monkeypatch.setitem(__import__("sys").modules, "wandb", SimpleNamespace(init=init))
    cache = SimpleNamespace(cache_id="cache-1", patch_hw=(14, 14))
    actual = _wandb_run(_config(), tmp_path, cache, 768, 100, 20)

    assert actual is run
    assert captured["project"] == "rcr-test"
    assert set(captured["config"]) == {
        "model",
        "train",
        "optimizer",
        "loss",
        "evaluation",
        "cache",
        "num_train_samples",
        "num_val_samples",
    }
    assert "data" not in captured["config"]
    assert captured["config"]["cache"]["patch_hw"] == [14, 14]
    assert len(run.metrics) == 6


def test_disabled_wandb_needs_no_dependency(tmp_path) -> None:
    cfg = _config()
    cfg["wandb"]["enabled"] = False
    assert _wandb_run(cfg, tmp_path, None, 768, 100, 20) is None


def test_gradient_norm_matches_all_modules() -> None:
    a = nn.Linear(2, 1, bias=False)
    b = nn.Linear(2, 1, bias=False)
    a.weight.grad = torch.tensor([[3.0, 4.0]])
    b.weight.grad = torch.tensor([[0.0, 12.0]])
    assert _gradient_norm((a, b)) == 13.0


def test_train_evaluation_subset_is_fixed_and_keeps_dataset_order() -> None:
    samples = [{"sample_id": str(index)} for index in range(20)]
    first = _fixed_subset(samples, 6, seed=7)
    second = _fixed_subset(samples, 6, seed=7)

    assert first == second
    selected = [int(sample["sample_id"]) for sample in first]
    assert selected == sorted(selected)
    assert len(selected) == 6


def test_evaluation_config_rejects_candidate_cutoff_beyond_shortlist() -> None:
    cfg = _config()["evaluation"]
    cfg["candidate_ks"] = [21]
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_evaluation_config(cfg)
