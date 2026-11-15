"""Interrupted training must preserve the uninterrupted numerical trajectory."""

import copy
import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rcr.common.io import sha256_file
from rcr.proposed import train as training
from rcr.proposed.resume import capture_rng, restore_rng, training_lock
from tests.helpers import _sample
from tools.run import main


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


class _Scaler:
    def __init__(self, *args, **kwargs):
        self.updates = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self.updates += 1

    def state_dict(self):
        return {"updates": self.updates, "scale": 65536.0}

    def load_state_dict(self, state):
        self.updates = state["updates"]


def _pause(cfg, monkeypatch):
    compute = training.compute_loss

    def stop(*args, **kwargs):
        result = compute(*args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(training, "compute_loss", stop)
        training.train(cfg)


def _configure(cfg, data):
    cfg["train"].update(epochs=4, context_warmup_epochs=1)
    cfg["train"]["sampling"].update(
        warmup_epochs=1,
        refresh_every_epochs=2,
        identity_fraction=1.0,
        hard_fraction=0.0,
        semantic_hard_fraction=0.5,
    )
    cfg["evaluation"]["calibration"] = {
        "coverage_weights": [0.05],
        "context_weights": [0.025],
        "fine_coarse_weights": [0.15],
    }
    cfg["checkpointing"] = {"every_steps": 2}
    cfg["wandb"]["log_every_steps"] = 1
    for key in ("train2", "train3"):
        row = _sample(key, "q", "a")
        data.samples.append(row)
        data.samples_by_id[key] = row
        data.splits["train"].append(key)


@pytest.mark.parametrize("fault", ["signal", "after_step", "validation", "selection"])
def test_resume_matches_uninterrupted_training(tiny_training_run, monkeypatch, fault):
    cfg, _, wandb_run, _ = tiny_training_run
    data = training.load_rcr_data(None)
    _configure(cfg, data)
    monkeypatch.setattr(torch.amp, "GradScaler", _Scaler, raising=False)
    compute, mine, retrieve = (
        training.compute_loss,
        training.mine_hard_negatives,
        training.retrieve_rankings,
    )
    observations = []
    mining = []
    retrievals = []

    def observe(*args, **kwargs):
        observations.append((random.random(), float(np.random.random())))
        return compute(*args, **kwargs)

    def observe_mining(*args, **kwargs):
        mining.append(True)
        return mine(*args, **kwargs)

    def observe_retrieval(*args, **kwargs):
        retrievals.append(True)
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(training, "compute_loss", observe)
    monkeypatch.setattr(training, "mine_hard_negatives", observe_mining)
    monkeypatch.setattr(training, "retrieve_rankings", observe_retrieval)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["output"]["dir"] += "-reference"
    reference = training.train(reference_cfg)
    reference_last = torch.load(reference.parent / "last.pt", weights_only=True)
    reference_best = torch.load(reference, weights_only=True)
    reference_history = (reference.parent / "history.jsonl").read_text()
    reference_observations = list(observations)
    reference_mining = len(mining)
    reference_retrievals = len(retrievals)
    observations.clear()
    mining.clear()
    retrievals.clear()
    original_log = wandb_run.log
    previous_handler = signal.getsignal(signal.SIGTERM)

    with monkeypatch.context() as interrupted:
        if fault == "signal":

            def stop(*args, **kwargs):
                result = observe(*args, **kwargs)
                if len(observations) == 5:
                    assert signal.getsignal(signal.SIGTERM) != previous_handler
                    signal.raise_signal(signal.SIGTERM)
                return result

            interrupted.setattr(training, "compute_loss", stop)
            training.train(cfg)
        else:
            if fault == "after_step":

                def fail_log(values):
                    if values.get("global_step") == 5:
                        raise RuntimeError("injected interruption")
                    original_log(values)

                interrupted.setattr(wandb_run, "log", fail_log)
            elif fault == "validation":

                def fail_validation(*args, **kwargs):
                    raise RuntimeError("injected interruption")

                interrupted.setattr(training, "_evaluate", fail_validation)
            else:
                save_selection = training.save_selection

                def fail_selection(*args, **kwargs):
                    save_selection(*args, **kwargs)
                    raise RuntimeError("injected interruption")

                interrupted.setattr(training, "save_selection", fail_selection)
            with pytest.raises(RuntimeError, match="injected interruption"):
                training.train(cfg)
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    output = Path(cfg["output"]["dir"])
    pending = torch.load(output / "last.pt", weights_only=True)
    state = pending["training_state"]
    saved_step = state["progress"]["global_step"]
    assert saved_step == (5 if fault == "signal" else 4 if fault == "after_step" else 6)
    if fault in ("validation", "selection"):
        assert state["progress"]["phase"] == "validation"
    observations[:] = observations[:saved_step]
    resumed = training.train(cfg, resume="auto")
    last = torch.load(output / "last.pt", weights_only=True)
    best = torch.load(resumed, weights_only=True)
    for key in (
        "model",
        "text_encoder",
        "best_epoch",
        "best_score",
        "state_supervised_pairs",
    ):
        _equal(last[key], reference_last[key])
        _equal(best[key], reference_best[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(last["training_state"][key], reference_last["training_state"][key])
    assert last["training_state"]["scaler"]["updates"] == 12
    assert (output / "history.jsonl").read_text() == reference_history
    assert observations == reference_observations
    assert len(mining) == reference_mining
    assert len(retrievals) == reference_retrievals
    assert not list((output / "evaluation").rglob("pending_validation.pt"))
    assert (
        last["training_state"]["progress"]["wandb_id"] == state["progress"]["wandb_id"]
    )
    assert json.loads((output / "training_status.json").read_text())["complete"]
    digest = sha256_file(output / "last.pt")
    assert training.train(cfg, resume="auto") == resumed
    assert sha256_file(output / "last.pt") == digest


def test_legacy_checkpoint_rejected_without_overwriting(tiny_training_run):
    cfg, _, _, _ = tiny_training_run
    output = Path(cfg["output"]["dir"])
    output.mkdir()
    path = output / "last.pt"
    torch.save({"epoch": 3, "model": {"weight": torch.ones(1)}}, path)
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="no complete optimizer/scaler/RNG"):
        training.train(cfg, resume="auto")
    assert sha256_file(path) == digest
    assert list(output.iterdir()) == [path]


def test_rng_state_roundtrip_uses_safe_checkpoint_values(tmp_path):
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(4))
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    restore_rng(torch.load(path, weights_only=True))
    actual = (random.random(), np.random.random(), torch.rand(4))
    _equal(actual, expected)


@pytest.mark.parametrize("change", ["config", "inputs", "source_sha256", "runtime"])
def test_incompatible_resume_leaves_run_untouched(
    tiny_training_run, monkeypatch, change
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"]["epochs"] = 2
    _pause(cfg, monkeypatch)
    output = Path(cfg["output"]["dir"])
    paths = [
        output / name
        for name in ("last.pt", "history.jsonl", "config.yaml", "training_status.json")
    ]
    digests = [sha256_file(path) for path in paths]
    signature = training.resume_signature

    def changed(*args):
        current = signature(*args)
        current[change] = "different"
        return current

    monkeypatch.setattr(training, "resume_signature", changed)
    with pytest.raises(ValueError, match=change):
        training.train(cfg, resume="auto")
    assert [sha256_file(path) for path in paths] == digests


def test_resume_accepts_changed_checkpoint_cadence(tiny_training_run, monkeypatch):
    cfg, _, _, _ = tiny_training_run
    cfg["train"]["epochs"] = 2
    _pause(cfg, monkeypatch)
    cfg.setdefault("checkpointing", {})["every_steps"] = 1
    checkpoint = training.train(cfg, resume="auto")
    assert checkpoint.name == "best.pt"
    last = torch.load(checkpoint.parent / "last.pt", weights_only=True)
    assert last["epoch"] == 2
    assert last["config"]["checkpointing"]["every_steps"] == 1


def test_training_lock_rejects_another_writer_and_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="another training process"):
        with training_lock(tmp_path):
            with training_lock(tmp_path):
                pytest.fail("acquired an active training lock")
    with training_lock(tmp_path):
        pass


def test_resume_survives_a_new_process_and_hash_seed(tmp_path):
    from rcr.common.config import load_config

    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.json"
    base.write_text(json.dumps(load_config(root / "configs/proposed.yaml")))
    worker = """
import copy
import json
import signal
import sys
from pathlib import Path
from pytest import MonkeyPatch
from tests import conftest
from tests.test_resume import _configure
from rcr.proposed import train as training

phase, directory, base = sys.argv[1:]
directory = Path(directory)
directory.mkdir(parents=True, exist_ok=True)
base = json.loads(Path(base).read_text())
with MonkeyPatch.context() as patch:
    patch.setattr(conftest, '_workflow_config', lambda: copy.deepcopy(base))
    cfg, _, _, _ = conftest.tiny_training_run.__wrapped__(directory, patch)
    _configure(cfg, training.load_rcr_data(None))
    cfg['wandb']['enabled'] = False
    if phase == 'pause':
        compute = training.compute_loss
        count = [0]
        def stop(*args, **kwargs):
            loss = compute(*args, **kwargs)
            count[0] += 1
            if count[0] == 5:
                signal.raise_signal(signal.SIGTERM)
            return loss
        patch.setattr(training, 'compute_loss', stop)
    training.train(cfg, resume='auto' if phase == 'resume' else None)
"""
    for phase, name, seed in (
        ("full", "reference", "11"),
        ("pause", "split", "23"),
        ("resume", "split", "37"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                worker,
                phase,
                str(tmp_path / name),
                str(base),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
                "PYTHONHASHSEED": seed,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        (tmp_path / f"{phase}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr[-4000:]
    reference = torch.load(tmp_path / "reference/run/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "split/run/last.pt", weights_only=True)
    for key in ("model", "text_encoder", "best_epoch", "best_score"):
        _equal(actual[key], reference[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(actual["training_state"][key], reference["training_state"][key])
    assert (tmp_path / "split/run/history.jsonl").read_text() == (
        tmp_path / "reference/run/history.jsonl"
    ).read_text()


@pytest.mark.parametrize(
    "resume", [[], ["--resume"], ["--resume", "runs/trial/last.pt"]]
)
def test_cli_dispatches_explicit_resume(monkeypatch, resume):
    calls = []
    monkeypatch.setattr(training, "train", lambda cfg, **kwargs: calls.append(kwargs))
    main(["train", "--config", "configs/proposed.yaml", *resume])
    assert calls == [
        {"resume": None if not resume else "auto" if len(resume) == 1 else resume[1]}
    ]
