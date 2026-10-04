"""Ablation/sweep integration using real training, rankings and tiny local encoders."""

import copy
import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from rcr.methods.proposed import ablation, inference
from tests.methods.proposed.test_runner import experiment as experiment
from tests.methods.proposed.test_runner import local_stages as local_stages
from tools.methods import run as cli


def test_coarse_suite_selects_on_val_reuses_encoder_and_freezes_test(
    experiment, local_stages, tmp_path, monkeypatch
):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    original_config = path.read_bytes()
    checkpoint = Path(cfg["checkpoint"])
    original_checkpoint = checkpoint.read_bytes()
    cache_index = Path(cfg["data"]["cache"]) / "index.pt"
    original_cache = cache_index.read_bytes()
    suite = yaml.safe_load(Path("configs/ablations/coarse.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "sweep"))
    suite["select"]["metric"] = "candidate_recall_1"
    suite_path = tmp_path / "coarse.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    loads, calls = [], []
    load = sys.modules["transformers"].AutoModel.from_pretrained
    retrieve, evaluate = ablation.retrieve_experiments, ablation.evaluate_run

    def count_loads(*args, **kwargs):
        loads.append(1)
        return load(*args, **kwargs)

    def record(configs):
        calls.append(configs)
        return retrieve(configs)

    def controlled_selection(cfg, **kwargs):
        result = evaluate(cfg, **kwargs)
        # Val's unique winner is beta=.5. Test would prefer beta=0; it must
        # never be used to choose a setting or trigger a second search.
        beta = cfg["retrieval"]["coarse_beta"]
        result["overall"]["candidate_recall_1"] = float(
            beta == (0.5 if cfg["split"] == "val" else 0)
        )
        return result

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel, "from_pretrained", count_loads
    )
    monkeypatch.setattr(ablation, "retrieve_experiments", record)
    monkeypatch.setattr(ablation, "evaluate_run", controlled_selection)
    rows = cli.main(["ablate", "--config", str(suite_path), "--splits", "test", "val"])
    assert len(loads) == 2  # One model load per split, not per beta.
    assert [len(call) for call in calls] == [12, 1]
    test_cfg = next(iter(calls[1].values()))
    assert test_cfg["split"] == "test" and test_cfg["retrieval"]["coarse_beta"] == 0.5
    assert len(rows) == 13
    root = Path(suite["output_dir"])
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert selection["split"] == "val" and selection["value"] == 1
    assert selected["retrieval"]["coarse_beta"] == 0.5
    assert selected["selected_checkpoint_sha256"] == selection["checkpoint_sha256"]
    assert selected["selected_validation_sha256"] == selection["validation_sha256"]
    with (root / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 13
    assert path.read_bytes() == original_config
    assert checkpoint.read_bytes() == original_checkpoint
    assert cache_index.read_bytes() == original_cache
    # The selected YAML is an ordinary method config usable without the suite.
    cli.main(["run", "--config", str(root / "selected.yaml"), "--splits", "test"])
    saved = json.loads((root / "selected/test/run.json").read_text())
    assert saved["retrieval"]["coarse_beta"] == 0.5
    assert saved["config"]["retrieval"]["rerank"] is False
    # The retrieval suite consumes the selection, including mixed candidate K columns.
    retrieval_suite = yaml.safe_load(
        Path("configs/ablations/retrieval.yaml").read_text()
    )
    retrieval_suite.update(
        base_config=str(root / "selected.yaml"), output_dir=str(tmp_path / "retrieval")
    )
    retrieval_rows = ablation.run_ablation(retrieval_suite, Path(cli.__file__))
    assert len(retrieval_rows) == 5
    assert {row["coarse_beta"] for row in retrieval_rows} == {0.5}
    assert (
        len(
            {
                row["coarse_full_map"]
                for row in retrieval_rows
                if row["coarse_mode"] == "identity_state"
            }
        )
        == 1
    )
    with (tmp_path / "retrieval/summary.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["candidate_recall_1000"] == ""
    assert table[-1]["candidate_recall_1000"] != ""
    # Changing val annotations invalidates selection even with the same checkpoint.
    samples_path = Path(cfg["data"]["final_dir"]) / "samples.jsonl"
    original_samples = samples_path.read_bytes()
    samples = [json.loads(line) for line in original_samples.splitlines()]
    next(s for s in samples if s["sample_id"] == "val")["final_change"] += " outdoors"
    samples_path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    with pytest.raises(ValueError, match="validation data changed"):
        inference.retrieve({**selected, "split": "test"})
    samples_path.write_bytes(original_samples)
    # A later checkpoint overwrite must not silently reuse a stale beta selection.
    checkpoint.write_bytes(original_checkpoint + b"changed")
    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel,
        "from_pretrained",
        lambda *args: pytest.fail("loaded model before selection check"),
    )
    with pytest.raises(ValueError, match="changed since val selection"):
        inference.retrieve({**selected, "split": "test"})


def test_loss_suite_retrains_each_variant_with_shared_protocol(
    experiment, local_stages, tmp_path
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    cache_index = Path(cfg["data"]["cache"]) / "index.pt"
    cache_before = cache_index.read_bytes()
    suite = yaml.safe_load(Path("configs/ablations/loss.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "loss"))
    suite_path = tmp_path / "loss.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    rows = cli.main(["ablate", "--config", str(suite_path)])
    trained = [settings for stage, settings in local_stages if stage == "train"]
    assert len(trained) == len(rows) == 4
    assert {row["split"] for row in rows} == {"val"}
    assert {row["seed"] for row in rows} == {0}
    assert len({run["checkpoint"] for run in trained}) == 4
    for run in trained:
        checkpoint = torch.load(run["checkpoint"], weights_only=True)
        assert checkpoint["config"]["loss"] == run["loss"]
        assert checkpoint["best_epoch"] == 1
        assert run["train"] == cfg["train"]
        assert run["data"] == cfg["data"]
    assert cache_index.read_bytes() == cache_before


def test_future_cartesian_sweep_is_only_a_config_change(experiment, tmp_path):
    _, path = experiment
    before = path.read_bytes()
    suite = dict(
        base_config=str(path),
        stage="train",
        output_dir=str(tmp_path / "future"),
        experiments=[
            dict(
                name="temperature",
                overrides={"loss.state_weight": 0.5},
                sweep={"loss.state_temperature": [0.1, 0.2], "train.seed": [0, 1]},
            )
        ],
    )
    untouched = copy.deepcopy(suite)
    expanded = ablation.expand_experiments(suite)
    assert len({e["name"] for e in expanded}) == 4
    assert {
        (e["config"]["loss"]["state_temperature"], e["config"]["train"]["seed"])
        for e in expanded
    } == {(0.1, 0), (0.1, 1), (0.2, 0), (0.2, 1)}
    assert all(e["config"]["loss"]["state_weight"] == 0.5 for e in expanded)
    assert path.read_bytes() == before and suite == untouched
    suite["experiments"][0]["overrides"] = {"loss.typo": 0}
    with pytest.raises(KeyError, match="unknown config key"):
        ablation.expand_experiments(suite)


def test_selection_cannot_run_on_test_only(experiment, tmp_path, monkeypatch):
    _, path = experiment
    suite = dict(
        base_config=str(path),
        stage="inference",
        output_dir=str(tmp_path / "bad"),
        experiments=[dict(name="beta", overrides={})],
        select=dict(group="beta", metric="full_map"),
    )
    monkeypatch.setattr(
        ablation, "retrieve_experiments", lambda *args: pytest.fail("retrieved test")
    )
    with pytest.raises(ValueError, match="inference on val"):
        ablation.run_ablation(suite, Path(cli.__file__), splits=["test"])


def test_failed_inference_does_not_leave_previous_selection(
    experiment, tmp_path, monkeypatch
):
    _, path = experiment
    root = tmp_path / "failed"
    root.mkdir()
    for name in ("summary.csv", "selection.json", "selected.yaml"):
        (root / name).write_text("previous run")
    suite = dict(
        base_config=str(path),
        stage="inference",
        output_dir=str(root),
        experiments=[dict(name="beta", overrides={})],
        select=dict(group="beta", metric="full_map"),
    )

    def fail(*args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(ablation, "retrieve_experiments", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        ablation.run_ablation(suite, Path(cli.__file__))
    assert all(
        not (root / name).exists()
        for name in ("summary.csv", "selection.json", "selected.yaml")
    )
