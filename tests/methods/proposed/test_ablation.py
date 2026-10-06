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


def test_joint_grid_covers_both_axes_and_current_pair(experiment, tmp_path):
    _, path = experiment
    suite = yaml.safe_load(Path("configs/ablations/fine_coarse.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "joint"))
    experiments = ablation.expand_experiments(suite)
    pairs = {
        (
            e["config"]["retrieval"]["coarse_beta"],
            e["config"]["retrieval"]["fine_coarse_weight"],
        )
        for e in experiments
    }
    betas = {0.0, 0.1, 0.25, 0.4, 0.5, 1.0, 2.0, 4.0}
    weights = {0.0, 0.1, 0.25, 0.4, 0.5, 1.0, 2.0}
    assert len(experiments) == len(pairs) == 56
    assert pairs == {(beta, weight) for beta in betas for weight in weights}
    assert (0.4, 0.4) in pairs
    assert suite["select"] == {"group": "joint_coarse_fine", "metric": "full_map"}
    assert all(
        e["config"]["retrieval"]["rerank"]
        and e["config"]["retrieval"]["coarse_normalization"] == "zscore"
        and e["config"]["retrieval"]["coarse_mode"] == "identity_state"
        and e["config"]["retrieval"]["top_m"] == 500
        for e in experiments
    )


def test_selection_uses_val_group_and_first_pair_on_ties():
    rows = [
        dict(
            split="val",
            group="joint",
            full_map=0.5,
            coarse_beta=0.0,
            fine_coarse_weight=0.0,
        ),
        dict(
            split="val",
            group="joint",
            full_map=0.5,
            coarse_beta=0.4,
            fine_coarse_weight=1.0,
        ),
        dict(split="val", group="diagnostic", full_map=1.0),
        dict(split="test", group="joint", full_map=1.0),
    ]
    selection = dict(group="joint", metric="full_map")
    assert ablation._select_validation_result(rows, selection) is rows[0]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_selection_rejects_missing_or_nonfinite_metric(value):
    row = dict(split="val", group="joint")
    if value is not None:
        row["full_map"] = value
    with pytest.raises(ValueError, match="finite validation metrics"):
        ablation._select_validation_result(
            [row], dict(group="joint", metric="full_map")
        )


def test_joint_suite_selects_pair_on_val_reuses_encoder_and_freezes_test(
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
    suite = yaml.safe_load(Path("configs/ablations/fine_coarse.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "sweep"))
    suite_path = tmp_path / "joint.yaml"
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
        # Candidate recall prefers beta=0, while FINAL val Full-mAP prefers
        # (.5, 1). Test would prefer (0, 0); neither can guide selection.
        beta = cfg["retrieval"]["coarse_beta"]
        weight = cfg["retrieval"]["fine_coarse_weight"]
        best_pair = (0.5, 1.0) if cfg["split"] == "val" else (0.0, 0.0)
        result["overall"]["full_map"] = float((beta, weight) == best_pair)
        result["overall"]["candidate_recall_500"] = float(beta == 0)
        ablation.write_json(ablation.output_directory(cfg) / "metrics.json", result)
        return result

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel, "from_pretrained", count_loads
    )
    monkeypatch.setattr(ablation, "retrieve_experiments", record)
    monkeypatch.setattr(ablation, "evaluate_run", controlled_selection)
    rows = cli.main(["ablate", "--config", str(suite_path), "--splits", "test", "val"])
    assert len(loads) == 2  # One model load per split, not per grid pair.
    assert [len(call) for call in calls] == [56, 1]
    test_cfg = next(iter(calls[1].values()))
    assert test_cfg["split"] == "test" and test_cfg["retrieval"]["coarse_beta"] == 0.5
    assert test_cfg["retrieval"]["fine_coarse_weight"] == 1.0
    assert len(rows) == 57
    root = Path(suite["output_dir"])
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert selection["split"] == "val" and selection["value"] == 1
    assert selected["retrieval"]["coarse_beta"] == 0.5
    assert selected["retrieval"]["fine_coarse_weight"] == 1.0
    assert selection["metric"] == "full_map"
    assert selection["sweep"] == {
        "retrieval.coarse_beta": 0.5,
        "retrieval.fine_coarse_weight": 1.0,
    }
    assert selected["selected_checkpoint_sha256"] == selection["checkpoint_sha256"]
    assert selected["selected_validation_sha256"] == selection["validation_sha256"]
    with (root / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 57
    assert path.read_bytes() == original_config
    assert checkpoint.read_bytes() == original_checkpoint
    assert cache_index.read_bytes() == original_cache
    # The selected YAML is an ordinary method config usable without the suite.
    cli.main(["run", "--config", str(root / "selected.yaml"), "--splits", "test"])
    saved = json.loads((root / "selected/test/run.json").read_text())
    assert saved["retrieval"]["coarse_beta"] == 0.5
    assert saved["retrieval"]["fine_coarse_weight"] == 1.0
    assert saved["config"]["retrieval"]["rerank"] is True
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
    assert {row["fine_coarse_weight"] for row in retrieval_rows} == {1.0}
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
    # Coarse diagnostics inherit the same pair and never create another selection.
    coarse_suite = yaml.safe_load(Path("configs/ablations/coarse.yaml").read_text())
    coarse_suite.update(
        base_config=str(root / "selected.yaml"), output_dir=str(tmp_path / "coarse")
    )
    coarse_rows = ablation.run_ablation(coarse_suite, Path(cli.__file__))
    assert len(coarse_rows) == 4 and "select" not in coarse_suite
    assert {row["coarse_beta"] for row in coarse_rows} == {0.5}
    assert {row["fine_coarse_weight"] for row in coarse_rows} == {1.0}
    assert all(not row["rerank"] for row in coarse_rows)
    assert not (tmp_path / "coarse/selected.yaml").exists()
    # Changing val annotations invalidates selection even with the same checkpoint.
    samples_path = Path(cfg["data"]["final_dir"]) / "samples.jsonl"
    original_samples = samples_path.read_bytes()
    samples = [json.loads(line) for line in original_samples.splitlines()]
    next(s for s in samples if s["sample_id"] == "val")["final_change"] += " outdoors"
    samples_path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    with pytest.raises(ValueError, match="validation data changed"):
        inference.retrieve({**selected, "split": "test"})
    samples_path.write_bytes(original_samples)
    # A later checkpoint overwrite must not silently reuse a stale pair selection.
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


def test_test_failure_keeps_completed_joint_validation_selection(
    experiment, local_stages, tmp_path, monkeypatch
):
    _, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    root = tmp_path / "joint"
    suite = yaml.safe_load(Path("configs/ablations/fine_coarse.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(root))
    suite["experiments"][0]["sweep"] = {
        "retrieval.coarse_beta": [0.0, 0.4],
        "retrieval.fine_coarse_weight": [0.0, 0.4],
    }
    retrieve = ablation.retrieve_experiments

    def fail_test(configs):
        if next(iter(configs.values()))["split"] == "test":
            assert len(configs) == 1
            raise RuntimeError("test unavailable")
        return retrieve(configs)

    monkeypatch.setattr(ablation, "retrieve_experiments", fail_test)
    with pytest.raises(RuntimeError, match="test unavailable"):
        ablation.run_ablation(suite, Path(cli.__file__), splits=["val", "test"])
    selection = json.loads((root / "selection.json").read_text())
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selection["split"] == "val"
    assert selected["retrieval"]["coarse_beta"] == selection["coarse_beta"]
    assert (
        selected["retrieval"]["fine_coarse_weight"] == selection["fine_coarse_weight"]
    )
    with (root / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and {row["split"] for row in rows} == {"val"}


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
