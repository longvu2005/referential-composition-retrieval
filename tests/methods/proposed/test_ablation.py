"""New representation/binding suites and joint validation calibration."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from rcr.methods.proposed import ablation, inference
from tests.methods.proposed.test_runner import experiment as experiment
from tests.methods.proposed.test_runner import local_stages as local_stages
from tools.methods import run as cli


def test_representation_suite_has_factorial_controls():
    suite = yaml.safe_load(Path("configs/ablations/representation.yaml").read_text())
    variants = ablation.expand_experiments(suite)
    assert {
        (
            v["config"]["person_encoder"]["backend"],
            v["config"]["model"]["representation"],
        )
        for v in variants
    } == {("dino", "shared"), ("fafa", "shared"), ("dino", "dual"), ("fafa", "dual")}
    assert len({v["config"]["checkpoint"] for v in variants}) == 4
    assert {v["config"]["train"]["seed"] for v in variants} == {0}
    assert all(
        v["config"]["model"]["binding_mode"] == "none"
        for v in variants
        if v["config"]["model"]["representation"] == "shared"
    )


def test_shared_and_dual_controls_train_and_retrieve(
    experiment, local_stages, tmp_path
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    suite = yaml.safe_load(Path("configs/ablations/representation.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "representation"))
    suite["experiments"] = [
        run for run in suite["experiments"] if run["name"].startswith("dino_")
    ]
    for run in suite["experiments"]:
        run["overrides"]["data.cache"] = cfg["data"]["cache"]
    rows = ablation.run_ablation(suite, Path(cli.__file__))
    assert {row["representation"] for row in rows} == {"shared", "dual"}
    assert {row["binding_mode"] for row in rows} == {"none", "both"}
    assert len({row["checkpoint"] for row in rows}) == 2


def test_joint_selection_reuses_work_and_test_cannot_select(
    experiment, local_stages, tmp_path, monkeypatch
):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    original = path.read_bytes()
    suite = yaml.safe_load(Path("configs/calibration/joint.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "calibration"))
    suite["experiments"][0]["sweep"] = {
        "retrieval.coarse_beta": [0.0, 0.4],
        "retrieval.fine_coarse_weight": [0.0, 0.4],
    }
    calls, loads = [], []
    retrieve, evaluate = ablation.retrieve_experiments, ablation.evaluate_run
    native_load = sys.modules["transformers"].AutoModel.from_pretrained

    def record(configs):
        calls.append(configs)
        return retrieve(configs)

    def load(*args, **kwargs):
        loads.append(1)
        return native_load(*args, **kwargs)

    def controlled(cfg, **kwargs):
        metrics = evaluate(cfg, **kwargs)
        preferred = (0.4, 0.4) if cfg["split"] == "val" else (0.0, 0.0)
        metrics["overall"]["full_map"] = float(
            (cfg["retrieval"]["coarse_beta"], cfg["retrieval"]["fine_coarse_weight"])
            == preferred
        )
        ablation.write_json(ablation.output_directory(cfg) / "metrics.json", metrics)
        return metrics

    monkeypatch.setattr(ablation, "retrieve_experiments", record)
    monkeypatch.setattr(ablation, "evaluate_run", controlled)
    monkeypatch.setattr(sys.modules["transformers"].AutoModel, "from_pretrained", load)
    rows = ablation.run_ablation(suite, Path(cli.__file__), splits=["val", "test"])
    assert len(rows) == 5 and len(loads) == 2
    assert [len(group) for group in calls] == [4, 1]
    root = tmp_path / "calibration"
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert (
        selected["retrieval"]["coarse_beta"]
        == selected["retrieval"]["fine_coarse_weight"]
        == 0.4
    )
    assert path.read_bytes() == original
    binding = yaml.safe_load(Path("configs/ablations/binding.yaml").read_text())
    binding.update(
        base_config=str(root / "selected.yaml"), output_dir=str(tmp_path / "binding")
    )
    before = len(loads)
    outputs = ablation.run_ablation(binding, Path(cli.__file__))
    assert len(outputs) == 4 and len(loads) == before + 1
    assert {row["binding_mode"] for row in outputs} == {
        "both",
        "identity",
        "semantic",
        "none",
    }
    assert len({row["coarse_full_map"] for row in outputs}) == 1
    assert len({row["checkpoint_sha256"] for row in outputs}) == 1
    assert {row["fine_coarse_weight"] for row in outputs} == {0.4}
    with pytest.raises(ValueError, match="inference on val"):
        ablation.run_ablation(suite, Path(cli.__file__), splits=["test"])
    # A calibrated config remains an ordinary method config.
    cli.main(["run", "--config", str(root / "selected.yaml"), "--splits", "test"])
    checkpoint = Path(cfg["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since val selection"):
        inference.retrieve(selected)


def test_training_binding_variants_retrain_independently(
    experiment, local_stages, tmp_path
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    suite = yaml.safe_load(Path("configs/ablations/binding_train.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "binding_train"))
    rows = ablation.run_ablation(suite, Path(cli.__file__))
    runs = [run for stage, run in local_stages if stage == "train"]
    assert len(rows) == len(runs) == 4
    assert {run["model"]["binding_mode"] for run in runs} == {
        "both",
        "identity",
        "semantic",
        "none",
    }
    assert all(
        run["train"] == cfg["train"] and run["data"] == cfg["data"] for run in runs
    )
    assert len({run["checkpoint"] for run in runs}) == 4


def test_sweeps_are_explicit_and_do_not_mutate_base(experiment, tmp_path):
    _, path = experiment
    original = path.read_bytes()
    suite = dict(
        base_config=str(path),
        stage="train",
        output_dir=str(tmp_path),
        experiments=[
            dict(
                name="paired",
                sweep={"train.seed": [0, 1], "model.binding_mode": ["both", "none"]},
            )
        ],
    )
    before = copy.deepcopy(suite)
    assert len(ablation.expand_experiments(suite)) == 4
    assert suite == before and path.read_bytes() == original
    suite["experiments"][0]["overrides"] = {"data.final_dir": "different"}
    with pytest.raises(ValueError, match="share benchmark"):
        ablation.expand_experiments(suite)


def test_failed_test_keeps_validation_selection(
    experiment, local_stages, tmp_path, monkeypatch
):
    _, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    suite = dict(
        base_config=str(path),
        stage="inference",
        output_dir=str(tmp_path / "joint"),
        experiments=[dict(name="one")],
        select=dict(metric="full_map"),
    )
    retrieve = ablation.retrieve_experiments

    def fail_test(configs):
        if next(iter(configs.values()))["split"] == "test":
            raise RuntimeError("test unavailable")
        return retrieve(configs)

    monkeypatch.setattr(ablation, "retrieve_experiments", fail_test)
    with pytest.raises(RuntimeError, match="test unavailable"):
        ablation.run_ablation(suite, Path(cli.__file__), splits=["val", "test"])
    root = Path(suite["output_dir"])
    assert (root / "selected.yaml").exists()
    assert json.loads((root / "selection.json").read_text())["split"] == "val"
    assert (root / "summary.csv").exists()
