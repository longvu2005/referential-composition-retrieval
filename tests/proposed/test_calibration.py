"""Validation calibration uses the real train/retrieve/evaluate flow."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from rcr.common.io import output_directory
from rcr.proposed import calibration, retrieve
from tests.proposed.test_runner import experiment, local_stages  # noqa: F401
from tools import run as cli


def test_calibration_loads_once_and_freezes_validation_choice(
    experiment,  # noqa: F811
    local_stages,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    original = path.read_bytes()
    settings = {
        "model_config": str(path),
        "output_dir": str(tmp_path / "calibration"),
        "coarse_beta": [0.0, 0.4],
        "fine_coarse_weight": [0.0, 0.4],
    }
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(yaml.safe_dump(settings))
    loads, splits = [], []
    load = sys.modules["transformers"].AutoModel.from_pretrained
    evaluate = calibration.evaluate_run

    def counted(*args, **kwargs):
        loads.append(1)
        return load(*args, **kwargs)

    def controlled(run_cfg, **kwargs):
        splits.append(run_cfg["split"])
        metrics = evaluate(run_cfg, **kwargs)
        metrics["overall"]["full_map"] = float(
            run_cfg["retrieval"]["coarse_beta"] == 0.4
            and run_cfg["retrieval"]["fine_coarse_weight"] == 0.4
        )
        calibration.write_json(output_directory(run_cfg) / "metrics.json", metrics)
        return metrics

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel, "from_pretrained", counted
    )
    monkeypatch.setattr(calibration, "evaluate_run", controlled)
    rows = cli.main(["calibrate", "--config", str(calibration_path)])
    assert len(rows) == 4 and loads == [1] and splits == ["val"] * 4
    assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert path.read_bytes() == original
    root = Path(settings["output_dir"])
    selected_path = root / "selected.yaml"
    selected = yaml.safe_load(selected_path.read_text())
    assert selected["retrieval"]["coarse_beta"] == 0.4
    assert selected["retrieval"]["fine_coarse_weight"] == 0.4
    assert json.loads((root / "selection.json").read_text())["split"] == "val"
    selected_bytes = selected_path.read_bytes()
    cli.main(["run", "--config", str(selected_path), "--splits", "test"])
    assert (root / "selected/test/metrics.json").exists()
    assert selected_path.read_bytes() == selected_bytes
    checkpoint = Path(cfg["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed since val selection"):
        retrieve.retrieve(selected)


@pytest.mark.parametrize("values", [[], [-1], [float("nan")], [True], "0.4"])
def test_invalid_grid_fails_before_creating_outputs(tmp_path, values):
    root = tmp_path / "calibration"
    with pytest.raises(ValueError, match="calibration weights"):
        calibration.calibrate(
            {
                "coarse_beta": values,
                "fine_coarse_weight": [0.4],
                "output_dir": str(root),
                "model_config": "missing.yaml",
            }
        )
    assert not root.exists()


def test_test_split_cannot_calibrate():
    with pytest.raises(ValueError, match="validation split"):
        retrieve.retrieve_weight_grid({"split": "test"}, [(0.4, 0.4)])
    with pytest.raises(SystemExit):
        cli.main(
            ["calibrate", "--config", "configs/calibration.yaml", "--splits", "test"]
        )


@pytest.mark.parametrize("metric", [float("nan"), 0.5])
def test_nonfinite_metrics_do_not_select_and_ties_keep_grid_order(
    experiment,  # noqa: F811
    local_stages,  # noqa: F811
    tmp_path,
    monkeypatch,
    metric,
):
    _, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    settings = {
        "model_config": str(path),
        "output_dir": str(tmp_path / "calibration"),
        "coarse_beta": [0.0, 0.4],
        "fine_coarse_weight": [0.0, 0.4],
    }
    evaluate = calibration.evaluate_run

    def controlled(cfg, **kwargs):
        result = evaluate(cfg, **kwargs)
        result["overall"]["full_map"] = metric
        return result

    monkeypatch.setattr(calibration, "evaluate_run", controlled)
    selected = Path(settings["output_dir"]) / "selected.yaml"
    if metric != metric:
        with pytest.raises(ValueError, match="finite validation"):
            calibration.calibrate(settings)
        assert not selected.exists()
    else:
        calibration.calibrate(settings)
        cfg = yaml.safe_load(selected.read_text())
        assert cfg["retrieval"]["coarse_beta"] == 0
        assert cfg["retrieval"]["fine_coarse_weight"] == 0
