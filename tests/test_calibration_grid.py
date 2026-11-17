"""Cartesian coefficient calibration and exact full-gallery selection."""

import copy
import json
from itertools import product
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.config import load_config
from rcr.evaluation.evaluate import (
    evaluate_retrieval_output,
    selection_metrics,
)
from rcr.proposed import calibration, retrieve
from rcr.proposed import train as training
from rcr.proposed.contracts import CALIBRATED_RETRIEVAL_KEYS
from rcr.proposed.replay import replay_rankings
from tests.helpers import _add_gallery_image, _sample


def test_configured_grid_matches_official_replay_with_fixed_tail(
    saved_pipeline, monkeypatch
):
    cfg, data, _ = saved_pipeline
    grid = load_config("configs/proposed.yaml")["evaluation"]["calibration"]
    cfg["evaluation"]["calibration"] = grid
    cfg["retrieval"]["top_m"] = 3
    cache = retrieve.GalleryCache()
    for image in ("vx", "vy"):
        _add_gallery_image(cache, data, image, "val")
    data.samples_by_id["val"]["positive_image_ids"] = ["va", "vy"]
    for row in (_sample("val2", "vn", "vq"), _sample("val3", "va", "vn")):
        row["case_type"] = "GROUP" if row["sample_id"] == "val2" else "INDIVIDUAL"
        if row["sample_id"] == "val3":
            row["positive_image_ids"] = ["vn", "vx"]
        data.samples.append(row)
        data.samples_by_id[row["sample_id"]] = row
        data.splits["val"].append(row["sample_id"])
    samples = [data.samples_by_id[key] for key in data.splits["val"]]
    source = retrieve.retrieve(cfg)
    order = torch.tensor(
        [
            [i for i in range(len(source["gallery_ids"])) if i != query]
            for query in source["query_indices"].tolist()
        ],
        dtype=torch.int32,
    )
    source["coarse_rankings"] = source["rankings"] = order
    source["coarse_topm"] = order[:, :3]
    source["coarse_topm_scores"] = torch.tensor(
        [[2.0, 1.0, 0.0], [-torch.inf, -torch.inf, -torch.inf], [5.0, 3.0, -torch.inf]]
    )
    binding = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.002, -0.001, 0.003]])
    for mode in ("penalized", "matched_mean", "subject_mean"):
        source[f"binding_{mode}"] = binding
    source["fafa_binding"] = binding
    source["context"] = torch.tensor(
        [[0.4, 0.8, 0.5], [0.0, 0.0, 0.0], [2.0, -1.0, 0.5]]
    )
    source["coverage"] = torch.tensor(
        [[1.0, 0.2, 0.3], [0.5, 0.5, 0.5], [0.0, 1.0, 0.5]]
    )
    source = replay_rankings(source)
    evaluate_calls, replay_calls = [], []
    original_evaluate, original_replay = (
        calibration.evaluate_retrieval_output,
        calibration.replay_rankings,
    )

    def counted_evaluate(*args, **kwargs):
        evaluate_calls.append(True)
        return original_evaluate(*args, **kwargs)

    def counted_replay(*args, **kwargs):
        replay_calls.append(True)
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(calibration, "evaluate_retrieval_output", counted_evaluate)
    monkeypatch.setattr(calibration, "replay_rankings", counted_replay)
    selected = calibration.calibrate_rankings(source, data, samples, cfg)
    assert len(evaluate_calls) == 2 and len(replay_calls) == 1
    assert [
        tuple(row[key] for key in CALIBRATED_RETRIEVAL_KEYS) for row in selected.rows
    ] == list(product(*grid.values()))
    for row in selected.rows:
        direct = replay_rankings(
            source, **{key: row[key] for key in CALIBRATED_RETRIEVAL_KEYS}
        )
        result = evaluate_retrieval_output(
            data, samples, direct, cfg["candidate_ks"], split="val"
        )
        assert {
            key: row[key] for key in selection_metrics(result)
        } == selection_metrics(result)
        for case, metrics in result["by_case"].items():
            assert row[f"{case.lower()}_full_map"] == metrics["full_map"]
        assert torch.equal(direct["rankings"][:, 3:], source["coarse_rankings"][:, 3:])
    expected = max(selected.rows, key=lambda row: row["checkpoint_score"])
    assert selected.best == expected
    assert (
        selected.metrics["overall"]["checkpoint_score"] == expected["checkpoint_score"]
    )
    assert cfg["retrieval"]["context_weight"] == 0.10
    assert source["score_config"]["context_weight"] == 0.10


def test_calibrated_linear_coefficients_deploy_and_remain_frozen(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    original = copy.deepcopy(cfg["retrieval"])
    root = tmp_path / "linear-calibration"
    rows = calibration.calibrate(
        {
            "rankings": [str(Path(cfg["output"]["dir"]) / "val/rankings.pt")],
            "output_dir": str(root),
            "coverage_weights": [0.05, 0.12],
            "context_weights": [0.02, 0.06],
            "fine_coarse_weights": [0.10, 0.20],
        }
    )
    assert len(rows) == 8
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    winner = max(rows, key=lambda row: row["checkpoint_score"])
    for key in CALIBRATED_RETRIEVAL_KEYS:
        assert (
            selected["retrieval"][key]
            == selected["selected_retrieval"][key]
            == winner[key]
        )
    assert selected["retrieval"]["coverage_weight"] != original["coverage_weight"]
    assert selected["retrieval"]["context_weight"] != original["context_weight"]
    assert json.loads((root / "selection.json").read_text())["num_candidates"] == 8
    expected = replay_rankings(
        source, **{key: winner[key] for key in CALIBRATED_RETRIEVAL_KEYS}
    )
    deployed = retrieve.retrieve(selected)
    assert torch.equal(deployed["rankings"], expected["rankings"])
    assert "fine_score_ablation" not in deployed["provenance"]
    tampered = copy.deepcopy(selected)
    tampered["retrieval"]["coverage_weight"] += 0.001
    with pytest.raises(ValueError, match="settings changed since val selection"):
        retrieve.retrieve(tampered)
    assert cfg["retrieval"] == original


def test_three_coefficient_checkpoint_does_not_change_training(
    tiny_training_run, monkeypatch
):
    cfg, _, wandb_run, _ = tiny_training_run
    cfg["evaluation"]["calibration"] = {
        "coverage_weights": [0.12],
        "context_weights": [0.02],
        "fine_coarse_weights": [0.1375],
    }
    original = copy.deepcopy(cfg["retrieval"])
    observed = []
    compute = training.compute_loss

    def observed_loss(*args, **kwargs):
        observed.append(
            (kwargs["coverage_score_weight"], kwargs["context_score_weight"])
        )
        return compute(*args, **kwargs)

    monkeypatch.setattr(training, "compute_loss", observed_loss)
    checkpoint = training.train(cfg)
    saved = torch.load(checkpoint, weights_only=True)
    assert observed == [(original["coverage_weight"], original["context_weight"])]
    assert saved["config"]["retrieval"] == original
    selected = yaml.safe_load((checkpoint.parent / "selected.yaml").read_text())
    for key, value in zip(CALIBRATED_RETRIEVAL_KEYS, (0.12, 0.02, 0.1375), strict=True):
        assert saved[f"best_{key}"] == saved["validation"][key] == value
        assert selected["selected_retrieval"][key] == value
        assert wandb_run.summary[f"best_{key}"] == value
    assert cfg["retrieval"] == original
