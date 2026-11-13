"""CPU workflow tests for score pipeline."""

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.data import split_fingerprint
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.runner import evaluate_run
from rcr.proposed import calibration, retrieve
from rcr.proposed.replay import replay_rankings


def test_standalone_retrieve_save_evaluate_and_ranking_only_evaluation(
    saved_pipeline,
):
    cfg, data, scale = saved_pipeline
    output = retrieve.retrieve(cfg)
    directory = Path(cfg["output"]["dir"]) / "val"
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert saved["binding_scale"] == scale == 1.0
    assert torch.equal(replay_rankings(saved)["rankings"], output["rankings"])
    current_metrics = evaluate_run(cfg, data=data)
    # Evaluation needs only rankings and IDs; raw score storage does not affect it.
    ranking_only = {
        k: saved[k]
        for k in (
            "sample_ids",
            "gallery_ids",
            "rankings",
            "coarse_rankings",
            "coarse_topm",
            "fine_rankings",
            "fafa_rankings",
            "fafa_binding",
            "context",
            "coverage",
        )
    }
    assert evaluate_retrieval_output(
        data, [data.samples[1]], ranking_only, [1, 2], split="val"
    ) == evaluate_retrieval_output(data, [data.samples[1]], saved, [1, 2], split="val")
    assert current_metrics["overall"]["num_queries"] == 1
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["score_config"] == saved["score_config"]
    assert metadata["score_schema_version"] == 2
    assert metadata["fine_score_mode"] == "bounded_residual"
    assert metadata["context_weight"] == cfg["retrieval"]["context_weight"]
    assert metadata["checkpoint_epoch"] == 2
    assert metadata["validation_sha256"] == split_fingerprint(data, "val")
    assert {
        "checkpoint_sha256",
        "dataset_sha256",
        "git_commit",
        "git_dirty",
    } <= metadata.keys()

    unselected_test = copy.deepcopy(cfg)
    unselected_test["split"] = "test"
    with pytest.raises(ValueError, match="validation-selected configuration"):
        retrieve.retrieve(unselected_test)

    incomplete_test = copy.deepcopy(unselected_test)
    incomplete_test.update(
        selected_checkpoint_sha256=metadata["checkpoint_sha256"],
        selected_validation_sha256=metadata["validation_sha256"],
        selected_retrieval={},
    )
    with pytest.raises(ValueError, match="retrieval contract is incomplete"):
        retrieve.retrieve(incomplete_test)


def test_calibration_freezes_full_scoring_config_and_retrieves_selected(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    retrieve.retrieve(cfg)
    source_path = Path(cfg["output"]["dir"]) / "val/rankings.pt"
    root = tmp_path / "calibration"
    rows = calibration.calibrate(
        {
            "rankings": str(source_path),
            "output_dir": str(root),
            "coarse_beta": [0.0, 0.4],
            "top_m": [cfg["retrieval"]["top_m"]],
        }
    )
    assert len(rows) == 3  # Source configuration plus two coarse candidates.
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["retrieval"]["fine_score_mode"] == "bounded_residual"
    assert selected["retrieval"]["context_weight"] == cfg["retrieval"]["context_weight"]
    assert selected["selected_validation_sha256"] == split_fingerprint(data, "val")
    assert len(selected["selected_checkpoint_sha256"]) == 64
    best = max(rows, key=lambda row: row["checkpoint_score"])
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    assert expected["score_config"]["coarse_beta"] == best["coarse_beta"]
    output = retrieve.retrieve(selected)
    evaluate_run(selected, data=data)
    assert torch.equal(output["rankings"], expected["rankings"])

    for key, value in (
        ("context_weight", 0.2),
        ("coarse_beta", 3.0),
        ("top_m", 1),
    ):
        tampered = copy.deepcopy(selected)
        tampered["split"] = "test"
        tampered["retrieval"][key] = value
        with pytest.raises(
            ValueError, match="settings changed since val selection|must match training"
        ):
            retrieve.retrieve(tampered)
    # Chunking is operational, and may change without selecting a new method.
    selected["retrieval"]["fine_batch_size"] = 2
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])
    del selected["selected_retrieval"]
    with pytest.raises(ValueError, match="predates the scoring contract"):
        retrieve.retrieve(selected)


def test_obsolete_checkpoint_scorer_is_rejected_before_loading_text(
    saved_pipeline, monkeypatch
):
    cfg, _, _ = saved_pipeline
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    checkpoint["config"]["retrieval"]["fine_score_mode"] = "context_zscore"
    torch.save(checkpoint, cfg["checkpoint"])
    monkeypatch.setattr(
        retrieve,
        "load_text_modules",
        lambda *a, **k: pytest.fail("loaded obsolete model"),
    )
    with pytest.raises(ValueError, match="obsolete fine scorer"):
        retrieve.retrieve(cfg)


@pytest.mark.parametrize(
    "key,value", [("context_weight", 0.2), ("pooling", "penalized")]
)
def test_calibration_rejects_diagnostic_changes_to_the_trained_fine_scorer(
    saved_pipeline, tmp_path, key, value
):
    cfg, _, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    changed = replay_rankings(source, **{key: value})
    path = tmp_path / "ablation.pt"
    torch.save(changed, path)
    root = tmp_path / "forbidden-selection"
    with pytest.raises(
        ValueError, match="requires the trained residual weight and pooling"
    ):
        calibration.calibrate(
            {
                "rankings": str(path),
                "output_dir": str(root),
                "coarse_beta": [0.4],
                "top_m": [2],
            }
        )
    assert not root.exists()
