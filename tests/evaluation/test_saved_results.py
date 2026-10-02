"""Cross-method ranking validation, ties and optional coarse diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.methods.common.results import save_results, scores_to_rankings


def inputs():
    sample = {
        "sample_id": "s",
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": ["a"],
        "subjects": [{"subject_id": 1, "identity_ids": ["p"]}],
    }
    data = SimpleNamespace(
        gallery_ids=["q", "a", "b", "left"],
        images_by_id={x: {"path": f"test/{x}.jpg"} for x in ["q", "a", "b"]}
        | {"left": {"path": "leftover/left.jpg"}},
        splits={"test": ["s"]},
        gt_head_boxes_by_image={
            "q": [{"identity_id": "p"}],
            "a": [{"identity_id": "p"}],
        },
    )
    return data, [sample]


def test_same_ranking_has_same_metrics_with_or_without_coarse():
    data, samples = inputs()
    output = scores_to_rankings(
        samples, ["q", "a", "b"], np.array([[10, 1, 1]], dtype=np.float32)
    )
    assert output["rankings"].tolist() == [[1, 2]]  # Self removed; stable tie.
    baseline = evaluate_retrieval_output(data, samples, output, [1], split="test")
    output["coarse_topm"] = torch.tensor([[1]], dtype=torch.int32)
    proposed = evaluate_retrieval_output(data, samples, output, [1], split="test")
    assert "candidate_recall_1" not in baseline["overall"]
    assert proposed["overall"].pop("candidate_recall_1") == 1
    assert baseline["overall"] == proposed["overall"]
    assert baseline["by_case"] == proposed["by_case"]


@pytest.mark.parametrize(
    "scores", [np.array([[0, np.nan, 1]]), np.array([[0, 1, np.inf]]), np.ones((1, 2))]
)
def test_invalid_scores_rejected(scores):
    _, samples = inputs()
    with pytest.raises(ValueError):
        scores_to_rankings(samples, ["q", "a", "b"], scores)


@pytest.mark.parametrize(
    "ranking",
    [
        torch.tensor([[1, 1]], dtype=torch.int32),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([[1.0, 2.0]]),
    ],
)
def test_incomplete_duplicate_self_or_float_rankings_rejected(ranking):
    data, samples = inputs()
    output = {"sample_ids": ["s"], "gallery_ids": ["q", "a", "b"], "rankings": ranking}
    with pytest.raises(ValueError):
        evaluate_retrieval_output(data, samples, output, split="test")


def test_reordered_gallery_rejected_even_when_indices_in_range():
    data, samples = inputs()
    output = {
        "sample_ids": ["s"],
        "gallery_ids": ["q", "b", "a"],
        "rankings": torch.tensor([[1, 2]], dtype=torch.int32),
    }
    with pytest.raises(ValueError, match="gallery_ids"):
        evaluate_retrieval_output(data, samples, output, split="test")


def test_retrieval_replaces_results_and_removes_previous_metrics(tmp_path):
    _, samples = inputs()
    output = scores_to_rankings(
        samples, ["q", "a", "b"], np.array([[0, 1, 2]], dtype=np.float32)
    )
    (tmp_path / "metrics.json").write_text('{"obsolete": true}', encoding="utf-8")
    save_results(tmp_path, output, {"num_queries": 1})
    assert not (tmp_path / "metrics.json").exists()
    saved = torch.load(tmp_path / "rankings.pt", weights_only=True)
    assert saved["rankings"].tolist() == [[2, 1]]
    assert json.loads((tmp_path / "run.json").read_text()) == {"num_queries": 1}
    assert not list(tmp_path.glob("*.tmp"))
