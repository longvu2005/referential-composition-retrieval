"""Cross-method ranking validation, ties and optional coarse diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rcr.common.io import save_results, scores_to_rankings
from rcr.evaluation.evaluate import evaluate_retrieval_output


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
    assert proposed["overall"].pop("candidate_hit_1") == 1
    assert baseline["overall"] == proposed["overall"]
    assert baseline["by_case"] == proposed["by_case"]


def test_coarse_metrics_use_full_ranking_and_hit_differs_from_recall():
    data, samples = inputs()
    data.images_by_id["d"] = {"path": "test/d.jpg"}
    data.gallery_ids.append("d")
    data.gt_head_boxes_by_image["b"] = [{"identity_id": "p"}]
    samples[0]["positive_image_ids"] = ["a", "b"]
    output = {
        "sample_ids": ["s"],
        "gallery_ids": ["q", "a", "b", "d"],
        "rankings": torch.tensor([[1, 3, 2]], dtype=torch.int32),
        "coarse_rankings": torch.tensor([[3, 1, 2]], dtype=torch.int32),
        "coarse_topm": torch.tensor([[3, 1]], dtype=torch.int32),
    }
    result = evaluate_retrieval_output(data, samples, output, [1, 2], split="test")
    metrics = result["overall"]
    assert metrics["full_map"] == pytest.approx(5 / 6)
    assert metrics["coarse_full_map"] == pytest.approx(7 / 12)
    assert metrics["candidate_recall_2"] == 0.5
    assert metrics["candidate_hit_2"] == 1.0
    assert metrics["candidate_hit_1"] == 0.0
    assert result["by_case"]["INDIVIDUAL"]["coarse_full_r1"] == 0.0
    assert result["per_query"][0]["coarse_full_ap"] == pytest.approx(7 / 12)


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


def test_saved_evaluation_rejects_changed_labels_without_overwriting_metrics(tmp_path):
    from rcr.common.data import split_fingerprint
    from rcr.evaluation.runner import evaluate_run

    data, samples = inputs()
    samples[0].update(target_image_id="a", final_change="Subject 1 is standing")
    data.samples_by_id = {"s": samples[0]}
    output = scores_to_rankings(
        samples, ["q", "a", "b"], np.array([[0, 2, 1]], dtype=np.float32)
    )
    cfg = {"method": "clip", "mode": "clip_image", "split": "test",
           "output": {"dir": str(tmp_path)}}
    directory = tmp_path / "clip_image" / "test"
    save_results(directory, output, {"split_sha256": split_fingerprint(data, "test")})
    evaluate_run(cfg, data=data, samples=samples)
    before = (directory / "metrics.json").read_bytes()
    samples[0]["final_change"] = "Subject 1 is sitting"
    with pytest.raises(ValueError, match="changed since retrieval"):
        evaluate_run(cfg, data=data, samples=samples)
    assert (directory / "metrics.json").read_bytes() == before
