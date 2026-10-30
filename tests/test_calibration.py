"""Calibration selection and prepared metrics must retain the official protocol."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from rcr.common.io import sha256_file
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.report import load_run
from rcr.proposed import calibration, retrieve
from rcr.proposed.contracts import SELECTED_RETRIEVAL_KEYS
from tests.test_score_pipeline import saved_pipeline as saved_pipeline


def _validation_output(seed, cases, dtype):
    rng = np.random.default_rng(seed)
    gallery = [f"image-{i}" for i in range(97)]
    heads = {
        image: [{"identity_id": f"{i % 5}-{role}"} for role in range(3)]
        for i, image in enumerate(gallery)
    }
    samples = []
    orders = [[] for _ in range(4)]
    for i, case in enumerate(cases):
        identities = [f"{i % 5}-{role}" for role in range(3)]
        subjects = [{"subject_id": 1, "identity_ids": identities[:1]}]
        if case == "GROUP":
            subjects[0]["identity_ids"] = identities[:2]
        elif case in ("DUAL", "RELATIONAL"):
            subjects.append({"subject_id": 2, "identity_ids": identities[1:2]})
        samples.append(
            {
                "sample_id": f"query-{i}",
                "case_type": case,
                "query_image_id": gallery[i],
                "subjects": subjects,
                "positive_image_ids": gallery[i + 5 : i + 5 * (2 + i % 3) : 5],
            }
        )
        eligible = np.delete(np.arange(len(gallery)), i)
        for order in orders:
            order.append(rng.permutation(eligible))
    data = SimpleNamespace(
        gallery_ids=gallery,
        images_by_id={image: {"path": f"val/{image}.jpg"} for image in gallery},
        gt_head_boxes_by_image=heads,
        splits={"val": [row["sample_id"] for row in samples]},
    )
    output = {
        "sample_ids": data.splits["val"],
        "gallery_ids": gallery,
        **{
            field: torch.tensor(np.stack(order), dtype=dtype)
            for field, order in zip(
                ("rankings", "coarse_rankings", "fine_rankings", "fafa_rankings"),
                orders,
                strict=True,
            )
        },
    }
    return data, samples, output


@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "cases",
    [
        ["INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL"] * 2,
        ["INDIVIDUAL", "INDIVIDUAL", "DUAL"],
    ],
)
def test_prepared_metrics_match_every_official_summary_field(seed, cases, dtype):
    data, samples, source = _validation_output(seed, cases, dtype)
    ks = [1, 5, 17, 100]
    evaluator = calibration._CalibrationEvaluator(data, samples, source, ks)
    for field in (
        None,
        "rankings",
        "fine_rankings",
        "fafa_rankings",
        "coarse_rankings",
    ):
        if field is not None:
            source = {**source, field: source[field].flip(1)}
        expected = evaluate_retrieval_output(data, samples, source, ks, split="val")
        actual = evaluator(source)
        assert actual["overall"] == expected["overall"]
        assert actual["by_case"] == expected["by_case"]


def test_prepared_metrics_reuse_only_unchanged_branches(monkeypatch):
    data, samples, source = _validation_output(5, ["INDIVIDUAL", "GROUP"], torch.int32)
    evaluator = calibration._CalibrationEvaluator(data, samples, source, [1, 5])
    aggregate = calibration._aggregate_overall
    calls = []

    def counted(*args):
        calls.append(1)
        return aggregate(*args)

    monkeypatch.setattr(calibration, "_aggregate_overall", counted)
    evaluator(source)
    assert len(calls) == 4
    evaluator(source)
    assert len(calls) == 4
    evaluator({**source, "rankings": source["rankings"].flip(1)})
    assert len(calls) == 5


def test_prepared_metrics_reject_invalid_labels_before_using_positive_ranks():
    data, samples, source = _validation_output(5, ["GROUP"], torch.int32)
    samples[0]["positive_image_ids"] = [source["gallery_ids"][1]]
    with pytest.raises(ValueError, match="every Full Positive"):
        calibration._CalibrationEvaluator(data, samples, source, [1, 5])


@pytest.mark.parametrize(
    "metric,winner",
    [("checkpoint_score", 1), ("full_map", 0), ("macro_full_map", 1)],
)
def test_selection_uses_requested_objective_and_records_baseline(
    tmp_path, metric, winner
):
    retrieval = {
        "pooling": "subject_mean",
        "fine_score_mode": "context_zscore",
        "context_weight": 0.0225,
        "coarse_beta": 1.8,
        "fine_coarse_weight": 0.05,
        "top_m": 500,
    }
    cfg = {"retrieval": retrieval, "output": {"dir": "unused"}}
    rows = [
        {
            "index": 0,
            **retrieval,
            "checkpoint_sha256": "a" * 64,
            "validation_sha256": "b" * 64,
            "full_map": 0.54,
            "macro_full_map": 0.40,
            "checkpoint_score": 0.47,
        }
    ]
    rows.append(
        {
            **rows[0],
            "index": 1,
            "context_weight": 0.01,
            "full_map": 0.53,
            "macro_full_map": 0.49,
            "checkpoint_score": 0.51,
        }
    )
    before = copy.deepcopy(cfg)
    selected = calibration._write_selection(
        cfg, tmp_path, rows, selection_metric=metric
    )
    assert cfg == before
    assert selected["retrieval"]["context_weight"] == rows[winner]["context_weight"]
    record = json.loads((tmp_path / "selection.json").read_text())
    assert record["metric"] == metric
    assert record["value"] == rows[winner][metric]
    assert record["baseline_value"] == rows[0][metric]
    assert record["improvement"] == rows[winner][metric] - rows[0][metric]


@pytest.mark.parametrize("baseline_in_grid", [False, True])
def test_source_is_retained_on_ties_and_duplicate_candidates_run_once(
    saved_pipeline, tmp_path, baseline_in_grid
):
    cfg, data, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    # Give the source perfect AP; no candidate can improve its selection score.
    sample = data.samples_by_id[source["sample_ids"][0]]
    positive = source["gallery_ids"][source["rankings"][0, 0]]
    sample["positive_image_ids"] = [positive]
    sample["target_image_id"] = positive
    source = retrieve.retrieve(cfg)
    path = tmp_path / "source.pt"
    torch.save(source, path)
    digest = sha256_file(path)
    parameters = {
        key: source["score_config"][key]
        for key in (
            "coarse_beta",
            "top_m",
            "context_weight",
            "fine_coarse_weight",
            "pooling",
        )
    }
    grid = {**parameters}
    if not baseline_in_grid:
        grid.update(coarse_beta=0.0, top_m=1, context_weight=0.0)
    root = tmp_path / "calibrated"
    rows = calibration.calibrate(
        {
            "rankings": str(path),
            "output_dir": str(root),
            **{key: [value, value] for key, value in grid.items()},
        }
    )
    assert len(rows) == (1 if baseline_in_grid else 2)
    assert rows[0]["index"] == 0
    assert {key: rows[0][key] for key in parameters} == parameters
    assert rows[0]["full_map"] == rows[0]["checkpoint_score"] == 1.0
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["selected_retrieval"] == {
        key: source["score_config"][key] for key in SELECTED_RETRIEVAL_KEYS
    }
    record = json.loads((root / "selection.json").read_text())
    assert record["index"] == 0
    assert record["metric"] == "checkpoint_score"
    assert record["baseline_value"] == record["value"] == 1.0
    assert record["improvement"] == 0.0
    assert sha256_file(path) == digest
    assert (
        load_run("proposed", root / "selected/val", "val")["metrics"]["overall"][
            "full_map"
        ]
        == 1.0
    )


@pytest.mark.parametrize("metric", ["full_mAP", "", None, True])
def test_invalid_selection_metric_fails_before_opening_scores(tmp_path, metric):
    with pytest.raises(ValueError, match="selection_metric"):
        calibration.calibrate(
            {
                "rankings": str(tmp_path / "not-present.pt"),
                "output_dir": str(tmp_path / "output"),
                "selection_metric": metric,
            }
        )
    assert not (tmp_path / "output").exists()
