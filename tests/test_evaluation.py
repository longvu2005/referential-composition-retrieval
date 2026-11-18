"""Benchmark metrics, report integrity and result export."""

import csv
import json
from pathlib import Path

import pytest

from rcr.evaluation.evaluate import evaluate_rankings
from rcr.evaluation.report import ReportError
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.retrieve import retrieve
from tools.report import main

# Identity and Full Positive metrics.


def test_full_and_identity_metrics_use_their_own_positive_sets():
    gallery = ["q", "p1", "p2", "wrong_id", "wrong_state"]
    identities = {image: {"A", "B"} for image in gallery}
    identities["wrong_id"] = {"A"}
    sample = {
        "sample_id": "query",
        "case_type": "DUAL",
        "query_image_id": "q",
        "subjects": [
            {"subject_id": 1, "identity_ids": ["A"]},
            {"subject_id": 2, "identity_ids": ["B"]},
        ],
        "positive_image_ids": ["p1", "p2"],
    }
    result = evaluate_rankings(
        [sample],
        gallery,
        identities,
        {"query": ["wrong_id", "p1", "wrong_state", "p2"]},
    )
    assert result["overall"]["full_map"] == pytest.approx(0.5)
    assert result["overall"]["id_map"] == pytest.approx((1 / 2 + 2 / 3 + 3 / 4) / 3)
    assert result["overall"]["full_r1"] == 0
    assert result["overall"]["full_r5"] == 1
    assert result["by_case"]["DUAL"]["full_map"] == pytest.approx(0.5)


# Report integrity and export.


def test_default_report_exports_evaluated_proposed_results(saved_pipeline, tmp_path):
    cfg, data, _ = saved_pipeline
    cfg["output"]["dir"] = str(tmp_path / "runs" / "proposed" / "selected")
    output = retrieve(cfg)
    metrics = evaluate_run(cfg, data=data, output=output)

    paths = main(
        [
            "--split",
            "val",
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "tables"),
        ]
    )
    assert {path.name for path in paths} == {
        f"{name}.{extension}"
        for name in ("overall", "by_case")
        for extension in ("csv", "md", "tex")
    }
    with (tmp_path / "tables/val/overall.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["Method"] == "proposed"
    assert float(rows[0]["Full-mAP"]) == metrics["overall"]["full_map"]


def test_report_rejects_modified_run_metadata_before_writing(saved_pipeline, tmp_path):
    from rcr.evaluation.report import generate_report

    cfg, data, _ = saved_pipeline
    output = retrieve(cfg)
    evaluate_run(cfg, data=data, output=output)
    directory = Path(cfg["output"]["dir"]) / "val"
    metadata = directory / "run.json"
    run = json.loads(metadata.read_text())
    run["checkpoint_sha256"] = "a" * 64
    metadata.write_text(json.dumps(run))
    report_root = tmp_path / "tables"

    with pytest.raises(ReportError, match="different run.json"):
        generate_report({"proposed": directory}, split="val", output_dir=report_root)
    assert not report_root.exists()
