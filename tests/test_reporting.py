"""Proposed result export and evaluation provenance validation."""

import csv
import json
from pathlib import Path

import pytest

from rcr.evaluation.report import ReportError
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.retrieve import retrieve
from tools.report import main


def test_default_report_exports_evaluated_proposed_results(saved_pipeline, tmp_path):
    cfg, data, _ = saved_pipeline
    cfg["output"]["dir"] = str(tmp_path / "runs" / "proposed")
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
