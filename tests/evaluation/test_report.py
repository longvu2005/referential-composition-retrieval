"""Saved-JSON reports against real evaluator outputs, never model inference."""

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rcr.common.io import save_results, scores_to_rankings, sha256_file, write_json
from rcr.dataset.cases import CASE_TYPES
from rcr.evaluation.provenance import BENCHMARK_FIELDS, benchmark_metadata
from rcr.evaluation.report import METHODS, ReportError, generate_report, load_run
from rcr.evaluation.runner import evaluate_run
from scripts import report as cli


def saved_runs(tmp_path, split="val", cases=None):
    cases = cases or ["INDIVIDUAL", *CASE_TYPES]
    samples = []
    for i, case in enumerate(cases):
        subjects = [{"subject_id": 1, "identity_ids": ["p1"]}]
        if case == "GROUP":
            subjects[0]["identity_ids"].append("p2")
        elif case in ("DUAL", "RELATIONAL"):
            subjects.append({"subject_id": 2, "identity_ids": ["p2"]})
        samples.append(
            {
                "sample_id": f"s{i}",
                "query_image_id": "q",
                "target_image_id": "a",
                "positive_image_ids": ["a", "b"] if i in (2, 4) else ["a"],
                "subjects": subjects,
                "case_type": case,
                "final_desc": "Identify Subject 1 as the person",
                "final_change": "Subject 1 is standing",
                "final_instruction": "x",
            }
        )
    ids = ["q", "a", "b", "n"]
    data = SimpleNamespace(
        samples=samples,
        samples_by_id={s["sample_id"]: s for s in samples},
        images_by_id={x: {"path": f"{split}/{x}.jpg"} for x in ids},
        gallery_ids=ids,
        manifest={"version": "fixture"},
        splits={
            s: [q["sample_id"] for q in samples] if s == split else []
            for s in ("train", "val", "test")
        },
        gt_head_boxes_by_image={
            x: [{"identity_id": p} for p in ("p1", "p2")] for x in ("q", "a", "b")
        },
    )
    score_rows = [[9, 3, 1, 2], [9, 1, 2, 3], [9, 2, 3, 1], [9, 3, 2, 1], [9, 1, 3, 2]]
    output = scores_to_rankings(
        samples, ids, np.array(score_rows[: len(samples)], dtype=np.float32)
    )
    directories = {}
    result = None
    for name in METHODS:
        method = "clip" if name in METHODS[:4] else name
        root = tmp_path / "runs" / ("clip" if method == "clip" else method)
        directory = root / name / split if method == "clip" else root / split
        cfg = {"method": method, "split": split, "output": {"dir": str(root)}}
        run = {
            "method": method,
            "query_subset": False,
            "rcr_training": False,
            "checkpoint_sha256": "a" * 64,
            **benchmark_metadata(data, split, output),
        }
        if method == "clip":
            cfg["mode"] = name
            run["mode"] = name
        if name in ("early_fusion", "late_fusion"):
            cfg["fusion"] = {"image_weight": 0.4, "text_weight": 0.6}
            run["fusion_selection"] = {
                "split": "val",
                "metric": "full_map",
                "context_sha256": "c" * 64,
                **cfg["fusion"],
            }
        if name == "fafa":
            run["adapter_selection"] = {
                "split": "val",
                "metric": "full_map",
                "context_sha256": "d" * 64,
            }
        run["config"] = cfg
        variant = copy.deepcopy(output)
        if method == "proposed":
            variant["coarse_rankings"] = variant["rankings"].clone()
            variant["coarse_topm"] = variant["rankings"][:, :2].clone()
        save_results(directory, variant, run)
        result = evaluate_run(cfg, data=data, samples=samples, output=variant)
        directories[name] = directory
    return directories, result, data, samples


def rebind(directory, run):
    """Make intentional fixture metadata edits without a stale-file binding."""
    write_json(directory / "run.json", run)
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["provenance"] = {
        "run_sha256": sha256_file(directory / "run.json"),
        **{k: run[k] for k in BENCHMARK_FIELDS if k in run},
    }
    write_json(directory / "metrics.json", metrics)


@pytest.mark.parametrize("split", ["val", "test"])
def test_all_six_methods_export_exact_paper_columns_and_evaluator_values(
    tmp_path, split
):
    directories, metrics, _, _ = saved_runs(tmp_path, split)
    original = {
        path: path.read_bytes()
        for directory in directories.values()
        for path in directory.iterdir()
        if path.is_file()
    }
    paths = generate_report(directories, split=split, output_dir=tmp_path / "reports")
    assert {p.name for p in paths} == {
        f"{table}.{ext}"
        for table in ("overall", "by_case")
        for ext in ("csv", "md", "tex")
    }
    root = tmp_path / "reports" / split
    with (root / "overall.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and len(rows[0]) == 9
    assert [row["Method"] for row in rows] == list(METHODS)
    assert float(rows[0]["ID-mAP"]) == pytest.approx(0.85)
    assert float(rows[0]["Full-mAP"]) == pytest.approx(5 / 6)
    assert float(rows[0]["Full-mAP"]) == metrics["overall"]["full_map"]
    with (root / "by_case.csv").open() as handle:
        cases = list(csv.DictReader(handle))
    assert len(cases[0]) == 17
    assert float(cases[0]["INDIVIDUAL Full-mAP"]) == pytest.approx(2 / 3)
    assert "83.33" in (root / "overall.md").read_text()
    latex = (root / "by_case.tex").read_text()
    assert all(r"\multicolumn{4}{c}{" + case + "}" in latex for case in CASE_TYPES)
    assert r"clip\_image" in latex and r"\\" in latex
    assert all(
        "coarse" not in p.read_text() and "candidate" not in p.read_text()
        for p in paths
    )
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("filename", ["run.json", "metrics.json"])
def test_missing_results_do_not_touch_existing_reports(tmp_path, filename):
    dirs, _, _, _ = saved_runs(tmp_path)
    report = tmp_path / "reports" / "val" / "overall.csv"
    report.parent.mkdir(parents=True)
    report.write_text("old table")
    (dirs["fafa"] / filename).unlink()
    with pytest.raises(ReportError, match=f"missing result files:.*|{filename}"):
        generate_report(dirs, split="val", output_dir=tmp_path / "reports")
    assert report.read_text() == "old table"
    assert not (report.parent / "by_case.csv").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_sha256", "f" * 64),
        ("dataset_sha256", "f" * 64),
        ("gallery_ids_sha256", "f" * 64),
        ("dataset_version", "different"),
        ("num_gallery", 5),
    ],
)
def test_incompatible_runs_fail_before_export(tmp_path, field, value):
    dirs, _, _, _ = saved_runs(tmp_path)
    run = json.loads((dirs["fafa"] / "run.json").read_text())
    run[field] = value
    rebind(dirs["fafa"], run)
    with pytest.raises(ReportError, match=f"incompatible {field}"):
        generate_report(dirs, split="val", output_dir=tmp_path / "reports")
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ("subset", "complete query"),
        ("split", "split metadata"),
        ("count", "overall.*query count"),
        ("fingerprint", "split_sha256"),
        ("case_count", "case counts"),
        ("protocol", "evaluation_protocol"),
        ("training", "rcr_training"),
        ("checkpoint", "checkpoint_sha256"),
    ],
)
def test_invalid_run_metadata(tmp_path, change, match):
    dirs, _, _, _ = saved_runs(tmp_path)
    directory = dirs["clip_image"]
    run = json.loads((directory / "run.json").read_text())
    if change == "subset":
        run["query_subset"] = True
    elif change == "split":
        run["config"]["split"] = "test"
    elif change == "count":
        run["num_queries"] += 1
    elif change == "fingerprint":
        del run["split_sha256"]
    elif change == "case_count":
        run["case_counts"]["GROUP"] = 20
    elif change == "protocol":
        run["evaluation_protocol"] = "other"
    elif change == "training":
        run["rcr_training"] = True
    else:
        del run["checkpoint_sha256"]
    rebind(directory, run)
    # Count changes also mismatch per-query row count, checked before the mean.
    if change == "count":
        match = "inconsistent|query count"
    with pytest.raises(ReportError, match=match):
        load_run("clip_image", directory, "val")


@pytest.mark.parametrize(
    "change", ["overall", "by_case", "missing_case", "duplicate", "nan"]
)
def test_inconsistent_aggregates_and_bad_metrics(tmp_path, change):
    dirs, _, _, _ = saved_runs(tmp_path)
    directory = dirs["proposed"]
    metrics = json.loads((directory / "metrics.json").read_text())
    if change == "overall":
        metrics["overall"]["full_map"] = 0.875  # Unweighted case mean is wrong.
    elif change == "by_case":
        metrics["by_case"]["INDIVIDUAL"]["full_map"] = 1.0
    elif change == "missing_case":
        del metrics["by_case"]["GROUP"]
    elif change == "duplicate":
        metrics["per_query"][1]["sample_id"] = "s0"
    else:
        metrics["per_query"][0]["full_ap"] = float("nan")
    write_json(directory / "metrics.json", metrics)
    with pytest.raises(ReportError):
        load_run("proposed", directory, "val")


def test_stale_metrics_binding_and_legacy_re_evaluation_without_inference(tmp_path):
    dirs, _, data, samples = saved_runs(tmp_path)
    directory = dirs["clip_image"]
    run = json.loads((directory / "run.json").read_text())
    for key in BENCHMARK_FIELDS:
        if key not in ("split_sha256", "num_queries", "num_gallery"):
            run.pop(key, None)
    write_json(directory / "run.json", run)
    with pytest.raises(ReportError, match="different run.json"):
        load_run("clip_image", directory, "val")
    evaluate_run(run["config"], data=data, samples=samples)
    assert (
        load_run("clip_image", directory, "val")["metrics"]["overall"]["num_queries"]
        == 5
    )
    assert list((directory / ".history").rglob("metrics.json"))


@pytest.mark.parametrize("change", ["missing", "test_metric", "weight"])
def test_fusion_requires_frozen_validation_full_map_selection(tmp_path, change):
    dirs, _, _, _ = saved_runs(tmp_path, "test")
    directory = dirs["early_fusion"]
    run = json.loads((directory / "run.json").read_text())
    if change == "missing":
        del run["fusion_selection"]
    elif change == "test_metric":
        run["fusion_selection"]["split"] = "test"
    else:
        run["config"]["fusion"]["image_weight"] = 0.6
    rebind(directory, run)
    with pytest.raises(ReportError, match="fusion|frozen"):
        load_run("early_fusion", directory, "test")


def test_unfrozen_fafa_test_is_rejected(tmp_path):
    dirs, _, _, _ = saved_runs(tmp_path, "test")
    directory = dirs["fafa"]
    run = json.loads((directory / "run.json").read_text())
    del run["adapter_selection"]
    rebind(directory, run)
    with pytest.raises(ReportError, match="validation-frozen"):
        load_run("fafa", directory, "test")


def test_absent_cases_are_not_fabricated_zero_and_reports_are_preserved(tmp_path):
    dirs, _, _, _ = saved_runs(tmp_path, cases=["INDIVIDUAL"])
    paths = generate_report(dirs, split="val", output_dir=tmp_path / "reports")
    previous = {path.name: path.read_bytes() for path in paths}
    root = paths[0].parent
    with (root / "by_case.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["GROUP Full-mAP"] == ""
    assert "—" in (root / "by_case.md").read_text()
    assert "--" in (root / "by_case.tex").read_text()
    generate_report(dirs, split="val", output_dir=tmp_path / "reports")
    for name, content in previous.items():
        assert any(
            path.read_bytes() == content for path in (root / ".history").rglob(name)
        )


def test_report_cli_reads_saved_json_without_rankings_or_models(tmp_path):
    dirs, _, _, _ = saved_runs(tmp_path)
    for directory in dirs.values():
        (directory / "rankings.pt").unlink()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report.py",
            "--split",
            "val",
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "reports" / "val").glob("*"))) == 6
    with pytest.raises(SystemExit):
        cli.main(["--split", "val", "--run", "fafa=x", "--run", "fafa=y"])
