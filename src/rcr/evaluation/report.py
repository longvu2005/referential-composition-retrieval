"""Validate saved evaluation JSON and export paper tables; no inference."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from statistics import fmean

from rcr.dataset.cases import CASE_TYPES
from rcr.evaluation.provenance import BENCHMARK_FIELDS, PROTOCOL, digest_json

METHODS = ("clip_image", "clip_text", "early_fusion", "late_fusion", "fafa", "proposed")
OVERALL_METRICS = (
    ("id_map", "ID-mAP", "id_ap"),
    ("id_r1", "ID-R@1", "id_r1"),
    ("id_r5", "ID-R@5", "id_r5"),
    ("id_r10", "ID-R@10", "id_r10"),
    ("full_map", "Full-mAP", "full_ap"),
    ("full_r1", "Full-R@1", "full_r1"),
    ("full_r5", "Full-R@5", "full_r5"),
    ("full_r10", "Full-R@10", "full_r10"),
)
CASE_METRICS = tuple(OVERALL_METRICS[i] for i in (0, 1, 4, 5))


class ReportError(ValueError):
    """An incomplete or incompatible experiment must not enter a paper table."""


def default_runs(root: str | Path, split: str) -> dict[str, Path]:
    root = Path(root)
    return {
        name: root / "clip" / name / split
        if name in METHODS[:4]
        else root / name / split
        for name in METHODS
    }


def _read(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReportError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(result, dict):
        raise ReportError(f"{path}: expected a JSON object")
    return result


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ReportError(f"{label}: expected an integer >= {minimum}")
    return value


def _metric(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ReportError(f"{label}: expected a finite metric in [0, 1]")
    return value


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReportError(f"{label}: missing/invalid SHA256 fingerprint")
    return value


def _close(actual, expected, label):
    _metric(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ReportError(
            f"{label}: saved {actual} disagrees with per-query mean {expected}"
        )


def load_run(name: str, directory: str | Path, split: str) -> dict:
    """Read only metrics.json/run.json and verify their evaluator provenance.

    Older rankings can be re-evaluated to attach the JSON binding without
    inference. The original split fingerprint remains mandatory; missing
    historical fingerprints cannot be reconstructed by this reporter.
    """
    directory = Path(directory)
    if name not in METHODS:
        raise ReportError(f"unsupported method {name!r}")
    run = _read(directory / "run.json")
    metrics = _read(directory / "metrics.json")
    label = str(directory)
    config = run.get("config")
    if not isinstance(config, dict):
        raise ReportError(f"{label}: missing run.config")
    method = "clip" if name in METHODS[:4] else name
    if run.get("method") != method or config.get("method") != method:
        raise ReportError(f"{label}: method metadata does not match {name}")
    if method == "clip":
        aliases = {"image": "clip_image", "text": "clip_text"}
        if any(
            aliases.get(value, value) != name
            for value in (run.get("mode"), config.get("mode"))
        ):
            raise ReportError(f"{label}: CLIP mode metadata does not match {name}")
    if config.get("split") != split or run.get("split", split) != split:
        raise ReportError(f"{label}: split metadata does not match requested {split}")
    if run.get("query_subset") is not False:
        raise ReportError(f"{label}: only complete query splits may enter paper tables")
    _hash(run.get("split_sha256"), f"{label}: split_sha256")
    _hash(run.get("checkpoint_sha256"), f"{label}: checkpoint_sha256")
    for key in ("dataset_sha256", "sample_ids_sha256", "gallery_ids_sha256"):
        if key in run:
            _hash(run[key], f"{label}: {key}")
    for key, expected in (
        ("evaluation_protocol", PROTOCOL),
        ("self_exclusion", "query_image"),
        ("ranking_scope", "complete_split_gallery"),
    ):
        if key in run and run[key] != expected:
            raise ReportError(f"{label}: incompatible {key}")
    nq = _integer(run.get("num_queries"), f"{label}: num_queries", 1)
    _integer(run.get("num_gallery"), f"{label}: num_gallery", 2)
    if method in ("clip", "fafa") and run.get("rcr_training") is not False:
        raise ReportError(f"{label}: baseline must declare rcr_training=false")
    if method == "fafa" and split == "test":
        selection = run.get("adapter_selection", {})
        if (
            not isinstance(selection, dict)
            or selection.get("split") != "val"
            or selection.get("metric") != "full_map"
        ):
            raise ReportError(
                f"{label}: FAFA test requires validation-frozen adapter settings"
            )
        _hash(selection.get("context_sha256"), f"{label}: FAFA context")
    if name in ("early_fusion", "late_fusion"):
        selection = run.get("fusion_selection", {})
        if (
            not isinstance(selection, dict)
            or selection.get("split") != "val"
            or selection.get("metric") != "full_map"
        ):
            raise ReportError(f"{label}: fusion requires validation Full-mAP selection")
        _hash(selection.get("context_sha256"), f"{label}: fusion context")
        fusion = config.get("fusion", {})
        weights = [fusion.get(key) for key in ("image_weight", "text_weight")]
        for key, weight in zip(("image_weight", "text_weight"), weights, strict=True):
            _metric(weight, f"{label}: {key}")
            if key in selection and selection[key] != weight:
                raise ReportError(
                    f"{label}: frozen fusion weights disagree with selection"
                )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            raise ReportError(f"{label}: selected fusion weights must sum to one")

    provenance = metrics.get("provenance")
    if not isinstance(provenance, dict):
        raise ReportError(
            f"{label}: missing evaluation provenance; evaluate saved rankings "
            "with scripts/run.py evaluate (no inference)"
        )
    from rcr.common.io import sha256_file

    if provenance.get("run_sha256") != sha256_file(directory / "run.json"):
        raise ReportError(f"{label}: metrics.json belongs to a different run.json")
    for key in BENCHMARK_FIELDS:
        if key in run and provenance.get(key) != run[key]:
            raise ReportError(f"{label}: metrics/run metadata disagree on {key}")

    rows = metrics.get("per_query")
    overall, by_case = metrics.get("overall"), metrics.get("by_case")
    if (
        not isinstance(rows, list)
        or len(rows) != nq
        or not isinstance(overall, dict)
        or not isinstance(by_case, dict)
    ):
        raise ReportError(
            f"{label}: missing/inconsistent overall, by_case or per_query"
        )
    if overall.get("num_queries") != nq:
        raise ReportError(f"{label}: overall/run query count mismatch")
    sample_ids, case_rows = [], {case: [] for case in CASE_TYPES}
    for row in rows:
        if not isinstance(row, dict) or row.get("case_type") not in CASE_TYPES:
            raise ReportError(f"{label}: invalid per-query case type")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ReportError(f"{label}: invalid per-query sample_id")
        sample_ids.append(sample_id)
        for _, _, key in OVERALL_METRICS:
            _metric(row.get(key), f"{label}: {sample_id}.{key}")
        case_rows[row["case_type"]].append(row)
    if len(set(sample_ids)) != nq:
        raise ReportError(f"{label}: duplicate per-query sample_ids")
    if (
        "sample_ids_sha256" in run
        and digest_json(sample_ids) != run["sample_ids_sha256"]
    ):
        raise ReportError(f"{label}: per-query sample order differs from run metadata")
    if set(by_case) != {case for case, subset in case_rows.items() if subset}:
        raise ReportError(
            f"{label}: by_case does not match the evaluated case coverage"
        )
    counts = {case: len(subset) for case, subset in case_rows.items()}
    if "case_counts" in run and run["case_counts"] != counts:
        raise ReportError(f"{label}: case counts disagree with run metadata")
    for key, _, query_key in OVERALL_METRICS:
        _close(
            overall.get(key), fmean(row[query_key] for row in rows), f"{label}: {key}"
        )
    for case, subset in case_rows.items():
        if not subset:
            continue
        aggregate = by_case[case]
        if not isinstance(aggregate, dict) or aggregate.get("num_queries") != len(
            subset
        ):
            raise ReportError(f"{label}: {case} query count mismatch")
        for key, _, query_key in CASE_METRICS:
            _close(
                aggregate.get(key),
                fmean(row[query_key] for row in subset),
                f"{label}: {case}.{key}",
            )
    return {
        "name": name,
        "directory": directory,
        "run": run,
        "metrics": metrics,
        "sample_ids": sample_ids,
        "case_counts": counts,
    }


def validate_runs(runs: list[dict]) -> None:
    if not runs:
        raise ReportError("no runs selected")
    if len({run["name"] for run in runs}) != len(runs):
        raise ReportError("duplicate method results")
    first = runs[0]
    for other in runs[1:]:
        for key in ("split_sha256", "num_queries", "num_gallery"):
            if first["run"][key] != other["run"][key]:
                raise ReportError(
                    f"incompatible {key}: {first['name']} vs {other['name']}"
                )
        for key in ("sample_ids", "case_counts"):
            if first[key] != other[key]:
                raise ReportError(
                    f"incompatible {key}: {first['name']} vs {other['name']}"
                )
    # A matching split digest already binds exact queries, labels and gallery.
    # Additional modern provenance must agree wherever recorded; never invent
    # a Git commit/runtime/dataset version for legacy runs.
    for key in (
        "dataset_version",
        "dataset_sha256",
        "gallery_ids_sha256",
        "evaluation_protocol",
        "self_exclusion",
        "ranking_scope",
    ):
        values = {
            json.dumps(run["run"][key], sort_keys=True)
            for run in runs
            if run["run"].get(key) is not None
        }
        if len(values) > 1:
            raise ReportError(f"incompatible {key} across selected runs")


def table_rows(runs, *, by_case=False):
    if not by_case:
        headers = ["Method", *(label for _, label, _ in OVERALL_METRICS)]
        rows = [
            [
                run["name"],
                *(run["metrics"]["overall"][key] for key, _, _ in OVERALL_METRICS),
            ]
            for run in runs
        ]
    else:
        headers = [
            "Method",
            *(f"{case} {label}" for case in CASE_TYPES for _, label, _ in CASE_METRICS),
        ]
        rows = [
            [
                run["name"],
                *(
                    run["metrics"]["by_case"].get(case, {}).get(key)
                    for case in CASE_TYPES
                    for key, _, _ in CASE_METRICS
                ),
            ]
            for run in runs
        ]
    return headers, rows


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_table(headers, rows, format, *, by_case=False, decimals=2):
    """CSV keeps raw fractions; Markdown/LaTeX show percentages, absent cases --."""
    if format == "csv":
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
        return buffer.getvalue()
    display = [
        [
            row[0],
            *(
                None if value is None else f"{100 * value:.{decimals}f}"
                for value in row[1:]
            ),
        ]
        for row in rows
    ]
    if format == "md":
        lines = [
            "Values are percentages; — means no queries for that case.",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
        ]
        lines += [
            "| " + " | ".join("—" if value is None else value for value in row) + " |"
            for row in display
        ]
        return "\n".join(lines) + "\n"
    if format != "tex":
        raise ReportError(f"unsupported table format {format!r}")
    lines = [
        "% Metrics in percent; -- denotes a case with no evaluated queries.",
        r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"\hline",
    ]
    if by_case:
        lines += [
            "Method & "
            + " & ".join(r"\multicolumn{4}{c}{" + case + "}" for case in CASE_TYPES)
            + r" \\",
            " & "
            + " & ".join(
                _latex_escape(label) for _ in CASE_TYPES for _, label, _ in CASE_METRICS
            )
            + r" \\",
        ]
    else:
        lines.append(" & ".join(_latex_escape(header) for header in headers) + r" \\")
    lines.append(r"\hline")
    lines += [
        " & ".join("--" if value is None else _latex_escape(value) for value in row)
        + r" \\"
        for row in display
    ]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_report(
    run_directories, *, split, output_dir, formats=("csv", "md", "tex"), decimals=2
) -> list[Path]:
    if split not in ("val", "test"):
        raise ReportError("report split must be val or test")
    if type(decimals) is not int or not 0 <= decimals <= 10:
        raise ReportError("decimals must be an integer from 0 to 10")
    if not formats or set(formats) - {"csv", "md", "tex"}:
        raise ReportError("formats must be csv, md and/or tex")
    missing = [
        str(Path(directory) / filename)
        for directory in run_directories.values()
        for filename in ("metrics.json", "run.json")
        if not (Path(directory) / filename).is_file()
    ]
    if missing:
        raise ReportError("missing result files:\n" + "\n".join(missing))
    runs = [
        load_run(name, directory, split) for name, directory in run_directories.items()
    ]
    validate_runs(runs)
    # Render everything before touching output, including on incompatible runs.
    rendered = {}
    for name, by_case in (("overall", False), ("by_case", True)):
        headers, rows = table_rows(runs, by_case=by_case)
        for format in dict.fromkeys(formats):
            rendered[f"{name}.{format}"] = render_table(
                headers, rows, format, by_case=by_case, decimals=decimals
            )
    from rcr.common.io import preserve_file

    output = Path(output_dir) / split
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, content in rendered.items():
        path = output / filename
        preserve_file(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        paths.append(path)
    return paths
