"""Collect a manual Label Studio JSON export into reviewed.jsonl."""

from pathlib import Path

from labelstudio.common import (
    choices,
    latest_annotation,
    load_tasks,
    merge_output,
    textarea,
)
from rcr.dataset.rewrite import CASE_TYPES, validate_rewrite_output
from rcr.utils.jsonl import load_jsonl

INPUT = Path("dataset/data/work/review/review_input.jsonl")
EXPORT = Path("dataset/data/work/review/labelstudio_export.json")
OUTPUT = Path("dataset/data/work/review/reviewed.jsonl")


def parse_task(task: dict, source: dict) -> dict | None:
    annotation = latest_annotation(task)
    if annotation is None:
        return None

    selected_cases = choices(annotation, "case_type")
    if len(selected_cases) > 1:
        raise ValueError(f"{source['submission_id']}: multiple case types selected")

    case_type = selected_cases[0] if selected_cases else source["case_type"]
    if case_type not in CASE_TYPES:
        raise ValueError(
            f"{source['submission_id']}: invalid case_type {case_type!r}"
        )

    output = {
        "final_desc": textarea(annotation, "final_desc"),
        "final_change": textarea(annotation, "final_change"),
    }
    if output["final_desc"] is None or output["final_change"] is None:
        return None

    final_desc, final_change = validate_rewrite_output(source, output)
    return {
        "submission_id": source["submission_id"],
        "case_type": case_type,
        "final_desc": final_desc,
        "final_change": final_change,
    }


def main() -> None:
    sources = {row["submission_id"]: row for row in load_jsonl(INPUT)}
    rows = []
    for task in load_tasks(EXPORT):
        task_key = task.get("data", {}).get("task_key")
        source = sources.get(task_key)
        if source is None:
            continue
        row = parse_task(task, source)
        if row is not None:
            rows.append(row)

    merge_output(OUTPUT, rows)
    print(f"Collected: {len(rows)}")
    print(f"Updated: {OUTPUT}")


if __name__ == "__main__":
    main()
