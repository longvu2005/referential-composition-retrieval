"""Build the offline Label Studio import file for rewrite review."""

from pathlib import Path

from labelstudio.common import (
    image_html,
    load_box_index,
    original_html,
    subjects_html,
    write_tasks,
)
from rcr.dataset.rewrite import CASE_TYPES
from rcr.utils.jsonl import load_jsonl

INPUT = Path("dataset/data/work/review/review_input.jsonl")
OUTPUT = Path("dataset/data/work/review/labelstudio_tasks.json")


def build_task(row: dict, boxes_by_image: dict[str, list[dict]]) -> dict:
    images = (
        '<div style="display:flex;gap:16px;flex-wrap:wrap">'
        + image_html(
            row["query_image_id"],
            row["query_image_url"],
            row["subjects"],
            boxes_by_image,
            "QUERY",
        )
        + image_html(
            row["target_image_id"],
            row["target_image_url"],
            row["subjects"],
            boxes_by_image,
            "TARGET",
        )
        + "</div>"
    )
    return {
        "data": {
            "task_key": row["submission_id"],
            "meta": f"{row['case_type']} · {row['submission_id']}",
            "case_choices": [
                {"value": case_type, "selected": case_type == row["case_type"]}
                for case_type in CASE_TYPES
            ],
            "images_html": images,
            "subjects_html": subjects_html(row["subjects"]),
            "final_desc": row.get("final_desc") or "",
            "final_change": row.get("final_change") or "",
            "original_html": original_html(row),
        }
    }


def main() -> None:
    boxes = load_box_index()
    tasks = [build_task(row, boxes) for row in load_jsonl(INPUT)]
    write_tasks(OUTPUT, tasks)
    print(f"Prepared: {len(tasks)}")
    print(f"Import into Label Studio: {OUTPUT}")


if __name__ == "__main__":
    main()
