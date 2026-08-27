"""Build the offline Label Studio import file for Full Positive selection."""

from pathlib import Path

from labelstudio.common import (
    image_html,
    load_box_index,
    subjects_html,
    write_tasks,
)
from rcr.utils.jsonl import load_jsonl

INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
OUTPUT = Path("dataset/data/work/positives/labelstudio_tasks.json")
GROUP_SIZE = 10


def build_tasks(row: dict, boxes_by_image: dict[str, list[dict]]) -> list[dict]:
    seed = next(candidate for candidate in row["candidates"] if candidate["is_seed"])
    candidates = [
        candidate for candidate in row["candidates"] if not candidate["is_seed"]
    ]
    groups = [
        candidates[index : index + GROUP_SIZE]
        for index in range(0, len(candidates), GROUP_SIZE)
    ]

    reference = (
        '<div style="display:flex;gap:16px;flex-wrap:wrap">'
        + image_html(
            row["query_image_id"],
            row["query_image_url"],
            row["subjects"],
            boxes_by_image,
            "QUERY",
        )
        + image_html(
            seed["image_id"],
            seed["image_url"],
            row["subjects"],
            boxes_by_image,
            "SEED TARGET",
        )
        + "</div>"
    )
    instruction = f"{row['final_desc']}; {row['final_change']}."

    tasks = []
    for group_index, group in enumerate(groups):
        options = []
        for candidate in group:
            options.append(
                {
                    "value": candidate["image_id"],
                    "html": image_html(
                        candidate["image_id"],
                        candidate["image_url"],
                        row["subjects"],
                        boxes_by_image,
                        candidate["image_id"],
                    ),
                }
            )

        tasks.append(
            {
                "data": {
                    "task_key": f"{row['submission_id']}::{group_index:03d}",
                    "submission_id": row["submission_id"],
                    "group_index": group_index,
                    "group_count": len(groups),
                    "reference_html": reference,
                    "subjects_html": subjects_html(row["subjects"]),
                    "instruction": instruction,
                    "candidate_choices": options,
                }
            }
        )
    return tasks


def main() -> None:
    boxes = load_box_index()
    tasks = [task for row in load_jsonl(INPUT) for task in build_tasks(row, boxes)]
    write_tasks(OUTPUT, tasks)
    print(f"Prepared: {len(tasks)}")
    print(f"Import into Label Studio: {OUTPUT}")


if __name__ == "__main__":
    main()
