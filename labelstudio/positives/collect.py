"""Collect a manual Label Studio JSON export into positive_sets.jsonl."""

from collections import defaultdict
from pathlib import Path

from labelstudio.common import choices, latest_annotation, load_tasks, merge_output
from labelstudio.positives.prepare import GROUP_SIZE
from rcr.utils.jsonl import load_jsonl

INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
EXPORT = Path("dataset/data/work/positives/labelstudio_export.json")
OUTPUT = Path("dataset/data/work/positives/positive_sets.jsonl")


def _candidate_groups(row: dict) -> list[list[str]]:
    candidates = [
        candidate["image_id"]
        for candidate in row["candidates"]
        if not candidate["is_seed"]
    ]
    return [
        candidates[index : index + GROUP_SIZE]
        for index in range(0, len(candidates), GROUP_SIZE)
    ]


def aggregate(tasks: list[dict], inputs: list[dict]) -> list[dict]:
    inputs_by_id = {row["submission_id"]: row for row in inputs}
    groups = defaultdict(dict)

    for task in tasks:
        annotation = latest_annotation(task)
        if annotation is None:
            continue

        data = task.get("data", {})
        submission_id = data.get("submission_id")
        row = inputs_by_id.get(submission_id)
        if row is None:
            continue

        candidate_groups = _candidate_groups(row)
        group_index = data.get("group_index")
        if not isinstance(group_index, int) or not 0 <= group_index < len(
            candidate_groups
        ):
            raise ValueError(f"{submission_id}: invalid positive group")

        selected = choices(annotation, "positives")
        invalid = set(selected) - set(candidate_groups[group_index])
        if invalid:
            raise ValueError(
                f"{submission_id}: invalid positive choice {sorted(invalid)[0]}"
            )
        groups[submission_id][group_index] = selected

    outputs = []
    for submission_id, row in inputs_by_id.items():
        candidate_groups = _candidate_groups(row)
        labeled_groups = groups[submission_id]
        if len(labeled_groups) != len(candidate_groups):
            continue

        positives = [row["seed_target_image_id"]]
        for group_index in range(len(candidate_groups)):
            positives.extend(labeled_groups[group_index])
        outputs.append(
            {
                "submission_id": submission_id,
                "target_image_ids": list(dict.fromkeys(positives)),
            }
        )

    return outputs


def main() -> None:
    tasks = load_tasks(EXPORT) if EXPORT.exists() else []
    rows = aggregate(tasks, load_jsonl(INPUT))
    merge_output(OUTPUT, rows)
    print(f"Collected: {len(rows)}")
    print(f"Updated: {OUTPUT}")


if __name__ == "__main__":
    main()
