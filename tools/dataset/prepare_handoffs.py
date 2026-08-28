"""Prepare incremental inputs for the two human labeling handoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcr.dataset.handoff import (
    merge_handoff_catalog,
    prepare_positive_set_inputs,
    prepare_review_inputs,
    select_unfinished_records,
)
from rcr.utils.jsonl import load_jsonl, write_jsonl

SELECTED = Path("dataset/data/work/selection/selected.jsonl")
REWRITE_OUTPUT = Path("dataset/data/work/rewrite/rewrite_output.jsonl")
REVIEW_INPUT = Path("dataset/data/work/review/review_input.jsonl")
REVIEWED = Path("dataset/data/work/review/reviewed.jsonl")
POSITIVE_INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
POSITIVE_SETS = Path("dataset/data/work/positives/positive_sets.jsonl")
PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")
INDEX = Path("dataset/data/raw/metadata/index.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", choices=["review", "positives"])
    return parser.parse_args()


def load_optional(path: Path) -> list[dict]:
    return load_jsonl(path) if path.exists() else []


def main() -> None:
    args = parse_args()
    selected = load_jsonl(SELECTED)
    pair_data = json.loads(PAIR_DATA.read_text(encoding="utf-8"))
    index_lines = INDEX.read_text(encoding="utf-8").splitlines()

    if args.handoff == "review":
        completed_ids = {row["submission_id"] for row in load_optional(REVIEWED)}
        selected_ids = {row["submission_id"] for row in selected}
        current_rewrites = [
            row
            for row in load_jsonl(REWRITE_OUTPUT)
            if row["submission_id"] in selected_ids
        ]
        prepared = prepare_review_inputs(
            selected=selected,
            rewrite_outputs=current_rewrites,
            pair_data=pair_data,
            index_lines=index_lines,
        )
        outputs = merge_handoff_catalog(load_optional(REVIEW_INPUT), prepared)
        output_path = REVIEW_INPUT
    else:
        completed_ids = {row["submission_id"] for row in load_optional(POSITIVE_SETS)}
        reviewed = select_unfinished_records(
            load_jsonl(REVIEWED),
            completed_ids,
        )
        outputs = prepare_positive_set_inputs(
            selected=selected,
            reviewed=reviewed,
            pair_data=pair_data,
            index_lines=index_lines,
        )
        output_path = POSITIVE_INPUT

    write_jsonl(output_path, outputs)
    if args.handoff == "review":
        print(f"Review catalog: {len(outputs)}")
        print(f"Already reviewed: {len(completed_ids)}")
    else:
        print(f"Prepared: {len(outputs)}")
        print(f"Already completed: {len(completed_ids)}")
    print(f"Wrote handoff input to {output_path}")


if __name__ == "__main__":
    main()
