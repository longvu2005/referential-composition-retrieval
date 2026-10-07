"""Validate and copy the already QC-approved Stage 2 export."""

import json
from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.selection import select_samples

ANNOTATIONS = Path("dataset/data/raw/annotations/export_stage2.jsonl")
PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")

SELECTED_OUTPUT = Path("dataset/data/work/selection/selected.jsonl")


def main() -> None:
    records = load_jsonl(ANNOTATIONS)

    pair_data = json.loads(PAIR_DATA.read_text(encoding="utf-8"))
    selected = select_samples(records, pair_data)

    write_jsonl(SELECTED_OUTPUT, selected)

    print(f"Total: {len(records)}")
    print(f"Selected: {len(selected)}")


if __name__ == "__main__":
    main()
