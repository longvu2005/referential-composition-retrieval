"""Validate and copy the already QC-approved Stage 2 export."""

import argparse
import json
from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.selection import select_samples

ANNOTATIONS = Path("dataset/data/raw/annotations/export_stage2.jsonl")
PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")

SELECTED_OUTPUT = Path("dataset/data/work/selection/selected.jsonl")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--pair-data", type=Path, default=PAIR_DATA)
    parser.add_argument("--output", type=Path, default=SELECTED_OUTPUT)
    args = parser.parse_args(argv)
    records = load_jsonl(args.annotations)

    pair_data = json.loads(args.pair_data.read_text(encoding="utf-8"))
    selected = select_samples(records, pair_data)

    write_jsonl(args.output, selected)

    print(f"Total: {len(records)}")
    print(f"Selected: {len(selected)}")


if __name__ == "__main__":
    main()
