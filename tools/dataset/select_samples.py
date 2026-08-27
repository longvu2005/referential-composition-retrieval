"""Select samples that passed Stage 2 QC."""

from pathlib import Path

from rcr.dataset.selection import split_by_qc
from rcr.utils.jsonl import load_jsonl, write_jsonl

ANNOTATIONS = Path("dataset/data/raw/annotations/export_stage2.jsonl")

SELECTED_OUTPUT = Path("dataset/data/work/selection/selected.jsonl")

FAILED_OUTPUT = Path("dataset/data/work/selection/failed_qc.jsonl")


def main() -> None:
    records = load_jsonl(ANNOTATIONS)

    selected, failed = split_by_qc(records)

    write_jsonl(SELECTED_OUTPUT, selected)
    write_jsonl(FAILED_OUTPUT, failed)

    print(f"Total: {len(records)}")
    print(f"Selected: {len(selected)}")
    print(f"Failed QC: {len(failed)}")


if __name__ == "__main__":
    main()
