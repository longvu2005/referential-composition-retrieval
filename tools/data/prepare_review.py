"""Prepare the cumulative human-review catalog from accepted Stage 2 rows."""

import argparse
from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.handoff import merge_handoff_catalog, prepare_review_inputs

SELECTED = Path("dataset/data/work/selection/selected.jsonl")
REVIEW_INPUT = Path("dataset/data/work/review/review_input.jsonl")
REVIEWED = Path("dataset/data/work/review/reviewed.jsonl")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=SELECTED)
    parser.add_argument("--output", type=Path, default=REVIEW_INPUT)
    parser.add_argument("--reviewed", type=Path, default=REVIEWED)
    args = parser.parse_args(argv)

    prepared = prepare_review_inputs(load_jsonl(args.selected))
    existing = load_jsonl(args.output) if args.output.exists() else []
    outputs = merge_handoff_catalog(existing, prepared)
    completed = load_jsonl(args.reviewed) if args.reviewed.exists() else []
    # Completion belongs to reviewed.jsonl; preparing tasks never writes labels.
    write_jsonl(args.output, outputs)
    print(f"Catalog: {len(outputs)}")
    print(f"Completed: {len(completed)}")
    print(f"Wrote review input to {args.output}")


if __name__ == "__main__":
    main()
