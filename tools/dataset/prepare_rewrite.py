"""Prepare structured inputs for annotation rewriting."""

from pathlib import Path

from rcr.dataset.rewrite import prepare_rewrite_inputs
from rcr.utils.jsonl import load_jsonl, write_jsonl

INPUT = Path("dataset/data/work/selection/selected.jsonl")
OUTPUT = Path("dataset/data/work/rewrite/rewrite_input.jsonl")


def main() -> None:
    records = load_jsonl(INPUT)
    rewrite_inputs = prepare_rewrite_inputs(records)

    write_jsonl(OUTPUT, rewrite_inputs)

    print(f"Input: {len(records)}")
    print(f"Prepared: {len(rewrite_inputs)}")
    print(f"Wrote rewrite inputs to {OUTPUT}")


if __name__ == "__main__":
    main()
