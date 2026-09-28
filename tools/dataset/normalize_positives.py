"""Check or normalize positive ordering without changing any positive set."""

import argparse
import os
from pathlib import Path

from rcr.dataset.positives import normalize_positive_sets
from rcr.utils.jsonl import load_jsonl, write_jsonl

CATALOG = Path("dataset/data/work/positives/positive_set_input.jsonl")
OUTPUT = Path("dataset/data/work/positives/positive_sets.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="save canonical ordering")
    args = parser.parse_args()
    original = load_jsonl(OUTPUT)
    normalized = normalize_positive_sets(load_jsonl(CATALOG), original)
    by_id = {row["sample_id"]: row for row in original}
    changed = sum(row != by_id[row["sample_id"]] for row in normalized)
    print(f"Decisions: {len(original)}; reordered positive lists: {changed}")
    if args.write and original != normalized:
        temporary = OUTPUT.with_name(f".{OUTPUT.name}.normalize.tmp")
        write_jsonl(temporary, normalized)
        os.replace(temporary, OUTPUT)
        print("Saved canonical ordering; rebuild final/partial before training.")


if __name__ == "__main__":
    main()
