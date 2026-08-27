"""Build the final RCR dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcr.dataset.finalize import build_final_dataset
from rcr.utils.jsonl import load_jsonl, write_jsonl

SELECTED = Path("dataset/data/work/selection/selected.jsonl")
REVIEWED = Path("dataset/data/work/review/reviewed.jsonl")
POSITIVES = Path("dataset/data/work/positives/positive_sets.jsonl")
PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")
INDEX = Path("dataset/data/raw/metadata/index.txt")
OUTPUT = Path("dataset/data/final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pair_data = json.loads(PAIR_DATA.read_text(encoding="utf-8"))
    dataset = build_final_dataset(
        selected=load_jsonl(SELECTED),
        reviewed=load_jsonl(REVIEWED),
        positive_sets=load_jsonl(POSITIVES),
        pair_data=pair_data,
        index_lines=INDEX.read_text(encoding="utf-8").splitlines(),
        version=args.version,
    )

    write_jsonl(OUTPUT / "samples.jsonl", dataset["samples"])
    write_jsonl(OUTPUT / "images.jsonl", dataset["images"])
    write_jsonl(OUTPUT / "gallery.jsonl", dataset["gallery"])
    write_jsonl(OUTPUT / "head_boxes.jsonl", dataset["head_boxes"])

    split_dir = OUTPUT / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, sample_ids in dataset["splits"].items():
        (split_dir / f"{name}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in sample_ids),
            encoding="utf-8",
        )

    (OUTPUT / "manifest.json").write_text(
        json.dumps(dataset["manifest"], indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Built {len(dataset['samples'])} samples in {OUTPUT}")


if __name__ == "__main__":
    main()
