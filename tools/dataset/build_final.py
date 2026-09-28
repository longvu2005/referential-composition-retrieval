"""Build the final RCR dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcr.dataset.finalize import build_final_dataset
from rcr.dataset.gallery import build_indexed_gallery
from rcr.dataset.positives import check_positive_context, normalize_positive_sets
from rcr.utils.jsonl import load_jsonl, write_jsonl

SELECTED = Path("dataset/data/work/selection/selected.jsonl")
REVIEWED = Path("dataset/data/work/review/reviewed.jsonl")
POSITIVES = Path("dataset/data/work/positives/positive_sets.jsonl")
POSITIVE_INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
INDEX = Path("dataset/data/raw/metadata/index.txt")
PAIR_DATA = Path("dataset/data/raw/metadata/pair_data.json")
IMAGE_ROOT = Path("dataset/data/raw/images")
OUTPUT = Path("dataset/data/final")
PARTIAL_OUTPUT = Path("dataset/data/final_partial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_lines = INDEX.read_text(encoding="utf-8").splitlines()
    pair_data = json.loads(PAIR_DATA.read_text(encoding="utf-8"))
    selected = load_jsonl(SELECTED)
    reviewed = load_jsonl(REVIEWED)
    catalog = load_jsonl(POSITIVE_INPUT)
    positives = normalize_positive_sets(catalog, load_jsonl(POSITIVES))
    check_positive_context(catalog, positives, selected, reviewed)
    gallery_images = build_indexed_gallery(index_lines, IMAGE_ROOT)
    dataset = build_final_dataset(
        selected=selected,
        reviewed=reviewed,
        positive_sets=positives,
        gallery_images=gallery_images,
        index_lines=index_lines,
        version=args.version,
        pair_data=pair_data,
        allow_partial=args.allow_partial,
    )

    output = PARTIAL_OUTPUT if args.allow_partial else OUTPUT
    write_jsonl(output / "samples.jsonl", dataset["samples"])
    write_jsonl(output / "images.jsonl", dataset["images"])
    write_jsonl(output / "gallery.jsonl", dataset["gallery"])
    write_jsonl(output / "head_boxes.jsonl", dataset["head_boxes"])

    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, sample_ids in dataset["splits"].items():
        (split_dir / f"{name}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in sample_ids),
            encoding="utf-8",
        )

    (output / "manifest.json").write_text(
        json.dumps(dataset["manifest"], indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Built {len(dataset['samples'])} samples in {output}")


if __name__ == "__main__":
    main()
