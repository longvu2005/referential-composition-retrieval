"""Prepare cumulative inputs for the two human labeling handoffs."""

from __future__ import annotations

import argparse
from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.gallery import build_indexed_gallery
from rcr.dataset.handoff import (
    merge_handoff_catalog,
    prepare_positive_set_inputs,
    prepare_review_inputs,
)

SELECTED = Path("dataset/data/work/selection/selected.jsonl")
REWRITE_OUTPUT = Path("dataset/data/work/rewrite/rewrite_output.jsonl")
REVIEW_INPUT = Path("dataset/data/work/review/review_input.jsonl")
REVIEWED = Path("dataset/data/work/review/reviewed.jsonl")
POSITIVE_INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
POSITIVE_SETS = Path("dataset/data/work/positives/positive_sets.jsonl")
INDEX = Path("dataset/data/raw/metadata/index.txt")
IMAGE_ROOT = Path("dataset/data/raw/images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", choices=["review", "positives"])
    parser.add_argument(
        "--clip-rerank",
        action="store_true",
        help="rank only NEW positive tasks with CLIP; keep existing tasks unchanged",
    )
    args = parser.parse_args()
    if args.clip_rerank and args.handoff != "positives":
        parser.error("--clip-rerank is only valid for positives")
    return args


def load_optional(path: Path) -> list[dict]:
    return load_jsonl(path) if path.exists() else []


def main() -> None:
    args = parse_args()
    selected = load_jsonl(SELECTED)
    if args.handoff == "review":
        completed = load_optional(REVIEWED)
        prepared = prepare_review_inputs(
            selected=selected,
            rewrite_outputs=load_jsonl(REWRITE_OUTPUT),
        )
        outputs = merge_handoff_catalog(load_optional(REVIEW_INPUT), prepared)
        output_path = REVIEW_INPUT
    else:
        index_lines = INDEX.read_text(encoding="utf-8").splitlines()
        completed = load_optional(POSITIVE_SETS)
        existing = load_optional(POSITIVE_INPUT)
        known = {row["sample_id"] for row in existing}
        reviewed = load_jsonl(REVIEWED)
        review_by_id = {row["sample_id"]: row for row in reviewed}
        for row in existing:
            review = review_by_id[row["sample_id"]]
            tracked = ("case_type", "subjects", "final_desc", "final_change")
            if any(row[k] != review[k] for k in tracked):
                raise ValueError(
                    f"{row['sample_id']}: reviewed text/subjects differ from the "
                    "existing positive catalog; inspect its labels before refreshing"
                )
        new_reviews = [row for row in reviewed if row["sample_id"] not in known]
        if new_reviews:
            gallery_images = build_indexed_gallery(index_lines, IMAGE_ROOT)
            ranker = None
            if args.clip_rerank:
                from rcr.dataset.clip_rerank import ClipChangeRanker

                ranker = ClipChangeRanker()
            prepared = prepare_positive_set_inputs(
                selected=selected,
                reviewed=new_reviews,
                gallery_images=gallery_images,
                index_lines=index_lines,
                candidate_ranker=ranker,
            )
        else:
            prepared = []
        outputs = merge_handoff_catalog(existing, prepared)
        output_path = POSITIVE_INPUT

    write_jsonl(output_path, outputs)
    print(f"Catalog: {len(outputs)}")
    print(f"Completed: {len(completed)}")
    print(f"Wrote handoff input to {output_path}")


if __name__ == "__main__":
    main()
