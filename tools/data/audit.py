"""Audit final labels; optionally measure detected identity coverage in a cache."""

import argparse
from pathlib import Path

from rcr.common.data import load_rcr_data
from rcr.common.io import load_jsonl, sha256_file, write_json, write_jsonl
from rcr.dataset.audit import audit_data, audit_detection
from rcr.dataset.positive_audit import (
    audit_full_positives,
    full_positive_review_queue,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", default="dataset/data/final")
    parser.add_argument("--image-root", default="dataset/data/raw/images")
    parser.add_argument("--positive-catalog", help="Original positive_set_input.jsonl")
    parser.add_argument("--positive-decisions", help="Original positive_sets.jsonl")
    parser.add_argument(
        "--review-queue", help="Write a NEW JSONL catalog for manual review"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 after saving on invalid/conflicting labels or unproven scope",
    )
    parser.add_argument(
        "--detection-cache",
        help="Shared pixel-box cache; no scene or FAFA features needed",
    )
    parser.add_argument("--output", default="runs/data_audit.json")
    args = parser.parse_args()
    data = load_rcr_data(args.final_dir, args.image_root)
    full = audit_full_positives(
        data,
        load_jsonl(args.positive_catalog) if args.positive_catalog else None,
        load_jsonl(args.positive_decisions) if args.positive_decisions else None,
    )
    report = {
        "full_positives": full,
        "dataset_version": data.manifest.get("version"),
        "input_sha256": {
            name: sha256_file(Path(args.final_dir) / name)
            for name in (
                "samples.jsonl",
                "images.jsonl",
                "gallery.jsonl",
                "head_boxes.jsonl",
                "manifest.json",
                "splits/train.txt",
                "splits/val.txt",
                "splits/test.txt",
            )
        },
        "review_source_sha256": {
            "catalog": sha256_file(args.positive_catalog)
            if args.positive_catalog
            else None,
            "decisions": sha256_file(args.positive_decisions)
            if args.positive_decisions
            else None,
        },
    }
    try:
        report.update(audit_data(data))
    except ValueError as error:
        report["structural_error"] = str(error)
    report["detection"] = None
    if args.detection_cache:
        from rcr.common.detection import DetectionCache

        report["detection"] = audit_detection(
            data,
            DetectionCache(args.detection_cache),
        )
    write_json(args.output, report)
    if args.review_queue:
        write_jsonl(args.review_queue, full_positive_review_queue(data, full))
    for split, values in full["splits"].items():
        values = values["summary"]
        print(
            f"{split}: {values['queries']} queries, "
            f"{values['annotation_issue_queries']} invalid, "
            f"{values['conflict_groups']} conflict groups, "
            f"review scope {values['scope_status_counts']}"
        )
    print(f"Saved {args.output}")
    if args.strict and (
        report.get("structural_error")
        or any(
            row["annotation_issues"]
            or row["review_source_issues"]
            or row["disputed_image_ids"]
            or row["scope_status"] != "complete_catalog"
            for split in full["splits"].values()
            for row in split["queries"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
