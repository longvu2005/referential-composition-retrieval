"""Audit final labels; optionally measure detected identity coverage in a cache."""

import argparse

from rcr.common.data import load_rcr_data
from rcr.common.io import write_json
from rcr.dataset.audit import audit_data, audit_detection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", default="dataset/data/final")
    parser.add_argument("--cache", help="Optional existing proposed cache directory")
    parser.add_argument(
        "--dino-cache", help="DINO source for a separate FAFA person cache"
    )
    parser.add_argument("--output", default="runs/data_audit.json")
    args = parser.parse_args()
    data = load_rcr_data(args.final_dir)
    report = audit_data(data)
    report["detection"] = None
    if args.cache:
        from rcr.proposed.cache.store import GalleryCache

        report["detection"] = audit_detection(
            data, GalleryCache(args.cache, scene_root=args.dino_cache)
        )
    write_json(args.output, report)
    for split, values in report["splits"].items():
        print(
            f"{split}: {values['queries']} queries, "
            f"{values['conflict_groups']} label conflict groups"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
