"""Audit the Stage 2 dataset."""

import json
from pathlib import Path

from rcr.dataset.audit import audit_dataset
from rcr.utils.jsonl import load_jsonl

ANNOTATIONS = Path("dataset/data/raw/annotations/export_stage2.jsonl")
OUTPUT = Path("dataset/reports/audit/stage2_audit.json")


def main() -> None:
    records = load_jsonl(ANNOTATIONS)
    report = audit_dataset(records)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Samples: {report['num_samples']}")
    print(f"QC pass: {report['qc']['pass']}")
    print(f"QC fail: {report['qc']['fail']}")
    print(f"Wrote audit report to {OUTPUT}")


if __name__ == "__main__":
    main()
