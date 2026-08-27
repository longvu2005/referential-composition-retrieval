"""Dataset audit utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

JsonObject = dict[str, Any]


def audit_dataset(records: Iterable[JsonObject]) -> JsonObject:
    """Summarize the current Stage 2 dataset without modifying it."""

    records = list(records)

    case_types = Counter()
    qc_pass = 0
    qc_fail = 0
    qc_issues = Counter()
    failed_qc_submission_ids = []

    for record in records:
        annotation = record["annotation"]

        case_types[annotation["caseType"]] += 1

        qc = annotation["qc"]

        if qc["pass"]:
            qc_pass += 1
        else:
            qc_fail += 1
            failed_qc_submission_ids.append(record["submission_id"])

        for issue in qc.get("issues", []):
            qc_issues[issue["code"]] += 1

    return {
        "num_samples": len(records),
        "case_types": dict(sorted(case_types.items())),
        "qc": {
            "pass": qc_pass,
            "fail": qc_fail,
        },
        "qc_issues": dict(sorted(qc_issues.items())),
        "failed_qc_submission_ids": failed_qc_submission_ids,
    }
