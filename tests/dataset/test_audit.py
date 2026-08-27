"""Tests for dataset audit utilities."""

from rcr.dataset.audit import audit_dataset


def test_audit_dataset() -> None:
    records = [
        {
            "submission_id": "sample-1",
            "annotation": {
                "caseType": "SINGLE",
                "qc": {
                    "pass": True,
                    "issues": [],
                },
            },
        },
        {
            "submission_id": "sample-2",
            "annotation": {
                "caseType": "RELATIONAL",
                "qc": {
                    "pass": False,
                    "issues": [
                        {
                            "code": "caption_format",
                            "message": "Invalid caption format.",
                        }
                    ],
                },
            },
        },
    ]

    report = audit_dataset(records)

    assert report["num_samples"] == 2

    assert report["case_types"] == {
        "RELATIONAL": 1,
        "SINGLE": 1,
    }

    assert report["qc"] == {
        "pass": 1,
        "fail": 1,
    }

    assert report["qc_issues"] == {
        "caption_format": 1,
    }

    assert report["failed_qc_submission_ids"] == [
        "sample-2",
    ]
