"""Tests for dataset selection utilities."""

from rcr.dataset.selection import split_by_qc


def test_split_by_qc() -> None:
    records = [
        {
            "submission_id": "sample-1",
            "annotation": {
                "qc": {"pass": True},
            },
        },
        {
            "submission_id": "sample-2",
            "annotation": {
                "qc": {"pass": False},
            },
        },
        {
            "submission_id": "sample-3",
            "annotation": {
                "qc": {"pass": True},
            },
        },
    ]

    selected, failed = split_by_qc(records)

    assert [record["submission_id"] for record in selected] == [
        "sample-1",
        "sample-3",
    ]

    assert [record["submission_id"] for record in failed] == [
        "sample-2",
    ]
