"""Tests for JSONL utilities."""

from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"

    records = [
        {"sample_id": "sample-1", "text": "first"},
        {"sample_id": "sample-2", "text": "second"},
    ]

    write_jsonl(path, records)

    assert load_jsonl(path) == records
