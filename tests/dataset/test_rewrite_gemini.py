"""Tests for incremental Gemini rewrite state handling."""

import json
import runpy
from pathlib import Path

from rcr.utils.jsonl import write_jsonl

MODULE = runpy.run_path("tools/dataset/rewrite_gemini.py")
collected_chunk_outputs = MODULE["collected_chunk_outputs"]
completed_ids = MODULE["completed_ids"]
process_batch_results = MODULE["process_batch_results"]


def _make_chunk(work_dir: Path, number: int) -> Path:
    chunk_dir = work_dir / "chunks" / f"chunk_{number:06d}"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "manifest.json").write_text(
        json.dumps({"chunk_id": chunk_dir.name}),
        encoding="utf-8",
    )
    return chunk_dir


def test_completed_ids_includes_sample_errors(tmp_path: Path) -> None:
    work_dir = tmp_path / "rewrite"
    chunk_dir = _make_chunk(work_dir, 1)
    write_jsonl(
        chunk_dir / "output.jsonl",
        [
            {
                "submission_id": "success",
                "final_desc": "Identify Subject 1 as the man",
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling"
                ),
            }
        ],
    )
    write_jsonl(
        chunk_dir / "errors.jsonl",
        [{"submission_id": "blocked", "error": "PROHIBITED_CONTENT"}],
    )

    assert completed_ids(tmp_path / "missing.jsonl", work_dir) == {
        "success",
        "blocked",
    }


def test_collected_chunk_outputs_keeps_error_samples_as_empty_rewrites(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "rewrite"
    chunk_dir = _make_chunk(work_dir, 1)
    write_jsonl(
        chunk_dir / "output.jsonl",
        [
            {
                "submission_id": "success",
                "final_desc": "Identify Subject 1 as the man",
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling"
                ),
            }
        ],
    )
    write_jsonl(
        chunk_dir / "errors.jsonl",
        [
            {"submission_id": "success", "error": "stale error"},
            {"submission_id": "blocked", "error": "PROHIBITED_CONTENT"},
        ],
    )

    outputs = {row["submission_id"]: row for row in collected_chunk_outputs(work_dir)}

    assert outputs["success"]["final_desc"] == "Identify Subject 1 as the man"
    assert outputs["blocked"] == {
        "submission_id": "blocked",
        "final_desc": None,
        "final_change": None,
    }


def test_process_batch_results_keeps_sample_failures_for_review() -> None:
    source_records = [
        {
            "submission_id": "success",
            "case_type": "SINGLE",
            "subjects": [
                {
                    "subject_id": 1,
                    "description": "the man",
                    "change": "is smiling",
                }
            ],
            "pair_change": None,
        },
        {
            "submission_id": "blocked",
            "case_type": "SINGLE",
            "subjects": [
                {
                    "subject_id": 1,
                    "description": "the child",
                    "change": "is outside",
                }
            ],
            "pair_change": None,
        },
    ]
    success_output = {
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
    }
    results = [
        {
            "key": "success",
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(success_output)}]}}
                ]
            },
        },
        {
            "key": "blocked",
            "response": {
                "candidates": [
                    {
                        "content": {},
                        "finishReason": "PROHIBITED_CONTENT",
                    }
                ]
            },
        },
    ]
    raw_bytes = ("\n".join(json.dumps(result) for result in results) + "\n").encode()

    outputs, errors = process_batch_results(source_records, raw_bytes)

    assert outputs == [
        {
            "submission_id": "success",
            "final_desc": "Identify Subject 1 as the man",
            "final_change": ("then retrieve target images where Subject 1 is smiling"),
        },
        {
            "submission_id": "blocked",
            "final_desc": None,
            "final_change": None,
        },
    ]
    assert errors[0]["submission_id"] == "blocked"
    assert "PROHIBITED_CONTENT" in errors[0]["error"]
