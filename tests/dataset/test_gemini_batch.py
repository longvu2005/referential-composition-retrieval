"""Tests for pure Gemini batch helpers."""

import json

import pytest

from rcr.dataset.gemini_batch import (
    RESPONSE_SCHEMA,
    build_batch_request,
    build_model_contents,
    chunk_records,
    extract_response_text,
    merge_rewrite_outputs,
    parse_model_output,
    select_pending_records,
)

PROMPT = "Rewrite the annotation."
RECORD = {
    "submission_id": "sample-1",
    "case_type": "SINGLE",
    "subjects": [
        {
            "subject_id": 1,
            "description": "the man wearing a black jacket",
            "change": "is holding a cup",
        }
    ],
    "pair_change": None,
}


def test_build_model_contents_contains_only_model_fields() -> None:
    record = {**RECORD, "risk_tags": ["test-only"]}

    contents = build_model_contents(PROMPT, record)

    payload = json.loads(contents.split("Input annotation:\n", 1)[1])
    assert payload == {
        "case_type": "SINGLE",
        "subjects": RECORD["subjects"],
        "pair_change": None,
    }


def test_build_batch_request_uses_shared_generation_contract() -> None:
    request = build_batch_request(PROMPT, RECORD)

    assert request["key"] == "sample-1"
    generation_config = request["request"]["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseJsonSchema"] == RESPONSE_SCHEMA
    assert RESPONSE_SCHEMA["required"] == ["final_desc", "final_change"]
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "low"}


def test_select_pending_records_skips_completed_and_in_flight() -> None:
    records = [
        {**RECORD, "submission_id": "done"},
        {**RECORD, "submission_id": "running"},
        {**RECORD, "submission_id": "pending"},
    ]

    pending = select_pending_records(
        records,
        completed_ids={"done"},
        in_flight_ids={"running"},
    )

    assert [record["submission_id"] for record in pending] == ["pending"]


def test_select_pending_records_applies_filter_before_limit() -> None:
    records = [
        {**RECORD, "submission_id": "single-1"},
        {**RECORD, "submission_id": "rel-1", "case_type": "RELATIONAL"},
        {**RECORD, "submission_id": "rel-2", "case_type": "RELATIONAL"},
    ]

    pending = select_pending_records(
        records,
        completed_ids=set(),
        in_flight_ids=set(),
        case_type="RELATIONAL",
        limit=1,
    )

    assert [record["submission_id"] for record in pending] == ["rel-1"]


def test_chunk_records_is_deterministic() -> None:
    records = [{**RECORD, "submission_id": f"sample-{index}"} for index in range(5)]

    chunks = chunk_records(records, chunk_size=2)

    assert [[record["submission_id"] for record in chunk] for chunk in chunks] == [
        ["sample-0", "sample-1"],
        ["sample-2", "sample-3"],
        ["sample-4"],
    ]


def test_chunk_records_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_records([RECORD], chunk_size=0)


def test_merge_rewrite_outputs_preserves_existing_values() -> None:
    records = [
        {**RECORD, "submission_id": "old"},
        {**RECORD, "submission_id": "new"},
    ]
    existing = [{"submission_id": "old", "value": "keep"}]
    additions = [
        {"submission_id": "old", "value": "replace"},
        {"submission_id": "new", "value": "add"},
    ]

    merged = merge_rewrite_outputs(records, existing, additions)

    assert merged == [
        {"submission_id": "old", "value": "keep"},
        {"submission_id": "new", "value": "add"},
    ]


def test_parse_model_output_from_batch_result() -> None:
    payload = {
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
    }
    result = {
        "key": "sample-1",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(payload)}],
                    }
                }
            ]
        },
    }

    assert parse_model_output(result) == payload


def test_extract_response_text_reports_request_error() -> None:
    with pytest.raises(ValueError, match="Gemini request failed"):
        extract_response_text(
            {"key": "sample-1", "error": {"code": 429, "message": "quota"}}
        )


def test_extract_response_text_reports_finish_reason_without_content() -> None:
    result = {
        "key": "sample-1",
        "response": {
            "candidates": [
                {
                    "content": {},
                    "finishReason": "PROHIBITED_CONTENT",
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="PROHIBITED_CONTENT"):
        extract_response_text(result)
