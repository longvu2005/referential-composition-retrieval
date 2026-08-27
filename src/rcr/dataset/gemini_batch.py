"""Pure helpers for Gemini file-based rewrite batches."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

JsonObject = dict[str, Any]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_desc": {
            "type": "string",
            "description": "Single-line WHO/SELECT rewrite.",
        },
        "final_change": {
            "type": "string",
            "description": "Single-line WHAT/CHANGE rewrite.",
        },
    },
    "required": ["final_desc", "final_change"],
    "additionalProperties": False,
}


def build_model_contents(prompt: str, record: JsonObject) -> str:
    """Build the text payload shared by test and production requests."""

    model_input = {
        "case_type": record["case_type"],
        "subjects": record["subjects"],
        "pair_change": record["pair_change"],
    }

    return (
        f"{prompt.strip()}\n\n"
        "Input annotation:\n"
        f"{json.dumps(model_input, ensure_ascii=False)}"
    )


def build_batch_request(prompt: str, record: JsonObject) -> JsonObject:
    """Build one file-based Batch API request."""

    return {
        "key": record["submission_id"],
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_model_contents(prompt, record)}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": RESPONSE_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        },
    }


def select_pending_records(
    records: Iterable[JsonObject],
    completed_ids: set[str],
    in_flight_ids: set[str],
    case_type: str | None = None,
    limit: int | None = None,
) -> list[JsonObject]:
    """Select records that are neither completed nor already assigned."""

    pending = [
        record
        for record in records
        if record["submission_id"] not in completed_ids
        and record["submission_id"] not in in_flight_ids
    ]

    if case_type is not None:
        pending = [record for record in pending if record["case_type"] == case_type]

    if limit is not None:
        pending = pending[:limit]

    return pending


def chunk_records(
    records: list[JsonObject],
    chunk_size: int,
) -> list[list[JsonObject]]:
    """Split records into deterministic fixed-size chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    return [
        records[start : start + chunk_size]
        for start in range(0, len(records), chunk_size)
    ]


def merge_rewrite_outputs(
    records: Iterable[JsonObject],
    existing: Iterable[JsonObject],
    additions: Iterable[JsonObject],
) -> list[JsonObject]:
    """Merge rewrites in source order without replacing existing outputs."""

    merged = {row["submission_id"]: row for row in existing}
    for row in additions:
        merged.setdefault(row["submission_id"], row)

    return [
        merged[row["submission_id"]]
        for row in records
        if row["submission_id"] in merged
    ]


def extract_response_text(batch_result: JsonObject) -> str:
    """Extract the generated text from one successful batch result line."""

    if batch_result.get("error") is not None:
        raise ValueError(f"Gemini request failed: {batch_result['error']}")

    response = batch_result.get("response")
    if not isinstance(response, dict):
        raise ValueError("Batch result does not contain a response object.")

    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Batch response does not contain candidates.")

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("Batch response candidate must be a JSON object.")

    finish_reason = candidate.get("finishReason")
    detail = f" ({finish_reason})" if finish_reason else ""

    content = candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError(f"Batch response does not contain candidate content{detail}.")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError(f"Batch response candidate content has no parts{detail}.")

    texts = [
        part["text"] for part in parts if isinstance(part, dict) and "text" in part
    ]
    if not texts:
        raise ValueError("Batch response does not contain text output.")

    return "".join(texts)


def parse_model_output(batch_result: JsonObject) -> JsonObject:
    """Parse the structured JSON returned by Gemini for one request."""

    text = extract_response_text(batch_result)
    output = json.loads(text)

    if not isinstance(output, dict):
        raise ValueError("Gemini structured output must be a JSON object.")

    return output
