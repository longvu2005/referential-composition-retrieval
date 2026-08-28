"""Utilities for preparing and validating annotation rewrites."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

JsonObject = dict[str, Any]

SELECT_PREFIX = "Identify "
CHANGE_PREFIX = "then retrieve target images where "
SUBJECT_RE = re.compile(r"\bSubject\s+(\d+)\b")
TERMINAL_PUNCTUATION = ".;:!?"
CASE_TYPES = ("SINGLE", "MULTI", "RELATIONAL")
CASE_SUBJECT_IDS = {
    "SINGLE": (1,),
    "MULTI": (1, 2),
    "RELATIONAL": (1, 2),
}


def prepare_rewrite_inputs(records: Iterable[JsonObject]) -> list[JsonObject]:
    """Prepare structured annotations for rewriting."""

    outputs = []

    for record in records:
        annotation = record["annotation"]

        subjects = [
            {
                "subject_id": subject["subjectId"],
                "description": subject["desc"]["final"],
                "change": subject["change"]["final"],
            }
            for subject in sorted(
                annotation["subjects"],
                key=lambda x: x["subjectId"],
            )
        ]

        pair_change = annotation.get("pairChange")
        if pair_change is not None:
            pair_change = {
                "subject_1_id": subjects[0]["subject_id"],
                "subject_2_id": subjects[1]["subject_id"],
                "relation": pair_change["final"],
            }

        outputs.append(
            {
                "submission_id": record["submission_id"],
                "case_type": annotation["caseType"],
                "subjects": subjects,
                "pair_change": pair_change,
            }
        )

    return outputs


def _field(output: JsonObject, name: str) -> str:
    value = output.get(name)

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")

    value = value.strip()

    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exactly one non-empty line.")

    return value


def _subject_ids(text: str) -> list[int]:
    return [int(x) for x in SUBJECT_RE.findall(text)]


def _validate_select(text: str, subject_ids: list[int]) -> None:
    parts = re.split(
        r"\s+and\s+(?=Subject\s+\d+\s+as\s+)",
        text.removeprefix(SELECT_PREFIX),
    )

    expected = [f"Subject {subject_id} as " for subject_id in subject_ids]

    if len(parts) != len(expected):
        raise ValueError("final_desc must have one segment per Subject.")

    for part, prefix in zip(parts, expected, strict=True):
        if not part.startswith(prefix) or not part[len(prefix) :].strip():
            raise ValueError(
                "final_desc must use `Subject N as <description>` for every Subject."
            )


def _validate_output(
    output: JsonObject,
    subject_ids: list[int],
    required_change_ids: set[int],
) -> tuple[str, str]:
    if not isinstance(output, dict):
        raise ValueError("model output must be a JSON object.")

    if set(output) != {"final_desc", "final_change"}:
        raise ValueError(
            "model output must contain exactly final_desc and final_change."
        )

    final_desc = _field(output, "final_desc")
    final_change = _field(output, "final_change")

    if not final_desc.startswith(SELECT_PREFIX):
        raise ValueError(f"final_desc must begin with {SELECT_PREFIX!r}.")

    if not final_change.startswith(CHANGE_PREFIX):
        raise ValueError(f"final_change must begin with {CHANGE_PREFIX!r}.")

    if final_desc.endswith(tuple(TERMINAL_PUNCTUATION)):
        raise ValueError("final_desc must not end with punctuation.")

    if final_change.endswith(tuple(TERMINAL_PUNCTUATION)):
        raise ValueError("final_change must not end with punctuation.")

    if _subject_ids(final_desc) != subject_ids:
        raise ValueError(
            "final_desc must mention every Subject exactly once in input order."
        )

    _validate_select(final_desc, subject_ids)

    if set(_subject_ids(final_change)) != required_change_ids:
        raise ValueError(
            "final_change must mention exactly the Subjects required by the "
            "current structure."
        )

    return final_desc, final_change


def subject_ids_for_case(case_type: str) -> list[int]:
    """Return the canonical Subject IDs for a reviewed case type."""

    try:
        return list(CASE_SUBJECT_IDS[case_type])
    except KeyError as error:
        raise ValueError(f"invalid case_type {case_type!r}") from error


def validate_review_output(
    case_type: str,
    output: JsonObject,
) -> tuple[str, str]:
    """Validate human-reviewed text against the reviewed case structure."""

    subject_ids = subject_ids_for_case(case_type)
    return _validate_output(output, subject_ids, set(subject_ids))


def validate_rewrite_output(
    source: JsonObject,
    output: JsonObject,
) -> tuple[str, str]:
    """Validate one model rewrite against its source annotation."""

    subject_ids = [subject["subject_id"] for subject in source["subjects"]]
    required_change_ids = {
        subject["subject_id"]
        for subject in source["subjects"]
        if subject["change"] is not None
    }

    if pair_change := source.get("pair_change"):
        required_change_ids |= {
            pair_change["subject_1_id"],
            pair_change["subject_2_id"],
        }

    return _validate_output(output, subject_ids, required_change_ids)


def build_rewrite_result(
    source: JsonObject,
    output: JsonObject,
) -> JsonObject:
    """Build one validated rewrite record."""

    final_desc, final_change = validate_rewrite_output(source, output)

    return {
        "submission_id": source["submission_id"],
        "final_desc": final_desc,
        "final_change": final_change,
    }
