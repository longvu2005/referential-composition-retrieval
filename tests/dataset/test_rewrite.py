"""Tests for rewrite preparation and output validation."""

import pytest

from rcr.dataset.rewrite import (
    build_rewrite_result,
    prepare_rewrite_inputs,
    validate_rewrite_output,
)


def _single_source(change: str | None = "is holding a cup") -> dict:
    return {
        "submission_id": "sample-1",
        "case_type": "SINGLE",
        "subjects": [
            {
                "subject_id": 1,
                "description": "the man wearing a black jacket",
                "change": change,
            }
        ],
        "pair_change": None,
    }


def _single_output() -> dict:
    return {
        "final_desc": "Identify Subject 1 as the man wearing a black jacket",
        "final_change": "then retrieve target images where Subject 1 is holding a cup",
    }


def test_prepare_rewrite_inputs() -> None:
    records = [
        {
            "submission_id": "sample-1",
            "annotation": {
                "caseType": "RELATIONAL",
                "subjects": [
                    {
                        "subjectId": 2,
                        "desc": {"final": "the woman wearing a blue shirt"},
                        "change": {"final": "is smiling"},
                    },
                    {
                        "subjectId": 1,
                        "desc": {"final": "the man wearing a black jacket"},
                        "change": {"final": "is holding a cup"},
                    },
                ],
                "pairChange": {"final": "is standing next to"},
            },
        }
    ]

    outputs = prepare_rewrite_inputs(records)

    assert outputs == [
        {
            "submission_id": "sample-1",
            "case_type": "RELATIONAL",
            "subjects": [
                {
                    "subject_id": 1,
                    "description": "the man wearing a black jacket",
                    "change": "is holding a cup",
                },
                {
                    "subject_id": 2,
                    "description": "the woman wearing a blue shirt",
                    "change": "is smiling",
                },
            ],
            "pair_change": {
                "subject_1_id": 1,
                "subject_2_id": 2,
                "relation": "is standing next to",
            },
        }
    ]


def test_validate_rewrite_output_accepts_contract() -> None:
    assert validate_rewrite_output(_single_source(), _single_output()) == (
        "Identify Subject 1 as the man wearing a black jacket",
        "then retrieve target images where Subject 1 is holding a cup",
    )


def test_validate_rewrite_output_rejects_extra_fields() -> None:
    output = {**_single_output(), "final_instruction": "legacy"}

    with pytest.raises(ValueError, match="exactly final_desc and final_change"):
        validate_rewrite_output(_single_source(), output)


def test_validate_rewrite_output_requires_string_fields() -> None:
    output = _single_output()
    output["final_desc"] = None
    with pytest.raises(ValueError, match="final_desc must be a string"):
        validate_rewrite_output(_single_source(), output)

    output = _single_output()
    output["final_change"] = None
    with pytest.raises(ValueError, match="final_change must be a string"):
        validate_rewrite_output(_single_source(), output)


def test_validate_rewrite_output_requires_single_line_fields() -> None:
    output = _single_output()
    output["final_desc"] += "\nextra text"

    with pytest.raises(ValueError, match="exactly one non-empty line"):
        validate_rewrite_output(_single_source(), output)


def test_validate_rewrite_output_requires_prefixes() -> None:
    output = _single_output()
    output["final_desc"] = "Find Subject 1 as the man wearing a black jacket"

    with pytest.raises(ValueError, match="final_desc must begin"):
        validate_rewrite_output(_single_source(), output)


def test_validate_rewrite_output_requires_who_subject_order_once() -> None:
    source = {
        "submission_id": "sample-1",
        "case_type": "MULTI",
        "subjects": [
            {"subject_id": 1, "description": "the man", "change": "is smiling"},
            {"subject_id": 2, "description": "the woman", "change": "is waving"},
        ],
        "pair_change": None,
    }
    output = {
        "final_desc": "Identify Subject 2 as the woman and Subject 1 as the man",
        "final_change": (
            "then retrieve target images where Subject 1 is smiling "
            "and Subject 2 is waving"
        ),
    }

    with pytest.raises(ValueError, match="every Subject exactly once in input order"):
        validate_rewrite_output(source, output)


def test_validate_rewrite_output_respects_null_change() -> None:
    source = {
        "submission_id": "sample-1",
        "case_type": "MULTI",
        "subjects": [
            {"subject_id": 1, "description": "the man", "change": "is smiling"},
            {"subject_id": 2, "description": "the woman", "change": None},
        ],
        "pair_change": None,
    }
    output = {
        "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
        "final_change": (
            "then retrieve target images where Subject 1 is smiling "
            "and Subject 2 is waving"
        ),
    }

    with pytest.raises(ValueError, match="exactly the Subjects required"):
        validate_rewrite_output(source, output)


def test_validate_rewrite_output_requires_pair_participants() -> None:
    source = {
        "submission_id": "sample-1",
        "case_type": "RELATIONAL",
        "subjects": [
            {"subject_id": 1, "description": "the man", "change": None},
            {"subject_id": 2, "description": "the woman", "change": None},
        ],
        "pair_change": {
            "subject_1_id": 1,
            "subject_2_id": 2,
            "relation": "is standing to the left of",
        },
    }
    output = {
        "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
        "final_change": (
            "then retrieve target images where Subject 1 is standing on the left"
        ),
    }

    with pytest.raises(ValueError, match="exactly the Subjects required"):
        validate_rewrite_output(source, output)


def test_validate_rewrite_output_rejects_terminal_punctuation() -> None:
    output = _single_output()
    output["final_desc"] += "."

    with pytest.raises(ValueError, match="final_desc must not end"):
        validate_rewrite_output(_single_source(), output)


def test_build_rewrite_result_keeps_only_join_key_and_rewrite_fields() -> None:
    result = build_rewrite_result(_single_source(), _single_output())

    assert result == {
        "submission_id": "sample-1",
        "final_desc": "Identify Subject 1 as the man wearing a black jacket",
        "final_change": "then retrieve target images where Subject 1 is holding a cup",
    }
