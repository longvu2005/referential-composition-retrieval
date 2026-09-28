"""Tests for canonical Stage 2 text projection and validation."""

import pytest

from rcr.dataset.rewrite import prepare_rewrite_inputs, validate_review_output


def _row() -> dict:
    return {
        "sample_id": "train__1_1__1_2",
        "case_type": "RELATIONAL",
        "subjects": [
            {"subject_id": 1, "identity_ids": ["7"]},
            {"subject_id": 2, "identity_ids": ["8"]},
        ],
        "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
        "final_change": (
            "then retrieve target images where Subject 1 is beside Subject 2"
        ),
        "final_instruction": (
            "Identify Subject 1 as the man and Subject 2 as the woman; "
            "then retrieve target images where Subject 1 is beside Subject 2."
        ),
    }


def test_prepare_rewrite_inputs_uses_accepted_stage2_text() -> None:
    record = _row()
    assert prepare_rewrite_inputs([record]) == [
        {
            k: record[k]
            for k in (
                "sample_id",
                "case_type",
                "subjects",
                "final_desc",
                "final_change",
            )
        }
    ]


def test_validate_review_output_accepts_two_subjects() -> None:
    record = _row()
    assert validate_review_output(
        record["case_type"],
        {k: record[k] for k in ("final_desc", "final_change")},
    ) == (record["final_desc"], record["final_change"])


def test_validate_review_output_rejects_old_schema_fields() -> None:
    record = _row()
    with pytest.raises(ValueError, match="exactly final_desc and final_change"):
        validate_review_output(
            record["case_type"],
            {
                "final_desc": record["final_desc"],
                "final_change": record["final_change"],
                "pair_change": None,
            },
        )


def test_validate_review_output_rejects_missing_or_multiline_text() -> None:
    record = _row()
    with pytest.raises(ValueError, match="final_desc must be a string"):
        validate_review_output(
            "RELATIONAL", {"final_desc": None, "final_change": record["final_change"]}
        )
    with pytest.raises(ValueError, match="exactly one non-empty line"):
        validate_review_output(
            "RELATIONAL",
            {
                "final_desc": record["final_desc"] + "\nextra",
                "final_change": record["final_change"],
            },
        )


def test_validate_review_output_rejects_wrong_subject_order_or_relation() -> None:
    record = _row()
    with pytest.raises(ValueError, match="every Subject exactly once in input order"):
        validate_review_output(
            "RELATIONAL",
            {
                "final_desc": "Identify Subject 2 as the woman and Subject 1 as the man",
                "final_change": record["final_change"],
            },
        )
    with pytest.raises(ValueError, match="exactly the Subjects required"):
        validate_review_output(
            "RELATIONAL",
            {
                "final_desc": record["final_desc"],
                "final_change": "then retrieve target images where Subject 1 is smiling",
            },
        )


def test_validate_review_output_rejects_terminal_punctuation() -> None:
    record = _row()
    with pytest.raises(ValueError, match="final_desc must not end"):
        validate_review_output(
            "RELATIONAL",
            {
                "final_desc": record["final_desc"] + ".",
                "final_change": record["final_change"],
            },
        )
