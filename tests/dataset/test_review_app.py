"""Tests for the minimal identity-synchronized rewrite review UI backend."""

import pytest

import labelstudio.review.app as review_app
from labelstudio.review.app import (
    ReviewState,
    _image_relative_path,
    build_task_payload,
    validate_submission,
)
from rcr.utils.jsonl import load_jsonl, write_jsonl

BOXES = {
    "1_1": [
        {
            "image_id": "1_1",
            "label": "7",
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
        },
        {
            "image_id": "1_1",
            "label": "8",
            "x": 0.5,
            "y": 0.2,
            "width": 0.2,
            "height": 0.3,
        },
    ],
    "1_2": [
        {
            "image_id": "1_2",
            "label": "7",
            "x": 0.2,
            "y": 0.1,
            "width": 0.2,
            "height": 0.4,
        },
        {
            "image_id": "1_2",
            "label": "8",
            "x": 0.6,
            "y": 0.2,
            "width": 0.2,
            "height": 0.3,
        },
    ],
}


def _source() -> dict:
    return {
        "sample_id": "s1",
        "case_type": "INDIVIDUAL",
        "query_image_id": "1_1",
        "query_image_url": (
            "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
        ),
        "target_image_id": "1_2",
        "target_image_url": (
            "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_2.jpg"
        ),
        "candidate_identity_ids": ["7", "8"],
        "subjects": [
            {
                "subject_id": 1,
                "identity_ids": ["7"],
                "description": "the man",
                "change": "is smiling",
            }
        ],
        "query_boxes": [
            {
                "identity_id": box["label"],
                **{k: box[k] for k in ("x", "y", "width", "height")},
            }
            for box in BOXES["1_1"]
        ],
        "target_boxes": [
            {
                "identity_id": box["label"],
                **{k: box[k] for k in ("x", "y", "width", "height")},
            }
            for box in BOXES["1_2"]
        ],
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
    }


def test_build_task_payload_uses_same_identity_candidates_on_both_images() -> None:
    task = build_task_payload(_source())

    assert [box["identity_id"] for box in task["query"]["boxes"]] == ["7", "8"]
    assert [box["identity_id"] for box in task["target"]["boxes"]] == ["7", "8"]
    assert task["query"]["boxes"][0]["x"] == pytest.approx(10.0)
    assert task["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]


def test_validate_submission_single_requires_only_subject_1() -> None:
    row = validate_submission(
        _source(),
        {
            "sample_id": "s1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["8"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        },
    )

    assert row["subjects"] == [{"subject_id": 1, "identity_ids": ["8"]}]


def test_validate_submission_derives_group_from_multiple_s1_identities() -> None:
    row = validate_submission(
        _source(),
        {
            "sample_id": "s1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7", "8"]}],
            "final_desc": "Identify Subject 1 as the pair",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        },
    )

    assert row["case_type"] == "GROUP"


def test_validate_submission_derives_individual_from_single_group_identity() -> None:
    row = validate_submission(
        _source(),
        {
            "sample_id": "s1",
            "case_type": "GROUP",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        },
    )

    assert row["case_type"] == "INDIVIDUAL"


def test_validate_submission_keeps_grouped_subject_inside_dual() -> None:
    source = _source()
    source["candidate_identity_ids"] = ["7", "8", "9"]
    row = validate_submission(
        source,
        {
            "sample_id": "s1",
            "case_type": "DUAL",
            "subjects": [
                {"subject_id": 1, "identity_ids": ["7", "8"]},
                {"subject_id": 2, "identity_ids": ["9"]},
            ],
            "final_desc": "Identify Subject 1 as the pair and Subject 2 as the woman",
            "final_change": (
                "then retrieve target images where Subject 1 is smiling and "
                "Subject 2 is waving"
            ),
        },
    )

    assert row["case_type"] == "DUAL"


def test_validate_submission_allows_single_to_multi_case_change() -> None:
    row = validate_submission(
        _source(),
        {
            "sample_id": "s1",
            "case_type": "DUAL",
            "subjects": [
                {"subject_id": 1, "identity_ids": ["7"]},
                {"subject_id": 2, "identity_ids": ["8"]},
            ],
            "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
            "final_change": (
                "then retrieve target images where Subject 1 is smiling and "
                "Subject 2 is waving"
            ),
        },
    )

    assert row["case_type"] == "DUAL"
    assert [subject["subject_id"] for subject in row["subjects"]] == [1, 2]


def test_validate_submission_rejects_identity_in_two_subjects() -> None:
    with pytest.raises(ValueError, match="assigned to multiple subjects"):
        validate_submission(
            _source(),
            {
                "sample_id": "s1",
                "case_type": "DUAL",
                "subjects": [
                    {"subject_id": 1, "identity_ids": ["7"]},
                    {"subject_id": 2, "identity_ids": ["7"]},
                ],
                "final_desc": (
                    "Identify Subject 1 as the man and Subject 2 as the woman"
                ),
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling and "
                    "Subject 2 is waving"
                ),
            },
        )


def test_validate_submission_rejects_missing_subject_2_for_multi() -> None:
    with pytest.raises(ValueError, match="DUAL requires exactly S1, S2"):
        validate_submission(
            _source(),
            {
                "sample_id": "s1",
                "case_type": "DUAL",
                "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
                "final_desc": "Identify Subject 1 as the man",
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling"
                ),
            },
        )


def test_image_relative_path_accepts_local_files_url_and_rejects_parent() -> None:
    assert (
        _image_relative_path(
            "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
        ).as_posix()
        == "train/1_1.jpg"
    )

    with pytest.raises(ValueError, match="unsafe image path"):
        _image_relative_path("/images/../secret.jpg")


def test_review_state_tasks_exposes_lightweight_review_status() -> None:
    source = _source()
    other = {**_source(), "sample_id": "s2", "case_type": "DUAL"}
    state = ReviewState.__new__(ReviewState)
    state.rows = [source, other]
    state.rows_by_id = {row["sample_id"]: row for row in state.rows}
    state.index_by_id = {row["sample_id"]: i for i, row in enumerate(state.rows)}
    state.reviews = {
        "s1": {
            "sample_id": "s1",
            "case_type": "RELATIONAL",
            "subjects": [
                {"subject_id": 1, "identity_ids": ["7"]},
                {"subject_id": 2, "identity_ids": ["8"]},
            ],
            "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
            "final_change": (
                "then retrieve target images where Subject 1 is smiling and "
                "Subject 2 is waving"
            ),
        }
    }

    summary = state.tasks()

    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["pending"] == 1
    assert summary["case_types"] == ["INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL"]
    assert summary["editable_case_types"] == ["INDIVIDUAL", "DUAL", "RELATIONAL"]
    assert summary["one_subject_case_types"] == ["GROUP", "INDIVIDUAL"]
    assert summary["tasks"] == [
        {
            "index": 0,
            "sample_id": "s1",
            "case_type": "RELATIONAL",
            "reviewed": True,
        },
        {
            "index": 1,
            "sample_id": "s2",
            "case_type": "DUAL",
            "reviewed": False,
        },
    ]


def test_review_state_task_can_load_by_sample_id() -> None:
    state = ReviewState.__new__(ReviewState)
    state.rows = [_source()]
    state.rows_by_id = {"s1": state.rows[0]}
    state.index_by_id = {"s1": 0}
    state.reviews = {}

    payload = state.task(sample_id="s1")

    assert payload["index"] == 0
    assert payload["pending"] == 1
    assert payload["task"]["sample_id"] == "s1"


def test_review_save_preserves_order_when_editing_and_appending(tmp_path, monkeypatch):
    first = _source()
    second = {**first, "sample_id": "s2"}
    old = validate_submission(second, second)
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "reviewed.jsonl"
    write_jsonl(input_path, [first, second])
    write_jsonl(output_path, [old])
    monkeypatch.setattr(review_app, "INPUT", input_path)
    monkeypatch.setattr(review_app, "OUTPUT", output_path)

    state = ReviewState()
    state.save(first)
    edited = {
        **old,
        "final_change": "then retrieve target images where Subject 1 is sitting",
    }
    state.save(edited)
    saved = load_jsonl(output_path)

    assert [row["sample_id"] for row in saved] == ["s2", "s1"]
    assert saved[0] == edited
    assert saved[1] == validate_submission(first, first)
