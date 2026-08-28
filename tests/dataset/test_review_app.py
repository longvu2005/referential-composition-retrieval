"""Tests for the minimal identity-synchronized rewrite review UI backend."""

import pytest

from labelstudio.review.app import (
    ReviewState,
    _image_relative_path,
    build_task_payload,
    validate_submission,
)

BOXES = {
    "1_1": [
        {
            "image_id": "1_1", "label": "7", "x": 0.1, "y": 0.2,
            "width": 0.3, "height": 0.4,
        },
        {
            "image_id": "1_1", "label": "8", "x": 0.5, "y": 0.2,
            "width": 0.2, "height": 0.3,
        },
    ],
    "1_2": [
        {
            "image_id": "1_2", "label": "7", "x": 0.2, "y": 0.1,
            "width": 0.2, "height": 0.4,
        },
        {
            "image_id": "1_2", "label": "8", "x": 0.6, "y": 0.2,
            "width": 0.2, "height": 0.3,
        },
    ],
}


def _source() -> dict:
    return {
        "submission_id": "s1",
        "case_type": "SINGLE",
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
        "pair_change": None,
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
    }


def test_build_task_payload_uses_same_identity_candidates_on_both_images() -> None:
    task = build_task_payload(_source(), BOXES)

    assert [box["identity_id"] for box in task["query"]["boxes"]] == ["7", "8"]
    assert [box["identity_id"] for box in task["target"]["boxes"]] == ["7", "8"]
    assert task["query"]["boxes"][0]["x"] == pytest.approx(10.0)
    assert task["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]


def test_validate_submission_single_requires_only_subject_1() -> None:
    row = validate_submission(
        _source(),
        {
            "submission_id": "s1",
            "case_type": "SINGLE",
            "subjects": [{"subject_id": 1, "identity_ids": ["8"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        },
    )

    assert row["subjects"] == [{"subject_id": 1, "identity_ids": ["8"]}]


def test_validate_submission_allows_single_to_multi_case_change() -> None:
    row = validate_submission(
        _source(),
        {
            "submission_id": "s1",
            "case_type": "MULTI",
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

    assert row["case_type"] == "MULTI"
    assert [subject["subject_id"] for subject in row["subjects"]] == [1, 2]


def test_validate_submission_rejects_identity_in_two_subjects() -> None:
    with pytest.raises(ValueError, match="assigned to multiple subjects"):
        validate_submission(
            _source(),
            {
                "submission_id": "s1",
                "case_type": "MULTI",
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
    with pytest.raises(ValueError, match="MULTI requires exactly S1, S2"):
        validate_submission(
            _source(),
            {
                "submission_id": "s1",
                "case_type": "MULTI",
                "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
                "final_desc": "Identify Subject 1 as the man",
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling"
                ),
            },
        )


def test_image_relative_path_accepts_local_files_url_and_rejects_parent() -> None:
    assert _image_relative_path(
        "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
    ).as_posix() == "train/1_1.jpg"

    with pytest.raises(ValueError, match="unsafe image path"):
        _image_relative_path("/images/../secret.jpg")


def test_review_state_tasks_exposes_lightweight_review_status() -> None:
    source = _source()
    other = {**_source(), "submission_id": "s2", "case_type": "MULTI"}
    state = ReviewState.__new__(ReviewState)
    state.rows = [source, other]
    state.rows_by_id = {row["submission_id"]: row for row in state.rows}
    state.index_by_id = {row["submission_id"]: i for i, row in enumerate(state.rows)}
    state.reviews = {
        "s1": {
            "submission_id": "s1",
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
    assert summary["tasks"] == [
        {
            "index": 0,
            "submission_id": "s1",
            "case_type": "RELATIONAL",
            "reviewed": True,
        },
        {
            "index": 1,
            "submission_id": "s2",
            "case_type": "MULTI",
            "reviewed": False,
        },
    ]


def test_review_state_task_can_load_by_submission_id() -> None:
    state = ReviewState.__new__(ReviewState)
    state.rows = [_source()]
    state.rows_by_id = {"s1": state.rows[0]}
    state.index_by_id = {"s1": 0}
    state.reviews = {}
    state.boxes_by_image = BOXES

    payload = state.task(submission_id="s1")

    assert payload["index"] == 0
    assert payload["pending"] == 1
    assert payload["task"]["submission_id"] == "s1"
