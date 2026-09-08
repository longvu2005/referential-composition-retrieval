"""Tests for human-labeling handoff preparation."""

import pytest

from rcr.dataset.handoff import (
    merge_handoff_catalog,
    prepare_positive_set_inputs,
    prepare_review_inputs,
    select_unfinished_records,
)


def _selected() -> list[dict]:
    return [
        {
            "submission_id": "sample-1",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
            "annotation": {
                "caseType": "MULTI",
                "subjects": [
                    {
                        "subjectId": 1,
                        "desc": {
                            "queryGroupIds": ["7"],
                            "final": "the man in black",
                        },
                        "change": {"final": "is smiling"},
                    },
                    {
                        "subjectId": 2,
                        "desc": {
                            "queryGroupIds": ["8"],
                            "final": "the woman in blue",
                        },
                        "change": {"final": "is waving"},
                    },
                ],
            },
        }
    ]


def _pair_data() -> dict:
    return {
        "images": [
            {"image_id": "1_1", "url": "/images/1_1.jpg"},
            {"image_id": "1_2", "url": "/images/1_2.jpg"},
            {"image_id": "1_3", "url": "/images/1_3.jpg"},
            {"image_id": "1_4", "url": "/images/1_4.jpg"},
        ],
    }


def _index() -> list[str]:
    return [
        "1 1 10 20 30 40 7 1",
        "1 1 50 20 20 30 8 1",
        "1 1 75 20 15 20 9 1",
        "1 2 10 10 20 20 7 1",
        "1 2 40 10 20 20 8 1",
        "1 2 70 10 15 20 9 1",
        "1 3 20 20 20 20 7 1",
        "1 3 60 20 20 20 8 1",
        "1 4 30 30 20 20 7 1",
        "1 4 60 30 20 20 9 1",
    ]


def _reviewed() -> list[dict]:
    return [
        {
            "submission_id": "sample-1",
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
    ]


def test_prepare_review_inputs_adds_labeling_context() -> None:
    outputs = prepare_review_inputs(
        selected=_selected(),
        rewrite_outputs=_reviewed(),
        pair_data=_pair_data(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert output["query_image_url"] == "/images/1_1.jpg"
    assert output["target_image_url"] == "/images/1_2.jpg"
    assert [box["subject_id"] for box in output["query_boxes"]] == [1, 2]
    assert output["candidate_identity_ids"] == ["7", "8", "9"]
    assert output["final_desc"] == _reviewed()[0]["final_desc"]


def test_prepare_positive_set_inputs_filters_by_all_required_identities() -> None:
    outputs = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=_reviewed(),
        pair_data=_pair_data(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert [candidate["image_id"] for candidate in output["candidates"]] == [
        "1_2",
        "1_3",
    ]
    assert output["candidates"][0]["is_seed"] is True
    assert output["case_type"] == "RELATIONAL"
    assert output["positive_image_ids"] == ["1_2"]




def test_merge_handoff_catalog_keeps_order_updates_existing_and_appends_new() -> None:
    existing = [
        {"submission_id": "old", "value": 1},
        {"submission_id": "stable", "value": 2},
    ]
    prepared = [
        {"submission_id": "stable", "value": 20},
        {"submission_id": "new", "value": 3},
    ]

    outputs = merge_handoff_catalog(existing, prepared)

    assert [row["submission_id"] for row in outputs] == ["old", "stable", "new"]
    assert outputs[1]["value"] == 20


def test_merge_handoff_catalog_rejects_duplicate_prepared_ids() -> None:
    with pytest.raises(ValueError, match="prepared handoff rows contain duplicate"):
        merge_handoff_catalog(
            [],
            [{"submission_id": "dup"}, {"submission_id": "dup"}],
        )


def test_select_unfinished_records_skips_completed_ids() -> None:
    records = [
        {"submission_id": "old"},
        {"submission_id": "new"},
    ]

    outputs = select_unfinished_records(records, {"old"})

    assert outputs == [{"submission_id": "new"}]


def test_prepare_review_inputs_keeps_failed_rewrite_empty() -> None:
    outputs = prepare_review_inputs(
        selected=_selected(),
        rewrite_outputs=[
            {
                "submission_id": "sample-1",
                "final_desc": None,
                "final_change": None,
            }
        ],
        pair_data=_pair_data(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert output["final_desc"] is None
    assert output["final_change"] is None


def test_prepare_review_inputs_normalizes_legacy_error_record() -> None:
    outputs = prepare_review_inputs(
        selected=_selected(),
        rewrite_outputs=[
            {
                "submission_id": "sample-1",
                "error": "Batch response does not contain candidate content.",
            }
        ],
        pair_data=_pair_data(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert output["final_desc"] is None
    assert output["final_change"] is None


def test_prepare_positive_set_inputs_requires_completed_review() -> None:
    reviewed = _reviewed()
    reviewed[0]["final_desc"] = None

    with pytest.raises(ValueError, match="final_desc must be completed"):
        prepare_positive_set_inputs(
            selected=_selected(),
            reviewed=reviewed,
            pair_data=_pair_data(),
            index_lines=_index(),
        )


def test_prepare_positive_set_inputs_uses_reviewed_subject_identities() -> None:
    reviewed = _reviewed()
    reviewed[0]["subjects"][1]["identity_ids"] = ["9"]

    outputs = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=reviewed,
        pair_data=_pair_data(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert output["subjects"] == [
        {"subject_id": 1, "identity_ids": ["7"]},
        {"subject_id": 2, "identity_ids": ["9"]},
    ]
    assert [candidate["image_id"] for candidate in output["candidates"]] == [
        "1_2",
        "1_4",
    ]


def test_prepare_positive_set_inputs_rejects_identity_missing_from_query() -> None:
    reviewed = _reviewed()
    reviewed[0]["subjects"][0]["identity_ids"] = ["999"]

    with pytest.raises(ValueError, match="must appear in both query and seed target"):
        prepare_positive_set_inputs(
            selected=_selected(),
            reviewed=reviewed,
            pair_data=_pair_data(),
            index_lines=_index(),
        )


def test_prepare_positive_set_inputs_allows_multi_to_single_case_change() -> None:
    reviewed = [
        {
            "submission_id": "sample-1",
            "case_type": "SINGLE",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        }
    ]

    output = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=reviewed,
        pair_data=_pair_data(),
        index_lines=_index(),
    )[0]

    assert output["case_type"] == "SINGLE"
    assert output["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]
