"""Check Stage 2 selection against the source pair metadata."""

import copy

import pytest

from rcr.dataset.selection import select_samples


def _row(sample_id: str = "train__1_1__1_2") -> dict:
    return {
        "sample_id": sample_id,
        "split": "TRAIN",
        "annotator_email": "reviewer@example.org",
        "query_image_id": "1_1",
        "query_image_path": "train/1_1.jpg",
        "query_boxes": [
            {"identity_id": "7", "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
        ],
        "target_image_id": "1_2",
        "target_image_path": "train/1_2.jpg",
        "target_boxes": [
            {"identity_id": "7", "x": 0.2, "y": 0.1, "width": 0.2, "height": 0.4}
        ],
        "case_type": "INDIVIDUAL",
        "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
        "final_instruction": (
            "Identify Subject 1 as the man; "
            "then retrieve target images where Subject 1 is smiling."
        ),
    }


def _pair_data() -> dict:
    row = _row()
    return {
        "pairs": [
            {
                "pair_id": row["sample_id"],
                "split": "TRAIN",
                "query_image_id": "1_1",
                "target_image_id": "1_2",
            }
        ],
        "boxes": [
            {
                "image_id": "1_1",
                "label": "7",
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.4,
            },
            {
                "image_id": "1_2",
                "label": "7",
                "x": 0.2,
                "y": 0.1,
                "width": 0.2,
                "height": 0.4,
            },
        ],
    }


def test_select_samples_keeps_approved_rows_and_order() -> None:
    row = _row()
    assert select_samples([row], _pair_data()) == [row]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("sample_id", "legacy-id", "pair_data mismatch"),
        ("split", "VAL", "pair_data mismatch"),
        ("final_instruction", "incorrect", "final_instruction mismatch"),
    ],
)
def test_selection_rejects_drift_from_authoritative_pair(field, value, reason) -> None:
    row = _row()
    row[field] = value
    with pytest.raises(ValueError, match=reason):
        select_samples([row], _pair_data())


def test_selection_rejects_wrong_head_box() -> None:
    row = copy.deepcopy(_row())
    row["target_boxes"][0]["x"] = 0.3
    with pytest.raises(ValueError, match="incorrect box coordinates"):
        select_samples([row], _pair_data())


def test_selection_rejects_legacy_annotation_shape() -> None:
    row = _row()
    row["annotation"] = {"caseType": "SINGLE"}
    with pytest.raises(ValueError, match="Stage 2 fields differ"):
        select_samples([row], _pair_data())
