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
            "sample_id": "sample-1",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
            "query_image_path": "train/1_1.jpg",
            "target_image_path": "train/1_2.jpg",
            "case_type": "DUAL",
            "final_desc": "Identify Subject 1 as the man and Subject 2 as the woman",
            "final_change": (
                "then retrieve target images where Subject 1 is smiling and "
                "Subject 2 is waving"
            ),
            "subjects": [
                {"subject_id": 1, "identity_ids": ["7"]},
                {"subject_id": 2, "identity_ids": ["8"]},
            ],
            "query_boxes": [
                {
                    "identity_id": str(identity),
                    "x": x / 100,
                    "y": y / 100,
                    "width": w / 100,
                    "height": h / 100,
                }
                for identity, x, y, w, h in (
                    (7, 10, 20, 30, 40),
                    (8, 50, 20, 20, 30),
                    (9, 75, 20, 15, 20),
                )
            ],
            "target_boxes": [
                {
                    "identity_id": str(identity),
                    "x": x / 100,
                    "y": y / 100,
                    "width": w / 100,
                    "height": h / 100,
                }
                for identity, x, y, w, h in (
                    (7, 10, 10, 20, 20),
                    (8, 40, 10, 20, 20),
                    (9, 70, 10, 15, 20),
                )
            ],
        }
    ]


def _gallery_images() -> list[dict]:
    return [
        {
            "image_id": f"1_{i}",
            "url": f"/images/train/1_{i}.jpg",
            "path": f"train/1_{i}.jpg",
        }
        for i in range(1, 6)
    ]


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
        "1 5 15 15 20 20 7 1",
        "1 5 55 15 20 20 8 1",
    ]


def _reviewed() -> list[dict]:
    return [
        {
            "sample_id": "sample-1",
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
    outputs = prepare_review_inputs(selected=_selected())

    output = outputs[0]
    assert output["query_image_url"] == "/images/train/1_1.jpg"
    assert output["target_image_url"] == "/images/train/1_2.jpg"
    assert [box["identity_id"] for box in output["query_boxes"]] == ["7", "8", "9"]
    assert output["candidate_identity_ids"] == ["7", "8", "9"]
    assert output["final_desc"] == _reviewed()[0]["final_desc"]


def test_prepare_positive_set_inputs_uses_full_indexed_gallery() -> None:
    outputs = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=_reviewed(),
        gallery_images=_gallery_images(),
        index_lines=_index(),
    )

    output = outputs[0]
    assert [candidate["image_id"] for candidate in output["candidates"]] == [
        "1_2",
        "1_3",
        "1_5",
    ]
    assert output["candidates"][0]["is_seed"] is True
    assert output["candidates"][-1]["image_url"] == "/images/train/1_5.jpg"
    assert [
        box["identity_id"] for box in output["candidates"][-1]["subject_boxes"]
    ] == ["7", "8"]
    assert output["case_type"] == "RELATIONAL"
    assert output["positive_image_ids"] == ["1_2"]


def test_prepare_positive_set_inputs_derives_group_from_reviewed_subjects() -> None:
    reviewed = [
        {
            "sample_id": "sample-1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7", "8"]}],
            "final_desc": "Identify Subject 1 as the pair",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        }
    ]

    output = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=reviewed,
        gallery_images=_gallery_images(),
        index_lines=_index(),
    )[0]

    assert output["case_type"] == "GROUP"
    assert output["subjects"] == [{"subject_id": 1, "identity_ids": ["7", "8"]}]


def test_merge_handoff_catalog_keeps_order_updates_existing_and_appends_new() -> None:
    existing = [
        {"sample_id": "old", "value": 1},
        {"sample_id": "stable", "value": 2},
    ]
    prepared = [
        {"sample_id": "stable", "value": 20},
        {"sample_id": "new", "value": 3},
    ]

    outputs = merge_handoff_catalog(existing, prepared)

    assert [row["sample_id"] for row in outputs] == ["old", "stable", "new"]
    assert outputs[1]["value"] == 20


def test_merge_handoff_catalog_rejects_duplicate_prepared_ids() -> None:
    with pytest.raises(ValueError, match="prepared handoff rows contain duplicate"):
        merge_handoff_catalog(
            [],
            [{"sample_id": "dup"}, {"sample_id": "dup"}],
        )


def test_select_unfinished_records_skips_completed_ids() -> None:
    records = [
        {"sample_id": "old"},
        {"sample_id": "new"},
    ]

    outputs = select_unfinished_records(records, {"old"})

    assert outputs == [{"sample_id": "new"}]


def test_prepare_review_inputs_requires_unique_selected_rows() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        prepare_review_inputs(selected=_selected() + _selected())


def test_prepare_review_inputs_rejects_unaccepted_text() -> None:
    incomplete = _selected()
    incomplete[0]["final_change"] = None
    with pytest.raises(ValueError, match="final_change must be a string"):
        prepare_review_inputs(selected=incomplete)


def test_prepare_positive_set_inputs_requires_completed_review() -> None:
    reviewed = _reviewed()
    reviewed[0]["final_desc"] = None

    with pytest.raises(ValueError, match="final_desc must be a string"):
        prepare_positive_set_inputs(
            selected=_selected(),
            reviewed=reviewed,
            gallery_images=_gallery_images(),
            index_lines=_index(),
        )


def test_prepare_positive_set_inputs_uses_reviewed_subject_identities() -> None:
    reviewed = _reviewed()
    reviewed[0]["subjects"][1]["identity_ids"] = ["9"]

    outputs = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=reviewed,
        gallery_images=_gallery_images(),
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
            gallery_images=_gallery_images(),
            index_lines=_index(),
        )


def test_prepare_positive_set_inputs_allows_multi_to_single_case_change() -> None:
    reviewed = [
        {
            "sample_id": "sample-1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        }
    ]

    output = prepare_positive_set_inputs(
        selected=_selected(),
        reviewed=reviewed,
        gallery_images=_gallery_images(),
        index_lines=_index(),
    )[0]

    assert output["case_type"] == "INDIVIDUAL"
    assert output["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]
