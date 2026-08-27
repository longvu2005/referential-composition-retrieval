"""Tests for final dataset construction."""

import pytest

from rcr.dataset.finalize import build_final_dataset


def _selected() -> list[dict]:
    return [
        {
            "submission_id": "sample-1",
            "split": "TRAIN",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
            "annotation": {
                "caseType": "SINGLE",
                "subjects": [
                    {
                        "subjectId": 1,
                        "desc": {
                            "queryGroupIds": ["7"],
                            "final": "the man",
                        },
                        "change": {"final": "is smiling"},
                    }
                ],
            },
        }
    ]


def _reviewed() -> list[dict]:
    return [
        {
            "submission_id": "sample-1",
            "case_type": "RELATIONAL",
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        }
    ]


def _pair_data() -> dict:
    return {
        "images": [
            {"image_id": "1_1", "url": "/x/PIPA/images/train/1_1.jpg"},
            {"image_id": "1_2", "url": "/x/PIPA/images/train/1_2.jpg"},
            {"image_id": "1_3", "url": "/x/PIPA/images/train/1_3.jpg"},
        ]
    }


def _index() -> list[str]:
    return [
        "1 1 10 20 30 40 7 1",
        "1 2 11 21 31 41 7 1",
        "1 3 12 22 32 42 7 1",
    ]


def test_build_final_dataset_preserves_expanded_positives() -> None:
    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=_reviewed(),
        positive_sets=[
            {"submission_id": "sample-1", "target_image_ids": ["1_2", "1_3"]}
        ],
        pair_data=_pair_data(),
        index_lines=_index(),
        version="test",
    )

    sample = dataset["samples"][0]
    assert sample["target_image_ids"] == ["1_2", "1_3"]
    assert sample["case_type"] == "RELATIONAL"
    assert sample["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]
    assert sample["final_instruction"] == (
        "Identify Subject 1 as the man; "
        "then retrieve target images where Subject 1 is smiling."
    )
    assert dataset["splits"] == {"train": ["sample-1"], "val": [], "test": []}


def test_build_final_dataset_falls_back_to_original_case_for_legacy_review() -> None:
    reviewed = _reviewed()
    reviewed[0].pop("case_type")

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[{"submission_id": "sample-1", "target_image_ids": ["1_2"]}],
        pair_data=_pair_data(),
        index_lines=_index(),
        version="test",
    )

    assert dataset["samples"][0]["case_type"] == "SINGLE"


def test_build_final_dataset_requires_seed_positive() -> None:
    with pytest.raises(ValueError, match="seed positive"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[{"submission_id": "sample-1", "target_image_ids": ["1_3"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_duplicate_review_ids() -> None:
    reviewed = _reviewed() + _reviewed()

    with pytest.raises(ValueError, match="duplicate submission_id sample-1"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"submission_id": "sample-1", "target_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_extra_review_fields() -> None:
    reviewed = _reviewed()
    reviewed[0]["unexpected"] = "legacy field"

    with pytest.raises(ValueError, match="reviewed record must contain"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"submission_id": "sample-1", "target_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_malformed_positive_set() -> None:
    with pytest.raises(ValueError, match="target_image_ids must be"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[
                {
                    "submission_id": "sample-1",
                    "target_image_ids": ["1_2", "1_2"],
                }
            ],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_revalidates_reviewed_rewrite() -> None:
    reviewed = _reviewed()
    reviewed[0]["final_change"] = "Subject 1 is smiling"

    with pytest.raises(ValueError, match="final_change must begin"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"submission_id": "sample-1", "target_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )
