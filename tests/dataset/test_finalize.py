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
            "case_type": "SINGLE",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
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
            {"submission_id": "sample-1", "positive_image_ids": ["1_2", "1_3"]}
        ],
        pair_data=_pair_data(),
        index_lines=_index(),
        version="test",
    )

    sample = dataset["samples"][0]
    assert sample["target_image_id"] == "1_2"
    assert sample["positive_image_ids"] == ["1_2", "1_3"]
    assert sample["case_type"] == "SINGLE"
    assert sample["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]
    assert sample["final_instruction"] == (
        "Identify Subject 1 as the man; "
        "then retrieve target images where Subject 1 is smiling."
    )
    assert dataset["splits"] == {"train": ["sample-1"], "val": [], "test": []}


def test_build_final_dataset_falls_back_to_original_case_when_case_is_missing() -> None:
    reviewed = _reviewed()
    reviewed[0].pop("case_type")

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
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
            positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_3"]}],
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
            positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_invalid_review_case_type() -> None:
    reviewed = _reviewed()
    reviewed[0]["case_type"] = "INVALID"

    with pytest.raises(ValueError, match="invalid case_type"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_malformed_positive_set() -> None:
    with pytest.raises(ValueError, match="positive_image_ids must be"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[
                {
                    "submission_id": "sample-1",
                    "positive_image_ids": ["1_2", "1_2"],
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
            positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_uses_reviewed_subject_identities() -> None:
    reviewed = _reviewed()
    reviewed[0]["subjects"] = [{"subject_id": 1, "identity_ids": ["8"]}]
    index_lines = [
        *_index(),
        "1 1 20 20 20 20 8 1",
        "1 2 20 20 20 20 8 1",
        "1 3 20 20 20 20 8 1",
    ]

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[
            {"submission_id": "sample-1", "positive_image_ids": ["1_2", "1_3"]}
        ],
        pair_data=_pair_data(),
        index_lines=index_lines,
        version="test",
    )

    assert dataset["samples"][0]["subjects"] == [
        {"subject_id": 1, "identity_ids": ["8"]}
    ]


def test_build_final_dataset_rejects_reviewed_identity_missing_from_query() -> None:
    reviewed = _reviewed()
    reviewed[0]["subjects"] = [{"subject_id": 1, "identity_ids": ["8"]}]

    with pytest.raises(
        ValueError, match="reviewed identities must appear in the query"
    ):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
            pair_data=_pair_data(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_allows_single_to_multi_case_change() -> None:
    reviewed = [
        {
            "submission_id": "sample-1",
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
        }
    ]
    index_lines = [
        *_index(),
        "1 1 20 20 20 20 8 1",
        "1 2 20 20 20 20 8 1",
    ]

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[{"submission_id": "sample-1", "positive_image_ids": ["1_2"]}],
        pair_data=_pair_data(),
        index_lines=index_lines,
        version="test",
    )

    sample = dataset["samples"][0]
    assert sample["case_type"] == "MULTI"
    assert [subject["subject_id"] for subject in sample["subjects"]] == [1, 2]
