"""Tests for final dataset construction."""

import pytest

from rcr.dataset.finalize import build_final_dataset


def _selected() -> list[dict]:
    return [
        {
            "sample_id": "sample-1",
            "split": "TRAIN",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
        }
    ]


def _reviewed() -> list[dict]:
    return [
        {
            "sample_id": "sample-1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is smiling",
        }
    ]


def _gallery_images() -> list[dict]:
    return [{"image_id": f"1_{i}", "path": f"train/1_{i}.jpg"} for i in range(1, 5)]


def _index() -> list[str]:
    return [
        "1 1 10 20 30 40 7 1",
        "1 2 11 21 31 41 7 1",
        "1 3 12 22 32 42 7 1",
        "1 4 13 23 33 43 7 1",
    ]


def test_build_final_dataset_preserves_expanded_positives() -> None:
    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=_reviewed(),
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2", "1_4"]}],
        gallery_images=_gallery_images(),
        index_lines=_index(),
        version="test",
    )

    sample = dataset["samples"][0]
    assert sample["target_image_id"] == "1_2"
    assert sample["positive_image_ids"] == ["1_2", "1_4"]
    assert sample["case_type"] == "INDIVIDUAL"
    assert sample["subjects"] == [{"subject_id": 1, "identity_ids": ["7"]}]
    assert sample["final_instruction"] == (
        "Identify Subject 1 as the man; "
        "then retrieve target images where Subject 1 is smiling."
    )
    assert dataset["splits"] == {"train": ["sample-1"], "val": [], "test": []}
    assert [row["image_id"] for row in dataset["gallery"]] == [
        "1_1",
        "1_2",
        "1_3",
        "1_4",
    ]


def test_build_final_dataset_uses_pair_data_split_over_stale_export_split() -> None:
    selected = _selected()
    selected[0]["split"] = "TRAIN"
    pair_data = {
        "pairs": [
            {
                "query_image_id": "1_1",
                "target_image_id": "1_2",
                "split": "VAL",
            }
        ]
    }

    dataset = build_final_dataset(
        selected=selected,
        reviewed=_reviewed(),
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
        gallery_images=_gallery_images(),
        index_lines=_index(),
        version="test",
        pair_data=pair_data,
    )

    assert dataset["splits"] == {"train": [], "val": ["sample-1"], "test": []}


def test_build_final_dataset_rejects_pair_missing_from_pair_data() -> None:
    with pytest.raises(ValueError, match="missing from pair_data"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
            pair_data={"pairs": []},
        )


def test_build_final_dataset_rejects_duplicate_pair_data_key() -> None:
    pair = {
        "query_image_id": "1_1",
        "target_image_id": "1_2",
        "split": "TRAIN",
    }
    with pytest.raises(ValueError, match="duplicate ordered pair"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
            pair_data={"pairs": [pair, dict(pair)]},
        )


def test_build_final_dataset_falls_back_to_original_case_when_case_is_missing() -> None:
    reviewed = _reviewed()
    reviewed[0].pop("case_type")

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
        gallery_images=_gallery_images(),
        index_lines=_index(),
        version="test",
    )

    assert dataset["samples"][0]["case_type"] == "INDIVIDUAL"


def test_build_final_dataset_derives_group_from_reviewed_subjects() -> None:
    reviewed = _reviewed()
    reviewed[0]["case_type"] = "INDIVIDUAL"
    reviewed[0]["subjects"] = [{"subject_id": 1, "identity_ids": ["7", "8"]}]
    reviewed[0]["final_desc"] = "Identify Subject 1 as the pair"
    index_lines = [
        *_index(),
        "1 1 20 20 20 20 8 1",
        "1 2 20 20 20 20 8 1",
    ]

    dataset = build_final_dataset(
        selected=_selected(),
        reviewed=reviewed,
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
        gallery_images=_gallery_images(),
        index_lines=index_lines,
        version="test",
    )

    assert dataset["samples"][0]["case_type"] == "GROUP"


def test_build_final_dataset_requires_seed_positive() -> None:
    with pytest.raises(ValueError, match="seed positive"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_3"]}],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_rejects_duplicate_review_ids() -> None:
    reviewed = _reviewed() + _reviewed()

    with pytest.raises(ValueError, match="duplicate sample_id sample-1"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
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
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
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
                    "sample_id": "sample-1",
                    "positive_image_ids": ["1_2", "1_2"],
                }
            ],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_revalidates_reviewed_text() -> None:
    reviewed = _reviewed()
    reviewed[0]["final_change"] = "Subject 1 is smiling"

    with pytest.raises(ValueError, match="final_change must begin"):
        build_final_dataset(
            selected=_selected(),
            reviewed=reviewed,
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
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
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2", "1_3"]}],
        gallery_images=_gallery_images(),
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
            positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
        )


def test_build_final_dataset_allows_single_to_multi_case_change() -> None:
    reviewed = [
        {
            "sample_id": "sample-1",
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
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
        gallery_images=_gallery_images(),
        index_lines=index_lines,
        version="test",
    )

    sample = dataset["samples"][0]
    assert sample["case_type"] == "DUAL"
    assert [subject["subject_id"] for subject in sample["subjects"]] == [1, 2]


def test_build_final_dataset_allows_partial_handoffs() -> None:
    selected = _selected() + [
        {
            **_selected()[0],
            "sample_id": "sample-2",
            "query_image_id": "1_3",
            "target_image_id": "1_4",
        }
    ]

    dataset = build_final_dataset(
        selected=selected,
        reviewed=_reviewed(),
        positive_sets=[{"sample_id": "sample-1", "positive_image_ids": ["1_2"]}],
        gallery_images=_gallery_images(),
        index_lines=_index(),
        version="test",
        allow_partial=True,
    )

    assert [row["sample_id"] for row in dataset["samples"]] == ["sample-1"]
    assert dataset["manifest"]["num_samples"] == 1
    assert dataset["manifest"]["partial"] is True


def test_build_final_dataset_stays_strict_by_default() -> None:
    with pytest.raises(ValueError, match="positive sets do not match selected samples"):
        build_final_dataset(
            selected=_selected(),
            reviewed=_reviewed(),
            positive_sets=[],
            gallery_images=_gallery_images(),
            index_lines=_index(),
            version="test",
        )
