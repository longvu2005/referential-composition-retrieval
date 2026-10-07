"""Core tests for shared RCR method utilities."""

import json
from pathlib import Path

import pytest

from rcr.common.data import (
    load_rcr_data,
    required_identity_ids,
    sample_selection_texts,
    split_image_ids,
    split_samples,
)
from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.rewrite import parse_selection_texts


def _write_final_dataset(root: Path) -> None:
    write_jsonl(
        root / "samples.jsonl",
        [
            {
                "sample_id": "train-1",
                "case_type": "DUAL",
                "query_image_id": "1_1",
                "target_image_id": "1_2",
                "positive_image_ids": ["1_2", "1_3"],
                "subjects": [
                    {"subject_id": 1, "identity_ids": ["7"]},
                    {"subject_id": 2, "identity_ids": ["8", "9"]},
                ],
                "final_desc": (
                    "Identify Subject 1 as the man in black and white "
                    "and Subject 2 as the woman in blue"
                ),
                "final_change": (
                    "then retrieve target images where Subject 1 is sitting "
                    "and Subject 2 is standing"
                ),
                "final_instruction": "unused in method input",
            },
            {
                "sample_id": "val-1",
                "case_type": "INDIVIDUAL",
                "query_image_id": "1_4",
                "target_image_id": "1_5",
                "positive_image_ids": ["1_5"],
                "subjects": [{"subject_id": 1, "identity_ids": ["10"]}],
                "final_desc": "Identify Subject 1 as the man",
                "final_change": (
                    "then retrieve target images where Subject 1 is smiling"
                ),
                "final_instruction": "unused in method input",
            },
        ],
    )
    write_jsonl(
        root / "images.jsonl",
        [
            {"image_id": f"1_{i}", "path": f"{split}/1_{i}.jpg"}
            for i, split in enumerate(
                ("train", "train", "train", "val", "val", "train", "test", "leftover"),
                start=1,
            )
        ],
    )
    write_jsonl(root / "gallery.jsonl", [{"image_id": f"1_{i}"} for i in range(1, 9)])
    write_jsonl(
        root / "head_boxes.jsonl",
        [
            {
                "box_id": "1_1::pid7",
                "image_id": "1_1",
                "identity_id": "7",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
            }
        ],
    )
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "train.txt").write_text("train-1\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("val-1\n", encoding="utf-8")
    (root / "splits" / "test.txt").write_text("", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"version": "test", "num_samples": 2}), encoding="utf-8"
    )


def test_parse_selection_texts_keeps_inner_and() -> None:
    text = (
        "Identify Subject 1 as the man in black and white "
        "and Subject 2 as the woman in blue"
    )

    assert parse_selection_texts(text, [1, 2]) == [
        "the man in black and white",
        "the woman in blue",
    ]


def test_load_rcr_data_and_training_pool(tmp_path) -> None:
    final_dir = tmp_path / "final"
    image_root = tmp_path / "images"
    _write_final_dataset(final_dir)

    data = load_rcr_data(final_dir, image_root)
    sample = split_samples(data, "train")[0]

    assert sample_selection_texts(sample) == [
        "the man in black and white",
        "the woman in blue",
    ]
    assert required_identity_ids(sample) == ["7", "8", "9"]
    assert split_image_ids(data, "train") == ["1_1", "1_2", "1_3", "1_6"]
    assert split_image_ids(data, "val") == ["1_4", "1_5"]
    assert split_image_ids(data, "test") == ["1_7"]
    assert data.image_path("1_2") == image_root / "train/1_2.jpg"
    assert data.gt_head_boxes_by_image["1_1"][0]["identity_id"] == "7"


def test_load_rcr_data_rejects_split_overlap(tmp_path) -> None:
    final_dir = tmp_path / "final"
    _write_final_dataset(final_dir)
    (final_dir / "splits" / "val.txt").write_text("train-1\nval-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="more than one split"):
        load_rcr_data(final_dir, tmp_path / "images")


def test_load_rcr_data_rejects_duplicate_sample_ids(tmp_path) -> None:
    final_dir = tmp_path / "final"
    _write_final_dataset(final_dir)
    samples = load_jsonl(final_dir / "samples.jsonl")
    write_jsonl(final_dir / "samples.jsonl", [*samples, samples[0]])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_rcr_data(final_dir, tmp_path / "images")


def test_split_samples_scopes_legacy_positives_without_mutating_annotations(tmp_path):
    final_dir = tmp_path / "final"
    _write_final_dataset(final_dir)
    data = load_rcr_data(final_dir, tmp_path / "images")
    stored = data.samples_by_id["train-1"]
    stored["positive_image_ids"] += ["1_5", "1_7", "1_8"]

    sample = split_samples(data, "train")[0]

    assert sample["positive_image_ids"] == ["1_2", "1_3"]
    assert stored["positive_image_ids"] == ["1_2", "1_3", "1_5", "1_7", "1_8"]


def test_split_samples_rejects_seed_from_another_split(tmp_path):
    final_dir = tmp_path / "final"
    _write_final_dataset(final_dir)
    data = load_rcr_data(final_dir, tmp_path / "images")
    data.samples_by_id["train-1"]["target_image_id"] = "1_5"

    with pytest.raises(ValueError, match="query and seed must belong to train"):
        split_samples(data, "train")
