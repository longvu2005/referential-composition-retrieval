"""CPU workflow tests for core."""

import json
from pathlib import Path

from rcr.common.data import (
    inference_sample,
    load_rcr_data,
    load_rcr_inputs,
    required_identity_ids,
    sample_selection_texts,
    split_image_ids,
    split_inputs,
    split_samples,
)
from rcr.common.io import write_jsonl


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


def test_public_loader_needs_no_head_seed_case_or_positive_annotations(tmp_path):
    _write_final_dataset(tmp_path)
    expected = load_rcr_inputs(tmp_path)
    (tmp_path / "head_boxes.jsonl").unlink()
    # A public export can omit every private sample field.
    write_jsonl(
        tmp_path / "samples.jsonl", [inference_sample(s) for s in expected.samples]
    )
    actual = load_rcr_inputs(tmp_path)
    assert not hasattr(actual, "gt_head_boxes_by_image")
    assert split_inputs(actual, "train") == split_inputs(expected, "train")
    assert actual.samples[0]["subjects"] == [{"subject_id": 1}, {"subject_id": 2}]
    assert set(actual.samples[0]) == {
        "sample_id",
        "query_image_id",
        "final_desc",
        "final_change",
        "final_instruction",
        "subjects",
    }


def test_scene_cache_keeps_native_global_without_person_geometry(tmp_path):
    import torch
    from PIL import Image
    from torch import nn

    from rcr.proposed.cache import scene as scene_cache
    from rcr.proposed.cache.scene import SceneCache

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 50), color="white").save(image_path)

    class SceneEncoder(nn.Module):
        def forward(self, pixels):
            scene = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            return scene, scene.mean(dim=1)[:, :2] + 9, (1, 2)

    def scene_processor(*, images, return_tensors):
        assert images.size == (100, 50)
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(1, 3, 16, 32)}

    cache_root = tmp_path / "cache"
    scene_cache.build_cache(
        image_ids=["image-1"],
        image_paths=[image_path],
        root=cache_root,
        image_encoder=SceneEncoder(),
        scene_processor=scene_processor,
        device="cpu",
        encoder_metadata={
            "image_encoder": {"model": "fake"},
        },
    )

    cache = SceneCache(cache_root)
    assert cache._load_item(0)["scene"].shape == (2, 4)
    assert cache.global_dim == 2
    torch.testing.assert_close(
        cache.load_global_features(torch.tensor([0])), torch.tensor([[11.0, 12.0]])
    )
    index = torch.load(cache_root / "index.pt", weights_only=True)
    assert not {"persons", "mask", "detector", "boxes_pixel"} & index.keys()
    assert set(cache._load_item(0)) == {"scene"}
