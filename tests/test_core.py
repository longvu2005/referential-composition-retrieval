"""Core tests for shared RCR method utilities."""

import json
from pathlib import Path

import pytest

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
from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.text import parse_selection_texts


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


def test_inference_projection_is_allowlisted_and_detached(tmp_path):
    _write_final_dataset(tmp_path)
    sample = load_rcr_data(tmp_path).samples[0]
    sample["new_private_field"] = "do not forward"
    sample["subjects"][0]["member_count"] = 17
    public = inference_sample(sample)
    assert "new_private_field" not in public and "case_type" not in public
    assert all(set(s) == {"subject_id"} for s in public["subjects"])
    public["subjects"][0]["subject_id"] = 9
    assert sample["subjects"][0]["subject_id"] == 1


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


def test_interrupted_scene_cache_resumes_native_globals(tmp_path):
    import torch
    from PIL import Image
    from torch import nn

    from rcr.proposed.cache import scene
    from rcr.proposed.cache.scene import SceneCache

    path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 16), color="white").save(path)

    class Encoder(nn.Module):
        def __init__(self, fail=False):
            super().__init__()
            self.calls = 0
            self.fail = fail

        def forward(self, pixels):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("interrupted")
            return torch.zeros(1, 2, 4), torch.tensor([[7.0, 8.0]]), (1, 2)

    kwargs = dict(
        image_ids=["a", "b"],
        image_paths=[path, path],
        root=tmp_path / "cache",
        scene_processor=lambda **kw: {"pixel_values": torch.zeros(1, 3, 16, 32)},
        device="cpu",
        encoder_metadata={"image_encoder": {"model": "fake"}},
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        scene.build_cache(image_encoder=Encoder(fail=True), **kwargs)
    resumed = Encoder()
    scene.build_cache(image_encoder=resumed, **kwargs)
    assert resumed.calls == 1
    cache = SceneCache(kwargs["root"])
    assert cache.scene_dim == 4 and cache.global_dim == 2
    torch.testing.assert_close(
        cache.load_global_features(torch.tensor([0, 1])),
        torch.tensor([[7.0, 8.0], [7.0, 8.0]]),
    )


@pytest.mark.parametrize("global_dim", [2, 4])
def test_packed_fafa_cache_uses_native_clip_globals_without_scene_reads(
    tmp_path, global_dim
):
    import torch

    from rcr.proposed.cache.native import NATIVE_VERSION
    from rcr.proposed.cache.store import GalleryCache

    source_root, fafa_root = tmp_path / "scene", tmp_path / "fafa"
    source_root.mkdir()
    fafa_root.mkdir()
    shared = {
        "image_ids": ["image-1"],
        "mask": torch.tensor([[True]]),
        "patch_hw": (1, 2),
        "scene_dim": 4,
        "image_encoder": {"model": "fake-clip"},
        "detector": {"model": "fake-detector"},
    }
    torch.save(
        {
            **shared,
            "format_version": 2,
            "cache_id": "scene-id",
            "global_dim": 2,
            "global_features": torch.tensor([[11.0, 12.0]]),
            "persons": torch.empty(1, 1, 0),
            "person_encoder": {"backend": "clip"},
        },
        source_root / "index.pt",
    )
    torch.save(
        {
            **shared,
            "format_version": 6,
            "layout": "fafa",
            "cache_id": "fafa-id",
            "source_cache_id": "scene-id",
            "global_dim": global_dim,
            "persons": torch.tensor([[1.0, 2.0, 3.0]]),
            "person_offsets": torch.tensor([0, 1]),
            "person_encoder": {
                "backend": "fafa",
                "spec": {},
                "checkpoint_sha256": "fake-weights",
                "encoder_precision": "float32",
            },
            "native": {"version": NATIVE_VERSION, "dim": 3, "tokens": 2},
        },
        fafa_root / "index.pt",
    )
    if global_dim != 2:
        with pytest.raises(ValueError, match="global dimensions differ"):
            GalleryCache(fafa_root, scene_root=source_root)
        return
    # No feature shards exist: native globals must come directly from the index.
    cache = GalleryCache(fafa_root, scene_root=source_root)
    assert (cache.scene_dim, cache.global_dim, cache.person_dim) == (4, 2, 3)
    expected = torch.tensor([[11.0, 12.0], [11.0, 12.0]])
    torch.testing.assert_close(
        cache.load_global_features(torch.tensor([0, 0])), expected
    )
    torch.testing.assert_close(cache.global_features, expected[:1])
