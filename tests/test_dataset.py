"""Dataset loading, annotation handoff, positive ordering and finalization."""

import json
import sys
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rcr import dataset as dataset_package
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
from rcr.dataset.finalize import build_final_dataset
from rcr.dataset.handoff import prepare_positive_set_inputs


@pytest.fixture
def clip_rerank(monkeypatch):
    """Exercise the real ranker with the optional model backend mocked."""
    path = Path(dataset_package.__file__).with_name("clip_rerank.py")
    spec = spec_from_file_location("rcr.dataset._test_clip_rerank", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    with monkeypatch.context() as backend:
        backend.setitem(sys.modules, "open_clip", SimpleNamespace())
        spec.loader.exec_module(module)
    return module


# Dataset loading and public inference inputs.


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


# Annotation handoff and split-scoped candidates.


def _handoff_inputs(split="train"):
    # Every image has identity 7; only complete same-split matches may survive.
    identities = {
        "1_1": ["7", "8"],  # Query.
        "1_2": ["7", "8"],  # Seed.
        "1_3": ["7", "8", "9"],  # Extra identities are allowed.
        "1_4": ["7"],  # A partial identity match is insufficient.
        "1_5": ["7", "8"],
        "1_6": ["7", "8"],
        "1_7": ["7", "8"],
        "1_8": ["7", "8"],
    }
    splits = dict.fromkeys(identities, split)
    other_splits = [s for s in ("train", "val", "test") if s != split][:2]
    splits.update(zip(("1_5", "1_6"), other_splits, strict=True))
    splits["1_7"] = "leftover"
    gallery = [
        {
            "image_id": image_id,
            "path": f"{image_split}/{image_id}.jpg",
            "url": f"/images/{image_split}/{image_id}.jpg",
        }
        for image_id, image_split in splits.items()
    ]
    index_lines = [
        f"1 {image_id.split('_')[1]} 1 2 3 4 {identity_id} 0"
        for image_id, image_ids in identities.items()
        for identity_id in image_ids
    ]
    selected = [
        {
            "sample_id": "sample-1",
            "case_type": "DUAL",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
        }
    ]
    reviewed = [
        {
            "sample_id": "sample-1",
            "case_type": "DUAL",
            "subjects": [
                {"subject_id": 1, "identity_ids": ["7"]},
                {"subject_id": 2, "identity_ids": ["8"]},
            ],
            "final_desc": ("Identify Subject 1 as the man and Subject 2 as the woman"),
            "final_change": (
                "then retrieve target images where Subject 1 and Subject 2 are sitting"
            ),
        }
    ]
    return selected, reviewed, gallery, index_lines


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_positive_candidates_cover_exact_split_identity_matches(split):
    inputs = _handoff_inputs(split)
    original = deepcopy(inputs)
    task = prepare_positive_set_inputs(*inputs)[0]

    assert [c["image_id"] for c in task["candidates"]] == ["1_2", "1_3", "1_8"]
    assert [c["is_seed"] for c in task["candidates"]] == [True, False, False]
    assert task["positive_image_ids"] == ["1_2"]
    assert inputs == original
    assert task["query_image_url"] == f"/images/{split}/1_1.jpg"
    for candidate in task["candidates"]:
        assert candidate["image_url"].startswith(f"/images/{split}/")
        assert {
            (box["subject_id"], box["identity_id"])
            for box in candidate["subject_boxes"]
        } == {(1, "7"), (2, "8")}


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_reranker_uses_seed_target_and_only_non_seed_split_candidates(split):
    def ranker(reference_image, text, candidates):
        assert reference_image == ("1_2", f"/images/{split}/1_2.jpg")
        assert text == _handoff_inputs(split)[1][0]["final_change"]
        assert candidates == [
            ("1_3", f"/images/{split}/1_3.jpg"),
            ("1_8", f"/images/{split}/1_8.jpg"),
        ]
        return ["1_8", "1_3"]

    task = prepare_positive_set_inputs(
        *_handoff_inputs(split), candidate_ranker=ranker
    )[0]
    assert [c["image_id"] for c in task["candidates"]] == ["1_2", "1_8", "1_3"]


@pytest.mark.parametrize("image_id", ["1_5", "1_6", "1_7"])
def test_positive_handoff_rejects_seed_outside_query_split(image_id):
    selected, reviewed, gallery, index_lines = _handoff_inputs()
    selected[0]["target_image_id"] = image_id
    with pytest.raises(ValueError, match="seed target.*same split"):
        prepare_positive_set_inputs(selected, reviewed, gallery, index_lines)


@pytest.mark.parametrize("split", ["leftover", "unknown"])
def test_positive_handoff_rejects_invalid_query_split(split):
    with pytest.raises(ValueError, match="query.*split"):
        prepare_positive_set_inputs(*_handoff_inputs(split))


def test_positive_handoff_rejects_duplicate_reviews():
    selected, reviewed, gallery, index_lines = _handoff_inputs()
    with pytest.raises(ValueError, match="reviewed: duplicate sample_id"):
        prepare_positive_set_inputs(selected, reviewed * 2, gallery, index_lines)


def test_reranker_cannot_reintroduce_leftover_candidates():
    with pytest.raises(ValueError, match="every non-seed candidate exactly once"):
        prepare_positive_set_inputs(
            *_handoff_inputs(),
            candidate_ranker=lambda reference, text, candidates: ["1_3", "1_7"],
        )


# Final export validation.


def _finalize_inputs():
    selected = [
        {
            "sample_id": "train-1",
            "split": "TRAIN",
            "case_type": "INDIVIDUAL",
            "query_image_id": "1_1",
            "target_image_id": "1_2",
        }
    ]
    reviewed = [
        {
            "sample_id": "train-1",
            "case_type": "INDIVIDUAL",
            "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "then retrieve target images where Subject 1 is sitting",
        }
    ]
    splits = ("train", "train", "train", "val", "test", "leftover", "train")
    return {
        "selected": selected,
        "reviewed": reviewed,
        "positive_sets": [
            {"sample_id": "train-1", "positive_image_ids": ["1_2", "1_3"]}
        ],
        "gallery_images": [
            {"image_id": f"1_{i}", "path": f"{split}/1_{i}.jpg"}
            for i, split in enumerate(splits, start=1)
        ],
        "index_lines": [f"1 {i} 1 2 3 4 {8 if i == 7 else 7} 0" for i in range(1, 8)],
        "version": "test",
        "pair_data": {
            "pairs": [
                {
                    "query_image_id": "1_1",
                    "target_image_id": "1_2",
                    "split": "TRAIN",
                }
            ]
        },
    }


def test_finalize_keeps_valid_labels_and_complete_image_registry():
    inputs = _finalize_inputs()
    original = deepcopy(inputs)
    result = build_final_dataset(**inputs)
    assert result["samples"][0]["positive_image_ids"] == ["1_2", "1_3"]
    assert result["splits"] == {"train": ["train-1"], "val": [], "test": []}
    assert result["images"] == inputs["gallery_images"]
    assert result["manifest"]["num_images"] == 7
    assert inputs == original


@pytest.mark.parametrize("image_id", ["1_4", "1_5", "1_6"])
def test_finalize_rejects_positive_outside_query_split(image_id):
    inputs = _finalize_inputs()
    inputs["positive_sets"][0]["positive_image_ids"].append(image_id)
    with pytest.raises(ValueError, match="positive.*must belong to.*train"):
        build_final_dataset(**inputs)


@pytest.mark.parametrize("position", [0, 1])
def test_finalize_checks_query_and_seed_image_split(position):
    inputs = _finalize_inputs()
    image = inputs["gallery_images"][position]
    image["path"] = f"val/{image['image_id']}.jpg"
    with pytest.raises(ValueError, match="query and seed target.*train"):
        build_final_dataset(**inputs)


def test_finalize_checks_authoritative_pair_split_against_image_paths():
    inputs = _finalize_inputs()
    inputs["pair_data"]["pairs"][0]["split"] = "VAL"
    with pytest.raises(ValueError, match="query and seed target.*val"):
        build_final_dataset(**inputs)


def test_finalize_uses_authoritative_pair_split_over_selected_metadata():
    inputs = _finalize_inputs()
    inputs["selected"][0]["split"] = "VAL"
    assert build_final_dataset(**inputs)["splits"]["train"] == ["train-1"]


def test_finalize_supports_selected_split_without_pair_data():
    inputs = _finalize_inputs()
    inputs["pair_data"] = None
    assert build_final_dataset(**inputs)["splits"]["train"] == ["train-1"]


def test_finalize_partial_export_uses_only_completed_tasks():
    inputs = _finalize_inputs()
    inputs["selected"].append({**inputs["selected"][0], "sample_id": "pending"})
    inputs["allow_partial"] = True
    result = build_final_dataset(**inputs)
    assert [row["sample_id"] for row in result["samples"]] == ["train-1"]
    assert result["manifest"]["partial"] is True


@pytest.mark.parametrize(
    ("positive_ids", "error"),
    [
        ([], "non-empty and unique"),
        (["1_2", "1_2"], "non-empty and unique"),
        (["1_3"], "missing seed positive"),
        (["1_2", "1_1"], "query image is positive"),
        (["1_2", "1_7"], "invalid positive"),
    ],
)
def test_finalize_keeps_positive_integrity_checks(positive_ids, error):
    inputs = _finalize_inputs()
    inputs["positive_sets"][0]["positive_image_ids"] = positive_ids
    with pytest.raises(ValueError, match=error):
        build_final_dataset(**inputs)


# CLIP configuration, ordering and feature reuse.


def test_openai_weights_use_quickgelu(clip_rerank, monkeypatch):
    calls = []

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return Model(), None, "preprocess"

    monkeypatch.setattr(
        clip_rerank,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=create_model,
            get_tokenizer=lambda name: "tokenizer",
        ),
    )
    ranker = clip_rerank.ClipCandidateRanker()
    assert calls == [("ViT-B-32", {"pretrained": "openai", "force_quick_gelu": True})]
    assert ranker._cached_features == {}


def test_image_features_are_read_once_and_keep_requested_order(
    clip_rerank, tmp_path, monkeypatch
):
    monkeypatch.setattr(clip_rerank, "CACHE_ROOT", tmp_path)
    (tmp_path / "train").mkdir()
    np.save(tmp_path / "train/a.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / "train/b.npy", np.array([0.0, 1.0], dtype=np.float32))
    original_load = np.load
    loaded = []

    def load(path, **kwargs):
        loaded.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(clip_rerank.np, "load", load)
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    ranker._cached_features = {}
    candidates = [("a", "/images/train/a.jpg"), ("b", "/images/train/b.jpg")]
    np.testing.assert_array_equal(ranker._image_features(candidates), np.eye(2))
    np.testing.assert_array_equal(
        ranker._image_features(candidates[::-1]), np.eye(2)[::-1]
    )
    assert len(loaded) == 2


def test_clip_ranks_by_max_of_independent_image_and_text_cosines(
    clip_rerank, monkeypatch
):
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    ranker.device = "cpu"
    ranker.model = SimpleNamespace(
        encode_text=lambda text: torch.tensor([[0.0, 4.0, 0.0]])
    )
    text_inputs = []

    def tokenize(text):
        text_inputs.append(text)
        return torch.zeros(1, 1, dtype=torch.long)

    ranker.tokenizer = tokenize
    reference = ("target", "/images/target.jpg")
    candidates = [("a", "a"), ("b", "b"), ("c", "c")]

    def image_features(images):
        assert images == [reference, *candidates]
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.6],
                [0.0, 0.9, np.sqrt(0.19)],
                [0.7, 0.7, np.sqrt(0.02)],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(ranker, "_image_features", image_features)
    image_scores, text_scores = ranker._similarities(
        reference, "the condition", candidates
    )
    np.testing.assert_allclose(image_scores, [0.8, 0.0, 0.7])
    np.testing.assert_allclose(text_scores, [0.0, 0.9, 0.7])
    assert ranker(reference, "the condition", candidates) == ["b", "a", "c"]
    assert text_inputs == [["the condition"], ["the condition"]]


@pytest.mark.parametrize(
    ("image_scores", "text_scores", "expected"),
    [
        ([0.4, 0.8, 0.8], [0.8, 0.2, 0.1], ["a", "b", "c"]),
        ([-0.2, -0.8, -0.4], [-0.9, -0.6, -0.3], ["b", "c", "a"]),
    ],
)
def test_clip_max_keeps_raw_negative_scores_and_deterministic_ties(
    clip_rerank, monkeypatch, image_scores, text_scores, expected
):
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    monkeypatch.setattr(
        ranker,
        "_similarities",
        lambda *args: (np.array(image_scores), np.array(text_scores)),
    )
    assert ranker(("q", "q"), "text", [("b", "b"), ("a", "a"), ("c", "c")]) == expected


def test_clip_empty_candidates_need_no_image_or_text_encoding(clip_rerank):
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    assert ranker(("q", "missing.jpg"), "text", []) == []


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_clip_max_rejects_non_finite_scores(clip_rerank, monkeypatch, bad_score):
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    monkeypatch.setattr(
        ranker, "_similarities", lambda *args: (np.array([bad_score]), np.array([0.5]))
    )
    with pytest.raises(ValueError, match="non-finite"):
        ranker(("q", "q"), "text", [("a", "a")])


@pytest.mark.parametrize("interrupted_write", [False, True])
def test_clip_recovers_an_incomplete_image_cache_and_publishes_atomically(
    clip_rerank, monkeypatch, tmp_path, interrupted_write
):
    from PIL import Image

    image_root = tmp_path / "images"
    (image_root / "train").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image_root / "train/a.png")
    cache_root = tmp_path / "cache"
    (cache_root / "train").mkdir(parents=True)
    cache_path = cache_root / "train/a.npy"
    cache_path.write_bytes(b"incomplete previous cache")
    monkeypatch.setattr(clip_rerank, "IMAGE_ROOT", image_root)
    monkeypatch.setattr(clip_rerank, "CACHE_ROOT", cache_root)
    ranker = clip_rerank.ClipCandidateRanker.__new__(clip_rerank.ClipCandidateRanker)
    ranker._cached_features = {}
    ranker.device = "cpu"
    ranker.preprocess = lambda image: torch.zeros(3, 4, 4)
    ranker.model = SimpleNamespace(
        encode_image=lambda images: torch.tensor([[3.0, 4.0]])
    )
    images = [("a", "/images/train/a.png")]
    if interrupted_write:
        with monkeypatch.context() as failed_write:

            def fail_save(handle, feature):
                handle.write(b"unfinished new cache")
                raise RuntimeError("injected write failure")

            failed_write.setattr(clip_rerank.np, "save", fail_save)
            with pytest.raises(RuntimeError, match="injected write failure"):
                ranker._image_features(images)
        assert cache_path.read_bytes() == b"incomplete previous cache"
        assert not cache_path.with_suffix(".npy.tmp").exists()
    np.testing.assert_allclose(ranker._image_features(images), [[0.6, 0.8]])
    np.testing.assert_allclose(np.load(cache_path), [0.6, 0.8])
    assert not cache_path.with_suffix(".npy.tmp").exists()
