"""Prepare only train query/positive labels; batches never need image I/O."""

import json

import pytest
import torch
from PIL import Image

from rcr.proposed import supervision as alignment
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.supervision import TrainSupervision
from tests.proposed.test_supervision import _fixture


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("seed_only", [False, True])
def test_prepared_labels_reuse_without_raw_images_or_full_features(
    tmp_path, monkeypatch, legacy, seed_only
):
    data, cache = _fixture(tmp_path)
    if legacy:
        for i, image_id in enumerate(("q", "p")):
            Image.new("RGB", (20, 20)).save(data.image_path(image_id))
            path = tmp_path / "features" / f"{i}.pt"
            item = torch.load(path, weights_only=True)
            item.pop("boxes_pixel")
            torch.save(item, path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.pt")}
    sample = {"query_image_id": "q", "positive_image_ids": ["p"]}
    if seed_only:
        sample["target_image_id"] = sample.pop("positive_image_ids")[0]
    calls = []
    load_geometry = cache.load_geometry

    def read_boxes(index):
        calls.append(index)
        item = load_geometry(index)
        assert set(item) <= {"boxes_scene", "boxes_pixel"}
        return item

    def fail(*args, **kwargs):
        pytest.fail("prepared labels must not read images/features or realign GT")

    monkeypatch.setattr(cache, "load_geometry", read_boxes)
    monkeypatch.setattr(cache, "_get_item", fail)
    path = tmp_path / "train-supervision.json"
    first = TrainSupervision(data, cache, scene_size=[20, 20], cache_file=path)
    first.prepare([sample, sample])
    assert calls == [0, 1]
    assert json.loads(path.read_text())["identities"] == {
        "q": ["a", "b", None], "p": ["a", "b", None]
    }
    assert not cache._items
    assert {p: p.read_bytes() for p in before} == before

    monkeypatch.setattr(cache, "load_geometry", fail)
    monkeypatch.setattr(Image, "open", fail)
    monkeypatch.setattr(alignment, "align_identities", fail)
    monkeypatch.setattr(alignment, "atomic_write_text", fail)
    again = TrainSupervision(data, cache, scene_size=[20, 20], cache_file=path)
    again.prepare([sample])
    assert again.load_ids(["p", "q"], 4) == [["a", "b", None, None]] * 2


def test_interrupted_preparation_saves_completed_images_and_resumes(
    tmp_path, monkeypatch
):
    data, cache = _fixture(tmp_path)
    load_geometry = cache.load_geometry
    path = tmp_path / "train-supervision.json"
    sample = {"query_image_id": "q", "positive_image_ids": ["p"]}

    def interrupt(index):
        if index == 1:
            raise KeyboardInterrupt
        return load_geometry(index)

    monkeypatch.setattr(cache, "load_geometry", interrupt)
    first = TrainSupervision(data, cache, cache_file=path)
    with pytest.raises(KeyboardInterrupt):
        first.prepare([sample])
    assert set(json.loads(path.read_text())["identities"]) == {"q"}
    calls = []

    def resume(index):
        calls.append(index)
        return load_geometry(index)

    monkeypatch.setattr(cache, "load_geometry", resume)
    again = TrainSupervision(data, cache, cache_file=path)
    again.prepare([sample])
    assert calls == [1]
    assert again.load_ids(["q", "p"], 3) == [["a", "b", None]] * 2


@pytest.mark.parametrize(
    "sample",
    [
        {"query_image_id": "v", "positive_image_ids": ["p"]},
        {"query_image_id": "q", "positive_image_ids": ["v"]},
    ],
)
def test_preparation_rejects_non_train_images_before_reading_boxes(
    tmp_path, monkeypatch, sample
):
    data, cache = _fixture(tmp_path)

    def fail(*args, **kwargs):
        pytest.fail("non-train boxes must not be opened")

    monkeypatch.setattr(cache, "load_geometry", fail)
    path = tmp_path / "train-supervision.json"
    supervision = TrainSupervision(data, cache, cache_file=path)
    with pytest.raises(ValueError, match="non-train"):
        supervision.prepare([sample])
    assert not path.exists()


def test_preparation_keeps_empty_detections_empty(tmp_path):
    data, _ = _fixture(tmp_path)
    path = tmp_path / "features" / "1.pt"
    item = torch.load(path, weights_only=True)
    item["persons"] = item["persons"][:0]
    item["boxes_pixel"] = item["boxes_pixel"][:0]
    item["boxes_scene"] = item["boxes_scene"][:0]
    torch.save(item, path)
    index_path = tmp_path / "index.pt"
    index = torch.load(index_path, weights_only=True)
    index["mask"][1] = False
    torch.save(index, index_path)
    supervision = TrainSupervision(data, GalleryCache(tmp_path))
    supervision.prepare([{"query_image_id": "q", "positive_image_ids": ["p"]}])
    assert supervision._ids["p"] == []
    assert supervision.load_ids(["p"], 3) == [[None, None, None]]


def test_prepared_object_never_aligns_a_missing_label_in_a_batch(tmp_path, monkeypatch):
    data, cache = _fixture(tmp_path)
    supervision = TrainSupervision(data, cache)
    supervision.prepare([{"query_image_id": "q", "positive_image_ids": []}])

    def fail(*args, **kwargs):
        pytest.fail("a prepared object must only look up labels")

    monkeypatch.setattr(cache, "load_geometry", fail)
    assert supervision.load_ids(["q"], 3) == [["a", "b", None]]
    with pytest.raises(RuntimeError, match="not prepared for p"):
        supervision.load_ids(["p"], 3)


def test_geometry_reader_maps_shards_and_returns_only_immutable_boxes(
    tmp_path, monkeypatch
):
    _, cache = _fixture(tmp_path)
    load = torch.load
    calls = []

    def track(path, *args, **kwargs):
        calls.append(kwargs.get("mmap"))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", track)
    item = cache.load_geometry(0)
    assert calls == [True]
    assert set(item) == {"boxes_scene", "boxes_pixel"}
    assert not cache._items
    torch.testing.assert_close(
        item["boxes_pixel"],
        torch.tensor([[0, 0, 4, 10], [4, 0, 8, 10], [10, 0, 15, 10]]).float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("key", ["boxes_scene", "boxes_pixel"])
def test_geometry_reader_rejects_boxes_that_differ_from_person_index(tmp_path, key):
    _fixture(tmp_path)
    index_path = tmp_path / "index.pt"
    index = torch.load(index_path, weights_only=True)
    index["format_version"] = 2
    torch.save(index, index_path)
    cache = GalleryCache(tmp_path)
    path = tmp_path / "features" / "0.pt"
    item = torch.load(path, weights_only=True)
    item[key] = item[key][:-1]
    torch.save(item, path)
    with pytest.raises(ValueError, match="box count differs"):
        cache.load_geometry(0)
