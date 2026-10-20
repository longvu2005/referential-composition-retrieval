"""Train GT is aligned independently; legacy label cleanup preserves features."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from rcr.proposed import supervision as alignment
from rcr.proposed.cache.dino import boxes_to_scene
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.supervision import TrainSupervision, align_identities
from tools.data.clean_cache import clean_cache


def _fixture(tmp_path):
    (tmp_path / "features").mkdir()
    ids = ["q", "p", "v"]
    torch.save(
        {
            "cache_id": "same-crops",
            "image_ids": ids,
            "persons": torch.ones(3, 3, 4),
            "mask": torch.ones(3, 3, dtype=torch.bool),
            "patch_hw": (1, 2),
        },
        tmp_path / "index.pt",
    )
    pixel = torch.tensor([[0, 0, 4, 10], [4, 0, 8, 10], [10, 0, 15, 10]]).float()
    for n, image_id in enumerate(ids):
        torch.save(
            {
                "cache_id": "same-crops",
                "image_id": image_id,
                "scene": torch.ones(2, 4),
                "persons": torch.ones(3, 4),
                "boxes_pixel": pixel,
                "boxes_scene": pixel / 20,
                "identity_ids": ["poisoned legacy label"] * 3,
            },
            tmp_path / "features" / f"{n}.pt",
        )
    heads = [
        {"identity_id": "a", "x": 1, "y": 1, "width": 1, "height": 1},
        {"identity_id": "b", "x": 5, "y": 1, "width": 1, "height": 1},
    ]
    data = SimpleNamespace(
        gallery_ids=ids,
        images_by_id={
            i: {"path": f"{'val' if i == 'v' else 'train'}/{i}.jpg"} for i in ids
        },
        image_path=lambda i: tmp_path / f"{i}.jpg",
        gt_head_boxes_by_image={
            "q": heads,
            "p": list(reversed(heads)),
            "v": [{**heads[0], "identity_id": "private-val"}],
        },
    )
    return data, GalleryCache(tmp_path, lru_mib=1)


def test_train_alignment_ignores_legacy_labels_and_refuses_val(tmp_path):
    data, cache = _fixture(tmp_path)
    supervision = TrainSupervision(data, cache)
    assert set(supervision.heads) == {"q", "p"}
    assert supervision.load_ids(["q", "p"], 4) == [["a", "b", None, None]] * 2
    assert all("identity_ids" not in item for item, _ in cache._items.values())
    with pytest.raises(ValueError, match="non-train"):
        supervision.load_ids(["v"], 3)
    assert supervision.load_ids(["q"], 3) == [["a", "b", None]]


def test_legacy_pixel_alignment_matches_letterbox_alignment(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 10)).save(image)
    persons = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]])
    item = {
        "boxes_pixel": persons,
        "boxes_scene": boxes_to_scene(persons, (20, 10), (32, 48)),
    }
    heads = [{"identity_id": "a", "x": 1, "y": 1, "width": 2, "height": 2}]
    expected = align_identities(item, heads)
    del item["boxes_pixel"]
    assert (
        align_identities(item, heads, image_path=image, scene_size=[48, 32]) == expected
    )
    with pytest.raises(ValueError, match="legacy box alignment"):
        align_identities(item, heads)


def test_clean_cache_removes_only_gt_and_keeps_cache_identity_and_tensors(tmp_path):
    data, cache = _fixture(tmp_path)
    before_index = (tmp_path / "index.pt").read_bytes()
    before = [
        torch.load(tmp_path / "features" / f"{i}.pt", weights_only=True)
        for i in range(3)
    ]
    assert clean_cache(tmp_path) == 3
    assert clean_cache(tmp_path) == 0
    assert (tmp_path / "index.pt").read_bytes() == before_index
    new_cache = GalleryCache(tmp_path)
    assert new_cache.cache_id == cache.cache_id
    for n, old in enumerate(before):
        new = torch.load(tmp_path / "features" / f"{n}.pt", weights_only=True)
        assert set(old) - set(new) == {"identity_ids"}
        for key, value in new.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, old[key], rtol=0, atol=0)
            else:
                assert value == old[key]
    assert TrainSupervision(data, new_cache).load_ids(["q"], 3) == [["a", "b", None]]


def test_persisted_legacy_alignment_needs_no_raw_image_or_scene_reads(
    tmp_path, monkeypatch
):
    data, cache = _fixture(tmp_path)
    for image_id in ("q", "p"):
        Image.new("RGB", (20, 20)).save(data.image_path(image_id))
    for index in (0, 1):
        path = tmp_path / "features" / f"{index}.pt"
        item = torch.load(path, weights_only=True)
        item.pop("boxes_pixel")
        torch.save(item, path)
    path = tmp_path / "train-supervision.json"
    first = TrainSupervision(data, cache, scene_size=[20, 20], cache_file=path)
    expected = first.load_ids(["q", "p"], 3)
    assert expected == [["a", "b", None]] * 2
    first.save()
    saved = path.read_bytes()

    def fail(*args, **kwargs):
        pytest.fail("reused train alignment must not read a scene or raw image")

    monkeypatch.setattr(cache, "_get_item", fail)
    monkeypatch.setattr(Image, "open", fail)
    monkeypatch.setattr(alignment, "atomic_write_text", fail)
    again = TrainSupervision(data, cache, scene_size=[20, 20], cache_file=path)
    assert again.load_ids(["p", "q"], 4) == [[*row, None] for row in expected]
    again.save()
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match="non-train"):
        again.load_ids(["v"], 3)


@pytest.mark.parametrize(
    "change", ["train_heads", "cache_id", "scene_size", "image_path"]
)
def test_persisted_alignment_is_invalidated_when_train_inputs_change(
    tmp_path, monkeypatch, change
):
    data, cache = _fixture(tmp_path)
    path = tmp_path / "train-supervision.json"
    first = TrainSupervision(data, cache, scene_size=[20, 20], cache_file=path)
    first.load_ids(["q"], 3)
    first.save()
    size = [20, 20]
    if change == "train_heads":
        data.gt_head_boxes_by_image["q"][0]["identity_id"] = "fresh"
    elif change == "cache_id":
        cache.cache_id = "different-crops"
    elif change == "scene_size":
        size = [32, 48]
    else:
        data.images_by_id["q"]["path"] = "train/new-q.jpg"

    calls = []
    original = alignment.align_identities

    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment, "align_identities", track)
    again = TrainSupervision(data, cache, scene_size=size, cache_file=path)
    assert again._ids == {}
    actual = again.load_ids(["q"], 3)
    assert calls == [1]
    assert actual[0][0] == ("fresh" if change == "train_heads" else "a")


def test_val_annotations_do_not_invalidate_train_alignment(tmp_path, monkeypatch):
    data, cache = _fixture(tmp_path)
    path = tmp_path / "train-supervision.json"
    first = TrainSupervision(data, cache, cache_file=path)
    expected = first.load_ids(["q"], 3)
    first.save()
    data.gt_head_boxes_by_image["v"][0]["identity_id"] = "changed-val"

    def fail(*args, **kwargs):
        pytest.fail("validation annotations must not enter the train cache key")

    monkeypatch.setattr(cache, "_get_item", fail)
    again = TrainSupervision(data, cache, cache_file=path)
    assert again.load_ids(["q"], 3) == expected
