"""A gallery projection must not touch scene files outside its gallery."""

import pytest
import torch

from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import _encode_gallery_state


def _legacy(root, *, indexed=False, lru_mib=0, mmap_shards=False):
    (root / "features").mkdir()
    scenes = torch.arange(5 * 6 * 4).reshape(5, 6, 4).half()
    index = {
        "image_ids": [str(i) for i in range(5)],
        "persons": torch.empty(5, 0, 4),
        "mask": torch.empty(5, 0, dtype=torch.bool),
        "patch_hw": (2, 3),
    }
    if indexed:
        index["global_features"] = scenes.float().mean(1)
    torch.save(index, root / "index.pt")
    # Other splits deliberately have no shards on this fixture.
    for i in (1, 3):
        torch.save({"scene": scenes[i]}, root / "features" / f"{i}.pt")
    return GalleryCache(root, lru_mib=lru_mib, mmap_shards=mmap_shards), scenes


@pytest.mark.parametrize("mmap_shards", [False, True])
def test_scoped_means_preserve_order_and_survive_scene_eviction(
    tmp_path, monkeypatch, mmap_shards
):
    cache, scenes = _legacy(tmp_path, mmap_shards=mmap_shards)
    indices = torch.tensor([3, 1, 3])
    actual = cache.load_global_features(indices)
    torch.testing.assert_close(actual, scenes[indices].float().mean(1), rtol=0, atol=0)
    assert set(cache._global_rows) == {1, 3}
    assert not cache._items  # Retaining the small means needs no full-scene LRU.

    def fail(*args, **kwargs):
        pytest.fail("a previously pooled image must not be read again")

    monkeypatch.setattr(cache, "_get_item", fail)
    torch.testing.assert_close(cache.load_global_features(indices), actual)
    empty = cache.load_global_features(torch.tensor([], dtype=torch.long))
    assert empty.shape == (0, 4)


def test_gallery_state_reads_only_its_gallery(tmp_path):
    cache, scenes = _legacy(tmp_path)
    model = RCRModel(dim=4, identity_dim=2, state_dim=2, num_heads=2).eval()
    indices = torch.tensor([3, 1])
    with torch.no_grad():
        actual = _encode_gallery_state(cache, model, torch.device("cpu"), 1, indices)
        expected = model.encode_image_state(scenes[indices].float().mean(1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(cache._global_rows) == {1, 3}


def test_scoped_means_reuse_an_existing_scene_lru_entry(tmp_path, monkeypatch):
    cache, scenes = _legacy(tmp_path, lru_mib=1)
    cache._get_item(3)

    def fail(*args, **kwargs):
        pytest.fail("pooling must reuse the scene already loaded by training")

    monkeypatch.setattr(cache, "_read_feature", fail)
    torch.testing.assert_close(
        cache.load_global_features(torch.tensor([3])), scenes[3:4].float().mean(1)
    )


def test_indexed_means_need_no_scene_shards(tmp_path, monkeypatch):
    cache, scenes = _legacy(tmp_path, indexed=True)

    def fail(*args, **kwargs):
        pytest.fail("an index with global means must not load scene shards")

    monkeypatch.setattr(cache, "_get_item", fail)
    indices = torch.tensor([4, 0, 4])
    torch.testing.assert_close(
        cache.load_global_features(indices), scenes[indices].float().mean(1)
    )
