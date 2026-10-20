"""Packed image indices preserve evidence while bounding disk and CPU copies."""

import weakref
from pathlib import Path

import pytest
import torch

from rcr.proposed.cache import fafa as extraction
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import _encode_gallery_identity
from tests.proposed.test_runner import experiment as experiment
from tests.proposed.test_runner import local_stages as local_stages
from tests.proposed.test_split_cache import split_cache as split_cache


def _uneven_source(source):
    index = torch.load(source / "index.pt", weights_only=True)
    counts = [0, 3, 1, 2, 1, 0, 2, 1, 4]
    dim = index["persons"].shape[-1]
    index["persons"] = torch.zeros(len(counts), 6, dim)
    index["mask"] = torch.zeros(len(counts), 6, dtype=torch.bool)
    for i, count in enumerate(counts):
        index["mask"][i, [1, 3, 4, 5][:count]] = True
    torch.save(index, source / "index.pt")
    for i, count in enumerate(counts):
        path = source / "features" / f"{i}.pt"
        item = torch.load(path, weights_only=True)
        item["persons"] = torch.zeros(count, dim)
        for key in ("boxes_scene", "boxes_pixel"):
            item[key] = item[key][:1].repeat(count, 1)
        item["identity_ids"] = ["p1"] * count
        torch.save(item, path)
    return counts


@pytest.mark.parametrize("mmap_shards", [False, True])
def test_scene_and_target_io_mode_preserves_values(
    split_cache, monkeypatch, mmap_shards
):
    cfg, _, source, _, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    cache = GalleryCache(root, scene_root=source, mmap_shards=mmap_shards)
    reads = []
    original = torch.load

    def load(path, **kwargs):
        reads.append((Path(path).parent.name, kwargs.get("mmap")))
        return original(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    indices = torch.tensor([0, 1, 0])
    scene, persons, _, _ = cache.load(indices)
    targets = cache.native_targets(indices, persons.shape[1])
    assert reads and all(mode == mmap_shards for _, mode in reads)
    assert {group for group, _ in reads} == {"features", "targets"}
    expected = GalleryCache(root, scene_root=source, mmap_shards=not mmap_shards)
    torch.testing.assert_close(scene, expected.load(indices)[0], rtol=0, atol=0)
    torch.testing.assert_close(
        targets, expected.native_targets(indices, persons.shape[1]), rtol=0, atol=0
    )


@pytest.mark.parametrize("legacy_version", [4, 5])
def test_packed_and_legacy_indices_have_identical_coarse_inputs_and_outputs(
    split_cache, legacy_version
):
    cfg, _, source, _, _ = split_cache
    counts = _uneven_source(source)
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    index = torch.load(root / "index.pt", weights_only=True)
    assert index["format_version"] == 6
    assert index["persons"].shape == (sum(counts), 5)
    assert "global_features" not in index
    assert index["person_offsets"].tolist() == [0, 0, 3, 4, 6, 7, 7, 9, 10, 14]
    # Distinct rows make crop-order/offset mistakes observable.
    index["persons"] = torch.arange(sum(counts) * 5).reshape(-1, 5).half()
    torch.save(index, root / "index.pt")
    packed = GalleryCache(root, scene_root=source)
    dense = torch.zeros(len(counts), 6, 5, dtype=torch.float16)
    for i in range(len(counts)):
        dense[i, packed.mask[i]] = packed.person_row(i)
    old_root = root.with_name("old-images")
    old_root.mkdir()
    index.update(format_version=legacy_version, persons=dense)
    index.pop("person_offsets")
    if legacy_version == 4:
        index.pop("signature")
    torch.save(index, old_root / "index.pt")
    old = GalleryCache(old_root, scene_root=source)
    indices = torch.tensor([8, 0, 3, 8, 1, 5])
    useful = packed.mask[indices].any(dim=0)
    for cache in (packed, old):
        persons, mask = cache.load_persons(indices)
        torch.testing.assert_close(persons, dense[indices][:, useful], rtol=0, atol=0)
        assert torch.equal(mask, packed.mask[indices][:, useful])
        assert cache.load_persons(torch.tensor([0, 5]))[0].shape == (2, 0, 5)
    model = RCRModel(8, 6, 2, input_dim=packed.scene_dim, person_input_dim=5).eval()
    with torch.inference_mode():
        a = _encode_gallery_identity(packed, model, torch.device("cpu"), 2, indices)
        b = _encode_gallery_identity(old, model, torch.device("cpu"), 2, indices)
    for (a_id, a_mask), (b_id, b_mask) in zip(a, b, strict=True):
        torch.testing.assert_close(a_id, b_id, rtol=0, atol=0)
        assert torch.equal(a_mask, b_mask)


def test_large_gallery_padding_is_not_reserved_or_materialized(
    split_cache, monkeypatch
):
    cfg, _, source, _, _ = split_cache
    index = torch.load(source / "index.pt", weights_only=True)
    n, _, dim = index["persons"].shape
    index["persons"] = index["persons"].expand(n, 8192, dim)
    index["mask"] = torch.zeros(n, 8192, dtype=torch.bool)
    index["mask"][:, 0] = True
    torch.save(index, source / "index.pt")
    reservations = []
    monkeypatch.setattr(
        extraction, "require_disk_space", lambda _, size: reservations.append(size)
    )
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    saved = torch.load(root / "index.pt", weights_only=True)
    assert saved["persons"].shape == (n, 5)
    assert saved["persons"].untyped_storage().nbytes() == n * 5 * 2
    old_padded_bytes = n * 8192 * 5 * 2
    assert reservations[0] < old_padded_bytes // 3
    assert (root / "index.pt").stat().st_size < old_padded_bytes // 3


def test_invalid_or_unrelated_shards_do_not_reduce_space_reservation(
    split_cache, monkeypatch
):
    cfg, _, _, _, _ = split_cache
    sizes = []

    def stop(_, size):
        sizes.append(size)
        raise OSError("no space")

    monkeypatch.setattr(extraction, "require_disk_space", stop)
    with pytest.raises(OSError, match="no space"):
        extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    for path in (root / ".persons/0.pt", root / "targets/9999.pt"):
        path.write_bytes(b"invalid" * 100_000)
    with pytest.raises(OSError, match="no space"):
        extraction.finish_fafa_cache(cfg, stage="fafa-image")
    assert sizes[0] == sizes[1]


def test_publication_failure_resumes_without_reextracting_images(
    split_cache, monkeypatch
):
    cfg, _, source, model, _ = split_cache
    save = extraction.save_torch

    def interrupt(value, path):
        if path.name == "index.pt":
            raise OSError("publication interrupted")
        return save(value, path)

    monkeypatch.setattr(extraction, "save_torch", interrupt)
    with pytest.raises(OSError, match="publication interrupted"):
        extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    target = (root / "targets/0.pt").read_bytes()
    cache_id = (root / ".building").read_text()
    monkeypatch.setattr(extraction, "save_torch", save)
    monkeypatch.setattr(
        model, "extract_features", lambda *a, **kw: pytest.fail("re-extracted")
    )
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    assert GalleryCache(root, scene_root=source).cache_id == cache_id
    assert (root / "targets/0.pt").read_bytes() == target
    assert not (root / ".persons").exists()


@pytest.mark.parametrize("fault", ["missing", "count", "dtype", "length"])
def test_bad_packed_offsets_are_rejected(split_cache, fault):
    cfg, _, source, _, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    root = Path(cfg["data"]["fafa_image_cache"])
    index = torch.load(root / "index.pt", weights_only=True)
    if fault == "missing":
        index.pop("person_offsets")
    elif fault == "count":
        index["person_offsets"][1] += 1
    elif fault == "dtype":
        index["person_offsets"] = index["person_offsets"].float()
    else:
        index["person_offsets"] = index["person_offsets"][:-1]
    torch.save(index, root / "index.pt")
    with pytest.raises(ValueError, match="index"):
        GalleryCache(root, scene_root=source)


def test_target_loader_reads_unique_shards_once_and_releases_each(
    split_cache, monkeypatch
):
    cfg, _, source, _, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    gallery = GalleryCache(cfg["data"]["fafa_image_cache"], scene_root=source)
    read = gallery.native._read
    refs, calls = [], []

    def track(*args):
        assert all(ref() is None for ref in refs)
        features = read(*args)
        refs.append(weakref.ref(features))
        calls.append(args[1])
        return features

    monkeypatch.setattr(gallery.native, "_read", track)
    result = gallery.native_targets(torch.tensor([2, 0, 2, 1, 0]), 1)
    assert calls == [2, 0, 1]
    assert all(ref() is None for ref in refs)
    torch.testing.assert_close(result[0], result[2], rtol=0, atol=0)
    torch.testing.assert_close(result[1], result[4], rtol=0, atol=0)


def test_fafa_shards_have_a_separate_lru_without_changing_features(
    split_cache, monkeypatch
):
    from rcr.common.data import load_rcr_inputs

    cfg, _, source, _, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa")
    root = Path(cfg["data"]["fafa_image_cache"])
    gallery = GalleryCache(
        root,
        scene_root=source,
        query_root=cfg["data"]["fafa_query_cache"],
        lru_mib=1,
        native_lru_mib=1,
    )
    sample = load_rcr_inputs(
        cfg["data"]["final_dir"], cfg["data"]["image_root"]
    ).samples[:1]
    reads = []
    load = torch.load

    def track(path, **kwargs):
        reads.append(Path(path))
        return load(path, **kwargs)

    monkeypatch.setattr(torch, "load", track)
    indices = torch.tensor([0, 1, 0])
    scenes = gallery.load(indices)
    targets = gallery.native_targets(indices, 1)
    query = gallery.native_queries(sample, 1)
    first_reads = len(reads)
    assert targets.dtype == torch.float16
    # Public outputs are copies; mutating them cannot poison cached evidence.
    expected = targets.clone()
    targets.zero_()
    torch.testing.assert_close(
        gallery.native_targets(indices, 1), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(gallery.native_queries(sample, 1), query, rtol=0, atol=0)
    torch.testing.assert_close(gallery.load(indices)[0], scenes[0], rtol=0, atol=0)
    assert len(reads) == first_reads == 5  # two scenes, two targets, one query
    assert gallery._bytes == sum(size for _, size in gallery._items.values())
    assert gallery._bytes <= 1024**2

    assert gallery._native_bytes == sum(
        size for _, size in gallery._native_items.values()
    )
    assert gallery._native_bytes <= 1024**2
    # Repeated large native shards may evict each other, never the scene working set.
    for n in range(3):
        gallery._cached(("fafa-target", "pressure", n), lambda: torch.zeros(200_000))
    torch.testing.assert_close(gallery.load(indices)[0], scenes[0], rtol=0, atol=0)
    assert len(reads) == first_reads
    assert all(isinstance(key, int) for key in gallery._items)
    assert gallery._native_bytes <= 1024**2
