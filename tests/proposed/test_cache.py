import pytest
import torch

from rcr.common.config import load_config
from rcr.proposed.cache.build import check_cache_config, fafa_spec
from rcr.proposed.cache.store import GalleryCache


def test_gallery_cache_loads_person_index_and_topm_features(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    torch.save(
        {
            "image_ids": ["a.jpg", "b.jpg"],
            "persons": torch.randn(2, 3, 8),
            "mask": torch.tensor([[True, True, False], [True, True, True]]),
            "patch_hw": (2, 3),
        },
        tmp_path / "index.pt",
    )

    for index, count in enumerate((2, 3)):
        torch.save(
            {
                "scene": torch.randn(6, 8),
                "persons": torch.randn(count, 8),
                "identity_ids": [str(i) for i in range(count)],
                "boxes_scene": torch.rand(count, 4),
            },
            feature_dir / f"{index}.pt",
        )

    cache = GalleryCache(tmp_path)
    assert cache.image_ids == ["a.jpg", "b.jpg"]
    assert cache.persons.shape == (2, 3, 8)
    assert cache.patch_hw == (2, 3)
    cache.validate_gallery(["a.jpg", "b.jpg"])
    with pytest.raises(ValueError, match="gallery IDs/order"):
        cache.validate_gallery(["b.jpg", "a.jpg"])

    scene, persons, boxes, ids, mask = cache.load(torch.tensor([1, 0]))
    assert scene.shape == (2, 6, 8)
    assert persons.shape == (2, 3, 8)
    assert boxes.shape == (2, 3, 4)
    assert ids[0] == ["0", "1", "2"]
    assert ids[1] == ["0", "1", None]
    assert mask.tolist() == [[True, True, True], [True, True, False]]


def test_gallery_cache_refuses_incomplete_or_mixed_build(tmp_path) -> None:
    (tmp_path / "features").mkdir()
    torch.save(
        {
            "cache_id": "new-build",
            "image_ids": ["photo"],
            "persons": torch.ones(1, 1, 4),
            "mask": torch.ones(1, 1, dtype=torch.bool),
            "patch_hw": (1, 2),
        },
        tmp_path / "index.pt",
    )
    torch.save(
        {
            "cache_id": "old-build",
            "image_id": "photo",
            "scene": torch.ones(2, 4),
            "persons": torch.ones(1, 4),
            "identity_ids": [None],
            "boxes_scene": torch.ones(1, 4),
        },
        tmp_path / "features" / "0.pt",
    )

    (tmp_path / ".building").write_text("new-build")
    with pytest.raises(RuntimeError, match="incomplete"):
        GalleryCache(tmp_path)
    (tmp_path / ".building").unlink()
    cache = GalleryCache(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        cache.load(torch.tensor([0]))
    with pytest.raises(ValueError, match="does not match"):
        _ = cache.global_features


def test_legacy_global_features_pool_once_without_writing(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "features").mkdir()
    scenes = torch.randn(2, 6, 4).half()
    torch.save(
        {
            "image_ids": ["a", "b"],
            "persons": torch.empty(2, 0, 4),
            "mask": torch.empty(2, 0, dtype=torch.bool),
            "patch_hw": (2, 3),
        },
        tmp_path / "index.pt",
    )
    for i, scene in enumerate(scenes):
        torch.save({"scene": scene}, tmp_path / "features" / f"{i}.pt")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    cache = GalleryCache(tmp_path)
    actual = cache.global_features
    torch.testing.assert_close(actual, scenes.float().mean(dim=1))

    def fail(*args, **kwargs):
        raise AssertionError("global vectors must not reread scene files")

    monkeypatch.setattr(torch, "load", fail)
    assert cache.global_features is actual
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_known_legacy_dino_source_reused_by_fafa_without_writes(tmp_path):
    source = tmp_path / "dino"
    source.mkdir()
    torch.save(
        {
            "cache_id": "known-legacy-dino",
            "image_ids": ["photo"],
            "persons": torch.ones(1, 1, 8),
            "mask": torch.ones(1, 1, dtype=torch.bool),
            "patch_hw": (14, 14),
        },
        source / "index.pt",
    )
    persons = tmp_path / "fafa"
    persons.mkdir()
    cfg = load_config("configs/proposed.yaml")
    native = load_config(cfg["person_encoder"]["fafa_config"])
    torch.save(
        {
            "format_version": 3,
            "layout": "persons",
            "cache_id": "known-fafa",
            "source_cache_id": "known-legacy-dino",
            "image_ids": ["photo"],
            "persons": torch.ones(1, 1, 16),
            "mask": torch.ones(1, 1, dtype=torch.bool),
            "patch_hw": (14, 14),
            "scene_dim": 8,
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
            "person_encoder": {"backend": "fafa", "spec": fafa_spec(native)},
        },
        persons / "index.pt",
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.pt")}
    cache = GalleryCache(persons, scene_root=source)
    check_cache_config(cache, cfg)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.pt")}
    with pytest.raises(ValueError, match="allow_legacy_dino"):
        check_cache_config(cache, {**cfg, "cache": {"allow_legacy_dino": False}})
    with pytest.raises(ValueError, match="allow_legacy_dino"):
        check_cache_config(cache._scene_cache, cfg)  # DINO cannot replace FAFA.
    cache._scene_cache.patch_hw = (7, 7)
    with pytest.raises(ValueError, match="patch grid"):
        check_cache_config(cache, cfg)
