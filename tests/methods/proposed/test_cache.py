import pytest
import torch

from rcr.methods.proposed.cache import GalleryCache


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
