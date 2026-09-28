"""Tests for canonical indexed-gallery construction."""

from pathlib import Path

import pytest

from rcr.dataset.gallery import build_indexed_gallery, indexed_image_ids


def _index() -> list[str]:
    return [
        "1 1 10 20 30 40 7 1",
        "1 1 50 20 20 30 8 1",
        "1 2 10 10 20 20 7 1",
        "2 3 20 20 20 20 9 1",
    ]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_indexed_image_ids_preserves_first_seen_order() -> None:
    assert indexed_image_ids(_index()) == ["1_1", "1_2", "2_3"]


def test_build_indexed_gallery_uses_all_and_only_indexed_images(tmp_path) -> None:
    _touch(tmp_path / "train" / "1_1.jpg")
    _touch(tmp_path / "val" / "1_2.jpg")
    _touch(tmp_path / "test" / "2_3.jpg")
    _touch(tmp_path / "train" / "9_9.jpg")

    gallery = build_indexed_gallery(_index(), tmp_path)

    assert gallery == [
        {
            "image_id": "1_1",
            "path": "train/1_1.jpg",
            "url": "/images/train/1_1.jpg",
        },
        {
            "image_id": "1_2",
            "path": "val/1_2.jpg",
            "url": "/images/val/1_2.jpg",
        },
        {
            "image_id": "2_3",
            "path": "test/2_3.jpg",
            "url": "/images/test/2_3.jpg",
        },
    ]


def test_build_indexed_gallery_rejects_missing_indexed_image(tmp_path) -> None:
    _touch(tmp_path / "train" / "1_1.jpg")
    _touch(tmp_path / "val" / "1_2.jpg")

    with pytest.raises(ValueError, match="2_3"):
        build_indexed_gallery(_index(), tmp_path)


def test_build_indexed_gallery_follows_split_symlinks(tmp_path) -> None:
    source = tmp_path / "source"
    _touch(source / "1_1.jpg")
    _touch(source / "1_2.jpg")
    _touch(source / "2_3.jpg")

    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "train").symlink_to(source, target_is_directory=True)

    gallery = build_indexed_gallery(_index(), image_root)

    assert [row["path"] for row in gallery] == [
        "train/1_1.jpg",
        "train/1_2.jpg",
        "train/2_3.jpg",
    ]
