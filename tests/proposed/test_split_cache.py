import io
from pathlib import Path

import pytest
import torch
import yaml

from rcr.proposed.cache import build as builder
from rcr.proposed.cache import fafa as fafa_cache
from rcr.proposed.cache.build import check_cache_config
from rcr.proposed.cache.dino import boxes_to_scene
from rcr.proposed.cache.store import GalleryCache
from tests.proposed.test_person_representation import FakeFAFA
from tests.proposed.test_runner import experiment, local_stages  # noqa: F401
from tools import run as cli


@pytest.fixture
def split_cache(experiment, local_stages, tmp_path, monkeypatch):  # noqa: F811
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "dino"])
    source = Path(cfg["data"]["cache"])
    cfg["data"].update(dino_cache=str(source), cache=str(tmp_path / "persons"))
    cfg["person_encoder"]["backend"] = "fafa"
    native = yaml.safe_load(Path("configs/fafa.yaml").read_text())
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture FAFA weights")
    native["checkpoint"]["path"] = str(weights)
    native_path = tmp_path / "fafa.yaml"
    native_path.write_text(yaml.safe_dump(native))
    cfg["person_encoder"]["fafa_config"] = str(native_path)
    path.write_text(yaml.safe_dump(cfg))
    model = FakeFAFA()
    monkeypatch.setattr(
        fafa_cache,
        "load_fafa",
        lambda *args: (model, {}, lambda crop: torch.ones(3, 4, 4), {}),
    )
    worker_calls = []

    def worker(cfg, **options):
        worker_calls.append(options)
        fafa_cache.finish_fafa_cache(cfg)

    monkeypatch.setattr(builder, "run_person_worker", worker)
    return cfg, path, source, model, worker_calls


def _bytes(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_persons_only_reuses_dino_then_train_and_retrieve(split_cache, monkeypatch):
    cfg, path, source, _, worker_calls = split_cache
    before = _bytes(source)
    monkeypatch.setattr(
        builder, "_prepare_dino", lambda *args: pytest.fail("built DINO")
    )
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "persons"])
    assert worker_calls == [{}]
    assert _bytes(source) == before
    native_path = Path(cfg["person_encoder"]["fafa_config"])
    native_config = native_path.read_bytes()
    native_path.unlink()
    rows = cli.main(["run", "--config", str(path), "--train"])
    native_path.write_bytes(native_config)
    assert [row["split"] for row in rows] == ["val", "test"]
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    assert checkpoint["input_dim"] == 6 and checkpoint["person_input_dim"] == 5
    root = Path(cfg["data"]["cache"])
    saved = _bytes(root)
    worker_calls.clear()
    # A complete cache works even after raw images and the worker disappear.
    cfg["person_encoder"]["python"] = "missing-python"
    cfg["data"]["image_root"] = "missing-images"
    builder.prepare_cache(cfg)
    assert not worker_calls and _bytes(root) == saved and _bytes(source) == before


def _make_legacy(cfg, source):
    index = torch.load(source / "index.pt", weights_only=True)
    for key in (
        "format_version",
        "image_paths",
        "scene_dim",
        "person_encoder",
        "image_encoder",
        "detector",
        "global_features",
    ):
        index.pop(key, None)
    # Tiny backbone has 2-pixel patches; use an equivalent legacy 16-pixel grid.
    cfg["image_encoder"]["scene_size"] = [64, 64]
    for i in range(len(index["image_ids"])):
        path = source / "features" / f"{i}.pt"
        item = torch.load(path, weights_only=True)
        item.pop("boxes_pixel")
        torch.save(item, path)
    torch.save(index, source / "index.pt")


def test_legacy_dino_without_pixel_boxes_is_read_only(split_cache):
    cfg, _, source, _, _ = split_cache
    _make_legacy(cfg, source)
    before = _bytes(source)
    fafa_cache.finish_fafa_cache(cfg)
    cache = GalleryCache(cfg["data"]["cache"], scene_root=source)
    check_cache_config(cache, cfg)
    assert _bytes(source) == before
    scene, persons, _, _, _ = cache.load(torch.tensor([0]))
    assert scene.shape == (1, 16, 6) and persons.shape == (1, 1, 5)
    torch.testing.assert_close(cache.global_features[0], scene[0].mean(0))
    cfg["cache"]["allow_legacy_dino"] = False
    with pytest.raises(ValueError, match="allow_legacy_dino"):
        check_cache_config(cache, cfg)


def test_legacy_dino_ablation_resolves_the_shared_source(split_cache, tmp_path):
    cfg, path, source, _, _ = split_cache
    _make_legacy(cfg, source)
    path.write_text(yaml.safe_dump(cfg))
    suite = yaml.safe_load(Path("configs/ablations/representation.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(tmp_path / "ablations"))
    suite["experiments"] = [suite["experiments"][0], suite["experiments"][2]]
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    before = _bytes(source)
    rows = cli.main(["ablate", "--config", str(suite_path), "--splits", "val"])
    assert {row["person_backend"] for row in rows} == {"dino"}
    assert {row["representation"] for row in rows} == {"shared", "dual"}
    assert _bytes(source) == before


@pytest.mark.parametrize("image_size", [(73, 129), (129, 73), (101, 101)])
def test_legacy_box_transform_roundtrip(image_size):
    width, height = image_size
    boxes = torch.tensor(
        [[0.0, 0.0, float(width), float(height)], [3.0, 4.0, width - 2.0, height - 3.0]]
    )
    scene_size = (224, 224)
    scene = boxes_to_scene(boxes, image_size, scene_size)
    torch.testing.assert_close(
        fafa_cache.boxes_to_pixels(scene, image_size, scene_size),
        boxes,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fafa_resumes_saved_images_and_never_publishes_partial(
    split_cache, monkeypatch
):
    cfg, _, source, model, _ = split_cache
    root = Path(cfg["data"]["cache"])
    extract = model.extract_features
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return extract(*args, **kwargs)

    monkeypatch.setattr(model, "extract_features", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        fafa_cache.finish_fafa_cache(cfg)
    first = (root / "features/0.pt").read_bytes()
    manifest = torch.load(root / ".build.pt", weights_only=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        GalleryCache(root, scene_root=source)
    calls = 0

    def count_remaining(*args, **kwargs):
        nonlocal calls
        calls += 1
        return extract(*args, **kwargs)

    monkeypatch.setattr(model, "extract_features", count_remaining)
    fafa_cache.finish_fafa_cache(cfg)
    cache = GalleryCache(root, scene_root=source)
    assert calls == len(cache.image_ids) - 1
    first_persons = torch.load(io.BytesIO(first), weights_only=True)["persons"]
    torch.testing.assert_close(cache.persons[0, cache.mask[0]], first_persons)
    assert not (root / "features").exists()
    assert cache.cache_id == manifest["cache_id"]
    assert not (root / ".build.pt").exists()
    assert not (root / ".persons.bin").exists()


@pytest.mark.parametrize("mismatch", ["source_id", "gallery", "spec", "weights"])
def test_existing_cache_mismatch_fails_without_rebuilding(
    split_cache, monkeypatch, mismatch
):
    cfg, _, source, _, _ = split_cache
    fafa_cache.finish_fafa_cache(cfg)
    if mismatch in ("source_id", "gallery"):
        index = torch.load(source / "index.pt", weights_only=True)
        if mismatch == "source_id":
            index["cache_id"] = "other-source"
        else:
            index["image_ids"] = list(reversed(index["image_ids"]))
        torch.save(index, source / "index.pt")
    elif mismatch == "spec":
        cfg["image_encoder"]["person_size"] = [32, 16]
    elif mismatch == "weights":
        native = yaml.safe_load(Path(cfg["person_encoder"]["fafa_config"]).read_text())
        Path(native["checkpoint"]["path"]).write_bytes(b"different checkpoint")
    monkeypatch.setattr(
        builder, "_prepare_dino", lambda *args: pytest.fail("rebuilt DINO")
    )
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *args, **kw: pytest.fail("FAFA")
    )
    with pytest.raises(ValueError):
        builder.prepare_cache(cfg)


def test_complete_person_index_needs_only_dino_shards(split_cache, monkeypatch):
    cfg, _, source, _, _ = split_cache
    fafa_cache.finish_fafa_cache(cfg)
    root = Path(cfg["data"]["cache"])
    assert not (root / "features").exists()
    monkeypatch.setattr(
        builder, "_prepare_dino", lambda *args: pytest.fail("rebuilt DINO")
    )
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *args, **kw: pytest.fail("FAFA")
    )
    builder.prepare_cache(cfg)
    cache = GalleryCache(root, scene_root=source)
    _, persons, _, _, _ = cache.load(torch.tensor([0]))
    torch.testing.assert_close(persons[0], cache.persons[0, cache.mask[0]].float())
    # Older completed caches can retain shards. They must not make fine
    # scoring read different persons from the coarse index.
    (root / "features").mkdir()
    torch.save({"persons": torch.full_like(persons[0], -999)}, root / "features/0.pt")
    _, legacy_persons, _, _, _ = cache.load(torch.tensor([0]))
    torch.testing.assert_close(legacy_persons, persons)
    (source / "features/0.pt").unlink()
    with pytest.raises(FileNotFoundError):
        GalleryCache(root, scene_root=source).load(torch.tensor([0]))


def test_dino_stage_never_needs_fafa_runtime(split_cache, monkeypatch):
    cfg, _, _, _, _ = split_cache
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *a, **k: pytest.fail("FAFA")
    )
    builder.prepare_cache(cfg, stage="dino")


def test_person_worker_uses_the_new_script_and_passes_config(
    experiment,  # noqa: F811
    monkeypatch,
):
    cfg, _ = experiment
    cfg["person_encoder"].update(backend="fafa", python="native-python")
    calls = []

    def execute(command, *, check):
        assert check
        calls.append(command)
        assert yaml.safe_load(Path(command[3]).read_text()) == cfg
        assert Path(command[1]) == Path("tools/cache_fafa.py").resolve()

    monkeypatch.setattr(builder.subprocess, "run", execute)
    builder.run_person_worker(cfg, prepare=True, force=True)
    assert calls[0][0] == "native-python"
    assert calls[0][-2:] == ["--prepare", "--force"]


def test_persons_stage_requires_complete_dino(split_cache):
    cfg, _, _, _, worker_calls = split_cache
    cfg["data"]["dino_cache"] = "missing-source"
    with pytest.raises(ValueError, match="DINO cache missing"):
        builder.prepare_cache(cfg, stage="persons")
    assert not worker_calls


def test_empty_person_gallery_still_publishes_scene_cache(split_cache):
    cfg, _, source, _, _ = split_cache
    index = torch.load(source / "index.pt", weights_only=True)
    index["persons"] = index["persons"][:, :0]
    index["mask"] = index["mask"][:, :0]
    torch.save(index, source / "index.pt")
    for i in range(len(index["image_ids"])):
        path = source / "features" / f"{i}.pt"
        item = torch.load(path, weights_only=True)
        for key in ("persons", "boxes_scene", "boxes_pixel"):
            item[key] = item[key][:0]
        item["identity_ids"] = []
        torch.save(item, path)
    fafa_cache.finish_fafa_cache(cfg)
    cache = GalleryCache(cfg["data"]["cache"], scene_root=source)
    scene, persons, boxes, ids, mask = cache.load(torch.tensor([0]))
    assert persons.shape == (1, 0, 5) and boxes.shape == (1, 0, 4)
    assert ids == [[]] and not mask.any()
    torch.testing.assert_close(cache.global_features[0], scene[0].mean(0))
