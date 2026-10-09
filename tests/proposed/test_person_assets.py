"""Release native base weights only after frozen FAFA features are reusable."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rcr.proposed.cache import build as builder
from rcr.proposed.cache.clip import FeatureCache, build_clip_cache
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import CLIPFeatures
from tests.proposed.test_text_and_pipeline import experiment as experiment
from tests.proposed.test_text_and_pipeline import tiny_clip as tiny_clip
from tools import run as cli


@pytest.fixture
def prepared_base_assets(experiment):
    cfg, data = experiment
    native_path = Path(cfg["person_encoder"]["fafa_config"])
    native = builder.load_fafa_config(cfg)
    root = data.final_dir.parent / "native-runtime"
    marker = data.final_dir.parent / "runtime-assets.json"
    native["checkpoint"].update(cache_root=str(root), runtime_assets_marker=str(marker))
    native_path.write_text(yaml.safe_dump(native))
    known = {
        "torch/hub/checkpoints/eva_vit_g.pth": b"EVA base weights",
        "torch/hub/checkpoints/blip2_pretrained.pth": b"BLIP2 base weights",
        "huggingface/hub/models--bert-base-uncased/blobs/weights": b"BERT weights",
    }
    unrelated = {
        "torch/hub/checkpoints/another-model.pth": b"keep this model",
        "other.pt": b"keep this file",
    }
    for name, contents in {**known, **unrelated}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    link = root / "huggingface/hub/models--bert-base-uncased/snapshots/main/model.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to("../../blobs/weights")
    saved = {
        "source_commit": native["source"]["commit"],
        "model_name": native["checkpoint"]["model_name"],
        "model_type": native["checkpoint"]["model_type"],
        "files": [
            {"path": str(path.relative_to(root)), "size": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()
        ],
    }
    marker.write_text(json.dumps(saved))
    return cfg, data, root, marker, native, known, unrelated, link


def test_release_keeps_features_tuned_weights_and_resumes_clip(
    prepared_base_assets, monkeypatch
):
    cfg, data, root, marker, native, known, unrelated, link = prepared_base_assets
    source = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    paths = [source.root / "index.pt", source._scene_cache.root / "index.pt"]
    before = [path.read_bytes() for path in paths]
    tuned = Path(native["checkpoint"]["path"]).read_bytes()
    original = CLIPFeatures.image
    calls = 0

    def interrupt(self, pixels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted CLIP cache")
        return original(self, pixels)

    monkeypatch.setattr(CLIPFeatures, "image", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_clip_cache(cfg, data, source)
    shard = Path(cfg["data"]["clip_cache"]) / "features/0.pt"
    shard_before = shard.read_bytes()
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *a, **kw: pytest.fail("native FAFA loaded")
    )
    config = data.final_dir.parent / "saved-run.yaml"
    config.write_text(yaml.safe_dump(cfg))
    cli.main(["prepare", "--config", str(config)])
    monkeypatch.setattr(CLIPFeatures, "image", original)
    # The normal clip-stage path performs the configured asset release itself.
    builder.prepare_cache(cfg, stage="clip")
    assert all(not (root / name).exists() for name in known)
    assert not link.is_symlink() and not marker.exists()
    assert all((root / name).read_bytes() == value for name, value in unrelated.items())
    assert Path(native["checkpoint"]["path"]).read_bytes() == tuned
    assert [path.read_bytes() for path in paths] == before
    assert shard.read_bytes() == shard_before
    loaded = FeatureCache(cfg, data).load(torch.tensor([0]))
    assert torch.isfinite(loaded["persons"]).all()
    # A later val recipe must not reacquire the released model assets.
    cli.main(["prepare", "--config", str(config)])
    assert not marker.exists()


def test_incomplete_person_cache_prevents_cleanup_and_still_prepares(
    prepared_base_assets, monkeypatch
):
    cfg, _, root, marker, _, known, _, _ = prepared_base_assets
    (Path(cfg["data"]["cache"]) / ".building").write_text("interrupted")
    with pytest.raises(ValueError, match="before its feature cache is complete"):
        builder.release_fafa_assets(cfg)
    calls = []
    monkeypatch.setattr(builder, "run_person_worker", lambda *a, **kw: calls.append(kw))
    builder.prepare_person_assets(cfg)
    assert calls == [{"prepare": True, "force": False}]
    assert marker.exists() and all((root / name).exists() for name in known)


def test_force_preparation_remains_explicit(prepared_base_assets, monkeypatch):
    cfg, *_ = prepared_base_assets
    calls = []
    monkeypatch.setattr(builder, "run_person_worker", lambda *a, **kw: calls.append(kw))
    builder.prepare_person_assets(cfg, force=True)
    assert calls == [{"prepare": True, "force": True}]


@pytest.mark.parametrize("case", ["marker", "traversal", "changed", "protected"])
def test_unsafe_asset_release_leaves_all_files(prepared_base_assets, case):
    cfg, _, root, marker, native, known, unrelated, _ = prepared_base_assets
    saved = json.loads(marker.read_text())
    if case == "marker":
        saved["source_commit"] = "another-source"
    elif case == "traversal":
        saved["files"].append({"path": "../important.pt", "size": 10})
    elif case == "changed":
        (root / next(iter(known))).write_bytes(b"changed by another consumer")
    else:
        native["checkpoint"]["cache_root"] = str(root.parent)
        Path(cfg["person_encoder"]["fafa_config"]).write_text(yaml.safe_dump(native))
    marker.write_text(json.dumps(saved))
    before = {name: (root / name).read_bytes() for name in (*known, *unrelated)}
    with pytest.raises(ValueError):
        builder.release_fafa_assets(cfg)
    assert all((root / name).read_bytes() == value for name, value in before.items())
    assert marker.exists()


def test_release_opt_out_keeps_model_assets(prepared_base_assets, monkeypatch):
    cfg, _, root, marker, _, known, _, _ = prepared_base_assets
    cfg["cache"]["release_fafa_assets"] = False
    monkeypatch.setattr(
        builder, "release_fafa_assets", lambda *a: pytest.fail("cleanup enabled")
    )
    builder.prepare_cache(cfg, stage="clip")
    assert marker.exists() and all((root / name).exists() for name in known)


def test_asset_parent_symlink_cannot_delete_external_files(prepared_base_assets):
    cfg, _, root, marker, _, known, unrelated, _ = prepared_base_assets
    outside = root.parent / "external-assets"
    (root / "torch").rename(outside)
    (root / "torch").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="parent escapes"):
        builder.release_fafa_assets(cfg)
    assert all((root / name).read_bytes() == value for name, value in known.items())
    assert all((root / name).read_bytes() == value for name, value in unrelated.items())
    assert marker.exists()


def test_read_only_assets_are_kept(prepared_base_assets, monkeypatch):
    cfg, _, root, marker, _, known, _, _ = prepared_base_assets
    monkeypatch.setattr(
        "rcr.proposed.cache.build.os.statvfs",
        lambda path: SimpleNamespace(f_flag=os.ST_RDONLY),
    )
    builder.release_fafa_assets(cfg)
    assert marker.exists()
    assert all((root / name).read_bytes() == value for name, value in known.items())
