"""Exercise checkpoint preparation against the pinned OpenAI CLIP API."""

import hashlib
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from rcr.baselines import prepare as prepare_baseline
from tools import run as cli


@pytest.mark.parametrize("name", ["ViT-L/14", "ViT-B/32"])
def test_prepare_clip_reads_native_registry(name, tmp_path, monkeypatch):
    native_clip = pytest.importorskip("clip")
    from clip.clip import _MODELS

    assert not hasattr(native_clip, "_MODELS")
    calls = []
    monkeypatch.setattr(
        prepare_baseline,
        "download",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    target = tmp_path / "checkpoint.pt"
    prepare_baseline.prepare_clip(name, str(target), force=False)
    url = _MODELS[name]
    assert calls == [
        ((url, target), {"force": False, "expected_hash": url.split("/")[-2]})
    ]


def test_prepare_cli_verifies_and_reuses_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("clip")
    from clip.clip import _MODELS

    contents = b"small checkpoint fixture"
    digest = hashlib.sha256(contents).hexdigest()
    source = tmp_path / "source" / digest / "model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(contents)
    monkeypatch.setitem(_MODELS, "ViT-L/14", source.as_uri())
    target = tmp_path / "prepared" / "checkpoint.pt"
    cfg = tmp_path / "clip.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"method": "clip", "model": {"name": "ViT-L/14", "checkpoint": str(target)}}
        )
    )
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(cfg)])
    cli.main(["prepare", *sys.argv[1:]])
    assert target.read_bytes() == contents
    assert not target.with_suffix(".pt.part").exists()

    # Cached files are reused only after their checksum is checked.
    source.unlink()
    cli.main(["prepare", *sys.argv[1:]])
    assert target.read_bytes() == contents

    source.write_bytes(contents)
    target.write_bytes(b"corrupted checkpoint")
    cli.main(["prepare", *sys.argv[1:]])
    assert target.read_bytes() == contents


@pytest.fixture
def fafa_preparation(tmp_path, monkeypatch):
    cfg = yaml.safe_load(Path("configs/fafa.yaml").read_text())
    cfg["checkpoint"]["path"] = str(tmp_path / "prepared" / "fafa.pt")
    path = tmp_path / "fafa.yaml"
    api = ModuleType("rcr.baselines.fafa")
    api.official_source = lambda cfg, prepare=False: tmp_path
    api.prepare_runtime_assets = lambda cfg, force=False: None
    monkeypatch.setitem(sys.modules, "rcr.baselines.fafa", api)
    monkeypatch.setattr(prepare_baseline, "prepare_clip", lambda *args: None)
    monkeypatch.setattr(prepare_baseline, "download", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(path)])
    return cfg, path


@pytest.mark.parametrize(
    "url",
    [
        "https://drive.google.com/file/d/test_checkpoint/view",
        "https://drive.google.com/file/d/test_checkpoint/view?usp=sharing",
        "https://drive.google.com/uc?id=test_checkpoint",
    ],
)
def test_fafa_prepare_uses_supported_gdown_api(fafa_preparation, monkeypatch, url):
    gdown = pytest.importorskip("gdown")

    cfg, config_path = fafa_preparation
    cfg["checkpoint"]["source_url"] = url
    config_path.write_text(yaml.safe_dump(cfg))
    native_signature = inspect.signature(gdown.download)
    calls = []

    def download(**kwargs):
        native_signature.bind(**kwargs)
        assert kwargs["url"] == "https://drive.google.com/uc?id=test_checkpoint"
        assert "fuzzy" not in kwargs
        calls.append(kwargs)
        Path(kwargs["output"]).write_bytes(b"checkpoint fixture")
        return kwargs["output"]

    monkeypatch.setattr(gdown, "download", download)
    cli.main(["prepare", *sys.argv[1:]])
    target = Path(cfg["checkpoint"]["path"])
    assert target.read_bytes() == b"checkpoint fixture"
    assert not target.with_suffix(".part").exists()
    cli.main(["prepare", *sys.argv[1:]])
    assert len(calls) == 1  # Reuse the prepared checkpoint.


def test_fafa_failed_download_preserves_existing_checkpoint(
    fafa_preparation, monkeypatch
):
    gdown = pytest.importorskip("gdown")

    cfg, config_path = fafa_preparation
    config_path.write_text(yaml.safe_dump(cfg))
    target = Path(cfg["checkpoint"]["path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous checkpoint")

    def fail_download(**kwargs):
        Path(kwargs["output"]).write_bytes(b"")
        return None

    monkeypatch.setattr(gdown, "download", fail_download)
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--config", str(config_path), "--force"]
    )
    with pytest.raises(RuntimeError, match="did not produce a file"):
        cli.main(["prepare", *sys.argv[1:]])
    assert target.read_bytes() == b"previous checkpoint"
    assert not target.with_suffix(".part").exists()
