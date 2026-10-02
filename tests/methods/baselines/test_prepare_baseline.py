"""Exercise checkpoint preparation against the pinned OpenAI CLIP API."""

import hashlib
import sys

import pytest
import yaml

from tools.methods import prepare_baseline


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
    prepare_baseline.main()
    assert target.read_bytes() == contents
    assert not target.with_suffix(".pt.part").exists()

    # Cached files are reused only after their checksum is checked.
    source.unlink()
    prepare_baseline.main()
    assert target.read_bytes() == contents

    source.write_bytes(contents)
    target.write_bytes(b"corrupted checkpoint")
    prepare_baseline.main()
    assert target.read_bytes() == contents
