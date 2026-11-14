"""Frozen person encoder asset preparation and checkpoint loading."""

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rcr.common.config import load_config
from rcr.proposed.cache import person_encoder


@pytest.fixture
def encoder_config(tmp_path):
    cfg = load_config("configs/person_encoder.yaml")
    settings = cfg["checkpoint"]
    settings["path"] = str(tmp_path / "weights" / "encoder.pt")
    settings["cache_root"] = str(tmp_path / "runtime")
    settings["runtime_assets_marker"] = str(tmp_path / "ready.json")
    return cfg


def test_prepare_reuses_local_weights_without_downloading(
    encoder_config, tmp_path, monkeypatch
):
    cfg = encoder_config
    local = tmp_path / "mounted.pt"
    local.write_bytes(b"prepared weights")
    cfg["checkpoint"]["local_path"] = str(local)
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace())
    monkeypatch.setattr(person_encoder, "official_source", lambda *a, **k: tmp_path)
    monkeypatch.setattr(person_encoder, "assets_ready", lambda _: True)
    prepared = []
    monkeypatch.setattr(
        person_encoder,
        "prepare_runtime_assets",
        lambda cfg, *, force: prepared.append(force),
    )

    person_encoder.prepare_fafa_model(cfg)
    checkpoint = Path(cfg["checkpoint"]["path"])
    assert checkpoint.is_symlink() and checkpoint.resolve() == local
    assert checkpoint.read_bytes() == b"prepared weights"
    assert not checkpoint.with_suffix(".pt.part").exists()
    person_encoder.prepare_fafa_model(cfg)
    assert prepared == [False, False]

    other = tmp_path / "other.pt"
    other.write_bytes(b"different weights")
    cfg["checkpoint"]["local_path"] = str(other)
    with pytest.raises(ValueError, match="differs from checkpoint.local_path"):
        person_encoder.prepare_fafa_model(cfg)
    assert checkpoint.resolve() == local


def test_load_restores_weights_and_query_policy_without_network(
    encoder_config, tmp_path, monkeypatch
):
    cfg = encoder_config
    checkpoint = Path(cfg["checkpoint"]["path"])
    checkpoint.parent.mkdir()
    torch.save({"model": {"weight": torch.full((2, 2), 3.0)}}, checkpoint)
    model = nn.Linear(2, 2, bias=False)
    processor = SimpleNamespace(max_words=50)
    monkeypatch.setattr(person_encoder, "official_source", lambda _: tmp_path)
    monkeypatch.setattr(person_encoder, "assets_ready", lambda _: True)
    monkeypatch.setattr(person_encoder, "runtime_cache", lambda *a, **k: tmp_path)

    def load_model(**kwargs):
        with socket.socket() as connection:
            with pytest.raises(RuntimeError, match="Network blocked"):
                connection.connect(("127.0.0.1", 9))
        return model, {}, {"eval": processor}

    monkeypatch.setattr(
        person_encoder,
        "official_api",
        lambda _: (lambda size, *, need_size: (size, need_size), load_model),
    )
    connect = socket.socket.connect
    loaded, processors, preprocess, details = person_encoder.load_fafa(
        cfg, torch.device("cpu")
    )

    torch.testing.assert_close(loaded.weight, torch.full((2, 2), 3.0))
    assert not loaded.training
    assert loaded.max_txt_len == processors["eval"].max_words == 128
    assert details["text_policy"] == {"max_txt_len": 128, "max_words": 128}
    assert details["missing_keys"] == details["unexpected_keys"] == []
    assert preprocess == (224, (384, 192))
    assert socket.socket.connect is connect

    monkeypatch.setattr(person_encoder, "assets_ready", lambda _: False)
    with pytest.raises(RuntimeError, match="runtime assets missing/stale"):
        person_encoder.load_fafa(cfg, torch.device("cpu"))
