"""Text limits reach the native loader and invalidate only query features."""

from pathlib import Path

import pytest
import torch
import yaml

from rcr.baselines import fafa
from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from tests.baselines.test_baselines import TinyFAFA, preprocess
from tests.baselines.test_baselines import benchmark as benchmark


class CaptionProcessor:
    max_words = 50

    def __call__(self, caption):
        return " ".join(caption.lower().split()[: self.max_words])


@pytest.mark.parametrize("length,expected", [(None, 32), (32, 32), (128, 128)])
def test_native_loader_applies_text_and_processor_limits(
    tmp_path, monkeypatch, length, expected
):
    cfg = yaml.safe_load(Path("configs/fafa.yaml").read_text())
    cfg["checkpoint"].pop("max_txt_len")
    if length is not None:
        cfg["checkpoint"]["max_txt_len"] = length
    model = torch.nn.Linear(2, 2)
    model.max_txt_len = 32  # The pinned from_config default.
    processor = CaptionProcessor()
    checkpoint = tmp_path / "native.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    cfg["checkpoint"]["path"] = str(checkpoint)
    monkeypatch.setattr(fafa, "official_source", lambda cfg: tmp_path)
    monkeypatch.setattr(fafa, "assets_ready", lambda cfg: True)
    monkeypatch.setattr(fafa, "runtime_cache", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(
        fafa,
        "official_api",
        lambda path: (
            lambda *a, **kw: preprocess,
            lambda **kw: (model, {}, {"eval": processor}),
        ),
    )
    loaded, processors, _, info = fafa.load_fafa(cfg, torch.device("cpu"))
    assert loaded.max_txt_len == expected
    assert processors["eval"].max_words == max(50, expected)
    assert info["text_policy"] == {
        "max_txt_len": expected,
        "max_words": max(50, expected),
    }
    # A relation after the old 50-word cap reaches the tokenizer at length 128.
    text = " ".join(["person"] * 60 + ["relation"])
    assert ("relation" in processors["eval"](text)) == (expected == 128)


@pytest.mark.parametrize("value", [None, True, 0, 1, 513, 128.0, "128"])
def test_invalid_text_limit_fails_before_native_model_loading(monkeypatch, value):
    monkeypatch.setattr(
        fafa, "official_source", lambda *a: pytest.fail("loaded source")
    )
    with pytest.raises(ValueError, match="checkpoint.max_txt_len"):
        fafa.load_fafa({"checkpoint": {"max_txt_len": value}}, torch.device("cpu"))


def test_baseline_text_change_reuses_gallery_and_rebuilds_query(
    benchmark, tmp_path, monkeypatch
):
    clip_cfg, weights, marker = benchmark
    cfg = yaml.safe_load(Path("configs/fafa.yaml").read_text())
    cfg.update(split="test", data=clip_cfg["data"])
    cfg["checkpoint"].pop("max_txt_len")  # A genuine legacy cache signature.
    cfg["checkpoint"].update(path=str(weights), runtime_assets_marker=str(marker))
    cfg["cache"]["dir"] = str(tmp_path / "fafa-cache")
    cfg["output"]["dir"] = str(tmp_path / "fafa-run")
    cfg["localization"]["query_selector"]["checkpoint"] = str(weights)
    cfg["localization"]["detector"]["checkpoint"] = str(weights)
    cfg["runtime"].update(device="cpu", num_workers=0, gallery_feature_dtype="float32")
    monkeypatch.setattr(fafa, "official_source", lambda *a: tmp_path)
    monkeypatch.setattr(fafa, "assets_ready", lambda cfg: True)
    candidates = [
        [
            {"box": [0, 0, 4, 8], "score": 0.9, "fallback": False},
            {"box": [4, 0, 8, 8], "score": 0.8, "fallback": False},
        ]
    ] * 3
    monkeypatch.setattr(
        fafa, "detect_gallery", lambda *a: (candidates, tmp_path / "detections")
    )
    monkeypatch.setattr(
        fafa,
        "load_fafa",
        lambda *a: (
            TinyFAFA(),
            {"eval": str},
            preprocess,
            {"missing_keys": [], "unexpected_keys": []},
        ),
    )
    original = fafa.encode_crops
    calls = []

    def encode(*args, **kwargs):
        calls.append((kwargs["query"], kwargs["path"].is_file()))
        return original(*args, **kwargs)

    monkeypatch.setattr(fafa, "encode_crops", encode)
    data = load_rcr_data(**cfg["data"])
    samples, gallery = split_samples(data, "test"), split_image_ids(data, "test")

    def retrieve():
        scores, metadata = fafa.retrieve_fafa(
            data, samples, gallery, cfg, torch.device("cpu")
        )
        del scores
        return metadata

    old = retrieve()
    image_path = Path(old["gallery_cache"]) / "gallery.npy"
    images = image_path.read_bytes()
    cfg["checkpoint"]["max_txt_len"] = 128
    new = retrieve()
    assert new["gallery_cache"] == old["gallery_cache"]
    assert new["query_cache"] != old["query_cache"]
    assert image_path.read_bytes() == images
    assert calls == [(False, False), (True, False), (False, True), (True, False)]
