"""CPU workflow tests for detection cache."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from rcr.common import detection
from rcr.common.config import load_config
from rcr.common.detection import DetectionCache, build_detection_cache
from rcr.common.geometry import boxes_to_scene
from rcr.proposed.cache import build, fafa, scene
from rcr.proposed.cache.store import GalleryCache
from tests.helpers import _sample


@pytest.fixture
def inputs(tmp_path):
    cfg = load_config("configs/proposed.yaml")
    cfg["runtime"]["device"] = "cpu"
    for key in (
        "detection_cache",
        "scene_cache",
        "fafa_image_cache",
        "fafa_query_cache",
    ):
        cfg["data"][key] = str(tmp_path / key)
    paths = {}
    for name in ("a", "b"):
        paths[name] = tmp_path / f"{name}.png"
        Image.new("RGB", (100, 50), color="white").save(paths[name])
    data = SimpleNamespace(
        gallery_ids=["a", "b"],
        image_path=paths.__getitem__,
        samples=[_sample("q", "a", "b")],
    )
    return cfg, data


def detect_inputs(cfg, data, monkeypatch):
    boxes = torch.tensor([[7.25, 3.5, 58.75, 47.5], [15.0, 9.0, 80.0, 45.0]])
    values = iter([boxes, torch.empty(0, 4)])
    monkeypatch.setattr(detection, "detect_persons", lambda *a: next(values))
    build_detection_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        cfg["data"]["detection_cache"],
        cfg["detector"],
        model=nn.Identity(),
    )
    return DetectionCache(cfg["data"]["detection_cache"])


class FakeFAFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qformer = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
        self.vision_proj = nn.Linear(3, 2)
        self.query_tokens = torch.zeros(1, 2, 3)
        self.tokenizer = lambda *a, **k: {"input_ids": [1, 2, 3]}

    def extract_features(self, batch, mode):
        count = len(batch["image"])
        if mode == "image":
            return SimpleNamespace(
                image_embeds=torch.ones(count, 2, 3),
                image_embeds_proj=torch.ones(count, 2, 2),
            )
        return SimpleNamespace(multimodal_embeds=torch.ones(count, 2))


def test_fafa_and_queries_are_independent_of_scene_but_bind_exact_detections(
    inputs, monkeypatch, tmp_path
):
    cfg, data = inputs
    source = detect_inputs(cfg, data, monkeypatch)
    native = load_config("configs/fafa.yaml")
    checkpoint = tmp_path / "fake-fafa.pt"
    checkpoint.write_bytes(b"weights")
    native["checkpoint"]["path"] = str(checkpoint)
    monkeypatch.setattr(fafa, "load_fafa_config", lambda _: native)
    monkeypatch.setattr(build, "load_fafa_config", lambda _: native)
    monkeypatch.setattr(fafa, "load_rcr_inputs", lambda *a: data)
    monkeypatch.setattr(
        fafa,
        "load_fafa",
        lambda *a: (
            FakeFAFA(),
            {"eval": lambda text: text},
            lambda image: torch.zeros(3, 4, 4),
            {},
        ),
    )
    # Real cache orchestration runs with no scene files present.
    fafa.finish_fafa_cache(cfg)
    image_root = Path(cfg["data"]["fafa_image_cache"])
    query_root = Path(cfg["data"]["fafa_query_cache"])
    before = [
        p.read_bytes() for p in (image_root / "index.pt", query_root / "index.pt")
    ]
    image_index = torch.load(image_root / "index.pt", weights_only=True)
    assert (
        not {"scene_dim", "global_dim", "patch_hw", "image_encoder", "source_cache_id"}
        & image_index.keys()
    )
    assert image_index["detection_cache_id"] == source.cache_id
    monkeypatch.setattr(
        fafa, "load_fafa", lambda *a: pytest.fail("must reuse independent FAFA caches")
    )
    changed = copy.deepcopy(cfg)
    changed["image_encoder"] = {"model": "another-scene-encoder"}
    fafa.finish_fafa_cache(changed)
    assert before == [
        p.read_bytes() for p in (image_root / "index.pt", query_root / "index.pt")
    ]

    class Encoder(nn.Module):
        def forward(self, pixels):
            return (
                torch.arange(8.0).reshape(1, 2, 4),
                torch.tensor([[11.0, 12.0]]),
                (1, 2),
            )

    scene_root = Path(cfg["data"]["scene_cache"])
    scene.build_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        scene_root,
        Encoder(),
        lambda **kw: {"pixel_values": torch.zeros(1, 3, 224, 224)},
        "cpu",
        encoder_metadata={"image_encoder": cfg["image_encoder"]},
    )
    gallery = GalleryCache(
        image_root,
        detection_root=source.root,
        scene_root=scene_root,
        query_root=query_root,
    )
    patches, people, boxes, mask = gallery.load(torch.tensor([0, 1]))
    assert patches.shape == (2, 2, 4) and people.shape == (2, 2, 3)
    assert mask.tolist() == [[True, True], [False, False]]
    torch.testing.assert_close(
        boxes[0],
        boxes_to_scene(source.load_geometry(0)["boxes_pixel"], (100, 50), (224, 224)),
    )
    torch.testing.assert_close(
        gallery.load_global_features(torch.tensor([1, 0])),
        torch.tensor([[11.0, 12.0], [11.0, 12.0]]),
    )
    assert gallery.native_queries(data.samples, 2).shape == (1, 1, 2, 2)
    assert gallery.native_targets(torch.tensor([0, 1]), 2).shape == (2, 2, 2, 2)
    assert gallery.detection_cache_id == source.cache_id

    bad = tmp_path / "changed-detections"
    values = iter([source.load_geometry(0)["boxes_pixel"].flip(0), torch.empty(0, 4)])
    monkeypatch.setattr(detection, "detect_persons", lambda *a: next(values))
    build_detection_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        bad,
        cfg["detector"],
        model=nn.Identity(),
    )
    with pytest.raises(ValueError, match="source detection cache"):
        GalleryCache(image_root, detection_root=bad, scene_root=scene_root)
