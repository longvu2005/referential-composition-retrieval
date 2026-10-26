"""Independent extraction, immutable crop identity, and shared-box consumers."""

import copy
import shutil
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
from tests.test_train import _sample


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
    monkeypatch.setattr(detection, "detect", lambda *a: {"person": next(values)})
    build_detection_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        cfg["data"]["detection_cache"],
        cfg["detector"],
        model=nn.Identity(),
    )
    return DetectionCache(cfg["data"]["detection_cache"])


def test_detection_resume_reuses_completed_shards_and_rejects_changed_spec(
    inputs, monkeypatch
):
    cfg, data = inputs
    root = Path(cfg["data"]["detection_cache"])
    calls = []

    def detect(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return {"person": torch.tensor([[-1.25, 2.0, 102.0, 40.0]])}

    monkeypatch.setattr(detection, "detect", detect)
    kwargs = dict(
        image_ids=data.gallery_ids,
        image_paths=[data.image_path(i) for i in data.gallery_ids],
        root=root,
        detector_config=cfg["detector"],
        model=nn.Identity(),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        build_detection_cache(**kwargs)
    with pytest.raises(RuntimeError, match="incomplete"):
        DetectionCache(root)
    with pytest.raises(ValueError, match="settings differ"):
        build_detection_cache(
            **{**kwargs, "detector_config": {**cfg["detector"], "threshold": 0.4}}
        )
    build_detection_cache(**kwargs)
    assert len(calls) == 3
    cache = DetectionCache(root)
    relocated = root.with_name("relocated")
    shutil.copytree(root, relocated)
    assert DetectionCache(relocated).cache_id == cache.cache_id
    # Reusing a published cache performs no new model inference.
    build_detection_cache(**kwargs)
    assert len(calls) == 3


@pytest.mark.parametrize("pixels", [True, False])
def test_import_legacy_boxes_without_loading_any_model(inputs, monkeypatch, pixels):
    cfg, data = inputs
    legacy = Path(cfg["data"]["scene_cache"])
    (legacy / "features").mkdir(parents=True)
    boxes = torch.tensor([[7.25, 3.5, 58.75, 47.5]])
    mask = torch.tensor([[True], [False]])
    torch.save(
        {
            "format_version": 2,
            "cache_id": "legacy",
            "image_ids": data.gallery_ids,
            "mask": mask,
            "detector": cfg["detector"],
            "image_encoder": {"scene_size": [128, 256]},
        },
        legacy / "index.pt",
    )
    for i, name in enumerate(data.gallery_ids):
        value = boxes if i == 0 else boxes[:0]
        item = {
            "cache_id": "legacy",
            "image_id": name,
            "identity_ids": ["private"],
            "boxes_scene": boxes_to_scene(value, (100, 50), (256, 128)),
        }
        if pixels:
            item["boxes_pixel"] = value
        torch.save(item, legacy / "features" / f"{i}.pt")
    monkeypatch.setattr(
        detection,
        "detect",
        lambda *a: pytest.fail("no detector inference during import"),
    )
    monkeypatch.setattr(build, "load_rcr_inputs", lambda *a: data)
    monkeypatch.setattr(
        build,
        "resolve_device",
        lambda *a: pytest.fail("import only needs CPU/image headers"),
    )
    build.prepare_cache(cfg, stage="detections", import_scene=legacy)
    cache = DetectionCache(cfg["data"]["detection_cache"])
    torch.testing.assert_close(cache.load_geometry(0)["boxes_pixel"], boxes)
    assert cache.mask.tolist() == [[True], [False]]
    index = torch.load(cache.root / "index.pt", weights_only=True)
    assert not {"image_encoder", "scene", "identity_ids"} & index.keys()
    with pytest.raises(ValueError, match="matching detector provenance"):
        build_detection_cache(
            data.gallery_ids,
            [data.image_path(i) for i in data.gallery_ids],
            legacy.with_name("bad-import"),
            {**cfg["detector"], "threshold": 0.4},
            import_scene=legacy,
        )


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
    monkeypatch.setattr(detection, "detect", lambda *a: {"person": next(values)})
    build_detection_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        bad,
        cfg["detector"],
        model=nn.Identity(),
    )
    with pytest.raises(ValueError, match="source detection cache"):
        GalleryCache(image_root, detection_root=bad, scene_root=scene_root)


def test_scene_stage_needs_no_detection_cache_or_worker(inputs, monkeypatch):
    cfg, data = inputs
    monkeypatch.setattr(build, "load_rcr_inputs", lambda *a: data)
    calls = []
    monkeypatch.setattr(build, "_prepare_scene", lambda *a: calls.append("scene"))
    monkeypatch.setattr(
        build,
        "build_detection_cache",
        lambda *a, **k: pytest.fail("scene stage must not load detector"),
    )
    monkeypatch.setattr(
        build,
        "run_person_worker",
        lambda *a, **k: pytest.fail("scene stage must not load FAFA"),
    )
    build.prepare_cache(cfg, stage="scene")
    assert calls == ["scene"]


def test_detection_audit_and_supervision_need_no_scene_or_fafa(
    inputs, monkeypatch, tmp_path
):
    from rcr.common.data import load_rcr_data
    from rcr.dataset.audit import audit_detection
    from rcr.proposed.supervision import TrainSupervision, prepare_train_supervision
    from tests.test_core import _write_final_dataset

    cfg, _ = inputs
    final = tmp_path / "final"
    final.mkdir()
    _write_final_dataset(final)
    data = load_rcr_data(final, tmp_path / "images")
    for name in data.gallery_ids:
        path = data.image_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 50)).save(path)
    monkeypatch.setattr(
        detection,
        "detect",
        lambda *a: {"person": torch.tensor([[0.0, 0.0, 20.0, 20.0]])},
    )
    build_detection_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        cfg["data"]["detection_cache"],
        cfg["detector"],
        model=nn.Identity(),
    )
    cache = DetectionCache(cfg["data"]["detection_cache"])
    report = audit_detection(data, cache)
    assert report["cache_id"] == cache.cache_id
    cfg["data"]["final_dir"] = str(final)
    cfg["data"]["image_root"] = str(tmp_path / "images")
    cfg["cache"]["supervision_cache"] = str(tmp_path / "labels.json")
    prepare_train_supervision(cfg)
    labels = TrainSupervision(data, cache, cache_file=cfg["cache"]["supervision_cache"])
    changed = TrainSupervision(
        data,
        cache,
        scene_size=[1024, 1024],
        cache_file=cfg["cache"]["supervision_cache"],
    )
    assert labels.key == changed.key
    assert labels.load_ids(["1_1"], 1) == [["7"]]
