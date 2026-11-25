"""Image/text features, person encoder assets and independent feature caches."""

import copy
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from rcr.common import detection
from rcr.common.config import load_config
from rcr.common.detection import DetectionCache, build_detection_cache
from rcr.common.geometry import boxes_to_scene
from rcr.proposed.cache import build, fafa, person_encoder, scene
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import (
    ImageEncoder,
    TextEncoder,
    load_text_encoder_state,
    text_encoder_state,
)
from tests.helpers import _sample
from tools.profile_gpu import TransferAudit

# Independent scene cache.


def test_scene_cache_keeps_native_global_without_person_geometry(tmp_path):
    import torch
    from PIL import Image
    from torch import nn

    from rcr.proposed.cache import scene as scene_cache
    from rcr.proposed.cache.scene import SceneCache

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 50), color="white").save(image_path)

    class SceneEncoder(nn.Module):
        def forward(self, pixels):
            scene = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            return scene, scene.mean(dim=1)[:, :2] + 9, (1, 2)

    def scene_processor(*, images, return_tensors):
        assert images.size == (100, 50)
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(1, 3, 16, 32)}

    cache_root = tmp_path / "cache"
    scene_cache.build_cache(
        image_ids=["image-1"],
        image_paths=[image_path],
        root=cache_root,
        image_encoder=SceneEncoder(),
        scene_processor=scene_processor,
        device="cpu",
        encoder_metadata={
            "image_encoder": {"model": "fake"},
        },
    )

    cache = SceneCache(cache_root)
    assert cache._load_item(0)["scene"].shape == (2, 4)
    assert cache.global_dim == 2
    torch.testing.assert_close(
        cache.load_global_features(torch.tensor([0])), torch.tensor([[11.0, 12.0]])
    )
    index = torch.load(cache_root / "index.pt", weights_only=True)
    assert not {"persons", "mask", "detector", "boxes_pixel"} & index.keys()
    assert set(cache._load_item(0)) == {"scene"}


# Native CLIP image and text features.


def test_scene_patches_and_native_global_match_clip_with_different_widths():
    backbone = CLIPVisionModelWithProjection(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    ).eval()
    encoder = ImageEncoder(backbone).eval()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = backbone(pixel_values=pixels)
        patches, global_embedding, grid = encoder(pixels)
    assert patches.shape == (2, 4, 16)
    assert global_embedding.shape == (2, 8)
    assert grid == (2, 2)
    torch.testing.assert_close(patches, expected.last_hidden_state[:, 1:])
    torch.testing.assert_close(global_embedding, expected.image_embeds)
    assert not any(p.requires_grad for p in backbone.parameters())


def test_native_text_eot_survives_padding_caching_and_context_projection():
    backbone = CLIPTextModelWithProjection(
        CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=1,
            eos_token_id=31,
            pad_token_id=0,
        )
    ).eval()
    encoder = TextEncoder(backbone, dim=6, cache_mib=1)
    ids = torch.tensor([[1, 4, 31, 0, 0], [1, 5, 6, 7, 31]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        expected = backbone(input_ids=ids, attention_mask=mask).text_embeds
    context, actual_mask, native = encoder(ids, mask)
    torch.testing.assert_close(native, expected)
    assert context.shape == (2, 5, 6)
    assert torch.equal(actual_mask, mask.bool())
    # A shorter right-padded visit reuses the same frozen states.
    _, _, shorter = encoder(ids[:1, :3], mask[:1, :3])
    torch.testing.assert_close(shorter, expected[:1])
    assert len(encoder._hidden) == 2
    encoder.train()
    assert not backbone.training
    context.square().mean().backward()
    assert encoder.proj.weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    assert not native.requires_grad


@pytest.mark.parametrize("cache_mib", [0, 1])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_lora_text_keeps_native_padding_and_reuses_only_frozen_states(
    cache_mib, device
):
    backbone = CLIPTextModelWithProjection(
        CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            max_position_embeddings=77,
            eos_token_id=31,
        )
    )
    rng = torch.get_rng_state()
    frozen = TextEncoder(copy.deepcopy(backbone), dim=6).to(device).eval()
    after_projection = torch.get_rng_state()
    torch.set_rng_state(rng)
    lora = {"enabled": True, "rank": 4}
    adapted = TextEncoder(backbone, dim=6, cache_mib=cache_mib, lora=lora).to(device)
    assert torch.equal(torch.get_rng_state(), after_projection)
    adapters = dict(adapted.named_lora_parameters())
    assert len(adapters) == 8
    assert all(".layers.1." in key or ".layers.2." in key for key in adapters)
    assert all(".q_proj." in key or ".v_proj." in key for key in adapters)
    ids = torch.tensor([[1, 4, 31, 31, 31], [1, 5, 6, 7, 31], [0, 1, 0, 6, 31]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 1]])
    valid = mask.bool().to(device)
    with torch.inference_mode():
        ctx, _, expected = frozen(ids, mask)
        context, actual_mask, initial = adapted.eval()(ids, mask)
        torch.testing.assert_close(initial, expected)
        torch.testing.assert_close(context[valid], ctx[valid])
        assert torch.equal(actual_mask, valid)

    prefix_calls = []
    with backbone.text_model.encoder.layers[0].register_forward_hook(
        lambda *args: prefix_calls.append(True)
    ):
        adapted.train()
        optimizer = torch.optim.AdamW(adapters.values(), lr=1e-2)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with TransferAudit() as audit:
                context, _, pooled = adapted(ids, mask)
            (context[valid].square().mean() + pooled.square().mean()).backward()
            optimizer.step()
            if device == "cuda" and cache_mib:
                assert audit.counts["cuda_to_cpu"] == 0
                assert audit.counts["cuda_scalar_reads"] == 0
        assert len(prefix_calls) == (0 if cache_mib else 2)
    assert all(p.grad is not None and p.grad.norm() > 0 for p in adapters.values())
    assert all(p.grad is None for p in backbone.parameters() if not p.requires_grad)
    with torch.inference_mode():
        _, _, learned = adapted.eval()(ids, mask)
        assert not torch.allclose(learned, initial)
        # Reuse the cached right-padded prefix with a shorter batch width.
        torch.testing.assert_close(adapted(ids[:1, :3], mask[:1, :3])[2], learned[:1])
    state = text_encoder_state(adapted, None)
    assert state["lora"]["weights"].keys() == adapters.keys()
    restored = TextEncoder(copy.deepcopy(frozen.backbone).cpu(), dim=6, lora=lora)
    load_text_encoder_state(restored, None, state)
    with torch.inference_mode():
        torch.testing.assert_close(restored.to(device).eval()(ids, mask)[2], learned)
    with pytest.raises(ValueError, match="LoRA settings"):
        load_text_encoder_state(restored, None, text_encoder_state(frozen, None))
    with pytest.raises(ValueError, match="disabled"):
        load_text_encoder_state(frozen, None, state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_lora_amp_keeps_fp32_outputs_and_shares_prefix_with_retrieval():
    backbone = CLIPTextModelWithProjection(
        CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            eos_token_id=31,
        )
    )
    encoder = TextEncoder(
        backbone,
        dim=6,
        cache_mib=1,
        lora={"enabled": True, "last_layers": 1},
        train_amp=True,
    ).cuda()
    ids = torch.tensor([[1, 4, 31], [1, 5, 31]])
    mask = torch.ones_like(ids)
    context, _, pooled = encoder(ids, mask)
    assert context.dtype == pooled.dtype == torch.float32
    (context.square().mean() + pooled.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for _, p in encoder.named_lora_parameters())
    with (
        torch.inference_mode(),
        backbone.text_model.encoder.layers[0].register_forward_hook(
            lambda *args: pytest.fail("FP32 retrieval reran the frozen AMP prefix")
        ),
    ):
        _, _, reference = encoder.eval()(ids, mask)
    torch.testing.assert_close(pooled, reference, atol=3e-3, rtol=3e-3)


# Frozen person encoder assets and offline loading.


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


# Detection, FAFA and scene cache contracts.


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
    native = load_config("configs/person_encoder.yaml")
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
        lru_mib=1,
    )
    geometry_reads = []
    read_geometry = gallery.detections.load_geometry

    def count_geometry(index):
        geometry_reads.append(index)
        return read_geometry(index)

    monkeypatch.setattr(gallery.detections, "load_geometry", count_geometry)
    gallery.load_person_geometry(torch.tensor([0, 1]))
    patches, people, boxes, mask = gallery.load(torch.tensor([0, 1]))
    gallery.load_person_geometry(torch.tensor([1, 0]))
    assert geometry_reads == [0, 1]  # Scene loads reuse the matching geometry.
    assert gallery._bytes <= 1024**2
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
    packed = gallery.native_matched_targets(
        torch.tensor([0, 1]), torch.tensor([[1, -1, 0], [-1, -1, -1]])
    )
    assert len(packed.values) == 2
    aligned = packed.to_aligned("cpu")
    assert aligned.shape == (2, 3, 2, 2)
    assert aligned[0, 1].count_nonzero() == aligned[1].count_nonzero() == 0
    with monkeypatch.context() as no_scenes:
        no_scenes.setattr(
            gallery._scene_cache,
            "_load_item",
            lambda i: pytest.fail("unused scene read"),
        )
        person_index, geometry, index_mask = gallery.load_person_geometry(
            torch.tensor([0, 1])
        )
    torch.testing.assert_close(person_index.float(), people)
    torch.testing.assert_close(geometry, boxes)
    torch.testing.assert_close(index_mask, mask)

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
