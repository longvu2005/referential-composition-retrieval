"""Text windows and a complete on-disk CPU pipeline using tiny frozen backbones."""

import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.proposed.batch import build_batch, query_inputs
from rcr.proposed.cache import build as builder
from rcr.proposed.cache import fafa as fafa_cache
from rcr.proposed.cache.clip import (
    FeatureCache,
    align_identities,
    build_clip_cache,
    text_strings,
)
from rcr.proposed.cache.dino import boxes_to_pixels, boxes_to_scene, build_cache
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.losses import compute_loss
from rcr.proposed.nn.encoders import (
    CLIPFeatures,
    ImageEncoder,
    QueryTextCache,
    parse_subjects,
)
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import mine_hard_negatives, retrieve_rankings
from rcr.proposed.retrieve import load_checkpoint, retrieve
from rcr.proposed.train import train
from tests.proposed.test_build_cache import (
    Backbone,
    Detector,
    DetectorProcessor,
    Processor,
)
from tools import run as cli


class Tokenizer:
    is_fast, bos_token_id, eos_token_id = True, 0, 1

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        words = list(re.finditer(r"\S+", text))
        return {
            "input_ids": [2 + sum(map(ord, w[0])) % 28 for w in words],
            "offset_mapping": [(w.start(), w.end()) for w in words],
        }

    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


class TextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)

    def forward(self, input_ids):
        assert input_ids.shape[-1] <= 8
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 8)

    def forward(self, pixel_values):
        pooled = self.project(pixel_values.mean((-1, -2)))
        return SimpleNamespace(
            pooler_output=pooled, last_hidden_state=pooled[:, None].expand(-1, 5, -1)
        )


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=8, hidden_size=8),
            vision_config=SimpleNamespace(image_size=8, patch_size=4, hidden_size=8),
            projection_dim=8,
            _commit_hash="tiny-test-only",
        )
        self.text_model = TextBackbone()
        self.vision_model = VisionBackbone()
        self.visual_projection = nn.Linear(8, 8)


class PixelProcessor:
    def __call__(self, images, **kwargs):
        pixels = [
            torch.tensor(im.resize((1, 1)).getpixel((0, 0))).float() / 255
            for im in images
        ]
        return {
            "pixel_values": torch.stack(pixels)[:, :, None, None].expand(-1, -1, 8, 8)
        }

    def to_dict(self):
        return {"test_only": True, "size": 8}


class FakeFAFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qformer = SimpleNamespace(config=SimpleNamespace(hidden_size=5))
        self.weight = nn.Parameter(torch.ones(1))

    def extract_features(self, samples, mode):
        assert mode == "image" and set(samples) == {"image"}
        assert not self.training and not self.weight.requires_grad
        tokens = (
            torch.arange(10.0).reshape(1, 2, 5).expand(len(samples["image"]), -1, -1)
        )
        return SimpleNamespace(image_embeds=tokens)


@pytest.fixture
def tiny_clip(monkeypatch):
    def factory(function):
        return SimpleNamespace(from_pretrained=lambda *args, **kwargs: function())

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPModel=factory(TinyCLIP),
            CLIPTokenizerFast=factory(Tokenizer),
            CLIPImageProcessor=factory(PixelProcessor),
        ),
    )
    torch.set_num_threads(1)
    torch.manual_seed(7)


def test_full_text_windows_all_mentions_and_unequal_subjects():
    encoder = CLIPFeatures(TinyCLIP(), Tokenizer())
    encoder.train()
    assert not encoder.backbone.training
    assert all(not p.requires_grad for p in encoder.parameters())
    condition = (
        "Subject 1 " + "word " * 80 + " Subject 2 is beside Subject 1 and Subject 2"
    )
    samples = [
        {
            "final_desc": (
                "Identify Subject 1 as the two boys and Subject 2 as the woman"
            ),
            "final_change": condition,
        },
        {
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "Subject 1 stands",
        },
    ]
    features = {t: encoder.text(t) for t in text_strings(samples)}
    batch = QueryTextCache(features).batch(samples)
    assert batch["change_mask"][0].sum() == len(
        Tokenizer()(condition, truncation=False)["input_ids"]
    )
    assert batch["subject_token_mask"][0, 0].sum() == 4
    assert batch["subject_token_mask"][0, 1].sum() == 4
    assert batch["subject_mask"].tolist() == [[True, True], [True, False]]
    parsed = parse_subjects(samples[0]["final_desc"], condition)
    assert parsed[0] == [1, 2] and parsed[1] == ["the two boys", "the woman"]
    encoded = features[condition]
    torch.testing.assert_close(
        encoded["tokens"],
        encoder.backbone.text_model.embed(torch.tensor(encoded["input_ids"])),
    )


def tiny_native_clip():
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

    config = CLIPConfig(
        text_config=CLIPTextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=8,
        ).to_dict(),
        vision_config=CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            image_size=8,
            patch_size=4,
        ).to_dict(),
        projection_dim=6,
    )
    return CLIPModel(config).eval()


@pytest.mark.parametrize("safe_serialization", [False, True])
def test_real_clip_checkpoint_load_and_extraction(tmp_path, safe_serialization):
    from transformers import CLIPModel

    model = tiny_native_clip()
    model.save_pretrained(tmp_path, safe_serialization=safe_serialization)
    filename = "model.safetensors" if safe_serialization else "pytorch_model.bin"
    assert (tmp_path / filename).is_file()
    restored = CLIPModel.from_pretrained(tmp_path).eval()
    encoder = CLIPFeatures(restored, Tokenizer())
    pixels = torch.randn(2, 3, 8, 8)
    pooled, tokens = encoder.image(pixels)
    native = model.vision_model(pixel_values=pixels)
    torch.testing.assert_close(tokens, native.last_hidden_state)
    torch.testing.assert_close(
        pooled,
        torch.nn.functional.normalize(
            model.visual_projection(native.pooler_output), dim=-1
        ),
    )
    assert pooled.shape == (2, 6) and tokens.shape == (2, 5, 8)
    encoded = encoder.text("Subject 1 " + "word " * 20)
    assert encoded["tokens"].shape == (22, 8)


def test_real_clip_bin_cache_resumes_after_failed_build(experiment, monkeypatch):
    cfg, data = experiment
    # The fixture already prepared tiny source caches. Restore real Transformers
    # before testing its binary checkpoint loader; no CLIP model API is mocked.
    monkeypatch.undo()
    from transformers import CLIPImageProcessor, CLIPTokenizerFast

    checkpoint = data.final_dir.parent / "local-clip-bin"
    tiny_native_clip().save_pretrained(checkpoint, safe_serialization=False)
    cfg["clip_encoder"]["model"] = str(checkpoint)
    monkeypatch.setattr(
        CLIPTokenizerFast, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        CLIPImageProcessor, "from_pretrained", lambda *a, **kw: PixelProcessor()
    )
    source = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    indexes = [source.root / "index.pt", source._scene_cache.root / "index.pt"]
    before = [path.read_bytes() for path in indexes]
    original = CLIPFeatures.image
    calls = 0

    def interrupted(self, pixels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted native CLIP build")
        return original(self, pixels)

    monkeypatch.setattr(CLIPFeatures, "image", interrupted)
    with pytest.raises(RuntimeError, match="interrupted native CLIP build"):
        build_clip_cache(cfg, data, source)
    with pytest.raises(ValueError, match="incomplete"):
        FeatureCache(cfg, data)
    root = Path(cfg["data"]["clip_cache"])
    first_shard = (root / "features/0.pt").read_bytes()
    cache_id = (root / ".building").read_text()
    monkeypatch.setattr(CLIPFeatures, "image", original)
    builder.prepare_cache(cfg, stage="clip")
    cache = FeatureCache(cfg, data)
    loaded = cache.load(torch.tensor([0]))
    assert torch.isfinite(loaded["clip_tokens"]).all()
    assert loaded["clip_tokens"].shape == (1, 1, 5, 8)
    assert loaded["clip_pooled"].shape == (1, 1, 6)
    assert cache.cache_id == cache_id
    assert (root / "features/0.pt").read_bytes() == first_shard
    assert [path.read_bytes() for path in indexes] == before
    assert not (root / ".building").exists() and not (root / ".build.pt").exists()


@pytest.fixture
def experiment(tmp_path, tiny_clip, monkeypatch):
    final, images = tmp_path / "final", tmp_path / "images"
    (final / "splits").mkdir(parents=True)
    samples, gallery, heads = [], [], []
    for split in ("train", "val", "test"):
        for i, name in enumerate(("q", "p", "same", "wrong")):
            image_id = f"{split}_{name}"
            relative = f"{split}/{image_id}.png"
            path = images / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.new("RGB", (10, 10), (20 + 40 * i, 50, 90)).save(path)
            gallery.append({"image_id": image_id, "path": relative})
            heads.append(
                {
                    "image_id": image_id,
                    "identity_id": f"{split}_{1 if name != 'wrong' else 2}",
                    "x": 2,
                    "y": 2,
                    "width": 2,
                    "height": 2,
                }
            )
        sample = {
            "sample_id": split,
            "case_type": "INDIVIDUAL",
            "query_image_id": f"{split}_q",
            "target_image_id": f"{split}_p",
            "positive_image_ids": [f"{split}_p"],
            "subjects": [{"subject_id": 1, "identity_ids": [f"{split}_1"]}],
            "final_desc": "Identify Subject 1 as the man",
            "final_change": "Subject 1 is standing beside Subject 1",
        }
        samples.append(sample)
        (final / "splits" / f"{split}.txt").write_text(split + "\n")
    for filename, rows in [
        ("samples", samples),
        ("images", gallery),
        ("gallery", gallery),
        ("head_boxes", heads),
    ]:
        (final / f"{filename}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (final / "manifest.json").write_text('{"version":"synthetic-test-only"}')
    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["data"] = dict(
        final_dir=str(final),
        image_root=str(images),
        cache=str(tmp_path / "fafa"),
        dino_cache=str(tmp_path / "dino"),
        clip_cache=str(tmp_path / "clip"),
    )
    cfg["runtime"]["device"] = "cpu"
    cfg["image_encoder"].update(
        model="tiny-image", scene_size=[8, 8], person_size=[8, 4]
    )
    cfg["model"].update(dim=16, identity_dim=4, num_heads=2, dropout=0)
    cfg["train"].update(epochs=3, warmup_epochs=1, batch_size=1, candidates=3)
    cfg["train"]["sampling"]["positives_per_query"] = 1
    cfg["evaluation"]["every_epochs"] = 1
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=1
    )
    cfg["candidate_ks"] = [1, 2]
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["checkpoint"] = str(tmp_path / "run/best.pt")
    data = load_rcr_data(final, images)
    build_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        cfg["data"]["dino_cache"],
        Detector(),
        DetectorProcessor(),
        ImageEncoder(Backbone(), 8),
        Processor(8, 8),
        Processor(8, 4),
        "cpu",
        gt_heads_by_image=data.gt_head_boxes_by_image,
        encoder_metadata={
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
        },
    )
    native = yaml.safe_load(Path("configs/fafa.yaml").read_text())
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"synthetic-fafa-test-only")
    native["checkpoint"]["path"] = str(weights)
    native_path = tmp_path / "fafa.yaml"
    native_path.write_text(yaml.safe_dump(native))
    cfg["person_encoder"]["fafa_config"] = str(native_path)
    monkeypatch.setattr(
        fafa_cache,
        "load_fafa",
        lambda *args: (FakeFAFA(), {}, lambda crop: torch.ones(3, 4, 4), {}),
    )
    fafa_cache.finish_fafa_cache(cfg)
    return cfg, data


def test_cache_resume_reuse_and_provenance(experiment, monkeypatch):
    cfg, data = experiment
    source = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    before = (Path(cfg["data"]["cache"]) / "index.pt").read_bytes()
    original = CLIPFeatures.image
    calls = 0

    def interrupted(self, pixels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(self, pixels)

    monkeypatch.setattr(CLIPFeatures, "image", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_clip_cache(cfg, data, source)
    with pytest.raises(ValueError, match="incomplete"):
        FeatureCache(cfg, data)
    shard = (Path(cfg["data"]["clip_cache"]) / "features/0.pt").read_bytes()
    monkeypatch.setattr(CLIPFeatures, "image", original)
    builder.prepare_cache(cfg, stage="clip")
    assert (Path(cfg["data"]["clip_cache"]) / "features/0.pt").read_bytes() == shard
    assert (Path(cfg["data"]["cache"]) / "index.pt").read_bytes() == before
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *a, **kw: pytest.fail("FAFA rebuilt")
    )
    monkeypatch.setattr(CLIPFeatures, "image", lambda *a: pytest.fail("CLIP rebuilt"))
    builder.prepare_cache(cfg)
    cache = FeatureCache(cfg, data)
    assert cache.load(torch.tensor([0]))["clip_tokens"].shape == (1, 1, 5, 8)
    changed = copy.deepcopy(cfg)
    changed["clip_encoder"]["revision"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        FeatureCache(changed, data)
    changed = copy.deepcopy(cfg)
    changed["person_encoder"]["backend"] = "dino"
    with pytest.raises(ValueError, match="requires"):
        FeatureCache(changed, data)
    changed_data = copy.deepcopy(data)
    changed_data.images_by_id[data.gallery_ids[0]]["identities_complete"] = True
    changed_cache = FeatureCache(cfg, changed_data)
    with pytest.raises(ValueError, match="supervision changed"):
        changed_cache.supervision()
    build_clip_cache(cfg, changed_data, source)
    assert FeatureCache(cfg, changed_data).supervision()[data.gallery_ids[0]][
        "complete"
    ]
    path = Path(cfg["data"]["clip_cache"]) / "supervision.pt"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        build_clip_cache(cfg, changed_data, source)
    path = Path(cfg["data"]["clip_cache"]) / "features/0.pt"
    item = torch.load(path, weights_only=True)
    item["boxes_scene"] = item["boxes_scene"] + 0.1
    torch.save(item, path)
    with pytest.raises(ValueError, match="boxes"):
        cache.load(torch.tensor([0]))


def test_full_pipeline_case_independence_and_protocol(experiment, tmp_path):
    cfg, data = experiment
    builder.prepare_cache(cfg, stage="clip")
    cache = FeatureCache(cfg, data)
    samples = split_samples(data, "train")
    original = query_inputs(samples, cache)
    stripped = [
        {k: s[k] for k in ("query_image_id", "final_desc", "final_change")}
        for s in samples
    ]
    changed = [
        {**s, "case_type": "GROUP", "subjects": [], "target_boxes": [[999]]}
        for s in samples
    ]
    for rows in (stripped, changed):
        inputs = query_inputs(rows, cache)
        for group in original:
            for name in original[group]:
                torch.testing.assert_close(inputs[group][name], original[group][name])
    candidates = [["train_p", "train_same", "train_wrong"]]
    labels = cache.supervision()
    batch = build_batch(
        samples, candidates, cache, labels, {"train_1": 0, "train_2": 1}
    )
    m = RCRModel(**cache.dimensions, **cfg["model"]).eval()
    loss, _ = compute_loss(m, batch, cache.patch_hw)
    changed_case = [{**s, "case_type": "RELATIONAL"} for s in samples]
    other, _ = compute_loss(
        m,
        build_batch(
            changed_case, candidates, cache, labels, {"train_1": 0, "train_2": 1}
        ),
        cache.patch_hw,
    )
    torch.testing.assert_close(loss, other)
    # The full loss wrapper must not create zero P_id gradients that trigger
    # AdamW decay when identity supervision is explicitly disabled.
    m.zero_grad(set_to_none=True)
    isolated = copy.deepcopy(batch)
    isolated["supervision"]["grounding"].fill_(-1)
    ranking, _ = compute_loss(m, isolated, cache.patch_hw, identity_weight=0)
    ranking.backward()
    assert all(p.grad is None for p in m.identity_head.parameters())
    assert all(p.grad is None for p in m.grounding.parameters())
    best = train(cfg)
    assert best.name == "best.pt" and best.exists()
    assert (best.parent / "warmup.pt").exists()
    model, checkpoint = load_checkpoint(best, cfg, cache, torch.device("cpu"))
    assert checkpoint["epoch"] > 1 and checkpoint["best_full_map"] is not None
    gallery_ids = split_image_ids(data, "val")
    val = split_samples(data, "val")
    normal = retrieve(cfg)
    full_cfg = {
        **cfg,
        "retrieval": {**cfg["retrieval"], "mode": "full"},
        "output": {"dir": str(tmp_path / "full")},
    }
    full = retrieve(full_cfg)
    for output in (normal, full):
        assert output["rankings"].shape == (1, 3)
        assert set(output["rankings"][0].tolist()) == {1, 2, 3}
    all_top = retrieve_rankings(
        val,
        cache,
        model,
        torch.device("cpu"),
        gallery_ids=gallery_ids,
        **{**cfg["retrieval"], "top_m": 3},
    )
    torch.testing.assert_close(all_top["rankings"], full["rankings"])
    assert (
        normal["runtime"]["shortlist_approximation"]
        and not full["runtime"]["shortlist_approximation"]
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    cli.main(["evaluate", "--config", str(path), "--splits", "val"])
    metrics = json.loads((Path(cfg["output"]["dir"]) / "val/metrics.json").read_text())
    assert "candidate_hit_2" in metrics["overall"]
    with pytest.raises(ValueError, match="only train"):
        mine_hard_negatives(
            val,
            cache,
            model,
            torch.device("cpu"),
            gallery_ids=gallery_ids,
            retrieval=cfg["retrieval"],
            pool_size=2,
        )
    pools = mine_hard_negatives(
        samples,
        cache,
        model,
        torch.device("cpu"),
        gallery_ids=split_image_ids(data, "train"),
        retrieval=cfg["retrieval"],
        pool_size=2,
        excluded={"train": {"train_same"}},
    )
    assert pools["train"] == ["train_wrong"]
    bad = tmp_path / "old.pt"
    torch.save({"architecture_version": "old"}, bad)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(bad, cfg, cache, torch.device("cpu"))


def test_alignment_unknown_and_box_transform():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]])
    image_size, scene_size = (40, 20), (224, 224)
    mapped = boxes_to_scene(boxes, image_size, scene_size)
    torch.testing.assert_close(
        boxes_to_pixels(mapped, image_size, scene_size), boxes, atol=1e-5, rtol=1e-5
    )
    heads = [
        dict(x=1, y=1, width=2, height=2, identity_id="a"),
        dict(x=5, y=1, width=2, height=2, identity_id="b"),
    ]
    assert align_identities(mapped, heads, image_size, scene_size) == [None, None]
