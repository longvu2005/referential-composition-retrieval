"""Architectural regressions: gradients, same-person binding and frozen FAFA cache."""

from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from rcr.proposed.cache import fafa as fafa_cache
from rcr.proposed.cache.build import check_cache_config
from rcr.proposed.cache.dino import build_cache
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.losses import compute_loss, identity_loss
from rcr.proposed.matching import partial_identity_matching
from rcr.proposed.nn.encoders import ImageEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import fafa_binding_scores
from tests.proposed.test_build_cache import (
    Backbone,
    Detector,
    DetectorProcessor,
    Processor,
)
from tests.proposed.test_training import _batch


def test_retrieval_updates_semantic_and_scale_but_never_identity():
    torch.manual_seed(31)
    model = RCRModel(8, 6, 2)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    before = model.identity_head.proj.weight.detach().clone()
    loss, _ = compute_loss(model, _batch(), (2, 3), identity_weight=0, state_weight=0)
    loss.backward()
    assert model.identity_head.proj.weight.grad is None
    assert model.person_proj[0].weight.grad.abs().sum() > 0
    assert model.binding_log_scale.grad.abs() > 0
    optimizer.step()
    torch.testing.assert_close(before, model.identity_head.proj.weight)
    model.zero_grad(set_to_none=True)
    identity_loss(
        model.encode_identity(torch.randn(4, 8)), torch.tensor([1, 2, 1, 2])
    ).backward()
    assert model.identity_head.proj.weight.grad.abs().sum() > 0
    assert model.person_proj[0].weight.grad.abs().sum() > 0
    assert (
        model.person_proj[0].weight.data_ptr()
        != model.identity_head.proj.weight.data_ptr()
    )


def test_shared_person_supports_different_person_and_scene_widths():
    model = RCRModel(8, 6, 2, input_dim=10, person_input_dim=12).eval()
    batch = _batch()
    for key in ("query_scene", "target_scene"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 10)
    for key in ("query_persons", "target_persons"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 12)
    loss, _ = compute_loss(model, batch, (2, 3))
    assert torch.isfinite(loss)
    assert model.identity_head.proj.in_features == 8
    assert model.person_proj[0].in_features == 12


def test_identity_and_state_must_match_the_same_target_person():
    # Person A has identity=1, semantics=0. Person B has identity=0, semantics=1.
    matching = partial_identity_matching(
        torch.tensor([[[1.0, 0.0]]]),
        torch.eye(2)[None],
        torch.ones(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    result = fafa_binding_scores(
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
        matching=matching,
    )
    assert result.item() == 0.0  # ID selects A; FDA reads A's tokens.


def test_binding_respects_member_masks_and_empty_targets():
    torch.manual_seed(5)
    qid = torch.randn(2, 3, 4, requires_grad=True)
    tid = torch.randn(2, 2, 4, requires_grad=True)
    comp = torch.randn(2, 2, 3, 6, requires_grad=True)
    semantic = torch.randn(2, 2, 6, requires_grad=True)
    members = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
    subjects = torch.tensor([[True, False]] * 2)
    mask = torch.tensor([[True, False], [False, False]])
    matching = partial_identity_matching(qid, tid, members, subjects, mask)
    result = fafa_binding_scores(comp, semantic[:, :, None], matching=matching)
    assert result[1].item() == pytest.approx(-1.1)
    assert not result.requires_grad
    assert all(x.grad is None for x in (qid, tid, comp, semantic))


class FakeFAFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qformer = SimpleNamespace(config=SimpleNamespace(hidden_size=5))
        self.weight = nn.Parameter(torch.ones(1))
        self.vision_proj = nn.Linear(5, 3)
        self.query_tokens = nn.Parameter(torch.ones(1, 7, 5))

    def extract_target_features(self, pixels, mode):
        assert mode == "mean"
        assert not self.training and not self.weight.requires_grad
        features = torch.nn.functional.normalize(
            torch.arange(21.0).reshape(1, 7, 3).expand(len(pixels), -1, -1), dim=-1
        )
        return features, None

    def extract_features(self, samples, mode):
        assert not self.training and not self.weight.requires_grad
        if mode == "multimodal":
            assert len(samples["text_input"]) == len(samples["image"])
            return SimpleNamespace(
                multimodal_embeds=torch.tensor(
                    [[0.6, 0.8, 0.0]] * len(samples["image"])
                )
            )
        assert mode == "image" and set(samples) == {"image"}
        # Hidden tokens differ from the projected features on purpose.
        tokens = (
            torch.arange(10.0).reshape(1, 2, 5).expand(len(samples["image"]), -1, -1)
        )
        return SimpleNamespace(
            image_embeds=tokens, image_embeds_proj=torch.zeros(len(tokens), 2, 1)
        )


def test_fafa_cache_uses_hidden_image_tokens_and_preserves_scene(tmp_path, monkeypatch):
    image = tmp_path / "person.png"
    Image.new("RGB", (10, 10)).save(image)
    root = tmp_path / "persons"
    source_root = tmp_path / "dino"
    metadata = {"image_encoder": {"scene_size": [8, 8]}, "detector": {"model": "tiny"}}
    build_cache(
        ["image"],
        [image],
        source_root,
        Detector(),
        DetectorProcessor(2),
        ImageEncoder(Backbone(), 8),
        Processor(8, 8),
        Processor(8, 4),
        "cpu",
        storage_dtype="float16",
        encoder_metadata=metadata,
    )
    scene = torch.load(source_root / "features/0.pt", weights_only=True)[
        "scene"
    ].clone()
    before = {p: p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    native = yaml.safe_load(open("configs/fafa.yaml"))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture weights")
    native["checkpoint"]["path"] = str(weights)
    native_path = tmp_path / "fafa.yaml"
    native_path.write_text(yaml.safe_dump(native))
    cfg = {
        "data": {
            "cache": str(root),
            "dino_cache": str(source_root),
            "final_dir": "unused",
            "image_root": str(tmp_path),
        },
        **metadata,
        "cache": {"storage_dtype": "float16"},
        "runtime": {"device": "cpu"},
        "person_encoder": {
            "backend": "fafa",
            "fafa_config": str(native_path),
            "batch_size": 1,
        },
    }
    monkeypatch.setattr(
        fafa_cache,
        "load_rcr_data",
        lambda *args: SimpleNamespace(
            gallery_ids=["image"], image_path=lambda _: image
        ),
    )
    model = FakeFAFA()
    monkeypatch.setattr(
        fafa_cache,
        "load_fafa",
        lambda *args: (model, {}, lambda crop: torch.ones(3, 4, 4), {}),
    )
    extract = model.extract_features
    monkeypatch.setattr(
        model, "extract_features", lambda *args, **kwargs: SimpleNamespace()
    )
    with pytest.raises(AttributeError):
        fafa_cache.finish_fafa_cache(cfg)
    with pytest.raises(RuntimeError, match="incomplete"):
        GalleryCache(root)
    monkeypatch.setattr(model, "extract_features", extract)
    fafa_cache.finish_fafa_cache(cfg)
    cache = GalleryCache(root, scene_root=source_root)
    assert cache.scene_dim == 8 and cache.person_dim == 5
    assert cache.encoder_metadata["person_encoder"]["backend"] == "fafa"
    loaded_scene, persons, _, _, _ = cache.load(torch.tensor([0]))
    torch.testing.assert_close(scene.float(), loaded_scene[0])
    torch.testing.assert_close(
        persons[0], torch.tensor([[2.5, 3.5, 4.5, 5.5, 6.5]] * 2)
    )
    torch.testing.assert_close(cache.global_features, loaded_scene.mean(1))
    saved = torch.load(root / "index.pt", weights_only=True)["persons"]
    assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
    assert before == {p: p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    assert not (root / "features").exists()
    assert "scene" not in torch.load(root / "index.pt", weights_only=True)
    check_cache_config(cache, cfg)
    monkeypatch.setattr(
        fafa_cache, "load_fafa", lambda *args: pytest.fail("loaded FAFA")
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    fafa_cache.finish_fafa_cache(cfg)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
