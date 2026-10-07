"""Architectural regressions: gradients, same-person binding and frozen FAFA cache."""

import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from rcr.methods.baselines import fafa as native_fafa
from rcr.methods.proposed import fafa_cache
from rcr.methods.proposed.build_cache import build_cache
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import ImageEncoder
from rcr.methods.proposed.losses import identity_loss
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.objective import compute_loss
from rcr.methods.proposed.person_binding import binding_scores
from rcr.methods.proposed.person_encoder import run_person_worker
from tests.methods.proposed.test_build_cache import (
    Backbone,
    Detector,
    DetectorProcessor,
    Processor,
)
from tests.methods.proposed.test_training import _batch
from tools.methods import cache_fafa


@pytest.mark.parametrize("failure", [None, "checkpoint", "assets", "imports"])
def test_fafa_preflight_runs_before_scene_models(tmp_path, monkeypatch, failure):
    weights = tmp_path / "weights.pt"
    if failure != "checkpoint":
        weights.write_bytes(b"fixture")
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump({"checkpoint": {"path": str(weights)}}))
    cfg = {
        "person_encoder": {
            "backend": "fafa",
            "python": sys.executable,
            "fafa_config": str(native),
            "batch_size": 4,
        }
    }
    monkeypatch.setattr(native_fafa, "official_source", lambda cfg: tmp_path)
    monkeypatch.setattr(native_fafa, "assets_ready", lambda cfg: failure != "assets")

    def imports(directory):
        if failure == "imports":
            raise ImportError("native dependency missing")

    monkeypatch.setattr(native_fafa, "official_api", imports)
    monkeypatch.setattr(
        fafa_cache, "finish_fafa_cache", lambda cfg: pytest.fail("loaded FAFA weights")
    )

    def execute(command, *, check):
        assert check and command[-1] == "--check"
        cache_fafa.main(command[2:])

    monkeypatch.setattr("rcr.methods.proposed.person_encoder.subprocess.run", execute)
    if failure is None:
        run_person_worker(cfg, check=True)
    else:
        error = {
            "checkpoint": FileNotFoundError,
            "assets": RuntimeError,
            "imports": ImportError,
        }[failure]
        with pytest.raises(error):
            run_person_worker(cfg, check=True)


def test_retrieval_updates_semantic_and_scale_but_never_identity():
    torch.manual_seed(31)
    model = RCRModel(8, 6, 2)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    before = model.identity_head.proj.weight.detach().clone()
    loss, _ = compute_loss(model, _batch(), (2, 3), identity_weight=0, state_weight=0)
    loss.backward()
    assert model.identity_head.proj.weight.grad is None
    assert model.semantic_proj.weight.grad.abs().sum() > 0
    assert model.binding_log_scale.grad.abs() > 0
    optimizer.step()
    torch.testing.assert_close(before, model.identity_head.proj.weight)
    model.zero_grad(set_to_none=True)
    identity_loss(
        model.encode_identity(torch.randn(4, 8)), torch.tensor([1, 2, 1, 2])
    ).backward()
    assert model.identity_head.proj.weight.grad.abs().sum() > 0
    assert model.semantic_proj.weight.grad is None
    assert (
        model.semantic_proj.weight.data_ptr()
        != model.identity_head.proj.weight.data_ptr()
    )


def test_dual_heads_support_different_person_and_scene_widths():
    model = RCRModel(8, 6, 2, input_dim=10, person_input_dim=12).eval()
    batch = _batch()
    for key in ("query_scene", "target_scene"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 10)
    for key in ("query_persons", "target_persons"):
        batch[key] = torch.randn(*batch[key].shape[:-1], 12)
    loss, _ = compute_loss(model, batch, (2, 3))
    assert torch.isfinite(loss)
    assert model.target_builder.identity_proj is None
    assert model.identity_head.proj.in_features == 12
    assert model.semantic_proj.in_features == 12


def test_shared_control_keeps_the_original_identity_retrieval_path():
    model = RCRModel(8, 6, 2, representation="shared", binding_mode="none")
    loss, _ = compute_loss(model, _batch(), (2, 3), identity_weight=0, state_weight=0)
    loss.backward()
    assert model.identity_head.proj.weight.grad.abs().sum() > 0
    assert model.semantic_proj is None
    assert model.target_builder.identity_proj is not None
    assert model.binding_log_scale is None


def test_identity_and_state_must_match_the_same_target_person():
    # Person A has identity=1, semantics=0. Person B has identity=0, semantics=1.
    result = binding_scores(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        torch.ones(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert result["both"].item() == 1.0  # max(ID+SEM), never max(ID)+max(SEM)=2.
    assert result["identity"].item() == result["semantic"].item() == 1.0
    assert result["none"].item() == 0.0


def test_binding_respects_member_masks_and_empty_targets():
    torch.manual_seed(5)
    qid = torch.randn(2, 3, 4, requires_grad=True)
    tid = torch.randn(2, 2, 4, requires_grad=True)
    comp = torch.randn(2, 2, 3, 6, requires_grad=True)
    semantic = torch.randn(2, 2, 6, requires_grad=True)
    members = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
    subjects = torch.tensor([[True, False]] * 2)
    mask = torch.tensor([[True, False], [False, False]])
    result = binding_scores(qid, tid, comp, semantic, members, subjects, mask)
    assert all(value[1] == 0 for value in result.values())
    result["both"].sum().backward()
    assert qid.grad is tid.grad is None
    assert semantic.grad[:, 1].count_nonzero() == 0
    assert comp.grad[:, 1].count_nonzero() == 0
    assert all(torch.isfinite(x.grad).all() for x in (comp, semantic))


class FakeFAFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qformer = SimpleNamespace(config=SimpleNamespace(hidden_size=5))
        self.weight = nn.Parameter(torch.ones(1))

    def extract_features(self, samples, mode):
        assert mode == "image" and set(samples) == {"image"}
        assert not self.training and not self.weight.requires_grad
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
    native = yaml.safe_load(open("configs/methods/fafa.yaml"))
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
    saved = torch.load(root / "features/0.pt", weights_only=True)["persons"]
    assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
    assert before == {p: p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    assert "scene" not in torch.load(root / "features/0.pt", weights_only=True)
    cache.validate_encoders(cfg)
    monkeypatch.setattr(
        fafa_cache, "load_fafa", lambda *args: pytest.fail("loaded FAFA")
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    fafa_cache.finish_fafa_cache(cfg)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
