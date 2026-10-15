"""Native FDA equivalence, frozen gradients, stale-cache guards and resume."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rcr.baselines.fafa import fda_scores
from rcr.common.data import load_rcr_data
from rcr.proposed.cache import native_build
from rcr.proposed.cache.native import NativeFAFACache, query_key, subject_conditions
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.matching import partial_identity_matching
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import fafa_binding_scores
from tests.proposed.test_person_representation import FakeFAFA
from tests.proposed.test_runner import experiment as experiment
from tests.proposed.test_runner import local_stages as local_stages
from tests.proposed.test_training import _batch
from tools import run as cli


@pytest.mark.parametrize("tokens,k", [(3, 6), (8, 6), (8, 1)])
def test_native_binding_equals_official_fda_on_identity_assigned_pairs(tokens, k):
    generator = torch.Generator().manual_seed(18)
    query = F.normalize(torch.randn(1, 2, 2, 5, generator=generator), dim=-1)
    target = F.normalize(torch.randn(1, 2, tokens, 5, generator=generator), dim=-1)
    matching = partial_identity_matching(
        torch.eye(2)[None],
        torch.eye(2).flip(0)[None],
        torch.eye(2)[None],
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert matching.indices.tolist() == [[1, 0]]
    expected = (
        fda_scores(query[0, 0, 0:1], target[0], k)[0, 1]
        + fda_scores(query[0, 1, 1:2], target[0], k)[0, 0]
    ) / 2
    actual = fafa_binding_scores(query, target, matching=matching, k=k)
    torch.testing.assert_close(actual[0], expected)
    # Padding is not visual evidence, including NaN/Inf in unused slots.
    query[0, 0, 1] = float("nan")
    query[0, 1, 0] = float("inf")
    padded_target = torch.cat((target, torch.full_like(target[:, :1], float("nan"))), 1)
    torch.testing.assert_close(
        fafa_binding_scores(query, padded_target, matching=matching, k=k), actual
    )


def test_frozen_fafa_has_no_gradient_but_context_and_fusion_update():
    from rcr.proposed.losses import compute_loss

    torch.manual_seed(39)
    model = RCRModel(
        8, 6, 2, fine_matching={"identity_threshold": -1, "duplicate_iou": 1}
    )
    with torch.no_grad():
        model.grounding.score.weight.zero_()
        model.grounding.score.bias.fill_(5)
    batch = _batch()
    batch["fafa_query"].requires_grad_()
    batch["fafa_targets"].requires_grad_()
    loss, _ = compute_loss(
        model, batch, (2, 3), grounding_weight=0, identity_weight=0, state_weight=0
    )
    loss.backward()
    assert batch["fafa_query"].grad is batch["fafa_targets"].grad is None
    assert model.binding_log_scale.grad.abs() > 0
    assert model.reasoner.score[-1].weight.grad.abs().sum() > 0
    assert model.composition.attn.in_proj_weight.grad.abs().sum() > 0
    assert model.identity_head.proj.weight.grad is None


def _instruction(text):
    return {
        "sample_id": "q",
        "query_image_id": "image",
        "subjects": [{"subject_id": 1}, {"subject_id": 2}],
        "final_change": "then retrieve target images where " + text,
    }


def test_subject_conditions_split_independent_changes_and_keep_relations():
    row = _instruction("Subject 1 wears red and Subject 2 wears blue")
    assert subject_conditions(row) == [
        "the reference person wears red",
        "the reference person wears blue",
    ]
    row["subjects"].reverse()
    assert subject_conditions(row) == [
        "the reference person wears blue",
        "the reference person wears red",
    ]
    for text in (
        "Subject 1 is behind Subject 2 and Subject 2 wears blue",
        "Subject 1 and Subject 2 wear blue",
        "Subject 1 wears red while Subject 2 wears blue",
    ):
        row = _instruction(text)
        conditions = subject_conditions(row)
        assert all("other referenced person" in condition for condition in conditions)
    group = _instruction("the members of Subject 1 are standing together")
    group["subjects"] = [{"subject_id": 1}]
    assert subject_conditions(group) == ["the reference person is standing together"]


def test_query_key_ignores_annotations_but_tracks_instruction_and_subject_order():
    row = _instruction("Subject 1 wears red and Subject 2 wears blue")
    original = query_key(row)
    labeled = copy.deepcopy(row)
    labeled.update(case_type="DUAL", positive_image_ids=["x"])
    labeled["subjects"][0]["identity_ids"] = ["private label"]
    assert query_key(labeled) == original
    labeled["final_change"] += " shoes"
    assert query_key(labeled) != original
    labeled = copy.deepcopy(row)
    labeled["subjects"].reverse()
    assert query_key(labeled) != original


def test_native_cache_shapes_text_validation_and_readonly_reuse(
    experiment,
    local_stages,
    monkeypatch,  # noqa: F811
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    gallery = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    cache = NativeFAFACache(cfg["data"]["native_cache"], gallery)
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    query = cache.load_queries(data.samples, 2)
    targets = cache.load_targets(torch.tensor([2, 0, 2]), 2)
    assert query.shape == (3, 1, 2, 3)
    assert targets.shape == (3, 2, 7, 3)
    assert not query.requires_grad and not targets.requires_grad
    assert query[:, :, 1:].count_nonzero() == 0
    torch.testing.assert_close(query[0, 0, 0], torch.tensor([0.6, 0.8, 0.0]))
    torch.testing.assert_close(targets[0], targets[2])
    assert targets[:, 1:].count_nonzero() == 0
    changed = {
        **data.samples[0],
        "final_change": data.samples[0]["final_change"] + " now",
    }
    with pytest.raises(ValueError, match="missing/stale"):
        cache.load_queries([changed], 2)
    with pytest.raises(ValueError, match="person padding differs"):
        cache.load_targets(torch.tensor([0]), 0)
    root = Path(cfg["data"]["native_cache"])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        native_build, "load_fafa", lambda *a: pytest.fail("loaded FAFA")
    )
    cfg["data"]["image_root"] = "missing-images"
    native_build.finish_native_cache(cfg)
    assert all(path.read_bytes() == value for path, value in before.items())
    gallery.cache_id = "rebuilt"
    with pytest.raises(ValueError, match="differs from the person cache"):
        NativeFAFACache(root, gallery)


def test_interrupted_native_build_resumes_targets_and_query_shards(
    experiment,
    local_stages,
    monkeypatch,  # noqa: F811
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "dino"])
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "persons"])
    model = FakeFAFA()
    original = model.extract_features
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "extract_features", interrupt)
    monkeypatch.setattr(
        native_build,
        "load_fafa",
        lambda *a: (
            model,
            {"eval": str},
            lambda crop: torch.ones(3, 4, 4),
            {},
        ),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        native_build.finish_native_cache(cfg)
    root = Path(cfg["data"]["native_cache"])
    assert (root / ".building").is_file()
    gallery = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    with pytest.raises(RuntimeError, match="incomplete"):
        NativeFAFACache(root, gallery)
    saved = {p: p.read_bytes() for p in root.rglob("*.pt")}
    monkeypatch.setattr(model, "extract_features", original)
    monkeypatch.setattr(
        model,
        "extract_target_features",
        lambda *a, **k: pytest.fail("recomputed targets"),
    )
    native_build.finish_native_cache(cfg)
    assert not (root / ".building").exists()
    assert all(
        p.read_bytes() == value for p, value in saved.items() if p.name != ".build.pt"
    )
    cache = NativeFAFACache(root, gallery)
    assert (
        cache.cache_id
        == torch.load(next(root.glob("queries/*.pt")), weights_only=True)["cache_id"]
    )


def test_native_cache_handles_no_detection_without_inventing_a_crop(
    experiment,
    local_stages,
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "dino"])
    source = Path(cfg["data"]["dino_cache"])
    index = torch.load(source / "index.pt", weights_only=True)
    index["mask"][0] = False
    torch.save(index, source / "index.pt")
    item = torch.load(source / "features/0.pt", weights_only=True)
    for key in ("persons", "boxes_scene", "boxes_pixel"):
        item[key] = item[key][:0]
    item["identity_ids"] = []
    torch.save(item, source / "features/0.pt")
    cli.main(["build-cache", "--config", str(path)])
    gallery = GalleryCache(
        cfg["data"]["cache"], scene_root=source, native_root=cfg["data"]["native_cache"]
    )
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    assert gallery.native_queries(data.samples[:1], 0).shape == (1, 1, 0, 3)
    assert gallery.native_targets(torch.tensor([0]), 0).shape == (1, 0, 7, 3)
