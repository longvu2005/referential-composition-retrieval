"""CPU contracts for train-only, budgeted crop mining and SupCon gradients."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rcr.proposed import losses
from rcr.proposed import train as training
from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.identity import (
    IdentityCropSampler,
    identity_sampling_settings,
    select_identity_observations,
)
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.supervision import TrainSupervision
from tests.test_train import (
    TextBackbone,
    TinyCache,
    Tokenizer,
    _has_gradient,
    _only_loss,
    _sample,
)
from tests.test_train import (
    tiny_training_run as tiny_training_run,
)


@pytest.fixture
def crop_batch(tmp_path):
    cache = TinyCache()
    cache.image_ids = ["q", "p", "n", "x", "r", "s", "v"]
    cache.by_id = {image: i for i, image in enumerate(cache.image_ids)}
    generator = torch.Generator().manual_seed(13)
    cache.scenes = torch.randn(7, 4, 8, generator=generator)
    cache.persons = torch.randn(7, 6, 8, generator=generator)
    cache.boxes = torch.tensor(
        [[i, 0, i + 0.8, 1] for i in range(6)], dtype=torch.float32
    )[None].repeat(7, 1, 1)
    identities = {
        "q": ["A", "B", "Z"],
        "p": ["A", "B"],
        "n": ["A", "C", "D", None, "E"],
        "x": ["A", "B"],
        "r": ["C"],
        "s": ["C"],
        "v": ["A"],
    }
    cache.mask = torch.tensor(
        [[i < len(identities[image]) for i in range(6)] for image in cache.image_ids]
    )
    cache.load_geometry = lambda index: {
        "boxes_pixel": cache.boxes[index, cache.mask[index]]
    }
    heads = {
        image: [
            {
                "identity_id": identity,
                "x": i + 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
            }
            for i, identity in enumerate(ids)
            if identity is not None
        ]
        for image, ids in identities.items()
    }
    heads["n"].append(
        {"identity_id": "F", "x": 4.5, "y": 0.1, "width": 0.1, "height": 0.1}
    )
    sample = _sample("train", "q", "p")
    sample["subjects"] = [
        {"subject_id": 1, "identity_ids": ["A"]},
        {"subject_id": 2, "identity_ids": ["B"]},
    ]
    sample["final_desc"] = (
        "Identify Subject 1 as the first person and Subject 2 as the second person"
    )
    sample["final_change"] = "Subject 1 beside Subject 2"
    data = SimpleNamespace(
        gallery_ids=cache.image_ids,
        images_by_id={
            i: {"path": f"{'val' if i == 'v' else 'train'}/{i}.jpg"}
            for i in cache.image_ids
        },
        gt_head_boxes_by_image=heads,
    )
    supervision = TrainSupervision(data, cache, cache_file=tmp_path / "labels.json")
    supervision.prepare([sample], all_train=True, supplemental=True)
    settings = identity_sampling_settings(
        {"max_crops": 6, "hard_negative_fraction": 0.5}
    )
    extras = [("x", 0, "A"), ("x", 1, "B"), ("r", 0, "C"), ("s", 0, "C")]
    return cache, data, supervision, sample, settings, extras


def _visual(fixture, *, extras=None, settings=None, copies=1):
    cache, _, supervision, sample, default_settings, default_extras = fixture
    return prepare_visual_batch(
        [sample] * copies,
        [["p", "n"]] * copies,
        cache,
        supervision=supervision,
        identity_image_ids=[{"p", "n"}] * copies,
        identity_crops=default_extras if extras is None else extras,
        identity_sampling=default_settings if settings is None else settings,
    )


def _embeddings(batch):
    query = torch.zeros(*batch["query_identity_labels"].shape, 3, requires_grad=True)
    target = torch.zeros(*batch["target_identity_labels"].shape, 3, requires_grad=True)
    with torch.no_grad():
        query[:, :3] = torch.eye(3)
        target[:, 1, :3] = torch.tensor([[1.0, 0, 0], [0.98, 0.2, 0], [0, 0, 1.0]])
        target[:, 1, 3:] = torch.tensor([1.0, 0, 0])
    return query, target


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_hard_fraction(fraction):
    with pytest.raises(ValueError, match="hard_negative_fraction"):
        identity_sampling_settings({"hard_negative_fraction": fraction})
    assert identity_sampling_settings({})["hard_negative_fraction"] == 0


def test_safe_labels_reuse_cache_and_keep_original_alignment(crop_batch, monkeypatch):
    cache, data, supervision, sample, _, _ = crop_batch
    original = supervision.load_ids(["n"], 6)[0]
    assert original[4] in ("E", "F")  # Existing greedy/BCE labels stay unchanged.
    assert supervision.load_ids(["n"], 6, supplemental=True)[0] == [
        "A",
        "C",
        "D",
        None,
        None,
        None,
    ]
    # Read the old cache format, then prepare only the new geometry labels.
    stored = json.loads(supervision.cache_file.read_text())
    stored.pop("supplemental_identities")
    supervision.cache_file.write_text(json.dumps(stored))
    restored = TrainSupervision(data, cache, cache_file=supervision.cache_file)
    restored.prepare([sample], all_train=True, supplemental=True)
    assert restored.load_ids(["n"], 6) == [original]
    monkeypatch.setattr(
        cache, "load_geometry", lambda *a: pytest.fail("already cached")
    )
    cached = TrainSupervision(data, cache, cache_file=supervision.cache_file)
    cached.prepare([sample], all_train=True, supplemental=True)
    assert cached.load_ids(["n"], 6, supplemental=True) == restored.load_ids(
        ["n"], 6, supplemental=True
    )
    with pytest.raises(ValueError, match="non-train"):
        cached.load_ids(["v"], supplemental=True)


def test_ambiguous_query_anchors_and_extra_crops_are_excluded(crop_batch):
    cache, data, _, sample, settings, _ = crop_batch
    data.gt_head_boxes_by_image["q"].append(
        {"identity_id": "other", "x": 0.5, "y": 0.1, "width": 0.1, "height": 0.1}
    )
    supervision = TrainSupervision(data, cache)
    supervision.prepare([sample], all_train=True, supplemental=True)
    fixture = (cache, data, supervision, sample, settings, [])
    batch = _visual(fixture, extras=[("n", 4, "E"), ("n", 3, None), ("n", 5, None)])
    assert batch["identity_anchor_mask"].tolist() == [
        [False, True, False, False, False, False]
    ]
    assert "extra_identity_labels" not in batch


def test_hardness_uses_required_anchor_ids_and_masks_unknown_ambiguous_padding(
    crop_batch,
):
    settings = {**crop_batch[4], "max_crops": 2}
    # No reserved extras: quota=1. D is closest to the unrequested person Z;
    # C is the closest wrong ID to the requested A/B anchors.
    batch = _visual(crop_batch, extras=[], settings=settings)
    assert batch["identity_anchor_mask"].tolist() == [
        [True, True, False, False, False, False]
    ]
    assert batch["supplemental_target_indices"].tolist() == [6, 7, 8]
    query, target = _embeddings(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hard, extra = select_identity_observations(query, target, batch, settings)
    assert hard.tolist() == [7] and not extra.numel()
    assert not hard.requires_grad and not extra.requires_grad
    # The other requested Subject's ID is compared by identity, not by its slot.
    batch["identity_anchor_mask"][0, 0] = False
    with torch.no_grad():
        target[0, 1, 0] = query[0, 1]
    hard, _ = select_identity_observations(query, target, batch, settings)
    assert hard.tolist() == [6]  # A is a valid hard example for anchor B.


@pytest.mark.parametrize(
    "limit,fraction", [(1, 0.5), (4, 0.5), (5, 0.5), (6, 0.5), (6, 1.0)]
)
def test_positive_quota_budget_deduplication_and_fallback(crop_batch, limit, fraction):
    # C in a negative candidate is also present in the random pool. It must
    # appear once; C from r/s remains distinct cross-image identity evidence.
    extras = [*crop_batch[5], ("n", 1, "C"), ("q", 0, "A"), ("p", 0, "A")]
    settings = {**crop_batch[4], "max_crops": limit, "hard_negative_fraction": fraction}
    batch = _visual(crop_batch, extras=extras, settings=settings, copies=2)
    query, target = _embeddings(batch)
    hard, chosen = select_identity_observations(query, target, batch, settings)
    p = min(2, limit)
    assert chosen[:p].tolist() == list(range(p))
    assert len(hard) == min(3, int(fraction * (limit - p)))
    assert len(hard) + len(chosen) <= limit
    assert batch["query_identity_mask"].sum() == 3
    assert batch["target_identity_mask"].sum() == 2
    assert len(batch["supplemental_target_indices"]) == 3
    hard_keys = [("n", int(i) % 6) for i in hard]
    selected_keys = [extras[int(i)][:2] for i in chosen]
    assert len(set(hard_keys + selected_keys)) == len(hard_keys + selected_keys)
    if limit == 6 and fraction == 0.5:
        assert ("r", 0) in selected_keys and ("s", 0) in selected_keys
    if limit == 6 and fraction == 1:
        assert len(hard) == 3  # Fewer than quota=4; fill one random observation.
        assert len(chosen) == 3


def test_selection_uses_no_autograd_storage_and_fp32_similarity(
    crop_batch, monkeypatch
):
    batch = _visual(crop_batch)
    query, target = _embeddings(batch)
    normalize = torch.nn.functional.normalize
    saved, calls = [], []

    def checked_normalize(value, *args, **kwargs):
        calls.append(value.dtype)
        assert not torch.is_grad_enabled()
        return normalize(value, *args, **kwargs)

    monkeypatch.setattr("rcr.proposed.identity.F.normalize", checked_normalize)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda x: saved.append(x) or x, lambda x: x
    ):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            select_identity_observations(query, target, batch, crop_batch[4])
    assert calls == [torch.float32, torch.float32]
    assert saved == []


def test_repeated_query_image_preserves_requested_anchors_across_subject_sets(
    crop_batch,
):
    cache, _, supervision, sample, settings, _ = crop_batch
    first = {
        **sample,
        "subjects": [
            {"subject_id": 1, "identity_ids": ["B"]},
            {"subject_id": 2, "identity_ids": ["Z"]},
        ],
    }
    batch = prepare_visual_batch(
        [first, sample],
        [["p", "n"], ["p", "n"]],
        cache,
        supervision=supervision,
        identity_image_ids=[{"p", "n"}] * 2,
        identity_sampling=settings,
    )
    # A is only requested by the second query. Its physical SupCon observation
    # was already retained in the first row, but mining must still use A.
    assert batch["query_identity_mask"][1].sum() == 0
    assert batch["identity_anchor_mask"][:, :3].tolist() == [
        [False, True, True],
        [True, False, False],
    ]


@pytest.mark.parametrize("max_crops,fraction", [(6, 0.0), (0, 0.5)])
def test_disabled_mining_matches_baseline_without_reading_negative_labels(
    crop_batch, monkeypatch, max_crops, fraction
):
    cache, _, supervision, sample, settings, _ = crop_batch
    settings = {**settings, "max_crops": max_crops, "hard_negative_fraction": fraction}
    sampler = IdentityCropSampler(supervision, settings)
    crops = sampler.sample([sample], [["p", "n"]], torch.Generator().manual_seed(5))
    baseline = prepare_visual_batch(
        [sample],
        [["p", "n"]],
        cache,
        supervision=supervision,
        identity_image_ids=[{"p", "n"}],
        identity_crops=crops,
    )
    load = supervision.load_ids

    def baseline_labels(image_ids, *args, **kwargs):
        assert not kwargs.get("supplemental")
        assert set(image_ids) <= {"q", "p"}
        return load(image_ids, *args, **kwargs)

    monkeypatch.setattr(supervision, "load_ids", baseline_labels)
    actual = _visual(crop_batch, settings=settings, extras=crops)
    assert actual.keys() == baseline.keys()
    for key in baseline:
        torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)


def test_batch_loss_backward_keeps_same_id_negatives_positive_and_reuses_embeddings(
    crop_batch, monkeypatch
):
    cache, _, supervision, sample, settings, _ = crop_batch
    settings = {**settings, "max_crops": 4}
    crops = [("n", 0, "A"), ("x", 0, "A"), ("r", 0, "C"), ("s", 0, "C")]
    batch = build_batch(
        [sample],
        [["p", "n"]],
        cache,
        Tokenizer(),
        TextEncoder(TextBackbone(), dim=8),
        "cpu",
        supervision=supervision,
        identity_image_ids=[{"p", "n"}],
        identity_crops=crops,
        identity_sampling=settings,
    )
    assert batch["positive_mask"].tolist() == [[True, False]]
    assert batch["target_identity_mask"][0, 1].sum() == 0
    assert (
        batch["supplemental_target_labels"][0] == batch["query_identity_labels"][0, 0]
    )
    batch["target_persons"].requires_grad_()
    model = RCRModel(
        dim=8, identity_dim=6, num_heads=2, fine_matching={"background_logit": -3.5}
    ).eval()
    calls, observed = [], {}
    model.identity_head.register_forward_hook(lambda m, a, out: calls.append(out.shape))
    original_loss = losses.identity_loss

    def observe(features, labels, temperature):
        observed.update(features=features, labels=labels.clone())
        return original_loss(features, labels, temperature)

    monkeypatch.setattr(losses, "identity_loss", observe)
    select = losses.select_identity_observations

    def observe_selection(*args):
        hard, extra = select(*args)
        observed["hard"] = hard
        return hard, extra

    monkeypatch.setattr(losses, "select_identity_observations", observe_selection)
    total, parts = losses.compute_loss(
        model, batch, cache.patch_hw, **_only_loss("identity")
    )
    total.backward()
    assert torch.isfinite(total) and observed["features"].requires_grad
    assert (observed["labels"] == batch["query_identity_labels"][0, 0]).sum() == 4
    assert _has_gradient(model.identity_head) and _has_gradient(model.person_proj)
    assert batch["target_persons"].grad[0, 1, 0].abs().sum() > 0
    assert batch["target_persons"].grad.flatten(0, 2)[observed["hard"]].abs().sum() > 0
    assert batch["target_persons"].grad[0, 1, 3:].count_nonzero() == 0
    assert calls == [
        torch.Size([1, 6, 6]),
        torch.Size([1, 2, 6, 6]),
        torch.Size([2, 6]),
    ]
    assert parts["identity_hard_crops"] == 1
    assert parts["identity_random_crops"] == 3
    assert parts["identity_supplemental_crops"] == 4


@pytest.mark.parametrize("disabled", [None, "fraction", "identity", "budget"])
def test_train_warmup_wiring_checkpoint_and_epoch_counts(
    tiny_training_run, monkeypatch, disabled
):
    cfg, _, wandb_run, _ = tiny_training_run
    cfg = deepcopy(cfg)
    cfg["train"].update(epochs=2)
    cfg["retrieval"]["coarse_beta"] = 0  # This fixture has no same-ID state pair.
    cfg["train"]["sampling"].update(warmup_epochs=1, hard_fraction=0)
    cfg["train"]["identity_sampling"].update(max_crops=4, hard_negative_fraction=0.5)
    if disabled == "fraction":
        cfg["train"]["identity_sampling"]["hard_negative_fraction"] = 0
    if disabled == "identity":
        cfg["loss"]["identity_weight"] = 0
    if disabled == "budget":
        cfg["train"]["identity_sampling"]["max_crops"] = 0
    data = training.load_rcr_data(None)
    data.gt_head_boxes_by_image["n"][0]["identity_id"] = "p2"
    build, evaluate = training.build_batch, training._evaluate
    mining, backgrounds = [], []

    def observe_batch(*args, **kwargs):
        mining.append(kwargs["identity_sampling"] is not None)
        return build(*args, **kwargs)

    def observe_evaluate(*args, **kwargs):
        backgrounds.append(args[6].fine_matching["background_logit"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "build_batch", observe_batch)
    monkeypatch.setattr(training, "_evaluate", observe_evaluate)
    checkpoint = training.train(cfg)
    assert mining == [False, disabled is None]
    assert backgrounds == [-3.5, -3.5]
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["model"]["fine_matching"]["background_logit"] == -3.5
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["train_identity_hard_crops"] for row in history] == [
        0,
        int(disabled is None),
    ]
    for row in history:
        assert row["train_identity_supplemental_crops"] <= 4
        assert row["train_identity_supplemental_crops"] == (
            row["train_identity_random_crops"] + row["train_identity_hard_crops"]
        )
    assert all("train/identity_hard_crops" not in row for row in wandb_run.logs)
    assert any("epoch/identity_hard_crops" in row for row in wandb_run.logs)
