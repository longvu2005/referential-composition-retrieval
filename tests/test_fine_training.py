"""Train-only fine mining, candidate provenance and rejection gradients."""

import json
from copy import deepcopy

import pytest
import torch

from rcr.proposed import ranking
from rcr.proposed import train as training
from rcr.proposed.batch import prepare_visual_batch
from rcr.proposed.losses import compute_loss, loss_settings
from rcr.proposed.sampling import CandidateIndex, sample_candidates, sampling_settings
from tests.test_ranking import tiny_ranker as tiny_ranker
from tests.test_train import (
    Tokenizer,
    _only_loss,
    _sample,
)
from tests.test_train import (
    objective_batch as objective_batch,
)
from tests.test_train import (
    tiny_training_run as tiny_training_run,
)


def test_candidate_quotas_keep_provenance_after_shuffle_and_exclusions():
    sample = _sample("train", "q", "p0")
    sample["positive_image_ids"] = positives = [f"p{i}" for i in range(3)]
    semantic = [f"s{i}" for i in range(30)]
    wrong = [f"w{i}" for i in range(30)]
    index = CandidateIndex(
        [sample],
        ["q", *positives, *semantic, *wrong, "disputed"],
        {"train": [*positives, *semantic]},
        {"train": {"disputed"}},
    )
    pools = {
        "train": {
            "semantic": ["w29", *semantic],
            "identity": ["p0", "s0", "disputed", *wrong],
        }
    }
    options = dict(
        index=index,
        positives_per_query=2,
        identity_fraction=0.5,
        hard_fraction=0.1,
        semantic_hard_fraction=0.5,
        identity_rejection=True,
        hard_pools=pools,
    )
    first = sample_candidates([sample], 32, torch.Generator().manual_seed(3), **options)
    second = sample_candidates(
        [sample], 32, torch.Generator().manual_seed(3), **options
    )
    assert first == second
    row = set(first.images[0])
    assert len(first.images[0]) == len(row) == 32
    assert len(row & set(positives)) == 2
    assert len(row & set(semantic)) >= 15
    assert first.semantic_hard_count == 8
    assert len(row & set(semantic[:16])) >= 8
    assert first.identity_hard_count == 3
    assert first.hard_identity_ids[0] <= row & set(wrong[:6])
    assert len(first.hard_identity_ids[0]) == 3
    assert not row & {"q", "disputed"}


def test_small_pools_and_random_fallback_are_not_tagged_as_hard():
    sample = _sample("train", "q", "p")
    gallery = ["q", "p", "s0", "s1", *[f"w{i}" for i in range(40)]]
    index = CandidateIndex([sample], gallery, {"train": ["p", "s0", "s1"]})
    sampled = sample_candidates(
        [sample],
        32,
        torch.Generator().manual_seed(4),
        index=index,
        identity_fraction=0.5,
        hard_fraction=0.1,
        semantic_hard_fraction=0.5,
        identity_rejection=True,
        hard_pools={"train": {"semantic": ["s1", "s0"], "identity": []}},
    )
    assert {"s0", "s1", "p"} <= set(sampled.images[0])
    assert len(set(sampled.images[0])) == 32
    assert sampled.semantic_hard_count == sampled.identity_hard_count == 0
    assert sampled.hard_identity_ids == [set()]


def test_batch_rejection_mask_follows_ids_and_rejects_bad_provenance(tiny_ranker):
    cache, _, _, samples, _ = tiny_ranker
    from types import SimpleNamespace

    supervision = SimpleNamespace(
        load_ids=lambda ids, width: [["p1"] * width for _ in ids]
    )
    options = dict(
        supervision=supervision,
        identity_image_ids=[{"a", "n"}],
        hard_identity_image_ids=[{"va"}],
    )
    first = prepare_visual_batch([samples[0]], [["va", "a", "n"]], cache, **options)
    second = prepare_visual_batch([samples[0]], [["n", "va", "a"]], cache, **options)
    assert first["hard_identity_negative_mask"].tolist() == [[True, False, False]]
    assert second["hard_identity_negative_mask"].tolist() == [[False, True, False]]
    assert second["identity_positive_mask"].tolist() == [[True, False, True]]
    with pytest.raises(ValueError, match="cannot contain all required"):
        prepare_visual_batch(
            [samples[0]],
            [["n", "a", "va"]],
            cache,
            **{**options, "hard_identity_image_ids": [{"n"}]},
        )
    with pytest.raises(ValueError, match="provenance"):
        prepare_visual_batch(
            [samples[0]],
            [["n", "a", "va"]],
            cache,
            **{**options, "hard_identity_image_ids": [{"missing"}]},
        )


def test_rejection_adds_wrong_id_gradient_without_changing_other_targets(
    objective_batch,
):
    model, batch = objective_batch
    batch["hard_identity_negative_mask"][0, 2] = True
    settings = _only_loss("context")
    old, old_parts = compute_loss(model, batch, (2, 2), **settings)
    old_grad = torch.autograd.grad(old, batch["target_persons"])[0]
    new, parts = compute_loss(
        model, batch, (2, 2), **{**settings, "context_rejection_weight": 0.2}
    )
    new.backward()
    torch.testing.assert_close(parts["context_semantic"], old_parts["context"])
    torch.testing.assert_close(
        new.detach(), old.detach() + 0.2 * parts["context_rejection"]
    )
    assert old_grad[0, 2].count_nonzero() == 0
    assert batch["target_persons"].grad[0, 2].abs().sum() > 0
    assert parts["context_rejection_active_fraction"] == 1
    assert all(p.grad is None for p in model.identity_head.parameters())
    assert batch["identity_positive_mask"].tolist() == [[True, True, False]]


@pytest.mark.parametrize("coverage", [[0, 1, 1], [1, 1, 0], [0, 0, 0]])
def test_rejection_skips_unsupported_pairs_with_finite_zero(
    objective_batch, monkeypatch, coverage
):
    from types import SimpleNamespace

    model, batch = objective_batch
    batch["hard_identity_negative_mask"][0, 2] = True
    raw = torch.tensor([0.5, 0.7, 0.8], requires_grad=True)
    monkeypatch.setattr(
        model,
        "score_context",
        lambda *a: (raw, SimpleNamespace(coverage=torch.tensor(coverage).float())),
    )
    loss, parts = compute_loss(
        model,
        batch,
        (2, 2),
        **{**_only_loss("context"), "context_rejection_weight": 0.2},
    )
    loss.backward()
    assert parts["context_rejection"] == 0
    assert parts["context_rejection_active_fraction"] == 0
    assert torch.isfinite(raw.grad).all() and raw.grad[2] == 0


def test_zero_rejection_weight_preserves_loss_even_with_hard_tags(objective_batch):
    model, batch = objective_batch
    settings = _only_loss("context")
    expected, _ = compute_loss(model, batch, (2, 2), **settings)
    batch["hard_identity_negative_mask"][0, 2] = True
    actual, parts = compute_loss(model, batch, (2, 2), **settings)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert parts["context_rejection"] == 0


def test_fine_mining_uses_common_union_filters_and_restores_modes(
    tiny_ranker, monkeypatch
):
    cache, model, text, samples, settings = tiny_ranker
    sample = samples[0]
    second = {**deepcopy(sample), "sample_id": "train2"}
    samples = [sample, second]
    cache.scenes[:, 0, 0] = torch.arange(6)
    scored = []
    native = cache.native_queries

    def public_queries(rows, width):
        for row in rows:
            assert "positive_image_ids" not in row and "case_type" not in row
            assert all(set(s) == {"subject_id"} for s in row["subjects"])
        return native(rows, width)

    def identities(cache, model, device, batch_size, image_indices):
        features = image_indices.float()[:, None, None].expand(-1, 1, 6)
        return [(features, torch.ones(len(image_indices), 1, dtype=torch.bool))]

    def coarse(query, logits, gallery, *a, **kw):
        return (10 - gallery[:, 0, 0])[None].expand(len(query), -1)

    def fine(query, scene, *args, **kwargs):
        assert not model.training and not text.training and not torch.is_grad_enabled()
        ids = scene[:, 0, 0].long()
        scored.extend(ids.tolist())
        values = torch.tensor([9, 9, 0.1, 0.7, 0.8, 0.3])[ids]
        return {
            "context": torch.tensor([0, 0, -3, 5, 20, 1]).float()[ids],
            "coverage": (ids != 4).float(),
            **{f"binding_{mode}": values for mode in ranking.POOLING_MODES},
        }

    monkeypatch.setattr(cache, "native_queries", public_queries)
    monkeypatch.setattr(ranking, "_encode_gallery_identity", identities)
    monkeypatch.setattr(ranking, "identity_scores_batched", coarse)
    monkeypatch.setattr(model, "score_target", fine)
    model.train()
    text.eval()
    options = dict(
        gallery_ids=cache.image_ids,
        pool_size=2,
        identity_image_ids={r["sample_id"]: {"q", "a", "n", "vq"} for r in samples},
        semantic_hard=True,
        identity_rejection=True,
    )
    results = []
    normalized_sizes = []
    normalize = ranking.fine_score_components

    def observe_normalization(binding, *args, **kwargs):
        normalized_sizes.append(len(binding))
        return normalize(binding, *args, **kwargs)

    monkeypatch.setattr(ranking, "fine_score_components", observe_normalization)
    for batch_size in (1, 4):
        results.append(
            ranking.mine_hard_negatives(
                samples,
                cache,
                Tokenizer(),
                text,
                model,
                torch.device("cpu"),
                retrieval={**settings, "coarse_beta": 0, "fine_batch_size": batch_size},
                query_batch_size=1 if batch_size == 1 else 2,
                **options,
            )
        )
    assert results[0] == results[1]
    assert normalized_sizes == [4, 4, 4, 4]
    assert set(scored) == {2, 3, 4, 5}  # Self/Full positives were never fine-scored.
    for pools in results[0].values():
        assert pools["semantic"] == ["vq", "n"]
        assert pools["identity"] == ["vn"]  # High FDA with zero coverage is excluded.
    assert model.training and not text.training

    scored.clear()
    empty = ranking.mine_hard_negatives(
        samples[:1],
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        retrieval={**settings, "coarse_beta": 0},
        excluded={"train": {"n", "vq", "va", "vn"}},
        **options,
    )
    assert empty["train"] == {"semantic": [], "identity": []}
    assert scored == []


def test_train_checks_native_query_cache_before_preparing_labels(
    tiny_training_run, monkeypatch
):
    cfg, cache, _, _ = tiny_training_run
    cfg["train"].update(candidates=4)
    cfg["train"]["sampling"]["semantic_hard_fraction"] = 0.5

    def validate(rows):
        if rows[0]["sample_id"] == "train":
            raise ValueError("train query cache missing")

    monkeypatch.setattr(cache, "validate_samples", validate)
    monkeypatch.setattr(
        training.TrainSupervision,
        "prepare",
        lambda *a, **kw: pytest.fail("must check cache before label preparation"),
    )
    with pytest.raises(ValueError, match="train query cache missing"):
        training.train(cfg)
    from pathlib import Path

    assert not Path(cfg["output"]["dir"]).exists()


@pytest.mark.parametrize("variant", ["both", "semantic", "rejection"])
def test_train_runs_mining_and_validation_after_warmup(
    tiny_training_run, monkeypatch, variant
):
    cfg, cache, wandb, _ = tiny_training_run
    data = training.load_rcr_data(None)
    data.gt_head_boxes_by_image["n"][0]["identity_id"] = "p2"
    for name in ("s0", "s1", "s2"):
        cache.by_id[name] = len(cache.image_ids)
        cache.image_ids.append(name)
        cache.scenes = torch.cat((cache.scenes, cache.scenes[:1].clone() + 0.1))
        cache.persons = torch.cat((cache.persons, cache.persons[:1].clone()))
        cache.boxes = torch.cat((cache.boxes, cache.boxes[:1].clone()))
        cache.mask = torch.cat((cache.mask, cache.mask[:1].clone()))
        data.images_by_id[name] = {"path": f"train/{name}.jpg"}
        data.gt_head_boxes_by_image[name] = deepcopy(data.gt_head_boxes_by_image["q"])
    cfg["train"].update(epochs=3, context_warmup_epochs=1, candidates=4)
    cfg["train"]["sampling"].update(
        warmup_epochs=1,
        refresh_every_epochs=1,
        semantic_hard_fraction=0.5 if variant != "rejection" else 0,
        hard_fraction=0.34 if variant != "semantic" else 0,
    )
    cfg["model"]["fine_matching"].update(
        selection_threshold=0.0001, identity_threshold=-1
    )
    rejection_weight = 0.2 if variant != "semantic" else 0
    cfg["loss"]["context_rejection_weight"] = rejection_weight
    mine, observed = training.mine_hard_negatives, []

    def observe_mining(*args, **kwargs):
        assert set(kwargs["gallery_ids"]) == {"q", "a", "n", "s0", "s1", "s2"}
        result = mine(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(training, "mine_hard_negatives", observe_mining)
    checkpoint = training.train(cfg)
    history = [
        json.loads(s)
        for s in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert len(observed) == 2
    assert all(
        pools["train"].get("identity", []) == (["n"] if rejection_weight else [])
        for pools in observed
    )
    assert [row["context_rejection_weight"] for row in history] == [
        0,
        rejection_weight,
        rejection_weight,
    ]
    semantic_count, identity_count = (
        int(variant != "rejection"),
        int(rejection_weight > 0),
    )
    assert [row["train_candidate_semantic_hard"] for row in history] == [
        0,
        semantic_count,
        semantic_count,
    ]
    assert [row["train_candidate_identity_hard"] for row in history] == [
        0,
        identity_count,
        identity_count,
    ]
    assert [row["train_context_rejection_active_fraction"] for row in history] == [
        0,
        identity_count,
        identity_count,
    ]
    for row in history[1:]:
        assert row["train_context"] == pytest.approx(
            row["train_context_semantic"]
            + rejection_weight * row["train_context_rejection"]
        )
        assert "val_checkpoint_score" in row
    mined = json.loads((checkpoint.parent / "hard_negatives.json").read_text())
    assert mined["epoch"] == 3 and mined["mining"] == "coarse_then_fine"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["method"] == "proposed"
    assert saved["config"]["loss"]["context_rejection_weight"] == rejection_weight
    assert wandb.finished


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_semantic_sampling_fraction_fails(bad):
    with pytest.raises(ValueError, match="fractions"):
        sampling_settings({"semantic_hard_fraction": bad})


def test_new_settings_default_to_zero_and_rejection_is_not_a_standalone_task():
    assert sampling_settings({})["semantic_hard_fraction"] == 0
    assert loss_settings({})["context_rejection_weight"] == 0
    only_rejection = {key: 0 for key in _only_loss("context") if key.endswith("weight")}
    only_rejection["context_rejection_weight"] = 0.2
    with pytest.raises(ValueError, match="at least one loss"):
        loss_settings(only_rejection)
