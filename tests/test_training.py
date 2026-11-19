"""Training objectives, sampling, memory equivalence and checkpoint recovery."""

import copy
import json
import os
import random
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from rcr.common.io import sha256_file
from rcr.evaluation.runner import evaluate_run
from rcr.proposed import losses, ranking, retrieve
from rcr.proposed import train as training
from rcr.proposed.batch import build_batch
from rcr.proposed.losses import _context_scores, compute_loss, retrieval_loss
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.resume import capture_rng, restore_rng, training_lock
from rcr.proposed.sampling import CandidateIndex, sample_candidates
from rcr.proposed.supervision import TrainSupervision
from tests.helpers import (
    TextBackbone,
    Tokenizer,
    _add_gallery_image,
    _fusion_example,
    _has_gradient,
    _only_loss,
    _sample,
)
from tools.run import main

# Objectives, warmup, validation and checkpoint selection.


@pytest.mark.parametrize("warmup,owner_weight", [(0, 0.0), (1, 0.0), (1, 0.25)])
def test_training_warmup_and_validation_selected_checkpoint(
    tiny_training_run, monkeypatch, warmup, owner_weight
):
    cfg, cache, wandb_run, _ = tiny_training_run
    assert cfg["loss"]["grounding_owner_weight"] == 0
    cfg["loss"]["grounding_owner_weight"] = owner_weight
    cfg["train"].update(epochs=3, context_warmup_epochs=warmup)
    cfg["train"]["identity_sampling"]["max_crops"] = 0
    cfg["retrieval"]["context_weight"] = 0.23
    data = training.load_rcr_data()
    query = data.samples_by_id["train"]
    query["case_type"] = "DUAL"
    query["subjects"].append({"subject_id": 2, "identity_ids": ["p2"]})
    query["final_desc"] = (
        "Identify Subject 1 as the person standing and Subject 2 as the seated person"
    )
    query["final_change"] = "then retrieve images where Subject 1 is beside Subject 2"
    query["final_instruction"] = f"{query['final_desc']}; {query['final_change']}"
    for image_id in ("q", "a", "n"):
        data.gt_head_boxes_by_image[image_id].append(
            {"identity_id": "p2", "x": 1.2, "y": 0.1, "width": 0.2, "height": 0.2}
        )
    weights, native_tokens = [], []
    compute = training.compute_loss

    def observe(model, batch, *args, **settings):
        weights.append(settings)
        native_tokens.append("fafa_targets" in batch)
        assert ("grounding_owner_mask" in batch) == bool(owner_weight)
        if owner_weight:
            assert batch["grounding_owner_mask"].tolist() == [[True]]
        assert batch["grounding_targets"].flatten().tolist() == [1.0, 0.0]
        return compute(model, batch, *args, **settings)

    monkeypatch.setattr(training, "compute_loss", observe)
    checkpoint = training.train(cfg)
    expected = [0.0 if epoch <= warmup else 1.0 for epoch in range(1, 4)]
    assert [row["context_weight"] for row in weights] == expected
    assert native_tokens == [bool(value) for value in expected]
    assert all(row["context_score_weight"] == 0.23 for row in weights)
    assert all(row["coverage_score_weight"] == 0.04 for row in weights)
    assert all(row["grounding_owner_weight"] == owner_weight for row in weights)
    assert [row["context_swap_weight"] for row in weights] == [
        0.25 * value for value in expected
    ]
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in history] == [
        "warmup" if value == 0 else "joint" for value in expected
    ]
    assert all("train_context_fda_error" in row for row in history)
    assert all(
        (row["train_grounding_owner"] > 0) == bool(owner_weight) for row in history
    )
    assert all("val_checkpoint_score" not in row for row in history[:warmup])
    best = max(
        (row for row in history if "val_checkpoint_score" in row),
        key=lambda row: row["val_checkpoint_score"],
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    last = torch.load(
        checkpoint.parent / "last.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint.name == "best.pt"
    assert saved["epoch"] == saved["best_epoch"] == best["epoch"]
    assert last["epoch"] == 3
    assert saved["config"]["retrieval"]["fine_score_mode"] == "linear"
    assert saved["config"]["retrieval"]["context_weight"] == 0.23
    assert saved["config"]["loss"]["grounding_owner_weight"] == owner_weight
    assert saved["scene_cache_id"] == cache.scene_cache_id
    assert saved["detection_cache_id"] == cache.detection_cache_id
    assert saved["state_supervised_pairs"] > 0
    assert set(saved["text_encoder"]) == {"projection", "text_policy"}
    assert "optimizer" not in saved and "scaler" not in saved
    assert "training_state" not in saved
    assert last["training_state"]["progress"]["epoch"] == 3
    assert last["training_state"]["progress"]["phase"] == "epoch_end"
    cache_path = Path(cfg["cache"]["supervision_cache"])
    supplemental = json.loads(cache_path.read_text())["supplemental_identities"]
    assert set(supplemental) == ({"q"} if owner_weight else set())
    assert any(
        row.get("train/grounding_owner", 0) > 0 for row in wandb_run.logs
    ) == bool(owner_weight)
    assert wandb_run.finished


def test_grounding_owner_averages_only_supervised_queries_and_crops():
    logits = torch.zeros(3, 3, 4)
    logits[:, 2] = torch.nan
    logits[0, 1, 0] = 2.0
    logits[0, :2, 1:] = torch.nan
    logits.requires_grad_()
    targets = torch.tensor(
        [
            [[1, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
    )
    crops = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True] * 4]
    )
    subjects = torch.tensor([[True, True, False]] * 2 + [[True, False, False]])
    loss = losses.grounding_owner_loss(logits, targets, crops, subjects)
    expected = (torch.logaddexp(torch.tensor(2.0), torch.tensor(0.0)) + np.log(2)) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 0, 0] < 0 < logits.grad[0, 1, 0]
    assert logits.grad[0, :, 1:].count_nonzero() == 0
    assert logits.grad[:, 2].count_nonzero() == 0
    assert logits.grad[2].count_nonzero() == 0


@pytest.mark.parametrize(
    "component", ["grounding", "identity", "coarse", "context", "state"]
)
def test_each_objective_trains_its_intended_modules(objective_batch, component):
    model, batch = objective_batch
    total, parts = training.compute_loss(model, batch, (2, 2), **_only_loss(component))
    assert torch.isfinite(total)
    total.backward()
    assert _has_gradient(model.identity_head) == (component in ("identity", "coarse"))
    if component not in ("identity", "coarse"):
        assert all(p.grad is None for p in model.identity_head.parameters())
    assert _has_gradient(model.reasoner) == (component == "context")
    assert _has_gradient(model.composition) == (component == "context")
    assert _has_gradient(model.state_text_proj) == (component == "state")
    assert _has_gradient(model.state_image_proj) == (component == "state")
    assert model.binding_log_scale.grad is None
    assert not model.binding_log_scale.requires_grad
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if component == "context":
        assert parts["context_active_fraction"].item() == 1
        assert batch["target_persons"].grad[0, 2].count_nonzero() == 0
        assert batch["target_persons"].grad[0, :2].abs().sum() > 0
    if component == "state":
        assert batch["target_global"].grad[0, 2].count_nonzero() == 0
        assert batch["target_global"].grad[0, :2].abs().sum() > 0


def test_joint_objective_learns_on_a_fixed_supervised_batch(objective_batch):
    model, batch = objective_batch
    settings = training.loss_settings({})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        value, parts = training.compute_loss(model, batch, (2, 2), **settings)
        assert torch.isfinite(value)
        assert parts["context_active_fraction"].item() == 1
        losses.append(value.item())
        value.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.binding_scale.item() == 1.0


def test_verified_swap_trains_context_on_the_same_assignment(
    tiny_training_run, monkeypatch
):
    _, cache, _, _ = tiny_training_run
    data = training.load_rcr_data()
    sample = data.samples_by_id["train"]
    sample["subjects"].append({"subject_id": 2, "identity_ids": ["p2"]})
    sample["final_desc"] = (
        "Identify Subject 1 as the person on the left and Subject 2 as the other person"
    )
    sample["final_change"] = (
        "then retrieve images where Subject 1 is to the left of Subject 2"
    )
    sample["positive_image_ids"] = ["a", "n"]
    cache.boxes = torch.tensor([[0.0, 0.0, 0.45, 1.0], [0.55, 0, 1.0, 1.0]]).expand(
        6, 2, 4
    )
    cache.mask = torch.ones(6, 2, dtype=torch.bool)
    cache.persons = cache.persons[:2, 0].expand(6, -1, -1)
    for name, order in (("q", ("p1", "p2")), ("a", ("p1", "p2")), ("n", ("p2", "p1"))):
        data.gt_head_boxes_by_image[name] = [
            {"identity_id": identity, "x": x, "y": 0.25, "width": 0.1, "height": 0.1}
            for identity, x in zip(order, (0.15, 0.7), strict=True)
        ]
    supervision = TrainSupervision(data, cache)
    supervision.prepare([sample], include_direction_swaps=True)
    assert supervision.direction_swaps["train"] == (
        sample["final_change"].replace("left", "right"),
        {"a"},
    )

    class DirectionTokenizer(Tokenizer):
        def __call__(self, texts, **kwargs):
            tokens = super().__call__(texts, **kwargs)
            for text, ids, offsets in zip(
                texts, tokens["input_ids"], tokens["offset_mapping"], strict=True
            ):
                for i, (start, end) in enumerate(offsets):
                    if text[start:end] == "right":
                        ids[i] = 3
            return tokens

    text_encoder = TextEncoder(TextBackbone(), dim=8)
    batch = build_batch(
        [sample],
        [["n", "a"]],
        cache,
        DirectionTokenizer(),
        text_encoder,
        "cpu",
        supervision=supervision,
        identity_image_ids=[{"a", "n"}],
        fafa_pooling="subject_mean",
        include_direction_swaps=True,
    )
    assert batch["direction_swap"]["positions"].tolist() == [[0, 1]]
    model = training.RCRModel(dim=8, identity_dim=6, num_heads=2)
    monkeypatch.setattr(
        model.grounding,
        "forward",
        lambda scene, *args: scene.new_tensor([[[5, -5], [-5, 5]]]).expand(
            len(scene), -1, -1
        ),
    )
    seen = []
    score = model.score_aligned_context

    def observe(query, scene, persons, boxes, indices, patch_hw):
        result = score(query, scene, persons, boxes, indices, patch_hw)
        seen.append((indices.detach().clone(), result.detach()))
        return result

    monkeypatch.setattr(model, "score_aligned_context", observe)
    settings = {**_only_loss("context"), "context_swap_weight": 0.25}
    loss, parts = compute_loss(model, batch, (2, 2), context_batch_size=1, **settings)
    assert parts["context_swap_active_fraction"] == 1
    assert len(seen) == 3
    torch.testing.assert_close(seen[1][0], seen[2][0])
    torch.testing.assert_close(
        parts["context_swap"],
        torch.nn.functional.softplus(seen[2][1] - seen[1][1]).mean(),
    )
    loss.backward()
    assert _has_gradient(model.reasoner.pair_score)
    assert _has_gradient(text_encoder.proj)
    assert not _has_gradient(model.identity_head)

    batch["direction_swap"]["target_roles"] = batch["direction_swap"][
        "target_roles"
    ].flip(1)
    _, parts = compute_loss(model, batch, (2, 2), **settings)
    assert parts["context_swap"] == parts["context_swap_active_fraction"] == 0


def test_training_enables_identity_crop_mining_after_warmup(
    tiny_training_run, monkeypatch
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"].update(epochs=2)
    cfg["retrieval"]["coarse_beta"] = 0  # This fixture has no same-ID state pair.
    cfg["train"]["sampling"].update(warmup_epochs=1, hard_fraction=0)
    cfg["train"]["identity_sampling"].update(max_crops=4, hard_negative_fraction=0.5)
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
    assert mining == [False, True]
    assert backgrounds == [-3.5, -3.5]
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["model"]["fine_matching"]["background_logit"] == -3.5
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["train_identity_hard_crops"] for row in history] == [
        0,
        1,
    ]
    for row in history:
        assert row["train_identity_supplemental_crops"] <= 4
        assert row["train_identity_supplemental_crops"] == (
            row["train_identity_random_crops"] + row["train_identity_hard_crops"]
        )


def test_training_validation_matches_run_with_the_same_parameters(
    tiny_training_run, monkeypatch
):
    cfg, cache, _, _ = tiny_training_run
    cfg["candidate_ks"] = [1]
    checkpoint = training.train(cfg)
    data = training.load_rcr_data(None)
    monkeypatch.setattr(retrieve, "load_rcr_data", lambda *args: data)
    monkeypatch.setattr(retrieve, "GalleryCache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(retrieve, "check_query_text_config", lambda *args: None)
    monkeypatch.setattr(retrieve, "check_cache_config", lambda *args: None)
    run = yaml.safe_load((checkpoint.parent / "selected.yaml").read_text())
    retrieve.retrieve(run)
    result = evaluate_run(run, data=data)
    trained = json.loads(
        (checkpoint.parent / "evaluation/epoch_001/val_metrics.json").read_text()
    )
    assert f"candidate_recall_{cfg['retrieval']['top_m']}" in result["overall"]
    for key, value in result["overall"].items():
        assert trained["overall"][key] == pytest.approx(value, abs=1e-12), key


def test_training_selects_checkpoint_after_calibration_without_feedback(
    tiny_training_run, monkeypatch
):
    from rcr.evaluation.report import load_run

    cfg, cache, wandb_run, _ = tiny_training_run
    cfg["train"].update(epochs=2)
    cfg["retrieval"]["top_m"] = 3
    data = training.load_rcr_data(None)
    _add_gallery_image(cache, data, "vx", "val")
    retrieve_rankings = training.retrieve_rankings
    calls = []

    def controlled(*args, **kwargs):
        calls.append(kwargs["fine_coarse_weight"])
        source = retrieve_rankings(*args, **kwargs)
        context = [5.6, 6.2, 1.6] if len(calls) == 1 else [6.2, 5.6, 1.6]
        return _fusion_example(source, context)

    monkeypatch.setattr(training, "retrieve_rankings", controlled)
    checkpoint = training.train(cfg)
    best = torch.load(checkpoint, weights_only=True)
    last = torch.load(checkpoint.parent / "last.pt", weights_only=True)
    selected = yaml.safe_load((checkpoint.parent / "selected.yaml").read_text())
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert calls == [0.15, 0.15]
    assert [row["val_calibration_reference_score"] for row in history] == [0.5, 1.0]
    assert [row["val_checkpoint_score"] for row in history] == [1.0, 1.0]
    assert [row["val_fine_coarse_weight"] for row in history] == [0.25, 0.15]
    assert best["epoch"] == best["best_epoch"] == 1
    assert last["epoch"] == 2 and last["best_epoch"] == 1
    assert best["validation"]["fine_coarse_weight"] == 0.25
    assert last["validation"]["fine_coarse_weight"] == 0.15
    assert best["best_fine_coarse_weight"] == last["best_fine_coarse_weight"] == 0.25
    assert best["config"]["retrieval"] == cfg["retrieval"]
    assert selected["retrieval"]["fine_coarse_weight"] == 0.25
    assert selected["selected_retrieval"]["fine_coarse_weight"] == 0.25
    assert selected["selected_checkpoint_sha256"] == sha256_file(checkpoint)
    assert wandb_run.summary["best_fine_coarse_weight"] == 0.25
    assert wandb_run.summary["best_val_checkpoint_score"] == 1.0
    for epoch in (1, 2):
        directory = checkpoint.parent / "evaluation" / f"epoch_{epoch:03d}"
        assert (directory / "summary.csv").is_file()
        assert (directory / "selection.json").is_file()
    reported = load_run("proposed", checkpoint.parent / "selected/val", "val")
    assert reported["metrics"]["overall"]["full_map"] == 1.0
    metadata = json.loads((checkpoint.parent / "selected/val/run.json").read_text())
    assert metadata["checkpoint_epoch"] == metadata["selection"]["epoch"] == 1
    assert metadata["fine_coarse_weight"] == 0.25


@pytest.mark.parametrize("weights", [[], [0.1, 0.1], [-0.1], [float("nan")], [True]])
@pytest.mark.parametrize(
    "key", ["coverage_weights", "context_weights", "fine_coarse_weights"]
)
def test_training_rejects_invalid_calibration_before_reading_data(
    tiny_training_run, monkeypatch, weights, key
):
    cfg, _, _, _ = tiny_training_run
    cfg["evaluation"]["calibration"][key] = weights
    monkeypatch.setattr(
        training, "load_rcr_data", lambda *args: pytest.fail("read training data")
    )
    with pytest.raises(ValueError, match=key):
        training.train(cfg)


# Sampling, mining and fine-score supervision.


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
        semantic_hard_fraction=0.3,
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
    assert first.semantic_hard_count == 5
    assert len(row & set(semantic[:10])) >= 5
    assert first.identity_hard_count == 3
    assert first.hard_identity_ids[0] <= row & set(wrong[:6])
    assert len(first.hard_identity_ids[0]) == 3
    assert not row & {"q", "disputed"}


def test_rejection_adds_wrong_id_gradient_without_changing_other_targets(
    objective_batch,
):
    model, batch = objective_batch
    batch["hard_identity_negative_mask"][0, 2] = True
    settings = _only_loss("context")
    semantic_only, semantic_parts = compute_loss(model, batch, (2, 2), **settings)
    semantic_grad = torch.autograd.grad(semantic_only, batch["target_persons"])[0]
    total, parts = compute_loss(
        model, batch, (2, 2), **{**settings, "context_rejection_weight": 0.2}
    )
    total.backward()
    torch.testing.assert_close(parts["context_semantic"], semantic_parts["context"])
    torch.testing.assert_close(
        total.detach(), semantic_only.detach() + 0.2 * parts["context_rejection"]
    )
    assert semantic_grad[0, 2].count_nonzero() == 0
    assert batch["target_persons"].grad[0, 2].abs().sum() > 0
    assert parts["context_rejection_active_fraction"] == 1
    assert all(p.grad is None for p in model.identity_head.parameters())
    assert batch["identity_positive_mask"].tolist() == [[True, True, False]]


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
        assert kwargs["context_weight"] == 0
        ids = scene[:, 0, 0].long()
        scored.extend(ids.tolist())
        values = torch.tensor([9, 9, 0.1, 0.7, 0.8, 0.3])[ids]
        return {
            "context": context_sign
            * torch.tensor([0, 0, -3000, 5000, 20000, 1000])[ids],
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
    scored_sizes = []
    score_components = ranking.fine_score_components

    def observe_scores(binding, context, coverage, *args, **kwargs):
        scored_sizes.append(len(binding))
        result = score_components(binding, context, coverage, *args, **kwargs)
        torch.testing.assert_close(
            result["fine_scores"], binding + 0.04 * coverage, rtol=0, atol=0
        )
        assert result["context_contribution"].count_nonzero() == 0
        return result

    monkeypatch.setattr(ranking, "fine_score_components", observe_scores)
    for batch_size in (1, 4):
        context_sign = 1.0 if batch_size == 1 else -1.0
        results.append(
            ranking.mine_hard_negatives(
                samples,
                cache,
                Tokenizer(),
                text,
                model,
                torch.device("cpu"),
                retrieval={
                    **settings,
                    "coarse_beta": 0,
                    "fine_batch_size": batch_size,
                    "context_weight": 0.1 if batch_size == 1 else 10000.0,
                },
                query_batch_size=1 if batch_size == 1 else 2,
                **options,
            )
        )
    assert results[0] == results[1]
    assert scored_sizes == [4, 4, 4, 4]
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
    assert mined["retrieval"]["context_weight"] == 0
    assert mined["retrieval"]["coverage_weight"] == cfg["retrieval"]["coverage_weight"]
    assert (
        mined["retrieval"]["fine_coarse_weight"]
        == cfg["retrieval"]["fine_coarse_weight"]
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["method"] == "proposed"
    assert saved["config"]["loss"]["context_rejection_weight"] == rejection_weight
    assert saved["config"]["retrieval"]["context_weight"] == 0.1
    assert wandb.finished


def _encode(model, batch):
    return model.encode_query(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        batch["change"],
        batch["subject_pos"],
        (2, 2),
        selection_mask=batch.get("selection_mask"),
        change_mask=batch.get("change_mask"),
        query_person_mask=batch.get("query_person_mask"),
        subject_mask=batch.get("subject_mask"),
        subject_token_mask=batch.get("subject_token_mask"),
        subject_ids=batch.get("subject_ids"),
        fafa_query=batch["fafa_query"],
    )


@pytest.mark.parametrize("weight,temperature", [(0.1, 0.1), (0.2, 0.15), (0.0, 0.1)])
def test_context_loss_uses_scaled_context_and_full_score_for_rejection(
    objective_batch, weight, temperature
):
    model, batch = objective_batch
    batch["fafa_targets"][:, 0].neg_()
    batch["hard_identity_negative_mask"][0, 2] = True
    query = _encode(model, batch)
    raw_context, coverage, binding, _ = _context_scores(model, query, batch, (2, 2), 64)
    details = model.score_target(
        query.repeat_candidates(3),
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        (2, 2),
        batch["target_mask"].flatten(0, 1),
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
        pooling=batch["fafa_pooling"],
        context_weight=weight,
        coverage_weight=0.07,
        return_details=True,
    )
    torch.testing.assert_close(
        raw_context.flatten(), details["context"], rtol=0, atol=0
    )
    full_score = model.binding_scale * binding + 0.07 * coverage + weight * raw_context
    torch.testing.assert_close(full_score.flatten(), details["score"], rtol=0, atol=0)
    settings = {
        **_only_loss("context"),
        "context_temperature": temperature,
        "context_rejection_weight": 0.5,
    }
    total, parts = training.compute_loss(
        model,
        batch,
        (2, 2),
        **settings,
        context_score_weight=weight,
        coverage_score_weight=0.07,
    )
    valid = batch["identity_positive_mask"] & (coverage > 0)
    semantic = retrieval_loss(
        weight * raw_context / temperature, batch["positive_mask"], valid
    )
    rejection_valid = (
        batch["positive_mask"] | batch["hard_identity_negative_mask"]
    ) & (coverage > 0)
    rejection = retrieval_loss(
        full_score / temperature, batch["positive_mask"], rejection_valid
    )
    torch.testing.assert_close(parts["context_semantic"], semantic)
    torch.testing.assert_close(parts["context_rejection"], rejection)
    torch.testing.assert_close(total, semantic + 0.5 * rejection)
    assert not binding.requires_grad
    assert parts["context_fda_error"].item() == 1


def test_model_forward_uses_the_requested_linear_coefficients(objective_batch):
    model, batch = objective_batch
    query = _encode(model, batch)
    weight = 0.27
    target = {
        "target_scene": batch["target_scene"][:, 0],
        "target_persons": batch["target_persons"][:, 0],
        "target_boxes": batch["target_boxes"][:, 0],
        "fafa_targets": batch["fafa_targets"][:, 0],
    }
    _, actual = model(
        **{
            key: batch[key]
            for key in (
                "query_scene",
                "query_persons",
                "query_boxes",
                "selections",
                "change",
                "subject_pos",
                "selection_mask",
                "change_mask",
                "query_person_mask",
                "subject_mask",
                "subject_token_mask",
                "subject_ids",
                "fafa_query",
            )
            if key in batch
        },
        **target,
        target_mask=batch["target_mask"][:, 0],
        patch_hw=(2, 2),
        pooling=batch["fafa_pooling"],
        context_weight=weight,
        coverage_weight=0.07,
    )
    expected = model.score_target(
        query,
        target["target_scene"],
        target["target_persons"],
        target["target_boxes"],
        (2, 2),
        batch["target_mask"][:, 0],
        fafa_targets=target["fafa_targets"],
        pooling=batch["fafa_pooling"],
        context_weight=weight,
        coverage_weight=0.07,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# Chunked losses, gradients and activation checkpointing.


def _expanded_batch(batch):
    rows = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            rows[key] = value
            continue
        if value.ndim and value.shape[0] == 1:
            value = value.repeat(3, *([1] * (value.ndim - 1)))
        rows[key] = value.detach().clone().requires_grad_(value.requires_grad)
    rows["query_scene"][1] += 0.3
    rows["target_mask"][1, 1] = False
    rows["hard_identity_negative_mask"][:, 2] = True
    rows["change"].requires_grad_()
    rows["selections"].requires_grad_()
    return rows


def _backward(model, batch, size, settings, *, amp=False):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        total, parts = losses.compute_loss(
            model, batch, (2, 2), context_batch_size=size, **settings
        )
    total.backward()
    gradients = {
        key: None if p.grad is None else p.grad.clone()
        for key, p in model.named_parameters()
    }
    gradients.update(
        (f"batch.{key}", value.grad.clone())
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
        and value.requires_grad
        and value.grad is not None
    )
    return total.detach(), parts, gradients


def _assert_same(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    assert actual[1].keys() == expected[1].keys()
    for key in actual[1]:
        torch.testing.assert_close(
            actual[1][key], expected[1][key], rtol=2e-5, atol=2e-6
        )
    assert actual[2].keys() == expected[2].keys()
    for key, gradient in actual[2].items():
        other = expected[2][key]
        if gradient is None:
            assert other is None, key
        else:
            torch.testing.assert_close(gradient, other, rtol=3e-4, atol=2e-6, msg=key)


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_chunked_context_matches_full_loss_and_all_gradients(
    objective_batch, chunk_size
):
    model, batch = objective_batch
    settings = losses.loss_settings({"context_rejection_weight": 0.2})
    expected = _backward(deepcopy(model), _expanded_batch(batch), 64, settings)
    actual = _backward(deepcopy(model), _expanded_batch(batch), chunk_size, settings)
    _assert_same(actual, expected)
    assert actual[2]["batch.change"].abs().sum() > 0
    assert actual[2]["batch.selections"].abs().sum() > 0


@pytest.mark.parametrize("amp", [False, True])
def test_checkpoint_preserves_chunked_dropout_gradients_and_rng(
    objective_batch, monkeypatch, amp
):
    model, batch = objective_batch
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.1
    settings = _only_loss("context")
    torch.manual_seed(100)
    actual = _backward(deepcopy(model), _expanded_batch(batch), 4, settings, amp=amp)
    actual_rng = torch.rand(5)
    monkeypatch.setattr(
        losses, "checkpoint", lambda function, *args, **kwargs: function(*args)
    )
    torch.manual_seed(100)
    expected = _backward(deepcopy(model), _expanded_batch(batch), 4, settings, amp=amp)
    _assert_same(actual, expected)
    torch.testing.assert_close(actual_rng, torch.rand(5), rtol=0, atol=0)


# Checkpoint resume, RNG state and process recovery.


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _equal(left, right)
    else:
        assert actual == expected


class _Scaler:
    def __init__(self, *args, **kwargs):
        self.updates = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self.updates += 1

    def state_dict(self):
        return {"updates": self.updates, "scale": 65536.0}

    def load_state_dict(self, state):
        self.updates = state["updates"]


def _pause(cfg, monkeypatch):
    compute = training.compute_loss

    def stop(*args, **kwargs):
        result = compute(*args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(training, "compute_loss", stop)
        training.train(cfg)


def _configure(cfg, data):
    cfg["train"].update(epochs=4, context_warmup_epochs=1)
    cfg["train"]["sampling"].update(
        warmup_epochs=1,
        refresh_every_epochs=2,
        identity_fraction=1.0,
        hard_fraction=0.0,
        semantic_hard_fraction=0.5,
    )
    cfg["evaluation"]["calibration"] = {
        "coverage_weights": [0.05],
        "context_weights": [0.025],
        "fine_coarse_weights": [0.15],
    }
    cfg["checkpointing"] = {"every_steps": 2}
    cfg["wandb"]["log_every_steps"] = 1
    for key in ("train2", "train3"):
        row = _sample(key, "q", "a")
        data.samples.append(row)
        data.samples_by_id[key] = row
        data.splits["train"].append(key)


@pytest.mark.parametrize("fault", ["signal", "after_step", "validation", "selection"])
def test_resume_matches_uninterrupted_training(tiny_training_run, monkeypatch, fault):
    cfg, _, wandb_run, _ = tiny_training_run
    data = training.load_rcr_data(None)
    _configure(cfg, data)
    monkeypatch.setattr(torch.amp, "GradScaler", _Scaler, raising=False)
    compute, mine, retrieve = (
        training.compute_loss,
        training.mine_hard_negatives,
        training.retrieve_rankings,
    )
    observations = []
    mining = []
    retrievals = []

    def observe(*args, **kwargs):
        observations.append((random.random(), float(np.random.random())))
        return compute(*args, **kwargs)

    def observe_mining(*args, **kwargs):
        mining.append(True)
        return mine(*args, **kwargs)

    def observe_retrieval(*args, **kwargs):
        retrievals.append(True)
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(training, "compute_loss", observe)
    monkeypatch.setattr(training, "mine_hard_negatives", observe_mining)
    monkeypatch.setattr(training, "retrieve_rankings", observe_retrieval)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["output"]["dir"] += "-reference"
    reference = training.train(reference_cfg)
    reference_last = torch.load(reference.parent / "last.pt", weights_only=True)
    reference_best = torch.load(reference, weights_only=True)
    reference_history = (reference.parent / "history.jsonl").read_text()
    reference_observations = list(observations)
    reference_mining = len(mining)
    reference_retrievals = len(retrievals)
    observations.clear()
    mining.clear()
    retrievals.clear()
    original_log = wandb_run.log
    previous_handler = signal.getsignal(signal.SIGTERM)

    with monkeypatch.context() as interrupted:
        if fault == "signal":

            def stop(*args, **kwargs):
                result = observe(*args, **kwargs)
                if len(observations) == 5:
                    assert signal.getsignal(signal.SIGTERM) != previous_handler
                    signal.raise_signal(signal.SIGTERM)
                return result

            interrupted.setattr(training, "compute_loss", stop)
            training.train(cfg)
        else:
            if fault == "after_step":

                def fail_log(values):
                    if values.get("global_step") == 5:
                        raise RuntimeError("injected interruption")
                    original_log(values)

                interrupted.setattr(wandb_run, "log", fail_log)
            elif fault == "validation":

                def fail_validation(*args, **kwargs):
                    raise RuntimeError("injected interruption")

                interrupted.setattr(training, "_evaluate", fail_validation)
            else:
                save_selection = training.save_selection

                def fail_selection(*args, **kwargs):
                    save_selection(*args, **kwargs)
                    raise RuntimeError("injected interruption")

                interrupted.setattr(training, "save_selection", fail_selection)
            with pytest.raises(RuntimeError, match="injected interruption"):
                training.train(cfg)
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    output = Path(cfg["output"]["dir"])
    pending = torch.load(output / "last.pt", weights_only=True)
    state = pending["training_state"]
    saved_step = state["progress"]["global_step"]
    assert saved_step == (5 if fault == "signal" else 4 if fault == "after_step" else 6)
    if fault in ("validation", "selection"):
        assert state["progress"]["phase"] == "validation"
    observations[:] = observations[:saved_step]
    resumed = training.train(cfg, resume="auto")
    last = torch.load(output / "last.pt", weights_only=True)
    best = torch.load(resumed, weights_only=True)
    for key in (
        "model",
        "text_encoder",
        "best_epoch",
        "best_score",
        "state_supervised_pairs",
    ):
        _equal(last[key], reference_last[key])
        _equal(best[key], reference_best[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(last["training_state"][key], reference_last["training_state"][key])
    assert last["training_state"]["scaler"]["updates"] == 12
    assert (output / "history.jsonl").read_text() == reference_history
    assert observations == reference_observations
    assert len(mining) == reference_mining
    assert len(retrievals) == reference_retrievals
    assert not list((output / "evaluation").rglob("pending_validation.pt"))
    assert (
        last["training_state"]["progress"]["wandb_id"] == state["progress"]["wandb_id"]
    )
    assert json.loads((output / "training_status.json").read_text())["complete"]
    digest = sha256_file(output / "last.pt")
    assert training.train(cfg, resume="auto") == resumed
    assert sha256_file(output / "last.pt") == digest


def test_legacy_checkpoint_rejected_without_overwriting(tiny_training_run):
    cfg, _, _, _ = tiny_training_run
    output = Path(cfg["output"]["dir"])
    output.mkdir()
    path = output / "last.pt"
    torch.save({"epoch": 3, "model": {"weight": torch.ones(1)}}, path)
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="no complete optimizer/scaler/RNG"):
        training.train(cfg, resume="auto")
    assert sha256_file(path) == digest
    assert list(output.iterdir()) == [path]


def test_rng_state_roundtrip_uses_safe_checkpoint_values(tmp_path):
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(4))
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    restore_rng(torch.load(path, weights_only=True))
    actual = (random.random(), np.random.random(), torch.rand(4))
    _equal(actual, expected)


@pytest.mark.parametrize("change", ["config", "inputs", "source_sha256", "runtime"])
def test_incompatible_resume_leaves_run_untouched(
    tiny_training_run, monkeypatch, change
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"]["epochs"] = 2
    _pause(cfg, monkeypatch)
    output = Path(cfg["output"]["dir"])
    paths = [
        output / name
        for name in ("last.pt", "history.jsonl", "config.yaml", "training_status.json")
    ]
    digests = [sha256_file(path) for path in paths]
    signature = training.resume_signature

    def changed(*args):
        current = signature(*args)
        current[change] = "different"
        return current

    monkeypatch.setattr(training, "resume_signature", changed)
    with pytest.raises(ValueError, match=change):
        training.train(cfg, resume="auto")
    assert [sha256_file(path) for path in paths] == digests


def test_resume_accepts_changed_checkpoint_cadence(tiny_training_run, monkeypatch):
    cfg, _, _, _ = tiny_training_run
    cfg["train"]["epochs"] = 2
    _pause(cfg, monkeypatch)
    cfg.setdefault("checkpointing", {})["every_steps"] = 1
    checkpoint = training.train(cfg, resume="auto")
    assert checkpoint.name == "best.pt"
    last = torch.load(checkpoint.parent / "last.pt", weights_only=True)
    assert last["epoch"] == 2
    assert last["config"]["checkpointing"]["every_steps"] == 1


def test_training_lock_rejects_another_writer_and_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="another training process"):
        with training_lock(tmp_path):
            with training_lock(tmp_path):
                pytest.fail("acquired an active training lock")
    with training_lock(tmp_path):
        pass


def test_resume_survives_a_new_process_and_hash_seed(tmp_path):
    from rcr.common.config import load_config

    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.json"
    base.write_text(json.dumps(load_config(root / "configs/proposed.yaml")))
    worker = """
import copy
import json
import signal
import sys
from pathlib import Path
from pytest import MonkeyPatch
from tests import conftest
from tests.test_training import _configure
from rcr.proposed import train as training
from rcr.proposed.batch import build_batch

phase, directory, base = sys.argv[1:]
directory = Path(directory)
directory.mkdir(parents=True, exist_ok=True)
base = json.loads(Path(base).read_text())
with MonkeyPatch.context() as patch:
    patch.setattr(conftest, '_workflow_config', lambda: copy.deepcopy(base))
    cfg, _, _, _ = conftest.tiny_training_run.__wrapped__(directory, patch)
    _configure(cfg, training.load_rcr_data(None))
    cfg['wandb']['enabled'] = False
    if phase == 'pause':
        compute = training.compute_loss
        count = [0]
        def stop(*args, **kwargs):
            loss = compute(*args, **kwargs)
            count[0] += 1
            if count[0] == 5:
                signal.raise_signal(signal.SIGTERM)
            return loss
        patch.setattr(training, 'compute_loss', stop)
    training.train(cfg, resume='auto' if phase == 'resume' else None)
"""
    for phase, name, seed in (
        ("full", "reference", "11"),
        ("pause", "split", "23"),
        ("resume", "split", "37"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                worker,
                phase,
                str(tmp_path / name),
                str(base),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
                "PYTHONHASHSEED": seed,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        (tmp_path / f"{phase}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr[-4000:]
    reference = torch.load(tmp_path / "reference/run/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "split/run/last.pt", weights_only=True)
    for key in ("model", "text_encoder", "best_epoch", "best_score"):
        _equal(actual[key], reference[key])
    for key in ("optimizer", "scaler", "rng"):
        _equal(actual["training_state"][key], reference["training_state"][key])
    assert (tmp_path / "split/run/history.jsonl").read_text() == (
        tmp_path / "reference/run/history.jsonl"
    ).read_text()


@pytest.mark.parametrize(
    "resume", [[], ["--resume"], ["--resume", "runs/trial/last.pt"]]
)
def test_cli_dispatches_explicit_resume(monkeypatch, resume):
    calls = []
    monkeypatch.setattr(training, "train", lambda cfg, **kwargs: calls.append(kwargs))
    main(["train", "--config", "configs/proposed.yaml", *resume])
    assert calls == [
        {"resume": None if not resume else "auto" if len(resume) == 1 else resume[1]}
    ]
