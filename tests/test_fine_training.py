"""CPU workflow tests for fine training."""

import json
from copy import deepcopy

import pytest
import torch

from rcr.proposed import ranking
from rcr.proposed import train as training
from rcr.proposed.losses import compute_loss
from rcr.proposed.sampling import CandidateIndex, sample_candidates
from tests.helpers import Tokenizer, _only_loss, _sample


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
    scored_sizes = []
    score_components = ranking.fine_score_components

    def observe_scores(binding, *args, **kwargs):
        scored_sizes.append(len(binding))
        return score_components(binding, *args, **kwargs)

    monkeypatch.setattr(ranking, "fine_score_components", observe_scores)
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
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["method"] == "proposed"
    assert saved["config"]["loss"]["context_rejection_weight"] == rejection_weight
    assert wandb.finished
