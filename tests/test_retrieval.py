"""Retrieval scores, pooling, calibration, replay and frozen selection."""

import copy
import json
from itertools import product
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.config import load_config
from rcr.common.data import split_fingerprint
from rcr.common.io import save_results, sha256_file
from rcr.evaluation.evaluate import evaluate_retrieval_output, selection_metrics
from rcr.evaluation.runner import evaluate_run
from rcr.proposed import calibration, ranking, retrieve
from rcr.proposed import train as training
from rcr.proposed.contracts import CALIBRATED_RETRIEVAL_KEYS
from rcr.proposed.matching import PersonMatching
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.nn.reasoning import FineReasoner
from rcr.proposed.ranking import SCORE_SCHEMA_VERSION
from rcr.proposed.replay import replay_rankings
from rcr.proposed.scores import POOLING_MODES, pool_fafa_scores
from tests.helpers import _add_gallery_image, _rank, _sample

# Fine scores, shortlist ranking and replay.


def test_fine_score_matches_independent_reference():
    binding = torch.tensor([0.5, -0.4, 0.1, 0.8], dtype=torch.float64)
    context = torch.tensor([7.0, 2.0, -4.0, 5.0], dtype=torch.float64)
    coverage = torch.tensor([0.1, 1.0, 0.0, 0.8], dtype=torch.float64)
    scale, weight = 1.7, 0.03
    result = ranking.fine_score_components(
        binding, context, coverage, scale, context_weight=weight
    )
    expected = [
        scale * float(b) + 0.04 * float(c) + weight * float(v)
        for b, v, c in zip(binding, context, coverage, strict=True)
    ]
    torch.testing.assert_close(result["fine_scores"], torch.tensor(expected))
    assert all(value.dtype == torch.float32 for value in result.values())


def test_pair_score_has_unsaturated_local_gradients_and_coverage_bonus():
    raw = torch.tensor([-100.0, -4.42, 0.0, 4.0, 100.0], requires_grad=True)
    coverage = torch.linspace(0, 1, len(raw), requires_grad=True)
    full = ranking.fine_score_components(torch.zeros_like(raw), raw, coverage, 1.0)[
        "fine_scores"
    ]
    single = ranking.fine_score_components(
        torch.zeros_like(raw[:1]), raw[:1], coverage[:1], 1.0
    )["fine_scores"]
    torch.testing.assert_close(full[:1], single, rtol=0, atol=0)
    raw_grad, coverage_grad = torch.autograd.grad(full.sum(), (raw, coverage))
    torch.testing.assert_close(raw_grad, torch.full_like(raw, 0.1), rtol=0, atol=0)
    torch.testing.assert_close(
        coverage_grad, torch.full_like(raw, 0.04), rtol=0, atol=0
    )


def test_fusion_matches_fp64_reference_and_preserves_ties_and_tail():
    fine = torch.tensor([0.6, 0.6001, 0.2, 0.3])
    coarse = torch.tensor([1.0, 2.0, 0.0, -torch.inf])
    actual = ranking.fuse_fine_coarse_scores(fine, coarse, 0.15)
    valid = coarse.isfinite()
    f, c = fine[valid].double(), coarse[valid].double()
    expected = (f - f.mean()) / f.std(unbiased=False)
    expected += 0.15 * (c - c.mean()) / c.std(unbiased=False)
    torch.testing.assert_close(actual[valid].double(), expected, atol=2e-6, rtol=2e-6)
    assert actual[-1].isneginf()
    assert torch.equal(actual.argsort(descending=True), torch.tensor([1, 0, 2, 3]))
    assert torch.equal(ranking.fuse_fine_coarse_scores(fine, coarse, 0), fine)
    assert (
        ranking.fuse_fine_coarse_scores(fine, torch.full_like(coarse, -torch.inf), 0.15)
        .isneginf()
        .all()
    )
    constant = torch.full((3,), 0.1)
    tied = ranking.fuse_fine_coarse_scores(constant, constant, 0.15)
    assert torch.equal(tied, torch.zeros(3))
    order = torch.tensor([3, 1, 2, 4, 0])
    assert torch.equal(ranking.rerank_topm(order, order[:3], tied), order)
    reordered = ranking.rerank_topm(order, order[:3], torch.tensor([0.0, 2.0, 1.0]))
    assert reordered.tolist() == [1, 2, 3, 4, 0]
    for bad in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            ranking.fuse_fine_coarse_scores(fine, coarse, bad)


def test_complete_topm_is_independent_of_inference_chunks(tiny_ranker):
    first = _rank(
        tiny_ranker,
        fine_batch_size=1,
        identity_batch_size=1,
        coarse_batch_size=1,
    )
    second = _rank(
        tiny_ranker,
        fine_batch_size=5,
        identity_batch_size=6,
        coarse_batch_size=6,
    )
    for key in ("coarse_rankings", "fine_rankings", "rankings", "fafa_rankings"):
        assert torch.equal(first[key], second[key])
    for key in (
        "coarse_topm_scores",
        "fine_scores",
        "context_contribution",
    ):
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("top_m", [1, 4])
def test_save_load_and_replay_topm_contract(tiny_ranker, tmp_path, top_m):
    output = _rank(tiny_ranker, top_m=top_m)
    save_results(tmp_path, output, {"score_schema_version": SCORE_SCHEMA_VERSION})
    loaded = torch.load(tmp_path / "rankings.pt", weights_only=True)
    replayed = replay_rankings(loaded)
    m = min(top_m, len(output["gallery_ids"]) - 1)
    for field in (
        "coarse_topm_scores",
        "context_contribution",
        "fine_scores",
        "fused_topm_scores",
        "coverage_bonus",
        "fafa_binding",
        "context",
        "coverage",
        *(f"binding_{mode}" for mode in POOLING_MODES),
    ):
        assert loaded[field].shape == (2, m)
        assert loaded[field].dtype == torch.float32
        assert torch.equal(replayed[field], loaded[field])
    assert (
        loaded["score_schema_version"]
        == loaded["schema_version"]
        == SCORE_SCHEMA_VERSION
    )
    assert torch.equal(loaded["rankings"][:, m:], loaded["coarse_rankings"][:, m:])
    for field in ("rankings", "fine_rankings", "fafa_rankings"):
        assert loaded[field].dtype == replayed[field].dtype == torch.int32
        assert torch.equal(loaded[field], replayed[field])
        for sample, row in zip(tiny_ranker[3], loaded[field], strict=True):
            assert len(row) == len(set(row.tolist())) == 5
            assert {loaded["gallery_ids"][j] for j in row} == set(
                loaded["gallery_ids"]
            ) - {sample["query_image_id"]}
    assert {
        "positive_image_ids",
        "case_type",
        "identity_ids",
        "identity_scores",
        "state_scores",
    }.isdisjoint(loaded)
    (tmp_path / "metrics.json").write_text('{"old": true}')
    save_results(tmp_path, replayed, {})
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / ".history").exists()
    assert torch.equal(
        torch.load(tmp_path / "rankings.pt", weights_only=True)["rankings"],
        replayed["rankings"],
    )


def test_fusion_keeps_unsupported_gallery_candidates_last(tiny_ranker):
    cache = tiny_ranker[0]
    cache.mask[cache.by_id["vn"]] = False
    output = _rank(tiny_ranker, top_m=20)
    masked = output["coarse_topm"] == cache.by_id["vn"]
    assert masked.sum() == 2
    assert torch.isneginf(output["coarse_topm_scores"][masked]).all()
    assert torch.isfinite(output["fine_scores"]).all()
    assert torch.isneginf(output["fused_topm_scores"][masked]).all()
    assert (output["rankings"][:, -1] == cache.by_id["vn"]).all()
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


def test_every_split_only_fine_scores_the_shortlist(tiny_ranker, monkeypatch):
    cache, model, _, samples, _ = tiny_ranker
    pairs = []
    original = model.score_target

    def counted(query, scene, *args, **kwargs):
        pairs.extend(scene[:, 0, 0].tolist())
        return original(query, scene, *args, **kwargs)

    monkeypatch.setattr(model, "score_target", counted)
    output = _rank(tiny_ranker, top_m=1)
    assert len(pairs) == len(samples)
    assert output["fafa_binding"].shape == (len(samples), 1)
    assert len(cache.image_ids) - 1 == output["rankings"].shape[1]


def test_ranking_ignores_case_identities_positives_and_cardinality(
    tiny_ranker, monkeypatch
):
    original = _rank(tiny_ranker)
    cache, model, text, samples, settings = tiny_ranker
    changed = copy.deepcopy(samples)
    for sample in changed:
        sample["case_type"] = "untrusted"
        sample["positive_image_ids"] = ["untrusted"]
        sample["target_image_id"] = "untrusted"
        sample["subjects"][0].update(identity_ids=["untrusted"], member_count=999)
    native_queries = cache.native_queries

    def public_only(rows, width):
        for row in rows:
            assert {"case_type", "positive_image_ids", "target_image_id"}.isdisjoint(
                row
            )
            assert all(set(s) == {"subject_id"} for s in row["subjects"])
        return native_queries(rows, width)

    monkeypatch.setattr(cache, "native_queries", public_only)
    new = _rank((cache, model, text, changed, settings))
    assert torch.equal(new["rankings"], original["rankings"])
    assert torch.equal(new["fine_scores"], original["fine_scores"])


# Person and Subject pooling.


def _matching(member, indices, subjects=None):
    member = torch.tensor(member, dtype=torch.float32)
    indices = torch.tensor(indices)
    active = (
        torch.ones(member.shape[:2], dtype=torch.bool)
        if subjects is None
        else torch.tensor(subjects, dtype=torch.bool)
    )
    weights = member / member.sum(-1, keepdim=True).clamp_min(1e-6)
    per_subject = (weights * (indices >= 0)[:, None]).sum(-1)
    coverage = per_subject.masked_fill(~active, 0).sum(-1) / active.sum(-1).clamp_min(1)
    return PersonMatching(indices, member, active, per_subject, coverage)


def test_global_and_subject_means_have_distinct_weighting():
    match = _matching([[[0.9, 0.1, 0, 0], [0, 0, 0.2, 0.8]]], [[0, -1, 1, -1]])
    fda = torch.tensor([[[0.8, 0, 0, 0], [0, 0, 0.2, 0]]])
    scores = pool_fafa_scores(fda, match)
    assert scores["penalized"].item() == pytest.approx((0.61 - 0.84) / 2)
    assert scores["matched_mean"].item() == pytest.approx((0.72 + 0.04) / 1.1)
    assert scores["subject_mean"].item() == pytest.approx((0.8 + 0.2) / 2)


def test_multi_subject_export_retains_a_completely_missing_role(
    tiny_ranker, monkeypatch
):
    cache, model, _, samples, _ = tiny_ranker
    for sample in samples:
        sample["subjects"].append({"subject_id": 2, "identity_ids": ["p2"]})
        sample["final_desc"] = (
            "Identify Subject 1 as the person on the left and Subject 2 as "
            "the person on the right"
        )
        sample["final_change"] = (
            "then retrieve images where Subject 1 is standing and Subject 2 is seated"
        )

    def ground(scene, persons, boxes, selections, *args):
        return scene.new_tensor([5.0, -5.0])[None, :, None].expand(
            len(scene), 2, persons.shape[1]
        )

    monkeypatch.setattr(model.grounding, "forward", ground)
    monkeypatch.setattr(
        cache,
        "native_queries",
        lambda rows, people: torch.ones(len(rows), 2, people, 5),
    )
    model.fine_matching["identity_threshold"] = -1.0
    source = _rank(tiny_ranker)
    torch.testing.assert_close(
        source["binding_matched_mean"], torch.ones_like(source["coverage"])
    )
    torch.testing.assert_close(
        source["binding_subject_mean"], torch.full_like(source["coverage"], -0.05)
    )
    for mode in POOLING_MODES:
        direct = _rank(tiny_ranker, pooling=mode)
        replayed = replay_rankings(source, pooling=mode)
        assert torch.equal(direct["rankings"], replayed["rankings"])
        torch.testing.assert_close(direct["fafa_binding"], replayed["fafa_binding"])


def test_ordered_pair_scoring_preserves_missing_subject_weights():
    dim = 4
    reasoner = FineReasoner(dim, 2, 2, 0)
    subjects = torch.ones(1, 3, dtype=torch.bool)
    condition, weights = reasoner.encode_condition(
        torch.zeros(1, 3, dim), None, torch.eye(3, dtype=torch.bool)[None], subjects
    )
    assert condition.shape == (1, 6, dim)
    torch.testing.assert_close(weights.sum(1), torch.ones(1))
    reasoner.subject_score = torch.nn.Linear(2 * dim + 4, 1)
    reasoner.pair_score = torch.nn.Linear(4 * dim + 5, 1, bias=False)
    with torch.no_grad():
        reasoner.subject_score.weight.zero_()
        reasoner.subject_score.bias.fill_(1)
        reasoner.pair_score.weight.zero_()
        reasoner.pair_score.weight[0, 3 * dim] = 1

    weights = torch.tensor([[0.25, 0, 0, 0.75, 0, 0]])
    boxes = torch.tensor([[[0, 0, 0.25, 1], [0.5, 0, 0.75, 1], [0.75, 0, 1, 1]]])
    membership = torch.eye(3)[None]
    evidence = torch.zeros(1, 3, dim)
    scene = torch.zeros(1, 4, dim)
    full = reasoner(condition, weights, evidence, boxes, membership, scene, (2, 2))
    reverse = reasoner(
        condition, weights, evidence, boxes[:, [1, 0, 2]], membership, scene, (2, 2)
    )
    membership[:, 1] = 0
    missing = reasoner(condition, weights, evidence, boxes, membership, scene, (2, 2))
    torch.testing.assert_close(
        torch.cat((full, reverse, missing)),
        torch.tensor(
            [
                0.625,
                -0.125,
                0.25,
            ]
        ),
    )


# Configured calibration grid and frozen coefficients.


def test_configured_grid_matches_official_replay_with_fixed_tail(
    saved_pipeline, monkeypatch
):
    cfg, data, _ = saved_pipeline
    grid = load_config("configs/proposed.yaml")["evaluation"]["calibration"]
    cfg["evaluation"]["calibration"] = grid
    cfg["retrieval"]["top_m"] = 3
    cache = retrieve.GalleryCache()
    for image in ("vx", "vy"):
        _add_gallery_image(cache, data, image, "val")
    data.samples_by_id["val"]["positive_image_ids"] = ["va", "vy"]
    for row in (_sample("val2", "vn", "vq"), _sample("val3", "va", "vn")):
        row["case_type"] = "GROUP" if row["sample_id"] == "val2" else "INDIVIDUAL"
        if row["sample_id"] == "val3":
            row["positive_image_ids"] = ["vn", "vx"]
        data.samples.append(row)
        data.samples_by_id[row["sample_id"]] = row
        data.splits["val"].append(row["sample_id"])
    samples = [data.samples_by_id[key] for key in data.splits["val"]]
    source = retrieve.retrieve(cfg)
    order = torch.tensor(
        [
            [i for i in range(len(source["gallery_ids"])) if i != query]
            for query in source["query_indices"].tolist()
        ],
        dtype=torch.int32,
    )
    source["coarse_rankings"] = source["rankings"] = order
    source["coarse_topm"] = order[:, :3]
    source["coarse_topm_scores"] = torch.tensor(
        [[2.0, 1.0, 0.0], [-torch.inf, -torch.inf, -torch.inf], [5.0, 3.0, -torch.inf]]
    )
    binding = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.002, -0.001, 0.003]])
    for mode in ("penalized", "matched_mean", "subject_mean"):
        source[f"binding_{mode}"] = binding
    source["fafa_binding"] = binding
    source["context"] = torch.tensor(
        [[0.4, 0.8, 0.5], [0.0, 0.0, 0.0], [2.0, -1.0, 0.5]]
    )
    source["coverage"] = torch.tensor(
        [[1.0, 0.2, 0.3], [0.5, 0.5, 0.5], [0.0, 1.0, 0.5]]
    )
    source = replay_rankings(source)
    evaluate_calls, replay_calls = [], []
    original_evaluate, original_replay = (
        calibration.evaluate_retrieval_output,
        calibration.replay_rankings,
    )

    def counted_evaluate(*args, **kwargs):
        evaluate_calls.append(True)
        return original_evaluate(*args, **kwargs)

    def counted_replay(*args, **kwargs):
        replay_calls.append(True)
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(calibration, "evaluate_retrieval_output", counted_evaluate)
    monkeypatch.setattr(calibration, "replay_rankings", counted_replay)
    selected = calibration.calibrate_rankings(source, data, samples, cfg)
    assert len(evaluate_calls) == 2 and len(replay_calls) == 1
    assert [
        tuple(row[key] for key in CALIBRATED_RETRIEVAL_KEYS) for row in selected.rows
    ] == list(product(*grid.values()))
    for row in selected.rows:
        direct = replay_rankings(
            source, **{key: row[key] for key in CALIBRATED_RETRIEVAL_KEYS}
        )
        result = evaluate_retrieval_output(
            data, samples, direct, cfg["candidate_ks"], split="val"
        )
        assert {
            key: row[key] for key in selection_metrics(result)
        } == selection_metrics(result)
        for case, metrics in result["by_case"].items():
            assert row[f"{case.lower()}_full_map"] == metrics["full_map"]
        assert torch.equal(direct["rankings"][:, 3:], source["coarse_rankings"][:, 3:])
    expected = max(selected.rows, key=lambda row: row["checkpoint_score"])
    assert selected.best == expected
    assert (
        selected.metrics["overall"]["checkpoint_score"] == expected["checkpoint_score"]
    )
    assert cfg["retrieval"]["context_weight"] == 0.10
    assert source["score_config"]["context_weight"] == 0.10


def test_calibrated_linear_coefficients_deploy_and_remain_frozen(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    original = copy.deepcopy(cfg["retrieval"])
    root = tmp_path / "linear-calibration"
    rows = calibration.calibrate(
        {
            "rankings": [str(Path(cfg["output"]["dir"]) / "val/rankings.pt")],
            "output_dir": str(root),
            "coverage_weights": [0.05, 0.12],
            "context_weights": [0.02, 0.06],
            "fine_coarse_weights": [0.10, 0.20],
        }
    )
    assert len(rows) == 8
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    winner = max(rows, key=lambda row: row["checkpoint_score"])
    for key in CALIBRATED_RETRIEVAL_KEYS:
        assert (
            selected["retrieval"][key]
            == selected["selected_retrieval"][key]
            == winner[key]
        )
    assert selected["retrieval"]["coverage_weight"] != original["coverage_weight"]
    assert selected["retrieval"]["context_weight"] != original["context_weight"]
    assert json.loads((root / "selection.json").read_text())["num_candidates"] == 8
    expected = replay_rankings(
        source, **{key: winner[key] for key in CALIBRATED_RETRIEVAL_KEYS}
    )
    deployed = retrieve.retrieve(selected)
    assert torch.equal(deployed["rankings"], expected["rankings"])
    assert "fine_score_ablation" not in deployed["provenance"]
    tampered = copy.deepcopy(selected)
    tampered["retrieval"]["coverage_weight"] += 0.001
    with pytest.raises(ValueError, match="settings changed since val selection"):
        retrieve.retrieve(tampered)
    assert cfg["retrieval"] == original


def test_three_coefficient_checkpoint_does_not_change_training(
    tiny_training_run, monkeypatch
):
    cfg, _, wandb_run, _ = tiny_training_run
    cfg["evaluation"]["calibration"] = {
        "coverage_weights": [0.12],
        "context_weights": [0.02],
        "fine_coarse_weights": [0.1375],
    }
    original = copy.deepcopy(cfg["retrieval"])
    observed = []
    compute = training.compute_loss

    def observed_loss(*args, **kwargs):
        observed.append(
            (kwargs["coverage_score_weight"], kwargs["context_score_weight"])
        )
        return compute(*args, **kwargs)

    monkeypatch.setattr(training, "compute_loss", observed_loss)
    checkpoint = training.train(cfg)
    saved = torch.load(checkpoint, weights_only=True)
    assert observed == [(original["coverage_weight"], original["context_weight"])]
    assert saved["config"]["retrieval"] == original
    selected = yaml.safe_load((checkpoint.parent / "selected.yaml").read_text())
    for key, value in zip(CALIBRATED_RETRIEVAL_KEYS, (0.12, 0.02, 0.1375), strict=True):
        assert saved[f"best_{key}"] == saved["validation"][key] == value
        assert selected["selected_retrieval"][key] == value
        assert wandb_run.summary[f"best_{key}"] == value
    assert cfg["retrieval"] == original


# Shortlist evidence, offline tuning and selected test retrieval.


def _settings(paths, root):
    return {"rankings": [str(path) for path in paths], "output_dir": str(root)}


def test_retrieval_scores_each_shortlist_pair_once_and_stores_topm_evidence(
    tiny_ranker, monkeypatch
):
    cache, model, _, samples, _ = tiny_ranker
    active, pairs = [], []
    encode, score = model.encode_query, model.score_target

    def image_index(scene):
        return (cache.scenes == scene).flatten(1).all(1).nonzero().item()

    def encode_counted(scene, *args, **kwargs):
        active[:] = [image_index(scene[0])]
        return encode(scene, *args, **kwargs)

    def score_counted(query, scene, *args, **kwargs):
        pairs.extend((active[0], image_index(row)) for row in scene)
        return score(query, scene, *args, **kwargs)

    monkeypatch.setattr(model, "encode_query", encode_counted)
    monkeypatch.setattr(model, "score_target", score_counted)
    output = _rank(tiny_ranker, top_m=1, coarse_beta=0.0)
    assert (
        output["schema_version"]
        == output["score_schema_version"]
        == SCORE_SCHEMA_VERSION
        == 4
    )
    assert "score_alignment" not in output["score_config"]
    assert {"identity_scores", "state_scores", "tuning_scores"}.isdisjoint(output)
    expected = {
        (cache.by_id[sample["query_image_id"]], int(output["coarse_topm"][n, 0]))
        for n, sample in enumerate(samples)
    }
    assert len(pairs) == len(set(pairs)) == len(samples)
    assert set(pairs) == expected
    for key in (
        "fafa_binding",
        "context",
        "coverage",
        *(f"binding_{mode}" for mode in POOLING_MODES),
    ):
        assert output[key].shape == (len(samples), 1)
        assert output[key].dtype == torch.float32
        assert output[key].device.type == "cpu"
    binding, context, coverage = (
        output[key] for key in ("fafa_binding", "context", "coverage")
    )
    torch.testing.assert_close(
        output["fine_scores"],
        output["binding_scale"] * binding
        + output["score_config"]["coverage_weight"] * coverage
        + output["score_config"]["context_weight"] * context,
        rtol=0,
        atol=0,
    )
    pooling = output["score_config"]["pooling"]
    assert torch.equal(output["fafa_binding"], output[f"binding_{pooling}"])
    assert {
        "case_type",
        "identity_ids",
        "positive_image_ids",
        "target_image_id",
    }.isdisjoint(output)
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_replay_matches_direct_retrieval_for_fine_score_settings(tiny_ranker, pooling):
    source = _rank(tiny_ranker)
    parameters = dict(
        pooling=pooling,
        context_weight=0.13,
        coverage_weight=0.07,
        fine_coarse_weight=0.2,
    )
    replayed = replay_rankings(source, **parameters)
    direct = _rank(tiny_ranker, **parameters)
    for key in (
        "coarse_rankings",
        "coarse_topm",
        "fine_rankings",
        "fafa_rankings",
        "rankings",
    ):
        assert torch.equal(replayed[key], direct[key]), key
    for key in ("coarse_topm_scores", "fine_scores", "fafa_binding", "context"):
        torch.testing.assert_close(replayed[key], direct[key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        replayed["coverage"], direct["coverage"], atol=2e-6, rtol=2e-5
    )
    assert replayed["score_config"]["pooling"] == pooling
    assert bool(replayed["provenance"].get("fine_score_ablation")) == (
        pooling != source["score_config"]["pooling"]
    )
    assert "fine_score_ablation" not in replay_rankings(source)["provenance"]


def test_replay_rejects_settings_that_change_the_shortlist(tiny_ranker):
    source = _rank(tiny_ranker)
    recorded = source["score_config"]
    for parameters in (
        {"coarse_beta": recorded["coarse_beta"] + 1.0},
        {"top_m": recorded["top_m"] - 1},
    ):
        with pytest.raises(ValueError, match="changes the Top-M shortlist"):
            replay_rankings(source, **parameters)
    unchanged = replay_rankings(
        source, coarse_beta=recorded["coarse_beta"], top_m=recorded["top_m"]
    )
    assert torch.equal(unchanged["rankings"], source["rankings"])


def test_offline_calibration_selects_and_freezes_without_checkpoint_or_caches(
    saved_pipeline, tmp_path, monkeypatch
):
    from rcr.evaluation.report import load_run

    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    paths, sources = [], []
    for name, beta in (("zero", 0.0), ("tuned", 1.5)):
        cfg["retrieval"]["coarse_beta"] = beta
        sources.append(retrieve.retrieve(cfg))
        paths.append(tmp_path / f"{name}-scores.pt")
        torch.save(sources[-1], paths[-1])
    digests = [sha256_file(path) for path in paths]
    checkpoint = Path(cfg["checkpoint"])
    moved = tmp_path / "hidden.pt"
    checkpoint.rename(moved)
    root = tmp_path / "offline-calibration"
    original_load = torch.load
    opened = []

    def only_scores(path, *args, **kwargs):
        assert Path(path) in paths
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "offline calibration must not load a model or cache, or run inference"
        )

    with monkeypatch.context() as guard:
        guard.setattr(torch, "load", only_scores)
        guard.setattr(retrieve, "load_text_modules", forbidden)
        guard.setattr(retrieve, "GalleryCache", forbidden)
        guard.setattr(RCRModel, "from_config", forbidden)
        guard.setattr(RCRModel, "score_target", forbidden)
        rows = calibration.calibrate(_settings(paths, root))
    assert opened == paths
    grid = cfg["evaluation"]["calibration"]["fine_coarse_weights"]
    assert [row["coarse_beta"] for row in rows] == [0.0] * len(grid) + [1.5] * len(grid)
    assert [row["fine_coarse_weight"] for row in rows] == grid * 2
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    best = max(rows, key=lambda row: row["checkpoint_score"])
    for key in (
        "coarse_beta",
        "context_weight",
        "coverage_weight",
        "fine_coarse_weight",
        "top_m",
    ):
        assert selected["retrieval"][key] == best[key] == expected["score_config"][key]
        assert selected["selected_retrieval"][key] == best[key]
    assert selected["candidate_ks"] == cfg["candidate_ks"] == [1, 2]
    assert all("candidate_recall_2" in row for row in rows)
    source = sources[paths.index(Path(best["score_source"]))]
    assert torch.equal(expected["rankings"], source["rankings"])
    assert (
        selected["selected_checkpoint_sha256"]
        == source["provenance"]["checkpoint_sha256"]
    )
    assert (
        selected["selected_validation_sha256"]
        == source["provenance"]["validation_sha256"]
    )
    assert [sha256_file(path) for path in paths] == digests
    assert (
        load_run("proposed", root / "selected/val", "val")["metrics"]["overall"][
            "full_map"
        ]
        == best["full_map"]
    )
    assert {"summary.csv", "selection.json", "selected.yaml"} <= {
        path.name for path in root.iterdir()
    }
    moved.rename(checkpoint)
    selected["output"]["dir"] = str(tmp_path / "direct-selected")
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])


def test_selected_test_only_fine_scores_topm_and_checks_checkpoint(
    saved_pipeline, tmp_path, monkeypatch
):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    retrieve.retrieve(cfg)
    root = tmp_path / "calibration"
    calibration.calibrate(
        _settings([Path(cfg["output"]["dir"]) / "val/rankings.pt"], root)
    )
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    selected["split"] = "test"
    # Reuse the tiny cache's other images as a separate test gallery.
    sample = _sample("test", "q", "a")
    data.samples[0] = sample
    data.samples_by_id = {s["sample_id"]: s for s in data.samples}
    data.splits.update(train=[], test=["test"])
    for image in ("q", "a", "n"):
        data.images_by_id[image]["path"] = f"test/{image}.jpg"
    score = RCRModel.score_target
    pairs = []

    def counted(self, query, scene, *args, **kwargs):
        pairs.extend(scene)
        return score(self, query, scene, *args, **kwargs)

    monkeypatch.setattr(RCRModel, "score_target", counted)
    output = retrieve.retrieve(selected)
    assert len(pairs) == 1
    assert output["rankings"].shape == (1, 2)
    assert output["fafa_binding"].shape == (1, 1)
    checkpoint = torch.load(selected["checkpoint"], weights_only=True)
    checkpoint["epoch"] += 1
    torch.save(checkpoint, selected["checkpoint"])
    with pytest.raises(ValueError, match="checkpoint changed since val selection"):
        retrieve.retrieve(selected)


def test_calibration_cli_runs_with_no_model_imports_and_only_real_annotations(
    saved_pipeline, tmp_path
):
    import subprocess
    import sys

    from rcr.common.io import write_json, write_jsonl

    cfg, data, _ = saved_pipeline
    final = tmp_path / "annotations"
    cfg["data"]["final_dir"] = str(final)
    cfg["data"]["image_root"] = str(tmp_path / "images-not-present")
    data.gt_head_boxes_by_image = {
        image: [{"image_id": image, **box} for box in boxes]
        for image, boxes in data.gt_head_boxes_by_image.items()
    }
    write_jsonl(final / "samples.jsonl", data.samples)
    write_jsonl(final / "images.jsonl", data.images_by_id.values())
    write_jsonl(
        final / "gallery.jsonl", [{"image_id": image} for image in data.gallery_ids]
    )
    write_jsonl(
        final / "head_boxes.jsonl",
        [
            {"image_id": image, **box}
            for image, boxes in data.gt_head_boxes_by_image.items()
            for box in boxes
        ],
    )
    write_json(final / "manifest.json", {})
    (final / "splits").mkdir()
    for split, ids in data.splits.items():
        (final / "splits" / f"{split}.txt").write_text("\n".join(ids))
    source = retrieve.retrieve(cfg)
    scores = tmp_path / "standalone-scores.pt"
    torch.save(source, scores)
    Path(cfg["checkpoint"]).unlink()
    settings = _settings([scores], tmp_path / "calibrated")
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(settings))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('transformers', 'rcr.proposed.nn', 'rcr.proposed.cache',
                                'rcr.proposed.retrieve')):
            raise AssertionError('offline calibration imported ' + fullname)
sys.meta_path.insert(0, NoModels())
from tools.run import main
main(['calibrate', '--config', sys.argv[1]])
""",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = yaml.safe_load(
        (Path(settings["output_dir"]) / "selected.yaml").read_text()
    )
    assert selected["candidate_ks"] == cfg["candidate_ks"]
    assert (
        selected["selected_checkpoint_sha256"]
        == source["provenance"]["checkpoint_sha256"]
    )


# Retrieve, save, calibrate and evaluate integration.


def test_standalone_retrieve_save_evaluate_and_ranking_only_evaluation(
    saved_pipeline,
):
    cfg, data, scale = saved_pipeline
    cfg["candidate_ks"] = [1]
    torch.use_deterministic_algorithms(False)
    output = retrieve.retrieve(cfg)
    assert torch.are_deterministic_algorithms_enabled()
    directory = Path(cfg["output"]["dir"]) / "val"
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert saved["binding_scale"] == scale == 1.0
    assert torch.equal(replay_rankings(saved)["rankings"], output["rankings"])
    current_metrics = evaluate_run(cfg, data=data)
    # Evaluation needs only rankings and IDs; raw score storage does not affect it.
    ranking_only = {
        k: saved[k]
        for k in (
            "sample_ids",
            "gallery_ids",
            "rankings",
            "coarse_rankings",
            "coarse_topm",
            "fine_rankings",
            "fafa_rankings",
            "fafa_binding",
            "context",
            "coverage",
        )
    }
    assert evaluate_retrieval_output(
        data, [data.samples[1]], ranking_only, [1, 2], split="val"
    ) == evaluate_retrieval_output(data, [data.samples[1]], saved, [1, 2], split="val")
    assert current_metrics["overall"]["num_queries"] == 1
    # The shortlist size is always a candidate-recall cutoff, as in training.
    assert f"candidate_recall_{cfg['retrieval']['top_m']}" in current_metrics["overall"]
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["score_config"] == saved["score_config"]
    assert metadata["score_schema_version"] == SCORE_SCHEMA_VERSION
    assert metadata["fine_score_mode"] == "linear"
    assert metadata["context_weight"] == cfg["retrieval"]["context_weight"]
    assert metadata["checkpoint_epoch"] == 2
    assert metadata["validation_sha256"] == split_fingerprint(data, "val")
    assert {
        "checkpoint_sha256",
        "dataset_sha256",
        "git_commit",
        "git_dirty",
    } <= metadata.keys()

    unselected_test = copy.deepcopy(cfg)
    unselected_test["split"] = "test"
    with pytest.raises(ValueError, match="validation-selected configuration"):
        retrieve.retrieve(unselected_test)

    incomplete_test = copy.deepcopy(unselected_test)
    incomplete_test.update(
        selected_checkpoint_sha256=metadata["checkpoint_sha256"],
        selected_validation_sha256=metadata["validation_sha256"],
        selected_retrieval={},
    )
    with pytest.raises(ValueError, match="retrieval contract is incomplete"):
        retrieve.retrieve(incomplete_test)


def test_calibration_freezes_full_scoring_config_and_retrieves_selected(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    paths = []
    for name, beta in (("zero", 0.0), ("tuned", 0.4)):
        cfg["retrieval"]["coarse_beta"] = beta
        retrieve.retrieve(cfg)
        paths.append(tmp_path / f"{name}.pt")
        (Path(cfg["output"]["dir"]) / "val/rankings.pt").rename(paths[-1])
    root = tmp_path / "calibration"
    rows = calibration.calibrate(
        {
            "rankings": [str(path) for path in paths],
            "output_dir": str(root),
            "fine_coarse_weights": [0.15, 0.0, 0.25],
        }
    )
    assert [row["coarse_beta"] for row in rows] == [0.0] * 3 + [0.4] * 3
    assert [row["fine_coarse_weight"] for row in rows] == [0.15, 0.0, 0.25] * 2
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["retrieval"]["fine_score_mode"] == "linear"
    assert selected["retrieval"]["context_weight"] == cfg["retrieval"]["context_weight"]
    assert selected["selected_validation_sha256"] == split_fingerprint(data, "val")
    assert len(selected["selected_checkpoint_sha256"]) == 64
    best = max(rows, key=lambda row: row["checkpoint_score"])
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    assert expected["score_config"]["coarse_beta"] == best["coarse_beta"]
    assert (
        selected["selected_retrieval"]["fine_coarse_weight"]
        == best["fine_coarse_weight"]
    )
    assert selected["selected_retrieval"]["coverage_weight"] == 0.04
    assert json.loads((root / "selection.json").read_text())["index"] == best["index"]
    output = retrieve.retrieve(selected)
    evaluate_run(selected, data=data)
    assert torch.equal(output["rankings"], expected["rankings"])

    for key, value in (
        ("context_weight", 0.2),
        ("coverage_weight", 0.2),
        ("fine_coarse_weight", 0.5),
        ("coarse_beta", 3.0),
        ("top_m", 1),
    ):
        tampered = copy.deepcopy(selected)
        tampered["split"] = "test"
        tampered["retrieval"][key] = value
        with pytest.raises(
            ValueError, match="settings changed since val selection|must match training"
        ):
            retrieve.retrieve(tampered)
    # Chunking is operational, and may change without selecting a new method.
    selected["retrieval"]["fine_batch_size"] = 2
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])
    del selected["selected_retrieval"]
    with pytest.raises(ValueError, match="predates the scoring contract"):
        retrieve.retrieve(selected)

    with pytest.raises(ValueError, match="unknown calibration settings: top_m"):
        calibration.calibrate(
            {"rankings": [str(paths[0])], "output_dir": str(root), "top_m": [1]}
        )
    foreign = torch.load(paths[0], weights_only=True)
    foreign["provenance"]["checkpoint_sha256"] = "0" * 64
    torch.save(foreign, tmp_path / "foreign.pt")
    with pytest.raises(ValueError, match="must come from one checkpoint"):
        calibration.calibrate(
            {
                "rankings": [str(paths[1]), str(tmp_path / "foreign.pt")],
                "output_dir": str(tmp_path / "mixed"),
            }
        )


@pytest.mark.parametrize("obsolete", ["scorer", "architecture"])
def test_obsolete_checkpoint_is_rejected_before_loading_text(
    saved_pipeline, monkeypatch, obsolete
):
    cfg, _, _ = saved_pipeline
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    if obsolete == "scorer":
        checkpoint["config"]["retrieval"]["fine_score_mode"] = "context_zscore"
    else:
        checkpoint["architecture_version"] = "older-architecture"
    torch.save(checkpoint, cfg["checkpoint"])
    monkeypatch.setattr(
        retrieve,
        "load_text_modules",
        lambda *a, **k: pytest.fail("loaded obsolete model"),
    )
    with pytest.raises(ValueError, match=obsolete):
        retrieve.retrieve(cfg)


def test_calibration_selects_fusion_that_corrects_a_fine_ranking_error(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    from tests.helpers import _add_gallery_image, _fusion_example

    cache = retrieve.GalleryCache()
    _add_gallery_image(cache, data, "vx", "val")
    cfg["retrieval"]["top_m"] = 3
    source = retrieve.retrieve(cfg)
    source = _fusion_example(source, [5.6, 5.601, 1.6])
    source = replay_rankings(source, fine_coarse_weight=0)
    order = source["coarse_rankings"]
    assert source["rankings"][0, 0] == order[0, 1]
    path = tmp_path / "controlled-val.pt"
    torch.save(source, path)
    root = tmp_path / "fusion-selection"
    rows = calibration.calibrate(
        {
            "rankings": [str(path)],
            "output_dir": str(root),
            "fine_coarse_weights": [0, 0.15, 0.25],
        }
    )
    assert [row["full_map"] for row in rows] == [0.5, 1.0, 1.0]
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["selected_retrieval"]["fine_coarse_weight"] == 0.15
    assert selected["selected_retrieval"]["context_weight"] == 0.10
    assert selected["selected_retrieval"]["coverage_weight"] == 0.04


def test_calibration_rejects_diagnostic_pooling_changes(saved_pipeline, tmp_path):
    cfg, _, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    changed = replay_rankings(source, pooling="penalized")
    path = tmp_path / "ablation.pt"
    torch.save(changed, path)
    root = tmp_path / "forbidden-selection"
    with pytest.raises(
        ValueError, match="requires the checkpoint pooling and fine score mode"
    ):
        calibration.calibrate({"rankings": [str(path)], "output_dir": str(root)})
    assert not root.exists()
