"""CPU workflow tests for ranking."""

import copy
import math

import pytest
import torch

from rcr.common.io import save_results
from rcr.proposed import ranking
from rcr.proposed.ranking import SCORE_SCHEMA_VERSION
from rcr.proposed.replay import replay_rankings
from rcr.proposed.scores import POOLING_MODES
from tests.helpers import _rank


def test_fine_score_matches_independent_reference():
    binding = torch.tensor([0.5, -0.4, 0.1, 0.8], dtype=torch.float64)
    context = torch.tensor([7.0, 2.0, -4.0, 5.0], dtype=torch.float64)
    coverage = torch.tensor([0.1, 1.0, 0.0, 0.8], dtype=torch.float64)
    scale, weight = 1.7, 0.03
    result = ranking.fine_score_components(
        binding, context, coverage, scale, context_weight=weight
    )
    expected = [
        scale * float(b) + weight * (float(c) * math.tanh(float(v)) - (1 - float(c)))
        for b, v, c in zip(binding, context, coverage, strict=True)
    ]
    torch.testing.assert_close(result["fine_scores"], torch.tensor(expected))
    assert all(value.dtype == torch.float32 for value in result.values())


@pytest.mark.parametrize("raw", [-100.0, -4.0, 0.0, 4.0, 100.0])
def test_coverage_is_monotone_and_residual_is_bounded(raw):
    coverage = torch.linspace(0, 1, 101)
    scores = ranking.fine_score_components(
        torch.zeros_like(coverage),
        torch.full_like(coverage, raw),
        coverage,
        1.0,
        context_weight=0.2,
    )["fine_scores"]
    assert (scores[1:] >= scores[:-1] - 1e-7).all()
    assert scores.min() >= -0.2 and scores.max() <= 0.2
    assert scores[0].item() == pytest.approx(-0.2)


def test_pair_score_and_gradient_do_not_depend_on_shortlist():
    raw = torch.tensor([-0.4, 0.2, 3.0], requires_grad=True)
    binding = torch.tensor([0.1, 0.2, -0.3])
    coverage = torch.tensor([1.0, 0.5, 0.0])
    full = ranking.fine_score_components(binding, raw, coverage, 1.0)["fine_scores"]
    single = ranking.fine_score_components(binding[:1], raw[:1], coverage[:1], 1.0)[
        "fine_scores"
    ]
    torch.testing.assert_close(full[:1], single, rtol=0, atol=0)
    grad = torch.autograd.grad(full[0], raw, retain_graph=True)[0]
    single_grad = torch.autograd.grad(single.sum(), raw)[0]
    torch.testing.assert_close(grad, single_grad, rtol=0, atol=0)
    assert grad[1:].count_nonzero() == 0


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
        "gated_context",
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
        "gated_context",
        "fine_scores",
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
    for field in ("rankings", "fine_rankings"):
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


def test_masked_gallery_candidate_gets_a_finite_null_residual(tiny_ranker):
    cache = tiny_ranker[0]
    cache.mask[cache.by_id["vn"]] = False
    output = _rank(tiny_ranker, top_m=20)
    masked = output["coarse_topm"] == cache.by_id["vn"]
    assert masked.sum() == 2
    assert torch.isneginf(output["coarse_topm_scores"][masked]).all()
    assert torch.isfinite(output["fine_scores"]).all()
    assert torch.equal(output["rankings"], output["fine_rankings"])
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
