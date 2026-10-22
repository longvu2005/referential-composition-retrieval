"""Complete-shortlist scoring, saved-score alignment and offline replay contracts."""

import copy
import math

import pytest
import torch

from rcr.common.io import save_results
from rcr.proposed import ranking
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.replay import main as replay_main
from rcr.proposed.replay import replay_rankings
from tests.test_train import TextBackbone, TinyCache, Tokenizer, _sample


@pytest.fixture
def tiny_ranker():
    torch.manual_seed(19)
    cache = TinyCache()
    model = RCRModel(dim=8, identity_dim=6, num_heads=2, state_dim=4)
    text = TextEncoder(TextBackbone(), dim=8)
    samples = [_sample("train", "q", "a"), _sample("val", "vq", "va")]
    settings = dict(
        top_m=4,
        fine_batch_size=2,
        identity_batch_size=2,
        coarse_batch_size=2,
        coarse_beta=0.4,
        fine_coarse_weight=0.4,
        fine_score_mode="context_zscore",
        context_weight=0.03,
    )
    return cache, model, text, samples, settings


def _rank(tiny_ranker, **overrides):
    cache, model, text, samples, settings = tiny_ranker
    return ranking.retrieve_rankings(
        samples,
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        **{**settings, **overrides},
    )


@pytest.mark.parametrize(
    "mode", ["legacy_raw", "raw_gamma", "context_zscore", "both_zscore"]
)
def test_fine_score_matches_independent_reference(mode):
    binding = torch.tensor([0.5, -0.4, 0.1, 0.8], dtype=torch.float64)
    context = torch.tensor([7.0, 2.0, -4.0, 5.0], dtype=torch.float64)
    coverage = torch.tensor([0.1, 1.0, 0.0, 0.8], dtype=torch.float64)
    scale, weight = 1.7, 0.03
    result = ranking.fine_score_components(
        binding, context, coverage, scale, mode, weight
    )
    gated = [float(c * s) for c, s in zip(coverage, context, strict=True)]
    scaled = [scale * float(b) for b in binding]

    def z(values):
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        return [(v - mean) / max(std, 1e-6) for v in values]

    if mode == "legacy_raw":
        expected = [b + c for b, c in zip(scaled, gated, strict=True)]
    else:
        left = z(scaled) if mode == "both_zscore" else scaled
        right = gated if mode == "raw_gamma" else z(gated)
        expected = [b + weight * c for b, c in zip(left, right, strict=True)]
    torch.testing.assert_close(result["fine_scores"], torch.tensor(expected))
    assert all(value.dtype == torch.float32 for value in result.values())


@pytest.mark.parametrize("size", [1, 2, 500])
@pytest.mark.parametrize("coverage_value", [0.0, 0.3])
def test_constant_context_is_zero_after_normalization(size, coverage_value):
    binding = torch.linspace(-1, 1, size)
    result = ranking.fine_score_components(
        binding,
        torch.full((size,), 7.0),
        torch.full((size,), coverage_value),
        1.9,
    )
    torch.testing.assert_close(result["fine_scores"], 1.9 * binding, rtol=0, atol=0)
    assert result["context_std"].item() == 0


@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_components_are_errors_even_with_zero_coverage(field, bad):
    values = [torch.ones(3), torch.ones(3), torch.zeros(3)]
    values[field][0] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        ranking.fine_score_components(*values, 1.0)


def test_finite_input_overflow_is_an_explicit_error():
    with pytest.raises(ValueError, match="float32|finite"):
        ranking.fine_score_components(
            torch.ones(2), torch.full((2,), 3e38), torch.full((2,), 2.0), 1.0
        )


def test_permutation_and_stable_tie_convention():
    ids = torch.tensor([3, 2, 4, 1])
    binding = torch.tensor([0.1, 0.9, -0.4, 0.5])
    context = torch.tensor([2.0, -1.0, 0.3, 4.0])
    coverage = torch.tensor([1.0, 0.2, 0.4, 0.9])
    scores = ranking.fine_score_components(binding, context, coverage, 1.3)[
        "fine_scores"
    ]
    perm = torch.tensor([3, 0, 2, 1])
    perm_scores = ranking.fine_score_components(
        binding[perm], context[perm], coverage[perm], 1.3
    )["fine_scores"]
    assert torch.equal(
        ranking.rerank_topm(ids, ids, scores),
        ranking.rerank_topm(ids[perm], ids[perm], perm_scores),
    )
    assert torch.equal(ranking.rerank_topm(ids, ids, torch.zeros(4)), ids)


@pytest.mark.parametrize("weight", [0.0, 0.4])
@pytest.mark.parametrize(
    "coarse",
    [
        [0.3, -float("inf"), 0.1],
        [float("nan"), float("inf"), -float("inf")],
        [0.3, 0.3, 0.3],
    ],
)
def test_fusion_preserves_nonfinite_coarse_policy(weight, coarse):
    fine = torch.tensor([0.2, 20.0, -0.3])
    coarse = torch.tensor(coarse)
    result = ranking.fuse_fine_coarse_scores(fine, coarse, weight)
    if weight == 0:
        assert torch.equal(result, fine)
    else:
        valid = coarse.isfinite()
        assert torch.isneginf(result[~valid]).all()
        assert torch.isfinite(result[valid]).all()


@pytest.mark.parametrize("mode", ["legacy_raw", "context_zscore"])
def test_complete_topm_is_independent_of_inference_chunks(tiny_ranker, mode):
    first = _rank(
        tiny_ranker,
        fine_score_mode=mode,
        fine_batch_size=1,
        identity_batch_size=1,
        coarse_batch_size=1,
    )
    second = _rank(
        tiny_ranker,
        fine_score_mode=mode,
        fine_batch_size=5,
        identity_batch_size=6,
        coarse_batch_size=6,
    )
    for key in ("coarse_rankings", "fine_rankings", "rankings", "fafa_rankings"):
        assert torch.equal(first[key], second[key])
    for key in (
        "coarse_topm_scores",
        "fine_scores",
        "fine_raw_scores",
        "gated_context",
        "context_mean",
        "context_std",
    ):
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-5)


def test_artifact_alignment_matches_real_forward_and_coarse_scores(
    tiny_ranker, monkeypatch
):
    cache, model, text, samples, settings = tiny_ranker
    # Deliberately permuted, non-contiguous subset of cache images.
    gallery = ["vn", "a", "q", "va", "n"]
    coarse_values, target_values = [], {}
    combine, score_target = ranking.combine_scores, model.score_target

    def coarse(*args, **kwargs):
        scores = combine(*args, **kwargs)
        coarse_values.append(scores.clone())
        return scores

    def score(query, scene, *args, **kwargs):
        details = score_target(query, scene, *args, **kwargs)
        for i, patches in enumerate(scene):
            cache_index = (cache.scenes == patches).flatten(1).all(1).nonzero().item()
            target_values[cache.image_ids[cache_index]] = {
                key: value[i].clone() for key, value in details.items()
            }
        return details

    monkeypatch.setattr(ranking, "combine_scores", coarse)
    monkeypatch.setattr(model, "score_target", score)
    output = ranking.retrieve_rankings(
        samples[:1],
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        **{**settings, "top_m": 3},
    )
    top = output["coarse_topm"][0].long()
    torch.testing.assert_close(
        output["coarse_topm_scores"][0], coarse_values[0][top], rtol=0, atol=0
    )
    for field, source in (
        ("fafa_binding", "binding"),
        ("context", "context"),
        ("coverage", "coverage"),
        ("fine_raw_scores", "score"),
    ):
        expected = torch.stack(
            [target_values[gallery[j]][source] for j in top.tolist()]
        )
        torch.testing.assert_close(output[field][0], expected, rtol=0, atol=0)
    assert output["fine_scores"].shape == (1, 3)
    assert output["context_std"].shape == (1,)
    assert len(target_values) == 3


@pytest.mark.parametrize("top_m", [1, 4, 20])
def test_save_load_replay_and_full_gallery_contract(tiny_ranker, tmp_path, top_m):
    output = _rank(tiny_ranker, top_m=top_m)
    save_results(tmp_path, output, {"score_schema_version": 1})
    loaded = torch.load(tmp_path / "rankings.pt", weights_only=True)
    replayed = replay_rankings(loaded)
    m = min(top_m, len(output["gallery_ids"]) - 1)
    for field in (
        "fafa_binding",
        "context",
        "coverage",
        "coarse_topm_scores",
        "gated_context",
        "fine_raw_scores",
        "fine_scores",
        "fused_topm_scores",
    ):
        assert loaded[field].shape == (2, m)
        assert loaded[field].dtype == torch.float32
        torch.testing.assert_close(replayed[field], loaded[field], rtol=0, atol=0)
    for field in ("rankings", "fine_rankings"):
        assert torch.equal(loaded[field], replayed[field])
        for sample, row in zip(tiny_ranker[3], loaded[field], strict=True):
            assert len(row) == len(set(row.tolist())) == 5
            assert {loaded["gallery_ids"][j] for j in row} == set(
                loaded["gallery_ids"]
            ) - {sample["query_image_id"]}
    assert {"positive_image_ids", "case_type", "identity_ids"}.isdisjoint(loaded)
    save_results(tmp_path, replayed, {})
    assert len(list((tmp_path / ".history").glob("*/rankings.pt"))) == 1


@pytest.mark.parametrize(
    "mode", ["legacy_raw", "raw_gamma", "context_zscore", "both_zscore"]
)
def test_replay_ablations_keep_tail_and_use_real_coarse_scores(tiny_ranker, mode):
    output = _rank(tiny_ranker, top_m=3)
    replayed = replay_rankings(
        output, fine_score_mode=mode, context_weight=0.13, fine_coarse_weight=0.7
    )
    assert torch.equal(replayed["rankings"][:, 3:], output["coarse_rankings"][:, 3:])
    # Compare official inference for its supported modes at the same weights.
    if mode in ranking.FINE_SCORE_MODES:
        direct = _rank(
            tiny_ranker,
            top_m=3,
            fine_score_mode=mode,
            context_weight=0.13,
            fine_coarse_weight=0.7,
        )
        assert torch.equal(direct["rankings"], replayed["rankings"])
    # Finite-mask behavior is shared even for partially/all unsupported Coarse.
    unsupported = copy.deepcopy(output)
    unsupported["coarse_topm_scores"][0, 1:] = -torch.inf
    unsupported["coarse_topm_scores"][1] = -torch.inf
    masked = replay_rankings(unsupported, fine_score_mode=mode, fine_coarse_weight=0.7)
    assert torch.isneginf(masked["fused_topm_scores"][0, 1:]).all()
    assert torch.equal(masked["rankings"][1], unsupported["coarse_rankings"][1])


def test_replay_rejects_old_missing_truncated_or_misaligned_data(tiny_ranker):
    with pytest.raises(ValueError, match="score_schema_version"):
        replay_rankings({"rankings": torch.tensor([[1, 2]])})
    output = _rank(tiny_ranker)
    broken = copy.deepcopy(output)
    del broken["coarse_topm_scores"]
    with pytest.raises(ValueError, match="coarse_topm_scores"):
        replay_rankings(broken)
    broken = copy.deepcopy(output)
    broken["coarse_topm"] = broken["coarse_topm"].flip(1)
    with pytest.raises(ValueError, match="prefix"):
        replay_rankings(broken)
    broken = copy.deepcopy(output)
    broken["coarse_rankings"] = broken["coarse_rankings"][:, :3]
    with pytest.raises(ValueError, match="complete"):
        replay_rankings(broken)
    broken = copy.deepcopy(output)
    broken["score_config"]["eps"] = 0.1
    with pytest.raises(ValueError, match="incompatible"):
        replay_rankings(broken)


@pytest.mark.parametrize("weight", [0.0, 0.4])
def test_real_masked_gallery_candidate_keeps_fusion_policy(tiny_ranker, weight):
    cache = tiny_ranker[0]
    cache.mask[cache.by_id["vn"]] = False
    output = _rank(tiny_ranker, top_m=20, fine_coarse_weight=weight)
    masked = output["coarse_topm"] == cache.by_id["vn"]
    assert masked.sum() == 2
    assert torch.isneginf(output["coarse_topm_scores"][masked]).all()
    assert torch.isfinite(output["fine_scores"]).all()
    if weight:
        assert torch.isneginf(output["fused_topm_scores"][masked]).all()
    else:
        assert torch.equal(output["fused_topm_scores"], output["fine_scores"])
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


def test_replay_cli_needs_only_saved_rankings(tiny_ranker, tmp_path):
    output = _rank(tiny_ranker)
    source = tmp_path / "rankings.pt"
    torch.save(output, source)
    replay_main(
        [
            str(source),
            "--output-dir",
            str(tmp_path / "replayed"),
            "--mode",
            "raw_gamma",
            "--context-weight",
            "0.1",
            "--fine-coarse-weight",
            "0",
        ]
    )
    saved = torch.load(tmp_path / "replayed/rankings.pt", weights_only=True)
    assert saved["score_config"]["fine_score_mode"] == "raw_gamma"
    assert torch.equal(saved["fine_rankings"], saved["rankings"])


def test_weight_grid_reuses_each_query_target_score(tiny_ranker, monkeypatch):
    cache, model, text, samples, settings = tiny_ranker
    calls = []
    original = model.score_target

    def counted(query, scene, *args, **kwargs):
        calls.extend(scene[:, 0, 0].tolist())
        return original(query, scene, *args, **kwargs)

    monkeypatch.setattr(model, "score_target", counted)
    outputs = ranking.rank_weight_grid(
        samples[:1],
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        settings={**settings, "top_m": 3},
        weights=[(0.0, 0.0), (0.0, 0.4), (0.4, 0.0), (0.4, 0.7)],
    )
    union = set(outputs[0]["coarse_topm"][0].tolist()) | set(
        outputs[2]["coarse_topm"][0].tolist()
    )
    assert len(calls) == len(set(calls)) == len(union)
    for output in outputs:
        assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


def test_new_settings_are_accepted_by_hard_negative_mining(tiny_ranker):
    cache, model, text, samples, settings = tiny_ranker
    pools = ranking.mine_hard_negatives(
        samples,
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        retrieval=settings,
        pool_size=2,
        query_batch_size=2,
    )
    for sample in samples:
        assert len(pools[sample["sample_id"]]) == 2
        assert not set(pools[sample["sample_id"]]) & {
            sample["query_image_id"],
            *sample["positive_image_ids"],
        }


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


@pytest.mark.parametrize(
    "override",
    [
        {"fine_score_mode": "both_zscore"},
        {"context_weight": float("nan")},
        {"top_m": 0},
    ],
)
def test_invalid_retrieval_settings_fail_early(override):
    with pytest.raises(ValueError):
        ranking.retrieval_settings(override, 0.4)
