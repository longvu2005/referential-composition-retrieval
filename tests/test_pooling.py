"""FAFA pooling equations and inference/replay consistency."""

import copy
from itertools import product

import pytest
import torch
import yaml

from rcr.proposed import calibration, retrieve
from rcr.proposed.matching import PersonMatching
from rcr.proposed.replay import _replay_rankings, replay_rankings
from rcr.proposed.scores import POOLING_MODES, fafa_binding_scores, pool_fafa_scores
from tests.test_ranking import _rank
from tests.test_ranking import tiny_ranker as tiny_ranker
from tests.test_score_pipeline import saved_pipeline as saved_pipeline
from tests.test_tuning_export import _settings


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


def test_missing_subject_keeps_its_weight_in_subject_mean():
    match = _matching([[[1, 0], [0, 1]]], [[0, -1]])
    scores = pool_fafa_scores(torch.tensor([[[0.6, 0], [0, 0]]]), match)
    assert scores["matched_mean"].item() == pytest.approx(0.6)
    assert scores["subject_mean"].item() == pytest.approx((0.6 - 1.1) / 2)


@pytest.mark.parametrize(
    "subjects", [[[True, True]], [[True, False]], [[False, False]]]
)
@pytest.mark.parametrize("people", [0, 3])
def test_empty_evidence_has_a_finite_null_score(subjects, people):
    match = _matching(torch.ones(1, 2, people).tolist(), [[-1] * people], subjects)
    scores = pool_fafa_scores(torch.zeros(1, 2, people), match)
    for value in scores.values():
        torch.testing.assert_close(value, torch.tensor([-1.1]))


def test_inactive_subjects_and_unmatched_values_do_not_enter_conditional_means():
    match = _matching([[[0.7, 0.3], [0, 0]]], [[0, -1]], [[True, False]])
    scores = pool_fafa_scores(
        torch.tensor([[[0.4, float("nan")], [float("nan"), float("nan")]]]), match
    )
    assert scores["matched_mean"].item() == pytest.approx(0.4)
    assert scores["subject_mean"].item() == pytest.approx(0.4)


def test_single_subject_conditional_mean_matches_saved_score_identity():
    match = _matching([[[0.8, 0.2, 0.4]]], [[0, -1, 1]])
    scores = pool_fafa_scores(torch.tensor([[[0.6, 0, 0.3]]]), match)
    restored = (scores["penalized"] + 1.1 * (1 - match.coverage)) / match.coverage
    torch.testing.assert_close(scores["matched_mean"], restored)
    torch.testing.assert_close(scores["subject_mean"], restored)


def test_native_fda_remains_detached_and_default_preserves_penalized_mean():
    match = _matching([[[0.8, 0.2]]], [[0, -1]])
    query = torch.tensor([[[[1.0, 0], [0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0]]]], requires_grad=True)
    all_scores = fafa_binding_scores(
        query, target, matching=match, return_poolings=True
    )
    default = fafa_binding_scores(query, target, matching=match)
    torch.testing.assert_close(default, torch.tensor([0.8 - 0.2 * 1.1]))
    torch.testing.assert_close(all_scores["penalized"], default, rtol=0, atol=0)
    for mode in ("matched_mean", "subject_mean"):
        torch.testing.assert_close(all_scores[mode], torch.ones(1))
    assert not any(value.requires_grad for value in all_scores.values())
    with pytest.raises(ValueError, match="pooling"):
        fafa_binding_scores(query, target, matching=match, pooling="unknown")


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_pooling_replay_matches_direct_inference(tiny_ranker, mode):
    source = _rank(tiny_ranker)
    result = replay_rankings(source, pooling=mode)
    direct = _rank(tiny_ranker, pooling=mode)
    assert result["score_config"]["pooling"] == mode
    assert result["score_schema_version"] == result["schema_version"] == 1
    for key in ("fafa_binding", "fine_scores", "fused_topm_scores"):
        torch.testing.assert_close(result[key], direct[key], rtol=0, atol=0)
    assert torch.equal(result["rankings"], direct["rankings"])
    assert torch.equal(result["fafa_rankings"], direct["fafa_rankings"])
    assert source["score_config"]["pooling"] == "penalized"


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
    eligible = torch.ones_like(source["coverage"], dtype=torch.bool)
    eligible[torch.arange(len(samples)), source["query_indices"]] = False
    torch.testing.assert_close(
        source["binding_matched_mean"][eligible], torch.ones(eligible.sum())
    )
    torch.testing.assert_close(
        source["binding_subject_mean"][eligible], torch.full((eligible.sum(),), -0.05)
    )
    for mode in POOLING_MODES:
        direct = _rank(tiny_ranker, pooling=mode)
        replayed = replay_rankings(source, pooling=mode)
        assert torch.equal(direct["rankings"], replayed["rankings"])
        torch.testing.assert_close(direct["fafa_binding"], replayed["fafa_binding"])


def test_pooling_cache_can_switch_modes_and_shortlists_without_stale_scores(
    tiny_ranker,
):
    source = _rank(tiny_ranker)
    cache = {}
    for m, mode, gamma in product([1, 4], POOLING_MODES, [0, 0.1]):
        cfg = {
            **source["score_config"],
            "top_m": m,
            "pooling": mode,
            "context_weight": gamma,
        }
        cached = _replay_rankings(source, cfg, cache)
        fresh = _replay_rankings(source, cfg, {})
        assert torch.equal(cached["rankings"], fresh["rankings"])
        torch.testing.assert_close(
            cached["fine_scores"], fresh["fine_scores"], rtol=0, atol=0
        )


def test_old_artifact_replays_its_mode_but_cannot_invent_subject_scores(tiny_ranker):
    source = copy.deepcopy(_rank(tiny_ranker))
    source["score_config"].pop("pooling")
    for mode in POOLING_MODES:
        source.pop(f"binding_{mode}")
    assert torch.equal(replay_rankings(source)["rankings"], source["rankings"])
    with pytest.raises(ValueError, match="retrieve validation again"):
        replay_rankings(source, pooling="subject_mean")


def test_calibration_selects_and_freezes_pooling(saved_pipeline, tmp_path):
    cfg, _, _ = saved_pipeline
    retrieve.retrieve(cfg)
    settings = _settings(tmp_path / "run/val/rankings.pt", tmp_path / "calibration")
    settings["pooling"] = list(POOLING_MODES)
    rows = calibration.calibrate(settings)
    assert {row["pooling"] for row in rows} == set(POOLING_MODES)
    selected = yaml.safe_load((tmp_path / "calibration/selected.yaml").read_text())
    best = max(rows, key=lambda row: row["full_map"])
    assert selected["method"] == "proposed"
    assert selected["retrieval"]["pooling"] == best["pooling"]
    assert selected["selected_retrieval"]["pooling"] == best["pooling"]
    changed = copy.deepcopy(selected)
    changed["retrieval"]["pooling"] = next(
        mode for mode in POOLING_MODES if mode != best["pooling"]
    )
    with pytest.raises(ValueError, match="settings changed"):
        retrieve.retrieve(changed)
