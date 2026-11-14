"""CPU workflow tests for pooling."""

import pytest
import torch

from rcr.proposed.matching import PersonMatching
from rcr.proposed.replay import replay_rankings
from rcr.proposed.scores import POOLING_MODES, pool_fafa_scores
from tests.helpers import _rank


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
