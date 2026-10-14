"""Structural regressions for case-free selection, assignment and evidence ownership."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from rcr.proposed.matching import (
    exclusive_membership,
    matching_settings,
    partial_identity_matching,
)
from rcr.proposed.nn.model import QueryEncoding, RCRModel
from rcr.proposed.nn.reasoning import FineReasoner
from rcr.proposed.scores import binding_scores


def test_selection_supports_multiple_members_and_exclusive_roles():
    logits = torch.tensor([[[5.0, 4.0, -5.0], [-5.0, -4.0, 5.0]]], requires_grad=True)
    membership = exclusive_membership(logits, torch.ones(1, 2, dtype=torch.bool))
    assert (membership > 0).tolist() == [[[True, True, False], [False, False, True]]]
    assert not membership.requires_grad
    assert ((membership > 0).sum(1) <= 1).all()


def test_no_top1_fallback_and_padding_never_selects_a_person():
    logits = torch.tensor([[[-4.0, -5.0, 100.0], [-3.0, -5.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    membership = exclusive_membership(logits, torch.ones(1, 2, dtype=torch.bool), mask)
    assert membership.count_nonzero() == 0
    # A padded Subject must not alter the foreground/background distribution.
    logits[:, 0, 0] = 4
    full = exclusive_membership(logits, torch.tensor([[True, False]]), mask)
    short = exclusive_membership(logits[:, :1], torch.tensor([[True]]), mask)
    torch.testing.assert_close(full[:, :1], short)
    assert full[:, 1].count_nonzero() == 0


def test_duplicate_query_boxes_do_not_create_extra_group_members():
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 1.0]] * 2])
    membership = exclusive_membership(
        torch.tensor([[[5.0, 4.0]]]), torch.tensor([[True]]), boxes=boxes
    )
    assert (membership > 0).tolist() == [[[True, False]]]


def test_matching_is_global_instead_of_greedy():
    # Greedy chooses q0->t0, then q1 has no valid t1. Global chooses both.
    query = torch.tensor([[[1.0, 0.0], [0.8, 0.6]]])
    target = torch.tensor([[[1.0, 0.0], [0.6, -0.8]]])
    result = partial_identity_matching(
        query,
        target,
        torch.eye(2)[None],
        torch.ones(1, 2, dtype=torch.bool),
        identity_threshold=0.1,
    )
    assert result.indices.tolist() == [[1, 0]]
    torch.testing.assert_close(result.coverage, torch.ones(1))


@pytest.mark.parametrize("separate_subjects", [False, True])
def test_target_person_cannot_satisfy_two_query_people(separate_subjects):
    query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target = query[:, :1]
    membership = torch.eye(2)[None] if separate_subjects else torch.ones(1, 1, 2)
    subjects = torch.ones(1, membership.shape[1], dtype=torch.bool)
    result = partial_identity_matching(query, target, membership, subjects)
    assert result.matched.sum() == 1
    torch.testing.assert_close(result.coverage, torch.tensor([0.5]))
    scores = binding_scores(
        query,
        target,
        query[:, None].expand(-1, subjects.shape[1], -1, -1),
        target,
        membership,
        subjects,
        matching=result,
    )
    # One complete ID+state match scores 2, one missing member scores -2.5.
    torch.testing.assert_close(scores["both"], torch.tensor([-0.25]))


def test_duplicate_target_detections_cannot_inflate_coverage():
    query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 1.0]] * 2])
    result = partial_identity_matching(
        query, query, torch.ones(1, 1, 2), torch.tensor([[True]]), target_boxes=boxes
    )
    assert result.matched.sum() == 1
    assert result.coverage.item() == 0.5


@pytest.mark.parametrize("target_people", [0, 2])
def test_null_is_penalized_and_empty_targets_have_finite_gradients(target_people):
    query = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[-1.0, 0.0]] * target_people]).reshape(1, target_people, 2)
    composed = query.detach()[:, None].clone().requires_grad_()
    semantic = target.clone().requires_grad_()
    result = binding_scores(
        query, target, composed, semantic, torch.ones(1, 1, 1), torch.tensor([[True]])
    )
    assert all(value.item() == -2.5 for value in result.values())
    result["both"].sum().backward()
    assert query.grad is None
    assert all(
        x.grad is not None and torch.isfinite(x.grad).all()
        for x in (composed, semantic)
    )


def test_semantic_scoring_cannot_switch_the_identity_assignment():
    query = torch.tensor([[[1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    composed = query[:, None]
    semantic = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    scores = binding_scores(
        query, target, composed, semantic, torch.ones(1, 1, 1), torch.tensor([[True]])
    )
    assert scores["identity"].item() == 1
    assert scores["semantic"].item() == 0
    assert scores["both"].item() == 1
    assert scores["none"].item() == 0


def test_active_subject_with_no_grounded_member_remains_missing():
    query = torch.tensor([[[1.0, 0.0]]])
    members = torch.tensor([[[1.0], [0.0]]])
    subjects = torch.tensor([[True, True]])
    match = partial_identity_matching(query, query, members, subjects)
    assert match.subject_coverage.tolist() == [[1.0, 0.0]]
    assert match.coverage.item() == 0.5
    scores = binding_scores(
        query,
        query,
        query[:, None].expand(-1, 2, -1, -1),
        query,
        members,
        subjects,
        matching=match,
    )
    torch.testing.assert_close(scores["both"], torch.tensor([-0.25]))


def test_assignment_matches_bruteforce_on_small_sets():
    generator = torch.Generator().manual_seed(82)
    for q, t in itertools.product(range(1, 5), range(1, 4)):
        query = F.normalize(torch.randn(1, q, 4, generator=generator), dim=-1)
        target = F.normalize(torch.randn(1, t, 4, generator=generator), dim=-1)
        members = torch.rand(1, 1, q, generator=generator) + 0.1
        result = partial_identity_matching(
            query, target, members, torch.tensor([[True]]), identity_threshold=0.1
        )
        similarity = (query @ target.transpose(1, 2))[0]
        weight = members[0, 0] / members.sum()

        def objective(assignment, weight=weight, similarity=similarity):
            return sum(
                float(weight[i] * (similarity[i, j] - 0.1))
                for i, j in enumerate(assignment)
                if j >= 0
            )

        feasible = []
        for assignment in itertools.product(range(-1, t), repeat=q):
            real = [j for j in assignment if j >= 0]
            if len(real) != len(set(real)):
                continue
            if any(
                j >= 0 and similarity[i, j] <= 0.1 for i, j in enumerate(assignment)
            ):
                continue
            feasible.append(objective(assignment))
        assert objective(result.indices[0].tolist()) == pytest.approx(
            max(feasible), abs=1e-6
        )


def _controlled_pair():
    torch.manual_seed(28)
    model = RCRModel(8, 2, 2).eval()
    with torch.no_grad():
        model.identity_head.proj.weight.zero_()
        model.identity_head.proj.weight[:, :2] = torch.eye(2)
        model.identity_head.proj.bias.zero_()
    members = torch.eye(2)[None]
    reference = torch.randn(1, 6, 8)
    mask = torch.cat((torch.ones(1, 2, dtype=torch.bool), members.flatten(1).bool()), 1)
    prior = torch.zeros(1, 6).masked_fill(~mask, -torch.inf)
    query = QueryEncoding(
        torch.zeros_like(members),
        torch.eye(2)[None],
        reference,
        mask,
        prior,
        reference[:, 2:].reshape(1, 2, 2, 8),
        members,
        torch.ones(1, 2, dtype=torch.bool),
    )
    persons = torch.randn(1, 3, 8)
    persons[0, :, :2] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.2, 0.8], [0.3, 0.0, 0.5, 0.8], [0.7, 0.0, 0.9, 0.8]]]
    )
    scene = torch.randn(1, 4, 8)
    return model, query, scene, persons, boxes


def test_unmatched_target_person_is_invisible_to_state_and_context():
    model, query, scene, persons, boxes = _controlled_pair()
    first = model.score_target(
        query, scene, persons, boxes, (2, 2), return_details=True
    )
    assert first["matched_indices"].tolist() == [[0, 1]]
    changed = persons.clone()
    changed[:, 2, 2:] = 1000
    changed.requires_grad_()
    actual = model.score_target(
        query, scene, changed, boxes, (2, 2), return_details=True
    )
    for key in ("score", "context", "binding", "coverage"):
        torch.testing.assert_close(first[key], actual[key])
    actual["score"].sum().backward()
    assert changed.grad[:, 2].count_nonzero() == 0
    assert changed.grad[:, :2].abs().sum() > 0
    assert model.identity_head.proj.weight.grad is None


def test_all_score_modes_share_one_assignment_and_aligned_context(monkeypatch):
    from rcr.proposed.nn import model as model_module

    model, query, scene, persons, boxes = _controlled_pair()
    original = model_module.partial_identity_matching
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "partial_identity_matching", counted)
    monkeypatch.setattr(
        model.reasoner.cross_attn,
        "forward",
        lambda *a, **k: pytest.fail("fresh target selection"),
    )
    result = model.score_target(
        query, scene, persons, boxes, (2, 2), return_components=True
    )
    assert set(result) == {"both", "identity", "semantic", "none"}
    assert calls == [1]


def test_missing_member_cannot_read_scene_as_a_substitute():
    model, query, scene, persons, boxes = _controlled_pair()
    details = model.score_target(
        query, scene, persons[:, :1], boxes[:, :1], (2, 2), return_details=True
    )
    assert details["matched_indices"].tolist() == [[0, -1]]
    assert details["coverage"].item() == 0.5
    assert details["subject_coverage"].tolist() == [[1.0, 0.0]]
    torch.testing.assert_close(details["gated_context"], details["context"] * 0.5)


def test_all_null_pair_has_no_context_rescue():
    model, query, scene, persons, boxes = _controlled_pair()
    persons[:, :, :2] = -1
    details = model.score_target(
        query, scene, persons, boxes, (2, 2), return_details=True
    )
    assert details["coverage"].item() == 0
    assert details["gated_context"].item() == 0
    assert details["binding"].item() == -2.5
    torch.testing.assert_close(details["score"], -2.5 * model.binding_scale[None])


def test_grounding_weights_cannot_be_lowered_by_retrieval_gradients():
    model, _, scene, persons, boxes = _controlled_pair()
    with torch.no_grad():
        model.grounding.score.weight.zero_()
        model.grounding.score.bias.fill_(5)
    query = model.encode_query(
        scene,
        persons[:, :2],
        boxes[:, :2],
        torch.randn(1, 1, 3, 8),
        torch.randn(1, 3, 8),
        torch.tensor([[1]]),
        (2, 2),
    )
    assert query.membership.count_nonzero() == 2
    assert not query.membership.requires_grad
    model.score_target(query, scene, persons, boxes, (2, 2)).sum().backward()
    assert model.grounding.score.weight.grad is None
    assert model.semantic_proj.weight.grad.abs().sum() > 0


def test_geometry_preserves_direction_and_masks_nulls():
    reasoner = FineReasoner(8, 2)
    with torch.no_grad():
        reasoner.relation_geo.weight.zero_()
        reasoner.relation_geo.weight[:, 0] = 1
    boxes = torch.tensor([[[0.0, 0.0, 0.2, 0.8], [0.6, 0.0, 0.8, 0.8]]])
    bias = reasoner._relation_bias(boxes, torch.ones(1, 2, dtype=torch.bool))
    assert (bias[:, 0, 1] > 0).all()
    torch.testing.assert_close(bias[:, 0, 1], -bias[:, 1, 0])
    masked = reasoner._relation_bias(boxes, torch.tensor([[True, False]]))
    assert masked[:, 1].count_nonzero() == masked[:, :, 1].count_nonzero() == 0


def test_relation_geometry_reaches_cls_score_and_receives_gradients():
    torch.manual_seed(19)
    reasoner = FineReasoner(8, 2).eval()
    query = torch.randn(1, 4, 8)
    target = torch.randn(1, 2, 8)
    boxes = torch.tensor([[[0.0, 0.0, 0.2, 0.8], [0.6, 0.0, 0.8, 0.8]]])
    score = reasoner(
        query,
        target,
        aligned_offset=2,
        aligned_boxes=boxes,
        matched_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    score.sum().backward()
    assert reasoner.relation_geo.weight.grad.abs().sum() > 0
    reversed_score = reasoner(
        query,
        target,
        aligned_offset=2,
        aligned_boxes=boxes.flip(1),
        matched_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert not torch.allclose(score, reversed_score, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "config",
    [
        {"selection_threshold": 0},
        {"identity_threshold": 2},
        {"missing_penalty": 2},
        {"duplicate_iou": 0},
        {"unknown": 1},
        {"missing_penalty": float("nan")},
    ],
)
def test_matching_config_rejects_invalid_settings(config):
    with pytest.raises(ValueError):
        matching_settings(config)


def test_retrieval_does_not_read_case_identities_or_positive_labels():
    from rcr.proposed.ranking import retrieve_rankings
    from tests.proposed.test_retrieval import _Cache, _sample, _TextEncoder, _Tokenizer

    torch.manual_seed(14)
    cache, encoder = _Cache(), _TextEncoder()
    model = RCRModel(8, 6, 2)
    sample = _sample()
    unlabeled = {
        key: value
        for key, value in sample.items()
        if key not in ("case_type", "positive_image_ids", "target_image_id")
    }
    unlabeled["subjects"] = [{"subject_id": 1}]

    def rank(row):
        return retrieve_rankings(
            [row],
            cache,
            _Tokenizer(),
            encoder,
            model,
            torch.device("cpu"),
            gallery_ids=cache.image_ids,
            top_m=2,
            fine_batch_size=2,
            identity_batch_size=2,
        )

    before, after = rank(sample), rank(unlabeled)
    for key in ("rankings", "fine_rankings", "coarse_rankings", "coarse_topm"):
        torch.testing.assert_close(before[key], after[key])
