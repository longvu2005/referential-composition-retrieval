"""Synthetic numerical contracts, without pretrained weights or quality claims."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from rcr.proposed.batch import build_supervision
from rcr.proposed.losses import (
    grounding_loss,
    identity_loss,
    matching_loss,
    retrieval_loss,
    score_candidates,
)
from rcr.proposed.nn.attention import local_scene
from rcr.proposed.nn.binding import EvidenceBinding
from rcr.proposed.nn.matching import partial_transport
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import rank_order
from rcr.proposed.sampling import (
    negative_exclusions,
    sample_candidates,
    training_batches,
)
from rcr.proposed.scores import coarse_scores, identity_score
from rcr.proposed.train import checkpoint_score


def visual(b=2, k=3, tokens=3):
    return dict(
        scene=torch.randn(b, 4, 8),
        persons=torch.randn(b, k, 8),
        clip_pooled=torch.randn(b, k, 8),
        clip_tokens=torch.randn(b, k, tokens, 8),
        boxes=torch.tensor([0.1, 0.2, 0.8, 0.9]).expand(b, k, 4).clone(),
        person_mask=torch.ones(b, k, dtype=torch.bool),
        clip_token_mask=torch.ones(b, k, tokens, dtype=torch.bool),
    )


def text(b=2, s=2):
    mentions = torch.zeros(b, s, 5, dtype=torch.bool)
    mentions[:, 0, 0] = True
    if s > 1:
        mentions[:, 1, 3] = True
    return dict(
        selections=torch.randn(b, s, 3, 8),
        selection_mask=torch.ones(b, s, 3, dtype=torch.bool),
        change=torch.randn(b, 5, 8),
        change_mask=torch.ones(b, 5, dtype=torch.bool),
        subject_ids=torch.arange(1, s + 1)[None].expand(b, -1),
        subject_mask=torch.ones(b, s, dtype=torch.bool),
        subject_token_mask=mentions,
    )


def model():
    return RCRModel(8, 8, 8, 8, 8, dim=16, identity_dim=4, num_heads=2, dropout=0)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(12)


def test_grounding_categorical_axis_group_and_background():
    m = model()
    v, txt = visual(), text()
    with torch.no_grad():
        m.grounding.subject_score[-1].weight.zero_()
        m.grounding.subject_score[-1].bias.fill_(3)
        m.grounding.background.weight.zero_()
        m.grounding.background.bias.fill_(-3)
    logits, a = m.grounding(v, txt, (2, 2))
    assert logits.shape == (2, 3, 3)
    assert (a.sum(1) <= 1 + 1e-6).all()
    assert (a[:, 0] > 0.4).all()  # One Subject can include every person.
    assert a[:, 0].sum() > 1
    v["person_mask"][0, 2] = False
    txt["subject_mask"][1, 1] = False
    _, a = m.grounding(v, txt, (2, 2))
    assert a[0, :, 2].count_nonzero() == a[1, 1].count_nonzero() == 0


@pytest.mark.parametrize("shape", [(3, 2), (6, 1), (0, 3), (3, 0)])
@pytest.mark.parametrize("zero", [False, True])
def test_transport_constraints_and_gradients(shape, zero):
    q, t = shape
    logits = torch.randn(2, q, t + 1, requires_grad=True)
    mass = torch.rand(2, q) * (not zero)
    real = torch.ones(2, t, dtype=torch.bool)
    if t:
        real[0] = False
    x, d = partial_transport(logits, mass, real)
    torch.testing.assert_close(x.sum(-1), mass, atol=1e-5, rtol=1e-5)
    assert (x[..., 1:].sum(1) <= 1.00001).all()
    assert d["capacity_violation"] <= 1e-5
    torch.testing.assert_close(x[0, :, 0], mass[0])
    if q:
        (x * torch.randn_like(x)).sum().backward()
        assert torch.isfinite(logits.grad).all()
        if t and not zero:
            assert logits.grad[1].abs().sum() > 0


def test_transport_matches_independent_constrained_objective():
    ell = torch.tensor([[[-0.8, -0.2, -0.5], [-0.8, -0.3, -0.6], [-0.8, -0.1, -0.2]]])
    mass = torch.tensor([[0.9, 0.8, 0.7]])
    x, _ = partial_transport(
        ell, mass, torch.ones(1, 2, dtype=torch.bool), tau=0.35, tolerance=1e-6
    )

    def objective(flat):
        y = flat.reshape(3, 3)
        return -float((y * ell.numpy()[0]).sum() - 0.35 * (y * np.log(y)).sum())

    constraints = [
        {"type": "eq", "fun": lambda flat: flat.reshape(3, 3).sum(1) - mass.numpy()[0]},
        {"type": "ineq", "fun": lambda flat: 1 - flat.reshape(3, 3)[:, 1:].sum(0)},
    ]
    solved = minimize(
        objective,
        np.repeat(mass.numpy()[0, :, None] / 3, 3, axis=1).flatten(),
        bounds=[(1e-10, None)] * 9,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    assert solved.success
    np.testing.assert_allclose(x.numpy().reshape(-1), solved.x, atol=2e-5)
    # Finite differences through the active capacity regime (FP32 tolerance).
    ell = ell.requires_grad_()
    probe = torch.arange(9.0).reshape_as(ell)
    score = (
        partial_transport(
            ell, mass, torch.ones(1, 2, dtype=torch.bool), tau=0.35, tolerance=1e-6
        )[0]
        * probe
    ).sum()
    analytic = torch.autograd.grad(score, ell)[0][0, 0, 1]
    high, low = ell.detach().clone(), ell.detach().clone()
    high[0, 0, 1] += 0.002
    low[0, 0, 1] -= 0.002

    def evaluate(e):
        return (
            partial_transport(
                e, mass, torch.ones(1, 2, dtype=torch.bool), tau=0.35, tolerance=1e-6
            )[0]
            * probe
        ).sum()

    numeric = (evaluate(high) - evaluate(low)) / 0.004
    torch.testing.assert_close(analytic, numeric, atol=0.002, rtol=0.02)


def test_transport_rejects_unconverged_coupling():
    with pytest.raises(RuntimeError, match="did not converge"):
        partial_transport(
            torch.tensor([[[-5.0, 0.0], [-5.0, 0.0], [-5.0, 0.0]]]),
            torch.ones(1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            max_iterations=1,
        )


@pytest.mark.parametrize("q,t", [(3, 2), (0, 2), (3, 0), (0, 0)])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_empty_padding_amp_are_finite(q, t, padding, amp):
    m = model()
    qv, tv, txt = visual(k=q), visual(k=t), text()
    txt["subject_mask"][1, 1] = False
    if padding:
        for v in (qv, tv):
            v["person_mask"].zero_()
            v["clip_token_mask"].zero_()
            for key in ("persons", "clip_pooled", "boxes", "clip_tokens"):
                v[key].fill_(float("nan"))
        txt["change_mask"][1].zero_()
        txt["selection_mask"][1].zero_()
        txt["change"][1].fill_(float("nan"))
        txt["selections"][1].fill_(float("nan"))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        out = m(qv, txt, tv, (2, 2))
    assert torch.isfinite(out["scores"]).all()
    if q == 0 or padding:
        assert (out["identity_score"] < -18).all()
    out["scores"].sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_rank_gradient_ownership_and_match_learning():
    m = model()
    qv, tv, txt = visual(), visual(), text()
    for v in (qv, tv, txt):
        for value in v.values():
            if value.is_floating_point():
                value.requires_grad_()
    result = m(qv, txt, tv, (2, 2))
    retrieval_loss(result["scores"][None], torch.tensor([[True, False]])).backward()
    for module in (m.grounding, m.identity_head):
        assert all(p.grad is None for p in module.parameters())
    for module in (m.composition, m.matching, m.binding, m.reasoner):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(v.grad is None for values in (qv, tv, txt) for v in values.values())
    m.zero_grad(set_to_none=True)
    query = m.encode_query(qv, txt, (2, 2))
    out = m.score_target(query, tv, (2, 2))
    positive = torch.zeros_like(out["transport"], dtype=torch.bool)
    positive[..., 1] = True
    matching_loss(
        out["transport"], positive, torch.ones(2, 3, dtype=torch.bool)
    ).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad) for p in m.matching.parameters()
    )
    assert m.matching.alpha > 0


def test_member_permutation_invariance_target_permutation_invariance():
    m = model().eval()
    qv, tv, txt = visual(), visual(), text()
    expected = m(qv, txt, tv, (2, 2))["scores"]
    for original, other in ((qv, tv), (tv, qv)):
        shuffled = {
            k: v if k == "scene" else v[:, [2, 0, 1]] for k, v in original.items()
        }
        result = (
            m(shuffled, txt, other, (2, 2))
            if original is qv
            else m(other, txt, shuffled, (2, 2))
        )
        torch.testing.assert_close(result["scores"], expected, atol=2e-5, rtol=2e-5)


def test_binding_preserves_null_mass_and_person_token_count():
    bind = EvidenceBinding(8, 8, 8, 2)
    members, tv = torch.randn(1, 1, 2, 8), visual(b=1, k=2, tokens=1)
    p = torch.tensor([[[0.8, 0.15, 0.05], [0.2, 0.3, 0.5]]])
    evidence, null = bind(members, tv, p, (2, 2))
    torch.testing.assert_close(null, p[..., 0])
    # More duplicate visual tokens for person 1 may change its within-person
    # summary, but must not change person/null allocation. Constant K/V proves it.
    with torch.no_grad():
        bind.value.weight.zero_()
        bind.value.bias.fill_(2.0)
        bind.null.fill_(5.0)
        bind.output.weight.copy_(torch.eye(8))
        bind.output.bias.zero_()
    one, _ = bind(members, tv, p, (2, 2))
    tv["clip_tokens"] = tv["clip_tokens"].expand(-1, -1, 7, -1)
    tv["clip_token_mask"] = torch.tensor([[[True] * 7, [True] + [False] * 6]])
    many, _ = bind(members, tv, p, (2, 2))
    torch.testing.assert_close(one, many)
    torch.testing.assert_close(one[0, 0, :, 0], 2 * (1 - p[0, :, 0]) + 5 * p[0, :, 0])
    assert evidence.shape == (1, 1, 2, 8)


def test_roi_matches_letterboxed_patch_centers():
    scene = torch.arange(4.0).reshape(1, 4, 1)
    full = local_scene(
        scene, torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), (2, 2), size=2, expansion=1
    )
    torch.testing.assert_close(full.flatten(), torch.arange(4.0))


def test_multi_positive_loss_and_unknown_grounding():
    scores = torch.tensor([[2.0, 1.0, -1.0, float("nan")]], requires_grad=True)
    positive = torch.tensor([[True, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    loss = retrieval_loss(scores, positive, valid, temperature=0.5)
    expected = -F.log_softmax(scores[:, :3] / 0.5, -1)[0, :2].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert scores.grad[0, 3] == 0
    logits = torch.randn(1, 3, 3, requires_grad=True)
    ground = grounding_loss(logits, torch.tensor([[1, 0, -1]]))
    ground.backward()
    assert logits.grad[..., 2].count_nonzero() == 0
    features = F.normalize(torch.randn(4, 6), dim=-1)
    _, active, total = identity_loss(features, torch.tensor([0, 0, 1, -1]))
    assert active == 2 and total == 3


def test_training_score_matches_pair_inference():
    m = model().eval()
    qv, txt, tv = visual(), text(), visual(b=6)
    inputs = {
        "query": {"visual": qv, "text": txt},
        "target": {k: v.reshape(2, 3, *v.shape[1:]) for k, v in tv.items()},
    }
    query, scores, _, _ = score_candidates(m, inputs, (2, 2), pair_batch_size=2)
    direct = m.score_target(query.repeat_candidates(3), tv, (2, 2))["scores"].reshape(
        2, 3
    )
    torch.testing.assert_close(scores, direct)


def sample(sid="a"):
    return {
        "sample_id": sid,
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "target_image_id": "p",
        "positive_image_ids": ["p"],
        "subjects": [{"subject_id": 1, "identity_ids": ["1"]}],
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "Subject 1 is standing beside Subject 1",
    }


def test_supervision_background_unknown_misses_dedup():
    row = sample()
    labels = {
        "q": {"identity_ids": ["1", "2", None]},
        "p": {"identity_ids": ["1", None], "gt_ids": ["1"], "complete": False},
        "miss": {"identity_ids": ["2"], "gt_ids": ["1", "2"], "complete": False},
        "unknown": {"identity_ids": [None], "gt_ids": ["1"], "complete": False},
        "absent": {"identity_ids": ["2"], "gt_ids": ["2"], "complete": False},
    }
    target = visual(b=4, k=2)
    result = build_supervision(
        [row],
        [["p", "miss", "unknown", "absent"]],
        visual(b=1),
        target,
        labels,
        {"1": 0, "2": 1},
    )
    assert result["grounding"].tolist() == [[1, 0, -1]]
    assert result["match_mask"][0, :, 0].tolist() == [True, True, False, False]
    assert result["match_positive"][0, 0, 0, 1]
    assert result["match_positive"][0, 1, 0, 0]
    assert result["query_ids"][0, 0] == result["target_ids"][0, 0, 0]
    labels["p"]["identity_ids"] = ["1", "1"]
    result = build_supervision(
        [row, row], [["p"], ["p"]], visual(), visual(b=2, k=2), labels, {"1": 0, "2": 1}
    )
    assert not result["query_unique"][1].any() and not result["target_unique"][1].any()
    assert result["match_positive"][0, 0, 0].tolist() == [False, True, True]


def test_case_cannot_change_sampling_exclusions_or_checkpoint_selection():
    rows = [sample("a"), {**sample("b"), "positive_image_ids": ["n"]}]
    changed = [{**s, "case_type": "RELATIONAL"} for s in rows]
    assert negative_exclusions(rows) == negative_exclusions(changed)

    def batches(samples):
        return [
            [s["sample_id"] for s in b]
            for b in training_batches(samples, 1, torch.Generator().manual_seed(2))
        ]

    assert batches(rows) == batches(changed)
    gallery = ["q", "p", "n", "same", "wrong1", "wrong2"]
    kwargs = dict(
        identity_pools={"a": ["p", "same"], "b": ["n", "same"]},
        identity_fraction=0.5,
        positives_per_query=1,
    )
    a = sample_candidates(rows, gallery, 3, torch.Generator().manual_seed(1), **kwargs)
    b = sample_candidates(
        changed, gallery, 3, torch.Generator().manual_seed(1), **kwargs
    )
    assert a == b
    assert (
        checkpoint_score(
            {"overall": {"full_map": 0.2}, "by_case": {"x": {"full_map": 0.9}}}
        )
        == 0.2
    )


def test_rank_tail_and_missing_policy():
    final, coarse = rank_order(
        torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
        torch.tensor([1, 2]),
        torch.tensor([-2.0, -1.0]),
        0,
    )
    assert final.tolist() == [2, 1, 3, 4] and coarse.tolist() == [1, 2, 3, 4]
    result = identity_score(
        torch.empty(2, 0),
        torch.empty(2, 2, 0),
        torch.tensor([[True, True], [False, False]]),
    )
    torch.testing.assert_close(result, torch.full((2,), math.log(1e-8)))
    m = model()
    query = m.encode_query(visual(b=1), text(b=1), (2, 2))
    result = coarse_scores(
        query, torch.empty(3, 0, 4), torch.empty(3, 0, dtype=torch.bool), m.matching
    )
    assert torch.isfinite(result).all() and (result < -18).all()


def test_tiny_condition_overfit():
    m = model().eval()
    qv, txt = visual(b=1, k=1), text(b=1, s=1)
    target = visual(b=2, k=1)
    # Identical identity evidence: only target condition features distinguish them.
    target["persons"][1] = target["persons"][0]
    optimizer = torch.optim.Adam(
        [
            p
            for n, p in m.named_parameters()
            if not n.startswith(("identity_head.", "grounding."))
        ],
        lr=0.01,
    )
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        query = m.encode_query(qv, txt, (2, 2))
        scores = m.score_target(query.repeat_candidates(2), target, (2, 2))["scores"]
        loss = retrieval_loss(
            scores[None], torch.tensor([[True, False]]), temperature=1
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 and losses[-1] < losses[0] / 5
