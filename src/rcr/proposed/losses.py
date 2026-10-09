"""Grounding, identity, matching and multi-positive ranking losses."""

import torch
import torch.nn.functional as F


def grounding_loss(logits, labels):
    valid = labels >= 0
    if not valid.any():
        return logits.new_zeros(())
    return F.cross_entropy(logits.float().transpose(1, 2)[valid], labels[valid])


def identity_loss(identity, labels, temperature=0.1):
    keep = labels >= 0
    identity, labels = F.normalize(identity[keep].float(), dim=-1), labels[keep]
    if len(identity) < 2:
        return identity.new_zeros(()), 0, len(identity)
    self_mask = torch.eye(len(identity), dtype=torch.bool, device=identity.device)
    positive = (labels[:, None] == labels[None]) & ~self_mask
    if temperature <= 0:
        raise ValueError("identity temperature must be positive")
    with torch.autocast(identity.device.type, enabled=False):
        logits = (identity @ identity.T / temperature).masked_fill(
            self_mask, -torch.inf
        )
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    count = positive.sum(-1)
    active = count > 0
    loss = -(log_prob.masked_fill(~positive, 0).sum(-1) / count.clamp_min(1))
    return (
        loss[active].mean() if active.any() else identity.new_zeros(()),
        int(active.sum()),
        len(identity),
    )


def retrieval_loss(scores, positive_mask, valid_mask=None, temperature=0.1):
    """Average positive log-softmax, with a complete valid-candidate denominator."""
    if temperature <= 0:
        raise ValueError("ranking temperature must be positive")
    valid = torch.ones_like(positive_mask) if valid_mask is None else valid_mask.bool()
    positive = positive_mask.bool() & valid
    active = positive.any(-1)
    if not active.any():
        return scores.masked_fill(~valid, 0).sum() * 0
    logits = (
        scores[active].float().masked_fill(~valid[active], -torch.inf) / temperature
    )
    log_prob = logits - torch.logsumexp(logits, -1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive[active], 0).sum(-1) / positive[active].sum(-1)
    ).mean()


def matching_loss(p, positive, mask):
    if not mask.any():
        return p.sum() * 0
    probability = (p * positive).sum(-1)
    return -probability[mask].clamp_min(1e-8).log().mean()


def score_candidates(model, inputs, patch_hw, pair_batch_size=16):
    """One scoring implementation shared by training and pairwise inference."""
    query = model.encode_query(
        inputs["query"]["visual"], inputs["query"]["text"], patch_hw
    )
    target = inputs["target"]
    b, c = target["scene"].shape[:2]
    repeated = query.repeat_candidates(c)
    from dataclasses import fields

    from rcr.proposed.nn.model import QueryEncoding

    scores, coupling, diagnostics = [], [], []
    for start in range(0, b * c, pair_batch_size):
        stop = start + pair_batch_size
        local_query = QueryEncoding(
            **{f.name: getattr(repeated, f.name)[start:stop] for f in fields(repeated)}
        )
        output = model.score_target(
            local_query,
            {k: v.flatten(0, 1)[start:stop] for k, v in target.items()},
            patch_hw,
        )
        scores.append(output["scores"])
        coupling.append(output["transport"])
        diagnostics.append(output["diagnostics"])
    return (
        query,
        torch.cat(scores).reshape(b, c),
        torch.cat(coupling).reshape(b, c, *coupling[0].shape[1:]),
        diagnostics,
    )


def compute_loss(
    model,
    batch,
    patch_hw,
    *,
    warmup=False,
    identity_weight=0.1,
    matching_weight=1.0,
    retrieval_weight=1.0,
    identity_temperature=0.1,
    rank_temperature=0.1,
    pair_batch_size=16,
):
    inputs, labels = batch["inputs"], batch["supervision"]
    # Warmup avoids transport/composition/reasoning altogether.
    if warmup:
        from rcr.proposed.nn.model import clean_visual

        visual = clean_visual(inputs["query"]["visual"])
        logits, _ = model.grounding(visual, inputs["query"]["text"], patch_hw)
        query_identity = model.encode_identity(visual["persons"])
        scores = logits.new_zeros(labels["positive_mask"].shape)
        rank = match = logits.new_zeros(())
        diagnostics = []
    else:
        query, scores, p, diagnostics = score_candidates(
            model, inputs, patch_hw, pair_batch_size
        )
        logits, query_identity = query.logits, query.identity
        match = matching_loss(
            p,
            labels["match_positive"],
            labels["match_mask"] & labels["candidate_mask"][..., None],
        )
        rank = retrieval_loss(
            scores, labels["positive_mask"], labels["candidate_mask"], rank_temperature
        )
    ground = grounding_loss(logits, labels["grounding"])
    q_keep = labels["query_unique"] & inputs["query"]["visual"]["person_mask"]
    t_keep = (
        labels["target_unique"]
        & inputs["target"]["person_mask"]
        & labels["candidate_mask"][..., None]
    )
    target_identity = model.encode_identity(inputs["target"]["persons"][t_keep])
    identity, active, anchors = identity_loss(
        torch.cat((query_identity[q_keep], target_identity)),
        torch.cat((labels["query_ids"][q_keep], labels["target_ids"][t_keep])),
        identity_temperature,
    )
    total = (
        ground
        + (identity_weight * identity if identity_weight else identity.detach() * 0)
        + matching_weight * match
        + retrieval_weight * rank
    )
    parts = {
        "grounding": ground.detach().item(),
        "identity": identity.detach().item(),
        "matching": match.detach().item(),
        "retrieval": rank.detach().item(),
        "identity_active_anchors": active,
        "identity_anchors": anchors,
        "match_supervised_rows": int(labels["match_mask"].sum()),
        "grounding_known_persons": int((labels["grounding"] >= 0).sum()),
    }
    for key in ("row_residual", "capacity_violation", "dual_residual", "iterations"):
        parts[f"transport_{key}"] = max((d[key] for d in diagnostics), default=0)
    return total, parts
