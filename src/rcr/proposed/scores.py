"""Coarse scores and fine scores on one shared partial person assignment."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from rcr.proposed.matching import (
    PersonMatching,
    gather_matched,
    partial_identity_matching,
)


def identity_scores(
    query_identity: Tensor,
    grounding_logits: Tensor,
    gallery_identity: Tensor,
    gallery_mask: Tensor | None = None,
    query_mask: Tensor | None = None,
    subject_mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Score gallery images from soft query-person membership and identity similarity.

    query_identity: [Kq, D]
    grounding_logits: [S, Kq]
    gallery_identity: [G, Kt, D]
    gallery_mask: [G, Kt], True for valid target persons
    """

    return identity_scores_batched(
        query_identity[None],
        grounding_logits[None],
        gallery_identity,
        gallery_mask,
        None if query_mask is None else query_mask[None],
        None if subject_mask is None else subject_mask[None],
        eps,
    )[0]


def identity_scores_batched(
    query_identity: Tensor,
    grounding_logits: Tensor,
    gallery_identity: Tensor,
    gallery_mask: Tensor | None = None,
    query_mask: Tensor | None = None,
    subject_mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Score [B,Kq,D] queries against one [G,Kt,D] gallery chunk, returning [B,G]."""
    b, g = query_identity.shape[0], gallery_identity.shape[0]
    if gallery_identity.shape[1] == 0:
        return gallery_identity.new_full((b, g), -torch.inf)
    if query_identity.shape[1] == 0:
        scores = gallery_identity.new_zeros(b, g)
        return (
            scores.masked_fill(~gallery_mask.any(-1)[None], -torch.inf)
            if gallery_mask is not None
            else scores
        )
    weight = grounding_logits.sigmoid()
    if query_mask is not None:
        weight = weight * query_mask[:, None].to(weight.dtype)
    if subject_mask is not None:
        weight = weight * subject_mask[:, :, None].to(weight.dtype)
    similarity = torch.einsum("bqd,gkd->bgqk", query_identity, gallery_identity)
    if gallery_mask is not None:
        similarity = similarity.masked_fill(~gallery_mask[None, :, None, :], -torch.inf)
    best = similarity.amax(dim=-1)
    empty = ~torch.isfinite(best).any(dim=-1)
    best = best.masked_fill(empty[..., None], 0)
    per_subject = (best[:, :, None, :] * weight[:, None]).sum(-1) / weight.sum(
        -1
    ).clamp_min(eps)[:, None]
    if subject_mask is not None:
        scores = per_subject.sum(-1) / subject_mask.sum(-1).clamp_min(1)[:, None]
    else:
        scores = per_subject.mean(-1)
    return scores.masked_fill(empty, -torch.inf)


def combine_scores(
    identity: Tensor,
    state: Tensor | None,
    *,
    beta: float = 0.4,
    exclude_index: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Combine raw scores AFTER scoring the complete split gallery for one query.

    Z-scores use population standard deviation on the same eligible images for
    both branches. Self and non-finite scores do not enter the statistics.
    Missing target persons retain the identity policy (-inf).
    Constant branches become 0. beta=0 skips the state branch.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("coarse_beta must be finite and nonnegative")
    branches = [(identity, 1.0)] + ([(state, beta)] if beta != 0 else [])
    if any(scores is None for scores, _ in branches):
        raise ValueError("coarse scoring requires its active score branches")
    first = branches[0][0]
    valid = torch.ones_like(first, dtype=torch.bool)
    for scores, _ in branches:
        valid &= torch.isfinite(scores)
    if exclude_index is not None:
        valid[exclude_index] = False
    result = torch.full_like(first, -torch.inf, dtype=torch.float32)
    if not valid.any():
        return result
    result[valid] = 0
    for scores, weight in branches:
        values = scores[valid].float()
        values = (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)
        result[valid] += weight * values
    return result


def binding_scores(
    query_identity: Tensor,
    target_identity: Tensor,
    composed: Tensor,
    target_semantic: Tensor,
    membership: Tensor,
    subject_mask: Tensor,
    target_mask: Tensor | None = None,
    *,
    matching: PersonMatching | None = None,
    identity_threshold: float = 0.2,
    missing_penalty: float = 2.5,
) -> Tensor:
    """Pool identity + semantic cosines over the same partial identity assignment.

    Pool all predicted members, including nulls. An active Subject with no query
    member also contributes the null penalty rather than disappearing.
    No gradient flows through identity, assignment or membership weights.
    """
    if not math.isfinite(missing_penalty) or missing_penalty <= 2:
        raise ValueError("missing_penalty must exceed 2 for normalized cosine scores")
    if matching is None:
        matching = partial_identity_matching(
            query_identity,
            target_identity,
            membership,
            subject_mask,
            target_mask,
            identity_threshold=identity_threshold,
        )
    with torch.autocast(composed.device.type, enabled=False):
        member = matching.membership
        selected = member > 0
        aligned_id = gather_matched(target_identity.detach().float(), matching)
        query_id = (
            query_identity.detach().float().masked_fill(~selected.any(1)[..., None], 0)
        )
        identity = (
            F.normalize(query_id, dim=-1) * F.normalize(aligned_id, dim=-1)
        ).sum(-1)[:, None]
        aligned_semantic = F.normalize(
            gather_matched(target_semantic.float(), matching), dim=-1
        )
        composed = composed.float().masked_fill(~selected[..., None], 0)
        semantic = (F.normalize(composed, dim=-1) * aligned_semantic[:, None]).sum(-1)
        denominator = member.sum(-1)
        weights = member / denominator[..., None].clamp_min(1e-6)
        active = matching.subjects

        value = (identity + semantic).masked_fill(
            ~matching.matched[:, None], -missing_penalty
        )
        per_subject = (value * weights).sum(-1)
        per_subject = per_subject.masked_fill(denominator == 0, -missing_penalty)
        per_subject = per_subject.masked_fill(~active, 0)
        result = per_subject.sum(-1) / active.sum(-1).clamp_min(1)
        return result.masked_fill(~active.any(-1), -missing_penalty)
