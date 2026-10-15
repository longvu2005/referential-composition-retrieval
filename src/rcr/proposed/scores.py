"""Coarse scores and fine scores on one shared partial person assignment."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from rcr.proposed.matching import (
    PersonMatching,
    gather_matched,
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


def fafa_binding_scores(
    query: Tensor,
    targets: Tensor,
    *,
    matching: PersonMatching,
    k: int = 6,
    missing_penalty: float = 1.1,
) -> Tensor:
    """Native FDA on identity-assigned pairs, pooled equally across Subjects.

    query [B,S,Kq,D] and targets [B,Kt,L,D] come from frozen FAFA. The
    learned Composition supplies reference tokens only to the Context Reasoner.
    Null members and empty active Subjects keep their contribution to the mean.
    """
    if not math.isfinite(missing_penalty) or missing_penalty <= 1:
        raise ValueError("missing_penalty must exceed 1 for normalized FDA scores")
    if type(k) is not int or k < 1:
        raise ValueError("fda_k must be a positive integer")
    if (
        query.ndim != 4
        or targets.ndim != 4
        or query.shape[:3] != matching.membership.shape
        or query.shape[0] != targets.shape[0]
        or query.shape[-1] != targets.shape[-1]
        or targets.shape[2] < 1
    ):
        raise ValueError("incompatible FAFA query/target/matching shapes")
    with torch.no_grad(), torch.autocast(query.device.type, enabled=False):
        member = matching.membership
        selected = member > 0
        length, dim = targets.shape[-2:]
        # Gather first: unused/padded detections (even NaN/Inf) cannot enter FDA.
        flat = targets.detach().float().flatten(2)
        aligned = gather_matched(flat, matching).reshape(
            targets.shape[0], query.shape[2], length, dim
        )
        query = query.detach().float().masked_fill(~selected[..., None], 0)
        similarity = torch.einsum(
            "bsqd,bqld->bsql",
            F.normalize(query, dim=-1),
            F.normalize(aligned, dim=-1),
        )
        fda = similarity.topk(min(k, length), dim=-1).values.mean(-1)
        denominator = member.sum(-1)
        weights = member / denominator[..., None].clamp_min(1e-6)
        active = matching.subjects

        value = fda.masked_fill(~matching.matched[:, None], -missing_penalty)
        per_subject = (value * weights).sum(-1)
        per_subject = per_subject.masked_fill(denominator == 0, -missing_penalty)
        per_subject = per_subject.masked_fill(~active, 0)
        result = per_subject.sum(-1) / active.sum(-1).clamp_min(1)
        return result.masked_fill(~active.any(-1), -missing_penalty)
