"""Coarse scores and explicit same-person fine binding."""

import math

import torch
from torch import Tensor


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


def coarse_scores(
    query_identity: Tensor,
    grounding_logits: Tensor,
    gallery_identity: Tensor,
    gallery_mask: Tensor | None = None,
    query_mask: Tensor | None = None,
    subject_mask: Tensor | None = None,
    eps: float = 1e-6,
    *,
    query_state: Tensor | None = None,
    gallery_state: Tensor | None = None,
    beta: float = 0.0,
) -> Tensor:
    """S_id + beta * dot(z_text, z_image), with normalized state inputs.

    State inputs are [Ds] and [G,Ds]. beta=0 preserves the original identity
    path exactly, including its empty-query/empty-target policy.
    """
    scores = identity_scores(
        query_identity,
        grounding_logits,
        gallery_identity,
        gallery_mask,
        query_mask,
        subject_mask,
        eps,
    )
    if beta == 0:
        return scores
    if query_state is None or gallery_state is None:
        raise ValueError("nonzero beta requires both text and image state embeddings")
    if (
        query_state.ndim != 1
        or gallery_state.ndim != 2
        or gallery_state.shape != (scores.shape[0], query_state.shape[0])
    ):
        raise ValueError("state embedding shapes must be [Ds] and [G,Ds]")
    return scores + beta * (gallery_state @ query_state)


def combine_scores(
    identity: Tensor | None,
    state: Tensor | None,
    *,
    mode: str = "identity_state",
    beta: float = 0.4,
    normalization: str = "zscore",
    exclude_index: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Combine raw scores AFTER scoring the complete split gallery for one query.

    Z-scores use population standard deviation on the same eligible images for
    both branches. Self and non-finite scores do not enter the statistics.
    Identity modes retain the missing-target-person policy (-inf); state_only
    ignores identity entirely, including that mask. Constant branches become 0.
    """
    if mode not in ("identity_only", "state_only", "identity_state"):
        raise ValueError(f"unknown coarse mode: {mode}")
    if normalization not in ("none", "zscore"):
        raise ValueError(f"unknown coarse normalization: {normalization}")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("coarse_beta must be finite and nonnegative")
    branches = (
        [(state, 1.0)]
        if mode == "state_only"
        else [(identity, 1.0)]
        + ([(state, beta)] if mode == "identity_state" and beta != 0 else [])
    )
    if any(scores is None for scores, _ in branches):
        raise ValueError(f"{mode} requires its active score branches")
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
        if normalization == "zscore":
            values = (values - values.mean()) / values.std(unbiased=False).clamp_min(
                eps
            )
        result[valid] += weight * values
    return result


BINDING_MODES = ("both", "identity", "semantic", "none")


def binding_scores(
    query_identity: Tensor,
    target_identity: Tensor,
    composed: Tensor,
    target_semantic: Tensor,
    membership: Tensor,
    subject_mask: Tensor,
    target_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return [B] scores; max follows pairwise fusion, then soft set pooling.

    composed/membership are [B,S,Kq,D] / [B,S,Kq]. Identities are [B,K,Di].
    Identity is detached here, at the only identity input to dual fine scoring.
    Empty person sets contribute zero. This optimistic match permits reuse of a
    target person; it is not a one-to-one assignment or an all-members guarantee.
    """
    with torch.autocast(composed.device.type, enabled=False):
        zero = composed.float().sum(dim=(1, 2, 3)) * 0
        if composed.shape[2] == 0 or target_semantic.shape[1] == 0:
            return dict.fromkeys(BINDING_MODES, zero)
        identity = torch.einsum(
            "bqd,btd->bqt",
            query_identity.detach().float(),
            target_identity.detach().float(),
        )[:, None]
        semantic = torch.einsum(
            "bsqd,btd->bsqt", composed.float(), target_semantic.float()
        )
        weights = membership.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        active = subject_mask.bool()

        def pool(scores):
            if target_mask is not None:
                scores = scores.masked_fill(
                    ~target_mask[:, None, None].bool(), -torch.inf
                )
            best = scores.amax(-1)
            # All-masked targets must not produce -inf * zero or NaN gradients.
            best = best.masked_fill(~torch.isfinite(best), 0)
            per_subject = (best * weights).sum(-1).masked_fill(~active, 0)
            return per_subject.sum(-1) / active.sum(-1).clamp_min(1)

        return {
            "both": pool(identity + semantic),
            "identity": pool(identity.expand_as(semantic)),
            "semantic": pool(semantic),
            "none": zero,
        }
