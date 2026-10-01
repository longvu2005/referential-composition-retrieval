"""Cheap identity plus global state retrieval for gallery shortlisting."""

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

    # Empty query policy: no identity evidence, so every gallery item ties at
    # zero except images with no detected target people (ranked last).
    if gallery_identity.shape[1] == 0:
        return gallery_identity.new_full((gallery_identity.shape[0],), -torch.inf)
    if query_identity.shape[0] == 0:
        scores = gallery_identity.new_zeros(gallery_identity.shape[0])
        return (
            scores.masked_fill(~gallery_mask.any(-1), -torch.inf)
            if gallery_mask is not None
            else scores
        )
    weight = grounding_logits.sigmoid()
    if query_mask is not None:
        weight = weight * query_mask[None].to(weight.dtype)
    if subject_mask is not None:
        weight = weight * subject_mask[:, None].to(weight.dtype)
    similarity = torch.einsum("qd,gkd->gqk", query_identity, gallery_identity)

    if gallery_mask is not None:
        similarity = similarity.masked_fill(~gallery_mask[:, None, :], -torch.inf)

    best = similarity.amax(dim=-1)
    empty = ~torch.isfinite(best).any(dim=-1)
    best = best.masked_fill(empty[:, None], 0)
    per_subject = (best[:, None, :] * weight[None]).sum(dim=-1) / weight.sum(
        dim=-1
    ).clamp_min(eps)[None]
    if subject_mask is not None:
        scores = per_subject.sum(-1) / subject_mask.sum().clamp_min(1)
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
