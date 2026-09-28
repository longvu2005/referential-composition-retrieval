"""Cheap soft identity retrieval for gallery shortlisting."""

import torch
from torch import Tensor


def coarse_scores(
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
