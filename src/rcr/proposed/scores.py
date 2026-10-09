"""The same finite identity aggregation in coarse and fine retrieval."""

import math

import torch

EPS = 1e-8


def identity_score(confidence, membership, subject_mask):
    """confidence [B,Kq], memberships [B,S,Kq]; empty Subjects score log(EPS)."""
    with torch.autocast(confidence.device.type, enabled=False):
        a = membership.detach().float()
        mass = a.sum(-1)
        per_subject = (a * confidence.float().clamp(EPS, 1).log()[:, None]).sum(-1) / (
            mass + EPS
        )
        per_subject = torch.where(mass > EPS, per_subject, math.log(EPS))
        per_subject = per_subject.masked_fill(~subject_mask, 0)
        count = subject_mask.sum(-1)
        score = per_subject.sum(-1) / count.clamp_min(1)
        return torch.where(count > 0, score, math.log(EPS))


def coarse_scores(query, gallery_identity, gallery_mask, matcher):
    """One query versus G images: cheap maximum identity confidence, no solver."""
    g, t = gallery_mask.shape
    identity = query.identity.expand(g, -1, -1)
    with torch.autocast(identity.device.type, enabled=False):
        if t:
            similarity = identity.float() @ gallery_identity.float().transpose(-1, -2)
            confidence = (
                matcher.confidence(similarity)
                .masked_fill(~gallery_mask[:, None], 0)
                .amax(-1)
            )
        else:
            confidence = identity.new_zeros(g, identity.shape[1])
        return identity_score(
            confidence,
            query.membership.expand(g, -1, -1),
            query.subject_mask.expand(g, -1),
        )
