"""Joint condition readout can access only identity-bound target evidence."""

import torch
from torch import nn

from rcr.proposed.nn.attention import AttentionBlock


class JointReasoner(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.evidence = nn.Linear(dim, dim)
        self.uncertainty = nn.Linear(4, dim)
        self.missing = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(
            [AttentionBlock(dim, heads, dropout) for _ in range(2)]
        )
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, query, evidence, p, null_mass):
        a = query.membership
        b, s, k = a.shape
        length = query.reference.shape[1] - s * k
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
        # Every member carries its role/reference, membership and matching uncertainty.
        features = torch.stack(
            (a, 1 - a, null_mass[:, None].expand_as(a), entropy[:, None].expand_as(a)),
            -1,
        )
        members = query.members + self.evidence(evidence) + self.uncertainty(features)
        tokens = torch.cat((query.reference[:, :length], members.flatten(1, 2)), 1)
        # One explicit missing marker per role, even for empty query person sets.
        mass = a.sum(-1)
        missing = mass <= 1e-8
        markers = query.roles + missing[..., None] * self.missing
        tokens = torch.cat((tokens, markers), 1)
        mask = torch.cat((query.reference_mask, query.subject_mask), 1)
        prior = torch.cat((query.prior, mass.new_zeros(b, s)), 1)
        for block in self.blocks:
            tokens = block(tokens, mask, prior)
        return self.score(tokens[:, 0]).squeeze(-1).float()
