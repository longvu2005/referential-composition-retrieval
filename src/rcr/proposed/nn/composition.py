"""Full-condition composition retaining every Subject/person and text token."""

import torch
from torch import nn

from rcr.proposed.nn.attention import AttentionBlock, geometry, position_encoding


class StructuredComposition(nn.Module):
    def __init__(self, clip_dim, text_dim, dim, heads, max_subjects=16, dropout=0.0):
        super().__init__()
        self.person = nn.Linear(clip_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.box = nn.Linear(4, dim)
        self.role = nn.Embedding(max_subjects, dim)
        self.sentinel = nn.Parameter(torch.zeros(1, 1, dim))
        self.block = AttentionBlock(dim, heads, dropout)

    def forward(self, visual, text, membership):
        b, s, k = membership.shape
        subjects = text["subject_mask"]
        if (text["subject_ids"] > self.role.num_embeddings).any():
            raise ValueError("text Subject ID exceeds model.max_subjects")
        role = self.role((text["subject_ids"] - 1).clamp_min(0)) * subjects[..., None]
        change = self.text(text["change"])
        change = change + position_encoding(
            change.shape[1], change.shape[2], change.device
        )
        change = change + torch.einsum(
            "bsl,bsd->bld", text["subject_token_mask"].float(), role
        )
        member = self.person(visual["clip_pooled"]) + self.box(
            geometry(visual["boxes"])
        )
        member = member[:, None] + role[:, :, None]
        member_mask = subjects[:, :, None] & visual["person_mask"][:, None]
        mask = torch.cat(
            (subjects.new_ones(b, 1), text["change_mask"], member_mask.flatten(1)), 1
        )
        prior = torch.cat(
            (
                change.new_zeros(b, 1 + change.shape[1]),
                membership.clamp_min(1e-8).log().flatten(1),
            ),
            1,
        )
        tokens = torch.cat(
            (self.sentinel.expand(b, -1, -1), change, member.flatten(1, 2)), 1
        )
        reference = self.block(tokens, mask, prior)
        return (
            reference,
            mask,
            prior,
            reference[:, 1 + change.shape[1] :].reshape(b, s, k, change.shape[-1]),
            role,
        )
