"""Hierarchical target attention preserving person-level transport and null mass."""

import math

import torch
from torch import nn

from rcr.proposed.nn.attention import geometry, local_scene


class EvidenceBinding(nn.Module):
    def __init__(
        self, token_dim, scene_dim, dim, heads, roi_size=2, roi_expansion=1.25
    ):
        super().__init__()
        self.visual = nn.Linear(token_dim, dim)
        self.scene = nn.Linear(scene_dim, dim)
        self.box = nn.Linear(4, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.null = nn.Parameter(torch.zeros(dim))
        self.heads, self.roi_size, self.roi_expansion = heads, roi_size, roi_expansion

    def forward(self, members, target, p, patch_hw):
        b, s, q, dim = members.shape
        t = target["person_mask"].shape[1]
        # Missing rows have P=0. Its residual (including epsilon rounding) is null,
        # never redistributed to real people.
        null_mass = p[..., 0] + (1 - p.sum(-1)).clamp_min(0)
        null_value = self.null.expand(b, s, q, -1)
        if t == 0 or q == 0:
            return null_mass[:, None, :, None] * null_value, null_mass
        local = local_scene(
            target["scene"],
            target["boxes"],
            patch_hw,
            self.roi_size,
            self.roi_expansion,
        )
        visual = self.visual(target["clip_tokens"])
        geom = self.box(geometry(target["boxes"]))
        tokens = torch.cat((visual, self.scene(local), geom[:, :, None]), -2)
        token_mask = torch.cat(
            (
                target["clip_token_mask"],
                target["person_mask"][..., None].expand(-1, -1, local.shape[-2] + 1),
            ),
            -1,
        )
        tokens = tokens.masked_fill(~token_mask[..., None], 0)
        # Padded people get a zero dummy key. Their transport mass remains zero.
        safe_mask = token_mask.clone()
        safe_mask[..., -1] |= ~safe_mask.any(-1)
        h, d = self.heads, dim // self.heads
        with torch.autocast(members.device.type, enabled=False):
            query = self.query(members.float()).reshape(b, s, q, h, d)
            key = self.key(tokens.float()).reshape(b, t, -1, h, d)
            value = self.value(tokens.float()).reshape(b, t, -1, h, d)
            logits = torch.einsum("bsqhd,btnhd->bsqhtn", query, key) / math.sqrt(d)
            weights = logits.masked_fill(
                ~safe_mask[:, None, None, None], -torch.inf
            ).softmax(-1)
            per_person = torch.einsum("bsqhtn,btnhd->bsqthd", weights, value).flatten(
                -2
            )
            per_person = per_person.masked_fill(
                ~target["person_mask"][:, None, None, :, None], 0
            )
            real = (per_person * p[:, None, :, 1:, None]).sum(-2)
            # Do not multiply real mass twice: output bias alone needs mass scaling.
            evidence = torch.nn.functional.linear(real, self.output.weight, None)
            if self.output.bias is not None:
                evidence = (
                    evidence + (1 - null_mass)[:, None, :, None] * self.output.bias
                )
        return evidence + null_mass[:, None, :, None] * null_value, null_mass
