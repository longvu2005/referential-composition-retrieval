"""Compose grounded identities with the requested change."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def composed_query_mask(
    change: Tensor,
    grounding_logits: Tensor,
    change_mask: Tensor | None = None,
    subject_mask: Tensor | None = None,
) -> Tensor:
    """Valid CLS, text and Subject-person tokens in composition output order."""

    batch, length = change.shape[:2]
    text = (
        torch.ones(batch, length, device=change.device, dtype=torch.bool)
        if change_mask is None
        else change_mask.bool()
    )
    identities = torch.isfinite(grounding_logits)
    if subject_mask is not None:
        identities = identities & subject_mask[:, :, None].bool()
    cls = torch.ones(batch, 1, device=change.device, dtype=torch.bool)
    return torch.cat((cls, text, identities.flatten(1)), dim=1)


def reference_key_bias(
    change: Tensor,
    grounding_logits: Tensor,
    change_mask: Tensor | None = None,
    subject_mask: Tensor | None = None,
) -> Tensor:
    """Soft membership prior aligned with [CLS, change, Subject identities]."""
    b, length = change.shape[:2]
    prior = F.logsigmoid(grounding_logits.float())
    if subject_mask is not None:
        prior = prior.masked_fill(~subject_mask[:, :, None], -torch.inf)
    neutral = change.new_zeros(b, 1 + length)
    bias = torch.cat((neutral, prior.flatten(1)), dim=1)
    return bias.masked_fill(
        ~composed_query_mask(change, grounding_logits, change_mask, subject_mask),
        -torch.inf,
    )


class StructuredComposition(nn.Module):
    """Ground each Subject with identity features, then compose the full query."""

    def __init__(
        self,
        dim: int,
        identity_dim: int,
        num_heads: int,
        max_subjects: int = 2,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.identity_proj = nn.Linear(identity_dim, dim)
        self.role = nn.Embedding(max_subjects, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

        self.bind_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        change: Tensor,
        identity: Tensor,
        grounding_logits: Tensor,
        subject_pos: Tensor,
        change_mask: Tensor | None = None,
        subject_mask: Tensor | None = None,
        subject_token_mask: Tensor | None = None,
        subject_ids: Tensor | None = None,
    ) -> Tensor:
        """Return composed tokens [B, 1+L+S*K, D]."""

        b, s, k = grounding_logits.shape
        d = change.shape[-1]

        active = (
            torch.ones(b, s, device=change.device, dtype=torch.bool)
            if subject_mask is None
            else subject_mask.bool()
        )
        membership = grounding_logits.float().sigmoid()
        membership = membership.masked_fill(~active[:, :, None], 0)
        weight = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if subject_ids is None:
            subject_ids = torch.arange(1, s + 1, device=change.device)[None].expand(
                b, -1
            )
        role = self.role(subject_ids.long().clamp_min(1) - 1)
        ids = self.identity_proj(identity)[:, None] + role[:, :, None]

        grounded = torch.sum(weight[..., None] * ids, dim=2)
        if subject_token_mask is None:
            subject_token_mask = F.one_hot(subject_pos, change.shape[1]).bool()
        mentions = subject_token_mask.bool() & active[:, :, None]
        if change_mask is not None:
            mentions = mentions & change_mask[:, None, :].bool()
        injection = torch.einsum("bsl,bsd->bld", mentions.to(change.dtype), grounded)
        bound = self.bind_norm(change + injection)
        change = torch.where(mentions.any(dim=1)[..., None], bound, change)

        ids = ids.reshape(b, s * k, d)
        tokens = torch.cat((self.cls.expand(b, -1, -1), change, ids), dim=1)
        key_bias = reference_key_bias(
            change, grounding_logits, change_mask, subject_mask
        )

        n = tokens.shape[1]
        attn_mask = key_bias[:, None, :].expand(-1, n, -1)
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        update, _ = self.attn(
            tokens,
            tokens,
            tokens,
            attn_mask=attn_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + update)
        return self.norm2(tokens + self.ffn(tokens))
