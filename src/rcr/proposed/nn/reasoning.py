"""Build target person tokens and score them against a composed query."""

import torch
from torch import Tensor, nn


class TargetPersonBuilder(nn.Module):
    """Context/geometry tokens; identity addition exists only in shared control."""

    def __init__(self, dim: int, identity_dim: int | None = None) -> None:
        super().__init__()
        self.evidence_proj = nn.Linear(dim, dim)
        self.identity_proj = (
            None if identity_dim is None else nn.Linear(identity_dim, dim)
        )
        self.box_proj = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        evidence: Tensor,
        boxes: Tensor,
        identity: Tensor | None = None,
    ) -> Tensor:
        x1, y1, x2, y2 = boxes.unbind(-1)
        geometry = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
        target = self.evidence_proj(evidence) + self.box_proj(geometry)
        if self.identity_proj is not None:
            if identity is None:
                raise ValueError("shared representation requires target identities")
            target = target + self.identity_proj(identity)
        return self.norm(target)


class FineReasoner(nn.Module):
    """Reason from composed query tokens over a target person set."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.norm_cross = nn.LayerNorm(dim)
        self.norm_self = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.Sequential(nn.GELU(), nn.Dropout(dropout)),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        query: Tensor,
        target: Tensor,
        target_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
        query_key_bias: Tensor | None = None,
    ) -> Tensor:
        """Return one fine retrieval score per query-target pair."""

        target_padding = None if target_mask is None else ~target_mask.bool()
        query_padding = None if query_mask is None else ~query_mask.bool()

        if target.shape[1] == 0:
            # An undetected target contributes no person evidence.
            update = torch.zeros_like(query)
        else:
            empty = None
            if target_padding is not None:
                # A key mask does not sanitize NaN/Inf in K/V. Remove padded
                # values before the projections/attention, including empty rows.
                target = target.masked_fill(target_padding[..., None], 0)
                empty = target_padding.all(dim=1)
                # Avoid an all-masked softmax, then remove its dummy update.
                target_padding = target_padding.clone()
                target_padding[empty, 0] = False
            update, _ = self.cross_attn(
                query,
                target,
                target,
                key_padding_mask=target_padding,
                need_weights=False,
            )
            if empty is not None:
                update = update.masked_fill(empty[:, None, None], 0)
        query = self.norm_cross(query + self.dropout(update))

        attn_mask = None
        if query_key_bias is not None:
            prior = query_key_bias
            if query_mask is not None:
                prior = prior.masked_fill(~query_mask.bool(), -torch.inf)
            attn_mask = prior[:, None, :].expand(-1, query.shape[1], -1)
            attn_mask = attn_mask.repeat_interleave(self.self_attn.num_heads, dim=0)
            query_padding = None
        update, _ = self.self_attn(
            query,
            query,
            query,
            key_padding_mask=query_padding,
            attn_mask=attn_mask,
            need_weights=False,
        )
        query = self.norm_self(query + self.dropout(update))
        query = self.norm_out(query + self.ffn(query))
        return self.score(query[:, 0]).squeeze(-1)
