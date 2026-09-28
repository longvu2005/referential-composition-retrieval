"""Build target person tokens and score them against a composed query."""

import torch
from torch import Tensor, nn


class TargetPersonBuilder(nn.Module):
    """Combine target evidence, identity, and geometry into person tokens."""

    def __init__(self, dim: int, identity_dim: int) -> None:
        super().__init__()
        self.evidence_proj = nn.Linear(dim, dim)
        self.identity_proj = nn.Linear(identity_dim, dim)
        self.box_proj = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, evidence: Tensor, identity: Tensor, boxes: Tensor) -> Tensor:
        """Return target person tokens [B,K,D] from normalized xyxy boxes."""

        x1, y1, x2, y2 = boxes.unbind(-1)
        box = torch.stack(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            dim=-1,
        )
        return self.norm(
            self.evidence_proj(evidence)
            + self.identity_proj(identity)
            + self.box_proj(box)
        )


class FineReasoner(nn.Module):
    """Reason from composed query tokens over a target person set."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(dim)
        self.norm_self = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
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
                empty = target_padding.all(dim=1)
                # Avoid an all-masked softmax, then remove its dummy update.
                target_padding = target_padding.clone()
                target_padding[empty, 0] = False
                target = target.masked_fill(empty[:, None, None], 0)
            update, _ = self.cross_attn(
                query,
                target,
                target,
                key_padding_mask=target_padding,
                need_weights=False,
            )
            if empty is not None:
                update = update.masked_fill(empty[:, None, None], 0)
        query = self.norm_cross(query + update)

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
        query = self.norm_self(query + update)
        query = self.norm_out(query + self.ffn(query))
        return self.score(query[:, 0]).squeeze(-1)
