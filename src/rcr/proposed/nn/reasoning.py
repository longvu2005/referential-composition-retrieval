"""Build target person tokens and score them against a composed query."""

import torch
from torch import Tensor, nn


class TargetPersonBuilder(nn.Module):
    """Build aligned target tokens from contextual evidence and geometry."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.evidence_proj = nn.Linear(dim, dim)
        self.box_proj = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        evidence: Tensor,
        boxes: Tensor,
    ) -> Tensor:
        x1, y1, x2, y2 = boxes.unbind(-1)
        geometry = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
        target = self.evidence_proj(evidence) + self.box_proj(geometry)
        return self.norm(target)


class FineReasoner(nn.Module):
    """Reason over query tokens and their aligned target evidence/null tokens."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
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
        # Ordered pair geometry: dx, dy, log width/height ratios, IoU.
        self.relation_geo = nn.Linear(5, num_heads, bias=False)

    def forward(
        self,
        query: Tensor,
        target: Tensor,
        target_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
        query_key_bias: Tensor | None = None,
        *,
        aligned_offset: int,
        aligned_boxes: Tensor | None = None,
        matched_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one fine retrieval score per query-target pair."""

        query_padding = None if query_mask is None else ~query_mask.bool()

        if query.shape[1] != aligned_offset + target.shape[1]:
            raise ValueError("aligned target tokens must match the query person tail")
        if target_mask is not None:
            target = target.masked_fill(~target_mask.bool()[..., None], 0)
        # Each person token receives only its assigned person's evidence. CLS/text
        # read relations in the following self-attention rounds.
        update = torch.cat((torch.zeros_like(query[:, :aligned_offset]), target), 1)
        query = self.norm_cross(query + self.dropout(update))

        attn_mask = None
        if query_key_bias is not None:
            prior = query_key_bias
            if query_mask is not None:
                prior = prior.masked_fill(~query_mask.bool(), -torch.inf)
            attn_mask = prior[:, None, :].expand(-1, query.shape[1], -1)
            attn_mask = attn_mask.repeat_interleave(self.self_attn.num_heads, dim=0)
            query_padding = None
        if aligned_boxes is not None:
            if matched_mask is None:
                raise ValueError(
                    "aligned geometry requires token offset and matched mask"
                )
            with torch.autocast(query.device.type, enabled=False):
                pair_bias = self._relation_bias(aligned_boxes, matched_mask)
                b, length = query.shape[:2]
                if attn_mask is None:
                    attn_mask = query.new_zeros(
                        b * self.self_attn.num_heads,
                        length,
                        length,
                        dtype=torch.float32,
                    )
                    if query_padding is not None:
                        attn_mask = attn_mask.masked_fill(
                            query_padding.repeat_interleave(
                                self.self_attn.num_heads, 0
                            )[:, None],
                            -torch.inf,
                        )
                        query_padding = None
                else:
                    attn_mask = attn_mask.float().clone()
                attn_mask[:, aligned_offset:, aligned_offset:] += pair_bias
                # Geometry attention is FP32, as in EvidenceBinding, including AMP.
                update, _ = self.self_attn(
                    query.float(),
                    query.float(),
                    query.float(),
                    key_padding_mask=query_padding,
                    attn_mask=attn_mask,
                    need_weights=False,
                )
                # The first round updates members; the second lets CLS/text
                # read their directed pair geometry.
                query = self.norm_self(query + self.dropout(update))
                update, _ = self.self_attn(
                    query.float(),
                    query.float(),
                    query.float(),
                    key_padding_mask=query_padding,
                    attn_mask=attn_mask,
                    need_weights=False,
                )
        else:
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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore cross-attention keys absent from the aligned-matching architecture.
        for key in list(state_dict):
            if key.startswith(f"{prefix}cross_attn."):
                state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _relation_bias(self, boxes: Tensor, mask: Tensor) -> Tensor:
        """Directed target geometry; null/unselected rows have neutral geometry."""
        boxes = boxes.float().masked_fill(~mask[..., None], 0)
        size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
        center = (boxes[..., 2:] + boxes[..., :2]) / 2
        delta = center[:, None] - center[:, :, None]
        ratio = size[:, None].log() - size[:, :, None].log()
        low = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
        high = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
        intersection = (high - low).clamp_min(0).prod(-1)
        area = size.prod(-1)
        union = area[:, :, None] + area[:, None] - intersection
        iou = intersection / union.clamp_min(1e-6)
        geometry = torch.cat((delta, ratio, iou[..., None]), -1)
        bias = self.relation_geo(geometry)
        valid = mask[:, :, None] & mask[:, None]
        bias = bias.masked_fill(~valid[..., None], 0)
        b, k = mask.shape
        return bias.permute(0, 3, 1, 2).reshape(b * self.self_attn.num_heads, k, k)
