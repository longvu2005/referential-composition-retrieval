"""Inject aligned person evidence and reason jointly with ordered pair tokens."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ReasoningLayout:
    """Valid token positions for one query, reused by all evaluation candidates."""

    reference_indices: Tensor
    token_indices: Tensor
    person_indices: Tensor
    person_offset: int


class TargetPersonBuilder(nn.Module):
    """Project contextual evidence and center/size geometry before injection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.evidence_proj = nn.Linear(dim, dim)
        self.box_proj = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, evidence: Tensor, boxes: Tensor) -> Tensor:
        with torch.autocast(boxes.device.type, enabled=False):
            x1, y1, x2, y2 = boxes.float().unbind(-1)
            geometry = torch.stack(
                ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1
            )
        return self.norm(self.evidence_proj(evidence) + self.box_proj(geometry))


class FineReasoner(nn.Module):
    """Read a single CLS score after person/ordered-relation self-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.pair_query = nn.Linear(2 * dim, dim)
        self.condition_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.condition_norm = nn.LayerNorm(dim)
        self.pair_proj = nn.Linear(3 * dim, dim)
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
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.relation_geo = nn.Linear(5, num_heads, bias=False)

    def encode_condition(self, tokens, text_mask, mentions, subjects):
        """Read each i<j condition once from composed text, before target injection."""
        first, second = _pairs(subjects)
        if not first.numel():
            return tokens.new_empty(len(tokens), 0, tokens.shape[-1])
        mentions = mentions.bool() & subjects[..., None]
        if text_mask is not None:
            text_mask = text_mask.bool()
            mentions = mentions & text_mask[:, None]
            tokens = tokens.masked_fill(~text_mask[..., None], 0)
        weights = mentions.float() / mentions.sum(-1, keepdim=True).clamp_min(1)
        markers = weights.to(tokens.dtype) @ tokens
        queries = self.pair_query(
            torch.cat((markers[:, first], markers[:, second]), -1)
        )
        padding = None if text_mask is None else ~text_mask
        if padding is not None:
            # An entirely padded text row reads zeros rather than an all-inf softmax.
            padding = padding & ~padding.all(1, keepdim=True)
        update, _ = self.condition_attn(
            queries, tokens, tokens, key_padding_mask=padding, need_weights=False
        )
        active = subjects[:, first] & subjects[:, second]
        return self.condition_norm(queries + update).masked_fill(~active[..., None], 0)

    def _read_endpoint(self, condition, persons, selected, null_person):
        """Read selected members, including those carrying missing-match evidence."""
        if not persons.shape[-2]:
            return null_person.expand(*condition.shape)
        with torch.autocast(persons.device.type, enabled=False):
            values = persons.float().masked_fill(~selected[..., None], 0)
            logits = (condition.float()[..., None, :] * values).sum(-1)
            logits = logits / math.sqrt(persons.shape[-1])
            present = selected.any(-1, keepdim=True)
            logits = logits.masked_fill(~selected, -torch.inf)
            # Never normalize an empty set: discard its neutral row and use n_miss.
            logits = logits.masked_fill(~present, 0)
            weights = logits.softmax(-1).masked_fill(~selected, 0)
            value = (weights[..., None] * values).sum(-2)
            return torch.where(present, value, null_person.float())

    def forward(
        self,
        reference: Tensor,
        target: Tensor,
        selected: Tensor,
        subjects: Tensor,
        condition: Tensor,
        null_person: Tensor,
        query_mask: Tensor | None = None,
        query_key_bias: Tensor | None = None,
        *,
        aligned_boxes: Tensor,
        matched_mask: Tensor,
        layout: ReasoningLayout | None = None,
    ) -> Tensor:
        """Inject each (Subject,person) slot, append relations, then read CLS only."""
        batch, count, people = selected.shape
        selected = selected.bool() & subjects[..., None]
        matched = matched_mask.bool() & selected
        offset = reference.shape[1] - count * people
        if offset < 1 or target.shape != (*selected.shape, reference.shape[-1]):
            raise ValueError("aligned target tokens must match the query person tail")
        target = target.masked_fill(~selected[..., None], 0).flatten(1, 2)
        injection = torch.cat((torch.zeros_like(reference[:, :offset]), target), 1)
        injected = self.norm_cross(reference + self.dropout(injection))
        persons = injected[:, offset:].unflatten(1, (count, people))

        first, second = _pairs(subjects)
        active_pairs = subjects[:, first] & subjects[:, second]
        if condition.shape != (batch, len(first), reference.shape[-1]):
            raise ValueError("ordered conditions must have one token per Subject pair")
        if first.numel():
            # Endpoint order is slot i, then slot j, independent of detection order.
            left = self._read_endpoint(
                condition, persons[:, first], selected[:, first], null_person
            )
            right = self._read_endpoint(
                condition, persons[:, second], selected[:, second], null_person
            )
            pairs = self.pair_proj(torch.cat((condition, left, right), -1))
            pairs = pairs.masked_fill(~active_pairs[..., None], 0)
        else:
            pairs = reference.new_empty(batch, 0, reference.shape[-1])
        tokens = torch.cat((injected, pairs), dim=1)

        valid = (
            torch.ones(reference.shape[:2], device=reference.device, dtype=torch.bool)
            if query_mask is None
            else query_mask.bool().clone()
        )
        valid[:, offset:] &= selected.flatten(1)
        valid = torch.cat((valid, active_pairs), dim=1)
        prior = (
            reference.new_zeros(reference.shape[:2], dtype=torch.float32)
            if query_key_bias is None
            else query_key_bias.float()
        )
        prior = torch.cat((prior, prior.new_zeros(batch, len(first))), 1)
        prior = prior.masked_fill(~valid, -torch.inf)
        geometry_mask = matched.flatten(1)
        geometry_offset, geometry_count = offset, count * people
        if layout is not None:
            # Masked rows never become keys in either round. Removing both axes
            # leaves every valid token's attention population and geometry intact.
            tokens = tokens.index_select(1, layout.token_indices)
            prior = prior.index_select(1, layout.token_indices)
            aligned_boxes = aligned_boxes.index_select(1, layout.person_indices)
            geometry_mask = geometry_mask.index_select(1, layout.person_indices)
            geometry_offset = layout.person_offset
            geometry_count = len(layout.person_indices)
        length = tokens.shape[1]
        heads = self.self_attn.num_heads
        with torch.autocast(reference.device.type, enabled=False):
            bias = prior[:, None, :].expand(-1, length, -1)
            bias = bias.repeat_interleave(heads, 0).clone()
            # Only matched person-to-person edges receive directed geometry.
            stop = geometry_offset + geometry_count
            bias[:, geometry_offset:stop, geometry_offset:stop] += self._relation_bias(
                aligned_boxes, geometry_mask
            )
            for _ in range(2):
                update, _ = self.self_attn(
                    tokens.float(),
                    tokens.float(),
                    tokens.float(),
                    attn_mask=bias,
                    need_weights=False,
                )
                tokens = self.norm_self(tokens + self.dropout(update))
        tokens = self.norm_out(tokens + self.ffn(tokens))
        score = self.score(tokens[:, 0]).squeeze(-1).float()
        # Partial coverage keeps the complete CLS score; no match is exactly zero.
        return torch.where(matched.flatten(1).any(-1), score, torch.zeros_like(score))

    def _relation_bias(self, boxes: Tensor, mask: Tensor) -> Tensor:
        """Per-head [a,b] bias uses b minus a; null/masked edges are neutral."""
        with torch.autocast(boxes.device.type, enabled=False):
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
            geometry = torch.cat(
                (delta, ratio, (intersection / union.clamp_min(1e-6))[..., None]), -1
            )
            bias = self.relation_geo(geometry)
            valid = mask[:, :, None] & mask[:, None]
            bias = bias.masked_fill(~valid[..., None], 0)
        batch, count = mask.shape
        return bias.permute(0, 3, 1, 2).reshape(
            batch * self.self_attn.num_heads, count, count
        )


def _pairs(values):
    count = values.shape[1]
    return torch.triu_indices(count, count, offset=1, device=values.device)
