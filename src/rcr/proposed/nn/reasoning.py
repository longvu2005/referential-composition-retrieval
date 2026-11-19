"""Score conditions over matched Subjects and ordered Subject pairs."""

import torch
from torch import Tensor, nn


class FineReasoner(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.pair_query = nn.Linear(2 * dim, dim)
        self.condition_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.condition_norm = nn.LayerNorm(dim)
        self.weight = nn.Linear(dim, 1)

        def scorer(width):
            return nn.Sequential(
                nn.Linear(width, dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mlp_ratio, 1),
            )

        self.subject_score = scorer(2 * dim + 4)
        self.pair_score = scorer(4 * dim + 5)

    def encode_condition(self, tokens, text_mask, mentions, subjects):
        """Read condition tokens with Subject and pair queries once per query."""
        first, second = _pairs(subjects)
        mentions = mentions.bool() & subjects[..., None]
        if text_mask is not None:
            text_mask = text_mask.bool()
            mentions = mentions & text_mask[:, None]
            tokens = tokens.masked_fill(~text_mask[..., None], 0)
        weights = mentions.float() / mentions.sum(-1, keepdim=True).clamp_min(1)
        markers = weights.to(tokens.dtype) @ tokens
        pairs = self.pair_query(torch.cat((markers[:, first], markers[:, second]), -1))
        queries = torch.cat((markers, pairs), dim=1)
        padding = None if text_mask is None else ~text_mask
        if padding is not None:
            padding = padding & ~padding.all(1, keepdim=True)
        update, _ = self.condition_attn(
            queries, tokens, tokens, key_padding_mask=padding, need_weights=False
        )
        condition = self.condition_norm(queries + update)
        active = torch.cat((subjects, subjects[:, first] & subjects[:, second]), dim=1)
        logits = self.weight(condition).squeeze(-1).float()
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        return condition, logits.softmax(1).masked_fill(~active, 0)

    def forward(
        self,
        condition: Tensor,
        weights: Tensor,
        evidence: Tensor,
        boxes: Tensor,
        membership: Tensor,
        scene: Tensor,
        patch_hw: tuple[int, int],
    ) -> Tensor:
        """Score matched role evidence at fixed query weights."""
        first, second = _pairs(membership)
        count = membership.shape[1]
        mass = membership.sum(-1, keepdim=True)
        matched = mass.squeeze(-1) > 0
        pooled = (membership / mass.clamp_min(1e-6)).to(evidence.dtype) @ evidence
        regions = _enclosing_boxes(boxes.float(), membership > 0)
        unary = self.subject_score(
            torch.cat((condition[:, :count], pooled, regions), dim=-1)
        ).squeeze(-1)
        if first.numel() == 0:
            return (weights * unary.float().masked_fill(~matched, 0)).sum(1)

        a, b = regions[:, first], regions[:, second]
        valid = matched[:, first] & matched[:, second]
        union = torch.cat(
            (
                torch.minimum(a[..., :2], b[..., :2]),
                torch.maximum(a[..., 2:], b[..., 2:]),
            ),
            dim=-1,
        )
        region = _region_features(
            scene, union.masked_fill(~valid[..., None], 0), patch_hw
        )
        pair = self.pair_score(
            torch.cat(
                (
                    condition[:, count:],
                    pooled[:, first],
                    pooled[:, second],
                    _geometry(a, b),
                    region,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        scores = torch.cat((unary, pair), dim=1).float()
        return (
            weights * scores.masked_fill(~torch.cat((matched, valid), dim=1), 0)
        ).sum(1)


def _pairs(values):
    count = values.shape[1]
    return torch.triu_indices(count, count, offset=1, device=values.device)


def _enclosing_boxes(boxes, members):
    if boxes.shape[1] == 0:
        return boxes.new_zeros((*members.shape[:2], 4))
    mask = members[..., None]
    low = boxes[:, None, :, :2].masked_fill(~mask, torch.inf).amin(2)
    high = boxes[:, None, :, 2:].masked_fill(~mask, -torch.inf).amax(2)
    return torch.cat((low, high), -1).masked_fill(~members.any(-1)[..., None], 0)


def _geometry(a, b):
    """Signed center and size offsets in ordered Subject coordinates."""
    a_size = (a[..., 2:] - a[..., :2]).clamp_min(1e-6)
    b_size = (b[..., 2:] - b[..., :2]).clamp_min(1e-6)
    delta = (b[..., :2] + b[..., 2:] - a[..., :2] - a[..., 2:]) * 0.5
    low = torch.maximum(a[..., :2], b[..., :2])
    high = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    union = a_size.prod(-1) + b_size.prod(-1) - intersection
    iou = intersection / union.clamp_min(1e-6)
    return torch.cat((delta, b_size.log() - a_size.log(), iou[..., None]), -1)


def _region_features(scene, boxes, patch_hw):
    """Pool cached scene tokens by their patch overlap with each pair region."""
    height, width = patch_hw
    y = torch.arange(height, device=boxes.device, dtype=torch.float32) / height
    x = torch.arange(width, device=boxes.device, dtype=torch.float32) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    patch_low = torch.stack((xx, yy), -1).reshape(-1, 2)
    patch_high = patch_low + boxes.new_tensor([1 / width, 1 / height])
    low = torch.maximum(boxes[..., None, :2], patch_low)
    high = torch.minimum(boxes[..., None, 2:], patch_high)
    overlap = (high - low).clamp_min(0).prod(-1)
    weights = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.to(scene.dtype) @ scene
