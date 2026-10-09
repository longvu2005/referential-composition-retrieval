"""Small attention blocks with explicit masks, membership priors and geometry."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def geometry(boxes):
    lo, hi = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((lo + hi) / 2, (hi - lo).clamp_min(0)), dim=-1)


def position_encoding(length, dim, device):
    position = torch.arange(length, device=device).float()[:, None]
    frequency = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000) / dim)
    )
    angles = position * frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :dim]


def local_scene(scene, boxes, patch_hw, size=2, expansion=1.25):
    """Bilinear ROI samples in normalized letterboxed patch coordinates.

    align_corners=False puts patch centers at (j+.5)/width, (i+.5)/height.
    Expanded boxes are clipped to the same letterboxed canvas used by DINO.
    """
    b, k = boxes.shape[:2]
    if k == 0:
        return scene.new_empty(b, 0, size * size, scene.shape[-1])
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    half = (boxes[..., 2:] - boxes[..., :2]) * (expansion / 2)
    lo, hi = (center - half).clamp(0, 1), (center + half).clamp(0, 1)
    axis = (torch.arange(size, device=boxes.device).float() + 0.5) / size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((xx, yy), -1).reshape(1, 1, size * size, 2)
    grid = lo[:, :, None] + unit * (hi - lo)[:, :, None]
    with torch.autocast(scene.device.type, enabled=False):
        image = scene.float().transpose(1, 2).reshape(b, -1, *patch_hw)
        out = F.grid_sample(
            image, grid.float() * 2 - 1, align_corners=False, padding_mode="border"
        )
    return out.permute(0, 2, 3, 1)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask, prior=None):
        tokens = tokens.masked_fill(~mask[..., None], 0)
        bias = (
            tokens.new_zeros(mask.shape, dtype=torch.float32)
            if prior is None
            else prior.float()
        )
        bias = bias.masked_fill(~mask, -torch.inf)
        bias = (
            bias[:, None]
            .expand(-1, tokens.shape[1], -1)
            .repeat_interleave(self.attention.num_heads, 0)
        )
        # Always called with a valid learned sentinel. FP32 attention is safe in AMP.
        with torch.autocast(tokens.device.type, enabled=False):
            update, _ = self.attention(
                tokens.float(),
                tokens.float(),
                tokens.float(),
                attn_mask=bias,
                need_weights=False,
            )
        tokens = self.norm1(tokens + self.dropout(update))
        return self.norm2(tokens + self.dropout(self.ffn(tokens))).masked_fill(
            ~mask[..., None], 0
        )
