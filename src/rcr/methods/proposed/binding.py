"""Shared reference-conditioned evidence binding."""

import torch
from torch import Tensor, nn


class EvidenceBinding(nn.Module):
    """Bind reference-conditioned scene evidence to person candidates."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        geo_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.ref_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.person_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.norm_f = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.Sequential(nn.GELU(), nn.Dropout(dropout)),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.geo = nn.Sequential(
            nn.Linear(4, geo_dim),
            nn.ReLU(),
            nn.Linear(geo_dim, num_heads),
        )

    def forward(
        self,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        reference: Tensor,
        patch_hw: tuple[int, int],
        reference_mask: Tensor | None = None,
        reference_key_bias: Tensor | None = None,
    ) -> Tensor:
        """scene [B,P,D], persons [B,K,D], boxes [B,K,4], reference [B,L,D]."""

        # Additive membership prior is applied to reference KEYS, before scene
        # reads the reference. CLS always provides a valid key for empty queries.
        attn_mask = None
        if reference_key_bias is not None or reference_mask is not None:
            prior = (
                torch.zeros(
                    reference.shape[:2], device=reference.device, dtype=reference.dtype
                )
                if reference_key_bias is None
                else reference_key_bias
            )
            if reference_mask is not None:
                prior = prior.masked_fill(~reference_mask.bool(), -torch.inf)
            attn_mask = prior[:, None, :].expand(-1, scene.shape[1], -1)
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        update, _ = self.ref_attn(
            scene,
            reference,
            reference,
            attn_mask=attn_mask,
            need_weights=False,
        )
        scene = self.norm_f(scene + self.dropout(update))

        if persons.shape[1] == 0:
            return persons

        # A float32 bias alone is insufficient: SDPA under FP16 autocast can
        # overflow on large geometry offsets. Keep this attention region FP32;
        # the text encoder, reference attention and FFNs still use outer AMP.
        with torch.autocast(persons.device.type, enabled=False):
            update, _ = self.person_attn(
                persons.float(),
                scene.float(),
                scene.float(),
                attn_mask=self._geo_bias(boxes, patch_hw),
                need_weights=False,
            )
        persons = self.norm_p(persons + self.dropout(update))
        return self.norm_out(persons + self.ffn(persons))

    def _geo_bias(self, boxes: Tensor, patch_hw: tuple[int, int]) -> Tensor:
        """Return additive geometry bias [B*H, K, P] for normalized xyxy boxes."""

        # Padding/degenerate boxes have no geometry. Give them a neutral bias
        # rather than dividing patch offsets by 1e-6. Valid small boxes retain
        # their original geometry, protected by the FP32 attention above.
        with torch.autocast(boxes.device.type, enabled=False):
            boxes = boxes.float()
            hp, wp = patch_hw
            assert hp * wp > 0

            y = (torch.arange(hp, device=boxes.device, dtype=boxes.dtype) + 0.5) / hp
            x = (torch.arange(wp, device=boxes.device, dtype=boxes.dtype) + 0.5) / wp
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            patches = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

            valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
            neutral = boxes.new_tensor([0.0, 0.0, 1.0, 1.0])
            boxes = torch.where(valid[..., None], boxes, neutral)
            x1, y1, x2, y2 = boxes.unbind(-1)
            w = (x2 - x1).clamp_min(1e-6)
            h = (y2 - y1).clamp_min(1e-6)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            dx = (patches[:, 0] - cx[..., None]) / w[..., None]
            dy = (patches[:, 1] - cy[..., None]) / h[..., None]
            dw = torch.log(w)[..., None].expand_as(dx)
            dh = torch.log(h)[..., None].expand_as(dy)

            bias = self.geo(torch.stack((dx, dy, dw, dh), dim=-1))
            bias = bias.masked_fill(~valid[..., None, None], 0)
            bias = bias.permute(0, 3, 1, 2)
            b, h, k, p = bias.shape
            return bias.reshape(b * h, k, p)
