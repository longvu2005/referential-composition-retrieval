"""Shared image/text encoders and identity projection."""

import torch.nn.functional as F
from torch import Tensor, nn


class ImageEncoder(nn.Module):
    """Extract DINOv3-style patch and CLS features with one shared backbone."""

    def __init__(self, backbone: nn.Module, dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.config.patch_size
        self.num_register_tokens = getattr(backbone.config, "num_register_tokens", 0)

        hidden_dim = backbone.config.hidden_size
        self.proj = nn.Identity() if hidden_dim == dim else nn.Linear(hidden_dim, dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        """Return patch features [B,P,D], CLS [B,D], and patch grid."""

        tokens = self.proj(self.backbone(pixel_values=images).last_hidden_state)
        patches = tokens[:, 1 + self.num_register_tokens :]
        patch_hw = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )
        return patches, tokens[:, 0], patch_hw


class TextEncoder(nn.Module):
    """Encode tokenized text with one shared language backbone."""

    def __init__(self, backbone: nn.Module, dim: int) -> None:
        super().__init__()
        self.backbone = backbone

        hidden_dim = backbone.config.hidden_size
        self.proj = nn.Identity() if hidden_dim == dim else nn.Linear(hidden_dim, dim)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return text tokens [B,L,D] and a boolean valid-token mask."""

        tokens = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.proj(tokens), attention_mask.bool()


class IdentityHead(nn.Module):
    """Project reference-free person features to normalized identity embeddings."""

    def __init__(self, dim: int, identity_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, identity_dim)

    def forward(self, person: Tensor) -> Tensor:
        return F.normalize(self.proj(person), dim=-1)
