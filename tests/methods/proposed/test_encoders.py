from types import SimpleNamespace

import torch
from torch import nn

from rcr.methods.proposed.encoders import IdentityHead, ImageEncoder, TextEncoder


class ImageBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=6,
            patch_size=2,
            num_register_tokens=2,
        )
        self.patch = nn.Conv2d(3, 6, kernel_size=2, stride=2)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        registers = torch.zeros(
            patches.shape[0],
            self.config.num_register_tokens,
            patches.shape[-1],
            device=patches.device,
            dtype=patches.dtype,
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls, registers, patches), dim=1)
        )


class TextBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.embedding = nn.Embedding(16, 6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_image_encoder_returns_patch_and_cls_features() -> None:
    encoder = ImageEncoder(ImageBackbone(), dim=8)
    images = torch.randn(2, 3, 8, 6, requires_grad=True)

    patches, cls, patch_hw = encoder(images)

    assert patches.shape == (2, 12, 8)
    assert cls.shape == (2, 8)
    assert patch_hw == (4, 3)

    (patches.mean() + cls.mean()).backward()
    assert images.grad is not None


def test_text_encoder_returns_tokens_and_mask() -> None:
    encoder = TextEncoder(TextBackbone(), dim=8)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    tokens, mask = encoder(input_ids, attention_mask)

    assert tokens.shape == (2, 3, 8)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(mask, attention_mask.bool())


def test_identity_head_normalizes_embeddings() -> None:
    head = IdentityHead(dim=8, identity_dim=5)
    features = torch.randn(4, 8, requires_grad=True)

    identity = head(features)

    torch.testing.assert_close(identity.norm(dim=-1), torch.ones(4))
    identity.mean().backward()
    assert features.grad is not None
