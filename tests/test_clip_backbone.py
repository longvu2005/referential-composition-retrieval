"""CPU workflow tests for clip backbone."""

import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from rcr.proposed.nn.encoders import (
    ImageEncoder,
    TextEncoder,
)


def test_scene_patches_and_native_global_match_clip_with_different_widths():
    backbone = CLIPVisionModelWithProjection(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    ).eval()
    encoder = ImageEncoder(backbone).eval()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = backbone(pixel_values=pixels)
        patches, global_embedding, grid = encoder(pixels)
    assert patches.shape == (2, 4, 16)
    assert global_embedding.shape == (2, 8)
    assert grid == (2, 2)
    torch.testing.assert_close(patches, expected.last_hidden_state[:, 1:])
    torch.testing.assert_close(global_embedding, expected.image_embeds)
    assert not any(p.requires_grad for p in backbone.parameters())


def test_native_text_eot_survives_padding_caching_and_context_projection():
    backbone = CLIPTextModelWithProjection(
        CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=1,
            eos_token_id=31,
            pad_token_id=0,
        )
    ).eval()
    encoder = TextEncoder(backbone, dim=6, cache_mib=1)
    ids = torch.tensor([[1, 4, 31, 0, 0], [1, 5, 6, 7, 31]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        expected = backbone(input_ids=ids, attention_mask=mask).text_embeds
    context, actual_mask, native = encoder(ids, mask)
    torch.testing.assert_close(native, expected)
    assert context.shape == (2, 5, 6)
    assert torch.equal(actual_mask, mask.bool())
    # A shorter right-padded visit reuses the same frozen states.
    _, _, shorter = encoder(ids[:1, :3], mask[:1, :3])
    torch.testing.assert_close(shorter, expected[:1])
    assert len(encoder._hidden) == 2
    encoder.train()
    assert not backbone.training
    context.square().mean().backward()
    assert encoder.proj.weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    assert not native.requires_grad
