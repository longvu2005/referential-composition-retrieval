"""Frozen BERT reuse must preserve projection learning and text/mask semantics."""

import copy

import pytest
import torch

from rcr.proposed.nn.encoders import TextEncoder
from tests.proposed.test_encoders import TextBackbone


@pytest.mark.parametrize("left_padding", [False, True])
def test_cached_bert_matches_uncached_outputs_and_projection_updates(left_padding):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    backbone = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    direct = TextEncoder(backbone, 6).train()
    cached = TextEncoder(copy.deepcopy(backbone), 6, cache_mib=1).train()
    cached.load_state_dict(direct.state_dict())
    ids = torch.tensor(
        [[0, 1, 2], [3, 4, 5]] if left_padding else [[1, 2, 0], [3, 4, 5]]
    )
    mask = (ids != 0).long()
    # Repeated rows within one batch should be extracted once.
    ids, mask = ids[[0, 1, 0]], mask[[0, 1, 0]]
    calls = []
    hook = cached.backbone.register_forward_hook(
        lambda module, args, kwargs, output: calls.append(len(kwargs["input_ids"])),
        with_kwargs=True,
    )
    optimizers = [
        torch.optim.SGD(model.proj.parameters(), lr=0.1) for model in (direct, cached)
    ]
    initial = cached.proj.weight.detach().clone()
    # Fill under inference_mode, as in mining/validation, then learn from hits.
    with torch.inference_mode():
        cached(ids, mask)
    for _ in range(2):
        outputs = []
        for encoder, optimizer in zip((direct, cached), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            output, valid = encoder(ids, mask)
            outputs.append(output)
            torch.testing.assert_close(valid, mask.bool())
            output[..., 0].square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(*outputs)
        torch.testing.assert_close(cached.proj.weight.grad, direct.proj.weight.grad)
        torch.testing.assert_close(cached.proj.weight, direct.proj.weight)
    hook.remove()
    assert calls == [2]
    assert not torch.equal(initial, cached.proj.weight)
    assert not cached.backbone.training
    assert all(p.grad is None for p in cached.backbone.parameters())


def test_cache_keys_include_tokens_masks_and_padding_positions():
    encoder = TextEncoder(TextBackbone(), 8, cache_mib=1)
    ids = torch.tensor([[1, 2, 0]])
    masks = torch.tensor([[1, 1, 0]])
    calls = []
    hook = encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    encoder(ids, masks)
    encoder(ids.clone(), masks.clone())
    encoder(ids, torch.ones_like(masks))
    encoder(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))
    encoder(torch.tensor([[1, 3, 0]]), masks)
    hook.remove()
    assert len(calls) == 4


def test_text_cache_separates_amp_features_from_fp32_validation():
    encoder = TextEncoder(TextBackbone(), 8, cache_mib=1)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    calls = []
    hook = encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    expected, _ = encoder(ids, mask)
    for _ in range(2):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded, _ = encoder(ids, mask)
        assert torch.isfinite(encoded).all()
    actual, _ = encoder(ids, mask)
    hook.remove()
    assert len(calls) == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loading_backbone_weights_invalidates_text_features():
    encoder = TextEncoder(TextBackbone(), 8, cache_mib=1)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    before, _ = encoder(ids, mask)
    state = {key: value.clone() for key, value in encoder.state_dict().items()}
    state["backbone.embedding.weight"] += 1
    encoder.load_state_dict(state)
    assert not encoder._hidden and encoder._hidden_bytes == 0
    after, _ = encoder(ids, mask)
    assert not torch.equal(after, before)


def test_frozen_text_lru_evicts_and_skips_oversized_rows():
    encoder = TextEncoder(TextBackbone(), 8, cache_mib=1)
    ids = torch.ones(1, 30_000, dtype=torch.long)
    mask = torch.ones_like(ids)
    calls = []
    hook = encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    for values in (ids, ids + 1, ids + 1, ids):
        encoder(values, mask)
        assert encoder._hidden_bytes <= 1024**2
        assert len(encoder._hidden) == 1
    assert len(calls) == 3
    oversized = torch.ones(1, 65_536, dtype=torch.long)
    encoder(oversized, oversized)
    encoder(oversized, oversized)
    hook.remove()
    assert len(calls) == 5
    assert encoder._hidden_bytes <= 1024**2


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_text_cache_budget_is_rejected(limit):
    with pytest.raises(ValueError, match="text_lru_mib"):
        TextEncoder(TextBackbone(), 8, cache_mib=limit)
