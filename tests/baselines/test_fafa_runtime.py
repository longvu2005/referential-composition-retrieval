"""Optional native FAFA integration with local tiny backbones, without weights."""

from pathlib import Path

import pytest
import torch
from torch import nn

from rcr.common.config import load_config


@pytest.fixture
def native():
    cfg = load_config("configs/fafa.yaml")
    if not Path(cfg["source"]["local_checkout"]).is_dir():
        pytest.skip("optional pinned FAFA source has not been prepared")
    pytest.importorskip("transformers")
    from rcr.baselines.fafa import official_api, official_source

    official_api(official_source(cfg))
    from lavis.models.blip2_models import Qformer

    return Qformer


def small_config(native):
    return native.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        encoder_width=12,
        add_cross_attention=True,
        cross_attention_freq=1,
        query_length=2,
    )


def test_native_qformer_safetensors_roundtrip(native, tmp_path):
    from safetensors.torch import save_file

    model = native.BertLMHeadModel(small_config(native)).eval()
    query, visual = torch.randn(2, 2, 16), torch.randn(2, 5, 12)
    inputs = {
        "query_embeds": query,
        "encoder_hidden_states": visual,
        "encoder_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "return_dict": True,
    }
    model.config.save_pretrained(tmp_path)
    save_file(
        {key: value.clone() for key, value in model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    restored = native.BertLMHeadModel.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        expected = model.bert(**inputs).last_hidden_state
        actual = restored.bert(**inputs).last_hidden_state
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("vocab_size", [31, 32, 33])
def test_native_qformer_pretrained_resize_loads_complete_checkpoint(
    native, tmp_path, vocab_size
):
    from safetensors.torch import save_file

    model = native.BertLMHeadModel(small_config(native)).eval()
    model.config.save_pretrained(tmp_path)
    save_file(
        {key: value.clone() for key, value in model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    restored = native.BertLMHeadModel.from_pretrained(tmp_path).eval()
    restored.resize_token_embeddings(vocab_size, mean_resizing=False)
    head = restored.cls.predictions
    assert head.bias is head.decoder.bias
    assert head.bias.shape == (vocab_size,)
    assert head.decoder.weight.shape == (vocab_size, 16)

    # A complete BLIP-2/FAFA vocabulary-sized checkpoint must load without
    # filtering keys or ignoring mismatches, including both saved bias aliases.
    checkpoint = {key: value.clone() for key, value in restored.state_dict().items()}
    checkpoint["cls.predictions.bias"] = torch.arange(vocab_size).float()
    checkpoint["cls.predictions.decoder.bias"] = checkpoint[
        "cls.predictions.bias"
    ].clone()
    restored.load_state_dict(checkpoint, strict=True)
    torch.testing.assert_close(head.bias, torch.arange(vocab_size).float())
    logits = head(torch.randn(2, 3, 16))
    assert logits.shape == (2, 3, vocab_size) and torch.isfinite(logits).all()


def test_native_fafa_image_and_composed_extraction(native, tmp_path, monkeypatch):
    from lavis.models.blip2_models.blip2 import Blip2Base
    from lavis.models.blip2_models.blip2_fafa_cpr import Blip2FAFACPR
    from transformers import BertTokenizer

    from rcr.proposed.cache.fafa import extract_person_features

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += [f"word{i}" for i in range(27)]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = BertTokenizer(vocab_file=str(tmp_path / "vocab.txt"))
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    class Vision(nn.Module):
        num_features = 12

        def __init__(self):
            super().__init__()
            self.project = nn.Linear(3, 12)

        def forward(self, pixels):
            return self.project(pixels.mean((-1, -2)))[:, None].expand(-1, 5, -1)

    monkeypatch.setattr(Blip2Base, "init_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(
        Blip2Base,
        "init_vision_encoder",
        lambda *args: (Vision(), nn.LayerNorm(12)),
    )
    monkeypatch.setattr(
        Blip2Base,
        "init_Qformer",
        lambda *args: (
            native.BertLMHeadModel(small_config(native)),
            nn.Parameter(torch.randn(1, 2, 16)),
        ),
    )
    model = Blip2FAFACPR(
        num_query_token=2, embed_dim=8, max_txt_len=8, use_mlm=False, use_dsu=False
    ).eval()
    assert model.Qformer.cls.predictions.bias.shape == (33,)
    pixels = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        features = model.extract_features({"image": pixels}, mode="image")
        pooled = extract_person_features(model, pixels)
        composed = model.extract_features(
            {"image": pixels, "text_input": ["word1 word2", "word3"]},
            mode="multimodal",
        ).multimodal_embeds
    assert features.image_embeds.shape == (2, 2, 16)
    torch.testing.assert_close(pooled, features.image_embeds.mean(1))
    assert composed.shape == (2, 8) and torch.isfinite(composed).all()
    torch.testing.assert_close(composed.norm(dim=-1), torch.ones(2))
