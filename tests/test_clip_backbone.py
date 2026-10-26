"""Native CLIP contracts with tiny real Transformers models; no downloads."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from rcr.proposed.nn.encoders import (
    ImageEncoder,
    QueryTextCache,
    TextEncoder,
    check_clip_config,
    encode_tokenized_text,
)
from rcr.proposed.nn.model import RCRModel
from tests.test_train import Tokenizer, _sample


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


def test_initial_state_cosine_is_native_clip_cosine():
    model = RCRModel(
        dim=8,
        input_dim=16,
        person_input_dim=8,
        global_dim=8,
        identity_dim=6,
        state_dim=8,
        num_heads=2,
    )
    image, text = torch.randn(3, 8), torch.randn(3, 8)
    expected = (F.normalize(image, dim=-1) * F.normalize(text, dim=-1)).sum(-1)
    actual = (model.encode_image_state(image) * model.encode_text_state(text)).sum(-1)
    torch.testing.assert_close(actual, expected)


def test_native_subject_spans_include_all_mentions_without_extending_vocabulary():
    sample = _sample("dual", "q", "a")
    sample["subjects"] = [{"subject_id": 1}, {"subject_id": 2}]
    sample["final_desc"] = "Identify Subject 1 as the man and Subject 2 as the woman"
    sample["final_change"] = "Subject 1 is left of Subject 2; Subject 1 faces Subject 2"
    tokenizer = Tokenizer()
    cache = QueryTextCache(tokenizer)
    batch = cache.batch([sample])
    assert batch["subject_token_mask"].sum(-1).tolist() == [[4, 4]]
    assert len(tokenizer) == 7
    assert cache.stats["change_truncated"] == 0


def test_truncation_preserves_eot_and_refuses_to_silently_drop_a_subject():
    sample = _sample("long", "q", "a")
    sample["final_change"] = "Subject 1 " + "standing " * 100
    cache = QueryTextCache(Tokenizer())
    batch = cache.batch([sample])
    assert batch["change_ids"].shape == (1, 77)
    assert batch["change_ids"][0, -1] == Tokenizer.eos_token_id
    assert cache.stats["change_truncated"] == 1
    sample["subjects"].append({"subject_id": 2})
    sample["final_desc"] += " and Subject 2 as the woman"
    sample["final_change"] += " Subject 2 is sitting"
    with pytest.raises(ValueError, match="Subject 2 is absent or truncated"):
        cache.batch([sample])


def test_global_text_stays_separate_from_learned_context_tokens():
    from tests.test_train import TextBackbone

    sample = _sample("one", "q", "a")
    tokens = QueryTextCache(Tokenizer()).batch([sample])
    encoder = TextEncoder(TextBackbone(), dim=6)
    before = encode_tokenized_text(tokens, encoder, "cpu")
    with torch.no_grad():
        encoder.proj.weight.add_(10)
    after = encode_tokenized_text(tokens, encoder, "cpu")
    torch.testing.assert_close(before["change_global"], after["change_global"])
    assert not torch.allclose(before["change"], after["change"])


def test_pair_contract_rejects_mixed_or_unpinned_backbones():
    from rcr.common.config import load_config

    config = load_config("configs/proposed.yaml")
    check_clip_config(config)
    for section, key, value in (
        ("model", "text_model", "google-bert/bert-base-uncased"),
        ("image_encoder", "revision", "main"),
        ("image_encoder", "scene_size", [256, 256]),
    ):
        wrong = copy.deepcopy(config)
        wrong[section][key] = value
        with pytest.raises(ValueError, match="CLIP"):
            check_clip_config(wrong)


def test_dino_cache_is_rejected_even_if_its_patch_width_matches_clip():
    from types import SimpleNamespace

    from rcr.common.config import load_config
    from rcr.proposed.cache.build import check_cache_config

    config = load_config("configs/proposed.yaml")
    old = SimpleNamespace(
        _scene_cache=None,
        format_version=2,
        encoder_metadata={
            "person_encoder": {"backend": "dino"},
            "image_encoder": {"model": "facebook/dinov3-vitb16-pretrain-lvd1689m"},
            "detector": config["detector"],
        },
    )
    with pytest.raises(ValueError, match="image encoder/preprocessing differs"):
        check_cache_config(old, config, person_backend="clip")


@pytest.mark.parametrize("eos_config", [2, 31])
def test_global_pooling_matches_native_clip_with_embedded_eot(eos_config):
    backbone = CLIPTextModelWithProjection(
        CLIPTextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=1,
            eos_token_id=eos_config,
            pad_token_id=0,
        )
    ).eval()
    encoder = TextEncoder(backbone, dim=6, cache_mib=1)
    ids = torch.tensor([[1, 6, 31, 9, 31]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        native = backbone(input_ids=ids, attention_mask=mask).text_embeds
    _, _, actual = encoder(ids, mask)
    torch.testing.assert_close(actual, native)


def test_loading_a_different_text_revision_fails_before_download():
    from rcr.proposed.nn.encoders import CLIP_MODEL, load_text_modules

    with pytest.raises(ValueError, match="text revision differs"):
        load_text_modules(CLIP_MODEL, 384, "cpu", revision="main")


@pytest.mark.parametrize("omit_key", [False, True])
def test_cache_only_validation_allows_missing_native_yaml_but_checks_encoders(
    tmp_path, omit_key
):
    from types import SimpleNamespace

    from rcr.common.config import load_config
    from rcr.proposed.cache.build import check_cache_config

    cfg = load_config("configs/proposed.yaml")
    cfg["person_encoder"]["fafa_config"] = str(tmp_path / "not-installed.yaml")
    if omit_key:
        cfg["person_encoder"].pop("fafa_config")
    cache = SimpleNamespace(
        _scene_cache=None,
        format_version=7,
        detections=SimpleNamespace(check_config=lambda _: None),
        native=SimpleNamespace(metadata={"version": "recorded-native-cache"}),
        encoder_metadata={
            "person_encoder": {"backend": "fafa"},
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
        },
    )
    cache._scene_cache = SimpleNamespace(encoder_metadata=cache.encoder_metadata)
    check_cache_config(cache, cfg)
    cache.encoder_metadata["image_encoder"] = {"model": "wrong"}
    with pytest.raises(ValueError, match="image encoder/preprocessing differs"):
        check_cache_config(cache, cfg)


def test_overlapping_cache_paths_fail_before_training_reads_dataset(monkeypatch):
    from rcr.common.config import load_config
    from rcr.proposed import train

    cfg = load_config("configs/proposed.yaml")
    cfg["runtime"]["device"] = "cpu"
    cfg["data"]["scene_cache"] = cfg["data"]["fafa_image_cache"]

    def forbidden(*args):
        raise AssertionError("bad cache paths must fail before dataset/model loading")

    monkeypatch.setattr(train, "load_rcr_data", forbidden)
    with pytest.raises(ValueError, match="caches must be separate"):
        train.train(cfg)
