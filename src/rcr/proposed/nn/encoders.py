"""Shared encoders, query text preparation, and identity projection."""

import math
import re
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.common.data import sample_selection_texts

SUBJECT_MARKERS = {1: "Subject 1", 2: "Subject 2"}
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_CONTEXT_LENGTH = 77
TEXT_POLICY = {"context_length": CLIP_CONTEXT_LENGTH, "subjects": "native_spans"}


def text_lora_settings(settings: dict | None = None) -> dict:
    """Resolve the late-block text adapters, including frozen legacy configs."""
    defaults = {"enabled": False, "last_layers": 2, "rank": 8, "alpha": 16}
    if settings is None:
        return defaults
    if not isinstance(settings, dict) or settings.keys() - defaults.keys():
        raise ValueError("model.text_lora accepts enabled, last_layers, rank and alpha")
    result = {**defaults, **settings}
    if type(result["enabled"]) is not bool:
        raise ValueError("model.text_lora.enabled must be true or false")
    for key in ("last_layers", "rank"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"model.text_lora.{key} must be a positive integer")
    alpha = result["alpha"]
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("model.text_lora.alpha must be finite and positive")
    return result


def check_clip_config(cfg: dict) -> None:
    image = cfg["image_encoder"]
    if (
        image["model"] != CLIP_MODEL
        or cfg["model"]["text_model"] != CLIP_MODEL
        or image.get("revision") != CLIP_REVISION
    ):
        raise ValueError(
            "image and text must use the same pinned CLIP ViT-B/32 checkpoint"
        )
    if image.get("scene_size") != [224, 224]:
        raise ValueError("CLIP ViT-B/32 scene_size must be [224, 224]")


def _autocast_dtype(device_type: str):
    """Return the active AMP dtype across supported PyTorch APIs."""
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        if device_type == "cuda":
            enabled = torch.is_autocast_enabled()
        elif device_type == "cpu":
            enabled = getattr(torch, "is_autocast_cpu_enabled", lambda: False)()
        else:
            enabled = False
    if not enabled:
        return None

    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        suffix = "gpu" if device_type == "cuda" else device_type
        getter = getattr(torch, f"get_autocast_{suffix}_dtype", None)
        return getter() if getter is not None else None


def load_text_modules(
    model_name: str,
    dim: int,
    device: torch.device | str,
    *,
    tokenizer_source=None,
    revision: str | None = None,
    cache_mib: int = 0,
    lora: dict | None = None,
    train_amp: bool = False,
):
    """Load the text half of the same pinned CLIP checkpoint as the scene cache."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    if model_name != CLIP_MODEL:
        raise ValueError(f"Proposed requires {CLIP_MODEL}")
    revision = revision or CLIP_REVISION
    if revision != CLIP_REVISION:
        raise ValueError("text revision differs from the pinned CLIP image/text pair")
    tokenizer = CLIPTokenizerFast.from_pretrained(
        tokenizer_source or model_name,
        **({} if tokenizer_source is not None else {"revision": revision}),
    )
    if tokenizer.padding_side != "right":
        raise ValueError("CLIP requires right-padded native text")
    backbone = CLIPTextModelWithProjection.from_pretrained(
        model_name, revision=revision
    )
    if backbone.config.max_position_embeddings != CLIP_CONTEXT_LENGTH:
        raise ValueError("CLIP text checkpoint must support exactly 77 positions")
    encoder = TextEncoder(
        backbone, dim, cache_mib=cache_mib, lora=lora, train_amp=train_amp
    )
    return tokenizer, encoder.to(device)


def text_encoder_state(text_encoder: "TextEncoder", tokenizer) -> dict:
    """Store the learned text weights and their encoding policy."""
    del tokenizer
    state = {
        "projection": text_encoder.proj.state_dict(),
        "text_policy": dict(TEXT_POLICY),
    }
    if text_encoder.lora["enabled"]:
        state["lora"] = {
            "settings": dict(text_encoder.lora),
            "weights": {
                name: param.detach()
                for name, param in text_encoder.named_lora_parameters()
            },
        }
    return state


def load_text_encoder_state(
    text_encoder: "TextEncoder", tokenizer, state: dict
) -> None:
    del tokenizer
    if state.get("text_policy") != TEXT_POLICY:
        raise ValueError("text checkpoint is not compatible with native CLIP spans")
    saved = state.get("lora")
    parameters = dict(text_encoder.named_lora_parameters())
    if text_encoder.lora["enabled"]:
        if not isinstance(saved, dict) or saved.get("settings") != text_encoder.lora:
            raise ValueError("CLIP text LoRA settings differ from the checkpoint")
        weights = saved.get("weights")
        if not isinstance(weights, dict) or weights.keys() != parameters.keys():
            raise ValueError(
                "CLIP text LoRA checkpoint has incompatible parameter keys"
            )
        for name, param in parameters.items():
            value = weights[name]
            if not isinstance(value, Tensor) or value.shape != param.shape:
                raise ValueError(
                    f"CLIP text LoRA checkpoint has invalid shape for {name}"
                )
    elif saved is not None:
        raise ValueError(
            "CLIP text LoRA is disabled but the checkpoint contains adapters"
        )
    text_encoder.proj.load_state_dict(state["projection"])
    if parameters:
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(weights[name])


class QueryTextCache:
    """CPU token IDs only; the text projection is recomputed at each step."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.rows: dict[tuple, dict] = {}
        self.stats = dict.fromkeys(
            (
                "selection_texts",
                "selection_truncated",
                "change_texts",
                "change_truncated",
            ),
            0,
        )

    @staticmethod
    def _key(sample: dict) -> tuple:
        return (
            sample["final_desc"],
            sample["final_change"],
            tuple(int(x["subject_id"]) for x in sample["subjects"]),
        )

    def prepare(self, samples: list[dict]) -> None:
        missing = {
            self._key(sample): sample
            for sample in samples
            if self._key(sample) not in self.rows
        }
        for key, sample in missing.items():
            row_stats = dict.fromkeys(self.stats, 0)
            texts = [*sample_selection_texts(sample), sample["final_change"]]
            encoded = self.tokenizer(
                texts,
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                verbose=False,
            )
            ids_rows, offset_rows = [], []
            for j, (ids, offsets) in enumerate(
                zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
            ):
                kind = "change" if j == len(texts) - 1 else "selection"
                row_stats[f"{kind}_texts"] += 1
                if len(ids) > CLIP_CONTEXT_LENGTH:
                    row_stats[f"{kind}_truncated"] += 1
                    ids = [*ids[: CLIP_CONTEXT_LENGTH - 1], self.tokenizer.eos_token_id]
                    offsets = [*offsets[: CLIP_CONTEXT_LENGTH - 1], (0, 0)]
                ids_rows.append(torch.tensor(ids, dtype=torch.long))
                offset_rows.append(offsets)
            subject_ids = [int(x["subject_id"]) for x in sample["subjects"]]
            positions = []
            for sid in subject_ids:
                spans = [
                    m.span()
                    for m in re.finditer(
                        rf"\bSubject\s+{sid}\b", texts[-1], re.IGNORECASE
                    )
                ]
                # A clipped "Subject" prefix alone must not count as a full role.
                spans = [
                    (a, b)
                    for a, b in spans
                    if any(left <= a < right for left, right in offset_rows[-1])
                    and any(left < b <= right for left, right in offset_rows[-1])
                ]
                found = [
                    i
                    for i, (left, right) in enumerate(offset_rows[-1])
                    if right > left and any(left < b and right > a for a, b in spans)
                ]
                if not found:
                    raise ValueError(
                        f"{sample['sample_id']}: Subject {sid} is absent or truncated "
                        "from CLIP's 77-token change; shorten the public change text"
                    )
                positions.append(torch.tensor(found, dtype=torch.long))
            self.rows[key] = {
                "selections": ids_rows[:-1],
                "change": ids_rows[-1],
                "positions": positions,
                "subject_ids": subject_ids,
                "text_stats": row_stats,
            }
            for name, count in row_stats.items():
                self.stats[name] += count

    def statistics(self, samples: list[dict]) -> dict:
        """Report this query set, independently of earlier visits to the cache."""
        self.prepare(samples)
        return {
            name: sum(self.rows[self._key(s)]["text_stats"][name] for s in samples)
            for name in self.stats
        }

    def batch(self, samples: list[dict]) -> dict[str, Tensor]:
        self.prepare(samples)
        rows = [self.rows[self._key(sample)] for sample in samples]
        b, subjects = len(rows), len(rows[0]["subject_ids"])
        if any(len(row["subject_ids"]) != subjects for row in rows):
            raise ValueError("text batches require equal Subject counts")
        left = getattr(self.tokenizer, "padding_side", "right") == "left"
        pad = getattr(self.tokenizer, "pad_token_id", 0)

        def padded(sequences):
            length = max(map(len, sequences))
            ids = torch.full((len(sequences), length), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            offsets = []
            for i, sequence in enumerate(sequences):
                offset = length - len(sequence) if left else 0
                ids[i, offset : offset + len(sequence)] = sequence
                mask[i, offset : offset + len(sequence)] = 1
                offsets.append(offset)
            return ids, mask, offsets

        selection_ids, selection_mask, _ = padded(
            [tokens for row in rows for tokens in row["selections"]]
        )
        change_ids, change_mask, offsets = padded([row["change"] for row in rows])
        subject_pos = torch.empty(b, subjects, dtype=torch.long)
        mentions = torch.zeros(b, subjects, change_ids.shape[1], dtype=torch.bool)
        for n, row in enumerate(rows):
            for j, positions in enumerate(row["positions"]):
                positions = positions + offsets[n]
                subject_pos[n, j] = positions[0]
                mentions[n, j, positions] = True
        return {
            "selection_ids": selection_ids,
            "selection_mask": selection_mask,
            "change_ids": change_ids,
            "change_mask": change_mask,
            "subject_pos": subject_pos,
            "subject_token_mask": mentions,
            "subject_ids": torch.tensor([row["subject_ids"] for row in rows]),
        }


def encode_tokenized_text(
    tokens: dict[str, Tensor], text_encoder: nn.Module, device: torch.device | str
) -> dict[str, Tensor]:
    """Return context tokens and the separate native CLIP global text embedding."""
    b, subjects = tokens["subject_ids"].shape
    selections, selection_mask, _ = text_encoder(
        tokens["selection_ids"], tokens["selection_mask"]
    )
    change, change_mask, change_global = text_encoder(
        tokens["change_ids"], tokens["change_mask"]
    )
    return {
        "selections": selections.reshape(b, subjects, *selections.shape[1:]),
        "selection_mask": selection_mask.reshape(b, subjects, -1),
        "change": change,
        "change_global": change_global,
        "change_mask": change_mask,
        **{
            key: tokens[key].to(device, non_blocking=True)
            for key in ("subject_pos", "subject_token_mask", "subject_ids")
        },
    }


def encode_query_text(
    samples: list[dict],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
    *,
    text_cache: QueryTextCache | None = None,
) -> dict[str, Tensor]:
    cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    return encode_tokenized_text(cache.batch(samples), text_encoder, device)


class ImageEncoder(nn.Module):
    """Keep native CLIP patch tokens and its separately projected global vector."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.patch_size = backbone.config.patch_size

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        output = self.backbone(pixel_values=images)
        patch_hw = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )
        patches = output.last_hidden_state[:, 1:]
        if patches.shape[1] != patch_hw[0] * patch_hw[1]:
            raise ValueError("CLIP patch grid differs from the preprocessed image")
        return patches, output.image_embeds, patch_hw


class _LoRALinear(nn.Module):
    """Add a low-rank update to a frozen attention projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        # Initialize on CPU so adding adapters never changes the main model's RNG.
        initial = torch.empty(rank, base.in_features, device="cpu", dtype=torch.float32)
        nn.init.kaiming_uniform_(initial, a=math.sqrt(5))
        self.lora_a = nn.Parameter(initial.to(base.weight))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

    def forward(self, value: Tensor) -> Tensor:
        update = F.linear(F.linear(value, self.lora_a), self.lora_b)
        return torch.add(self.base(value), update, alpha=self.scaling)


class TextEncoder(nn.Module):
    """Cache the frozen CLIP prefix and recompute the learned text path."""

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        *,
        cache_mib: int = 0,
        lora: dict | None = None,
        train_amp: bool = False,
    ) -> None:
        super().__init__()
        if type(cache_mib) is not int or cache_mib < 0:
            raise ValueError("cache.text_lru_mib must be a nonnegative integer")
        self.backbone = backbone.requires_grad_(False).eval()
        self.global_dim = backbone.config.projection_dim
        self.proj = nn.Linear(self.global_dim, dim)
        self.lora = text_lora_settings(lora)
        self.train_amp = train_amp
        self._lora_start = None
        if self.lora["enabled"]:
            layers = backbone.text_model.encoder.layers
            self._lora_start = len(layers) - self.lora["last_layers"]
            if self._lora_start < 0:
                raise ValueError("model.text_lora.last_layers exceeds CLIP text depth")
            if self.lora["rank"] > backbone.config.hidden_size:
                raise ValueError("model.text_lora.rank exceeds CLIP text width")
            with torch.random.fork_rng(devices=[]):
                for index in range(self._lora_start, len(layers)):
                    attention = layers[index].self_attn
                    for name in ("q_proj", "v_proj"):
                        setattr(
                            attention,
                            name,
                            _LoRALinear(
                                getattr(attention, name),
                                self.lora["rank"],
                                self.lora["alpha"],
                            ),
                        )

        # The LRU stores only states from layers with frozen weights.
        self._hidden: OrderedDict[tuple, Tensor] = OrderedDict()
        self._hidden_limit = cache_mib * 1024**2
        self._hidden_bytes = 0

    def named_lora_parameters(self):
        return (
            (name, param)
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self._hidden.clear()
        self._hidden_bytes = 0
        super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return context tokens, valid mask and native EOT embedding."""

        if input_ids.shape[1] > CLIP_CONTEXT_LENGTH:
            raise ValueError("CLIP text exceeds its 77-token position limit")
        device = self.proj.weight.device
        valid = attention_mask.to(device, dtype=torch.bool, non_blocking=True)
        if self._lora_start is None:
            tokens = self._encode_frozen(input_ids, attention_mask)
        else:
            # Share the same frozen prefix between AMP training and FP32 retrieval.
            with torch.autocast(device.type, enabled=False):
                hidden = self._encode_frozen(input_ids, attention_mask)
            amp = self.train_amp and self.training and device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                text_model = self.backbone.text_model
                hidden = self._run_clip_layers(
                    hidden,
                    valid,
                    range(self._lora_start, len(text_model.encoder.layers)),
                )
                tokens = self.backbone.text_projection(
                    text_model.final_layer_norm(hidden)
                )
            tokens = tokens.to(self.proj.weight.dtype)
        # CLIP's eos_token_id=2 convention pools the highest token ID in each row.
        if self.backbone.config.eos_token_id == 2:
            eot = input_ids.int().argmax(-1)
        else:
            eot = (input_ids == self.backbone.config.eos_token_id).int().argmax(-1)
        pooled = tokens[
            torch.arange(len(tokens), device=tokens.device),
            eot.to(device, non_blocking=True),
        ]
        return (
            self.proj(tokens),
            valid,
            pooled.float(),
        )

    @torch.no_grad()
    def _encode_frozen(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self._hidden_limit:
            return self._cached_backbone(input_ids, attention_mask)
        device = self.proj.weight.device
        return self._frozen_tokens(
            input_ids.to(device, non_blocking=True),
            attention_mask.to(device, non_blocking=True),
        )

    def _run_clip_layers(
        self, hidden: Tensor, attention_mask: Tensor, indices: range
    ) -> Tensor:
        from transformers.modeling_attn_mask_utils import (
            _create_4d_causal_attention_mask,
            _prepare_4d_attention_mask,
        )

        # CLIP eager/SDPA add the same masks in every block. Combine them once.
        mask = _prepare_4d_attention_mask(attention_mask, hidden.dtype)
        causal = _create_4d_causal_attention_mask(
            hidden.shape[:2], hidden.dtype, device=hidden.device
        )
        if causal is not None:
            mask = mask + causal
        layers = self.backbone.text_model.encoder.layers
        for index in indices:
            hidden = layers[index](
                hidden, attention_mask=mask, causal_attention_mask=None
            )[0]
        return hidden

    def _frozen_tokens(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self._lora_start is None:
            return self._backbone_tokens(input_ids, attention_mask)
        hidden = self.backbone.text_model.embeddings(input_ids=input_ids)
        return self._run_clip_layers(hidden, attention_mask, range(self._lora_start))

    def _backbone_tokens(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.backbone.text_projection(hidden)

    @torch.no_grad()
    def _cached_backbone(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Right padding does not change CLIP's causal token states. Canonicalize
        # that common case while retaining complete rows for left/irregular masks.
        device = self.proj.weight.device
        precision = _autocast_dtype(device.type)
        backbone_dtype = next(self.backbone.parameters()).dtype
        input_ids = input_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        id_rows = input_ids.tolist()
        mask_rows = attention_mask.tolist()
        keys, canonical = [], {}
        for ids, mask in zip(id_rows, mask_rows, strict=True):
            length = sum(bool(value) for value in mask)
            right_padded = length > 0 and all(mask[:length]) and not any(mask[length:])
            if right_padded:
                ids, mask = ids[:length], mask[:length]
            keys.append(
                (
                    str(device),
                    backbone_dtype,
                    precision,
                    tuple(ids),
                    tuple(mask),
                )
            )
            canonical[keys[-1]] = right_padded
        values, missing = {}, {}
        for n, key in enumerate(keys):
            if key in self._hidden:
                value = self._hidden.pop(key)
                self._hidden[key] = values[key] = value
            else:
                missing.setdefault(key, n)
        fresh = {}
        if missing:
            indices = torch.tensor(list(missing.values()))
            # Separate AMP/FP32 keys keep validation features at their own precision.
            ids_device = input_ids[indices].to(device, non_blocking=True)
            mask_device = attention_mask[indices].to(device, non_blocking=True)
            hidden_device = self._frozen_tokens(
                ids_device,
                mask_device,
            )
            if all(canonical[key] for key in missing):
                keep = mask_device.bool()
            else:
                keep = (
                    torch.arange(input_ids.shape[1])[None]
                    < torch.tensor([len(key[-2]) for key in missing])[:, None]
                )
                keep = keep.to(device, non_blocking=True)
            current = hidden_device.masked_fill(~keep[..., None], 0)
            # Persist CPU cache rows, but reuse fresh device states for this
            # forward instead of immediately uploading those same states again.
            hidden = hidden_device.cpu()
            fresh.update(zip(missing, current, strict=True))
            for key, row in zip(missing, hidden, strict=True):
                value = row[: len(key[-2])].clone()
                values[key] = value
                size = value.numel() * value.element_size()
                if size <= self._hidden_limit:
                    while (
                        self._hidden and self._hidden_bytes + size > self._hidden_limit
                    ):
                        _, removed = self._hidden.popitem(last=False)
                        self._hidden_bytes -= removed.numel() * removed.element_size()
                    self._hidden[key] = value
                    self._hidden_bytes += size
        length = input_ids.shape[1]
        hits, rows = [], []
        for key in dict.fromkeys(keys):
            if key in fresh:
                continue
            value = values[key]
            if value.shape[0] == length:
                rows.append(value)
            else:
                row = value.new_zeros(length, value.shape[-1])
                row[: value.shape[0]] = value
                rows.append(row)
            hits.append(key)
        if hits:
            uploaded = torch.stack(rows).to(device, non_blocking=True)
            if len(hits) == len(keys):
                return uploaded
            fresh.update(zip(hits, uploaded, strict=True))
        return torch.stack([fresh[key] for key in keys])


class IdentityHead(nn.Module):
    """Project reference-free person features to normalized identity embeddings."""

    def __init__(self, dim: int, identity_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, identity_dim)

    def forward(self, person: Tensor) -> Tensor:
        return F.normalize(self.proj(person).float(), dim=-1)
