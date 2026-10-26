"""Shared encoders, query text preparation, and identity projection."""

import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.common.data import sample_selection_texts

SUBJECT_MARKERS = {1: "Subject 1", 2: "Subject 2"}
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_CONTEXT_LENGTH = 77


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
    """Return the active AMP dtype across old and new PyTorch APIs."""
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
    return tokenizer, TextEncoder(backbone, dim, cache_mib=cache_mib).to(device)


def text_encoder_state(text_encoder: nn.Module, tokenizer) -> dict:
    """Store only the learned projection and native-tokenizer policy."""
    del tokenizer
    return {
        "projection": text_encoder.proj.state_dict(),
        "text_policy": {
            "context_length": CLIP_CONTEXT_LENGTH,
            "subjects": "native_spans",
        },
    }


def load_text_encoder_state(text_encoder: nn.Module, tokenizer, state: dict) -> None:
    expected = text_encoder_state(text_encoder, tokenizer)["text_policy"]
    if state.get("text_policy") != expected:
        raise ValueError("text checkpoint is not compatible with native CLIP spans")
    text_encoder.proj.load_state_dict(state["projection"])


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
    tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
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
            key: tokens[key]
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


class TextEncoder(nn.Module):
    """Frozen CLIP token projection followed by the learned context projection."""

    def __init__(self, backbone: nn.Module, dim: int, *, cache_mib: int = 0) -> None:
        super().__init__()
        if type(cache_mib) is not int or cache_mib < 0:
            raise ValueError("cache.text_lru_mib must be a nonnegative integer")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.global_dim = backbone.config.projection_dim
        self.proj = nn.Linear(self.global_dim, dim)
        # Only frozen native CLIP features are cached. Context-projection gradients and
        # optimizer updates must be recomputed on every visit to a text row.
        self._hidden: OrderedDict[tuple, Tensor] = OrderedDict()
        self._hidden_limit = cache_mib * 1024**2
        self._hidden_bytes = 0

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
        if self._hidden_limit:
            tokens = self._cached_backbone(input_ids, attention_mask)
        else:
            with torch.no_grad():
                tokens = self._backbone_tokens(input_ids, attention_mask)
        # Match CLIP's native pooling, including the legacy HF config whose
        # eos_token_id=2 represents the highest token ID rather than literal 2.
        if self.backbone.config.eos_token_id == 2:
            eot = input_ids.int().argmax(-1)
        else:
            eot = (input_ids == self.backbone.config.eos_token_id).int().argmax(-1)
        pooled = tokens[torch.arange(len(tokens), device=tokens.device), eot]
        return self.proj(tokens), attention_mask.bool(), pooled.float()

    def _backbone_tokens(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.backbone.text_projection(hidden)

    @torch.no_grad()
    def _cached_backbone(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Right padding does not change CLIP's causal token states. Canonicalize
        # that common case while retaining complete rows for left/irregular masks.
        precision = _autocast_dtype(input_ids.device.type)
        backbone_dtype = next(self.backbone.parameters()).dtype
        id_rows = input_ids.detach().cpu().tolist()
        mask_rows = attention_mask.detach().cpu().tolist()
        keys = []
        for ids, mask in zip(id_rows, mask_rows, strict=True):
            length = sum(bool(value) for value in mask)
            right_padded = length > 0 and all(mask[:length]) and not any(mask[length:])
            if right_padded:
                ids, mask = ids[:length], mask[:length]
            keys.append(
                (
                    str(input_ids.device),
                    backbone_dtype,
                    precision,
                    tuple(ids),
                    tuple(mask),
                )
            )
        values, missing = {}, {}
        for n, key in enumerate(keys):
            if key in self._hidden:
                value = self._hidden.pop(key)
                self._hidden[key] = values[key] = value
            else:
                missing.setdefault(key, n)
        if missing:
            indices = torch.tensor(list(missing.values()), device=input_ids.device)
            # Keep the caller's precision; AMP and FP32 visits have separate
            # keys so validation cannot reuse approximate training features.
            hidden = self._backbone_tokens(
                input_ids[indices], attention_mask[indices]
            ).cpu()
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
        rows = []
        for key in keys:
            value = values[key]
            if value.shape[0] == length:
                rows.append(value)
                continue
            row = value.new_zeros(length, value.shape[-1])
            row[: value.shape[0]] = value
            rows.append(row)
        return torch.stack(rows).to(input_ids.device)


class IdentityHead(nn.Module):
    """Project reference-free person features to normalized identity embeddings."""

    def __init__(self, dim: int, identity_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, identity_dim)

    def forward(self, person: Tensor) -> Tensor:
        return F.normalize(self.proj(person).float(), dim=-1)
