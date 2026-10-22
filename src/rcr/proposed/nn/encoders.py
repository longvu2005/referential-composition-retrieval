"""Shared encoders, query text preparation, and identity projection."""

import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.common.data import sample_selection_texts

SUBJECT_MARKERS = {1: "[S1]", 2: "[S2]"}


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
    add_subject_markers: bool = False,
    cache_mib: int = 0,
):
    """Build the tokenizer and frozen text encoder used by train and retrieve."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source or model_name)
    if add_subject_markers:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SUBJECT_MARKERS.values())}
        )
    backbone = AutoModel.from_pretrained(model_name)
    backbone.resize_token_embeddings(len(tokenizer))
    return tokenizer, TextEncoder(backbone, dim, cache_mib=cache_mib).to(device)


class QueryTextCache:
    """CPU token IDs only; the text projection is recomputed at each step."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.rows: dict[tuple, dict] = {}

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
        entries = list(missing.items())
        for start in range(0, len(entries), 64):
            chunk = entries[start : start + 64]
            selections, changes = [], []
            for _, sample in chunk:
                selections.extend(sample_selection_texts(sample))
                change = sample["final_change"]
                for subject in sample["subjects"]:
                    sid = int(subject["subject_id"])
                    change = re.sub(
                        rf"\bSubject\s+{sid}\b", SUBJECT_MARKERS[sid], change
                    )
                changes.append(change)

            def tokenize(texts):
                encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
                return [
                    ids[mask.bool()].clone()
                    for ids, mask in zip(
                        encoded["input_ids"], encoded["attention_mask"], strict=True
                    )
                ]

            selected, changed = tokenize(selections), tokenize(changes)
            offset = 0
            for (key, sample), ids in zip(chunk, changed, strict=True):
                subject_ids = [int(x["subject_id"]) for x in sample["subjects"]]
                positions = []
                for sid in subject_ids:
                    marker = SUBJECT_MARKERS[sid]
                    marker_id = self.tokenizer.convert_tokens_to_ids(marker)
                    found = (ids == marker_id).nonzero(as_tuple=True)[0]
                    if not len(found):
                        raise ValueError(
                            f"{marker} must be one tokenizer token and occur"
                        )
                    positions.append(found)
                count = len(subject_ids)
                self.rows[key] = {
                    "selections": selected[offset : offset + count],
                    "change": ids,
                    "positions": positions,
                    "subject_ids": subject_ids,
                }
                offset += count

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
    """Run frozen BERT and the trainable projection after CPU preparation."""
    tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
    b, subjects = tokens["subject_ids"].shape
    selections, selection_mask = text_encoder(
        tokens["selection_ids"], tokens["selection_mask"]
    )
    change, change_mask = text_encoder(tokens["change_ids"], tokens["change_mask"])
    return {
        "selections": selections.reshape(b, subjects, *selections.shape[1:]),
        "selection_mask": selection_mask.reshape(b, subjects, -1),
        "change": change,
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
    """Frozen, deterministic BERT features followed by a trainable projection."""

    def __init__(self, backbone: nn.Module, dim: int, *, cache_mib: int = 0) -> None:
        super().__init__()
        if type(cache_mib) is not int or cache_mib < 0:
            raise ValueError("cache.text_lru_mib must be a nonnegative integer")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        hidden_dim = backbone.config.hidden_size
        self.proj = nn.Identity() if hidden_dim == dim else nn.Linear(hidden_dim, dim)
        # Only frozen, unprojected features are cached. Projection gradients and
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
    ) -> tuple[Tensor, Tensor]:
        """Return text tokens [B,L,D] and a boolean valid-token mask."""

        if self._hidden_limit:
            tokens = self._cached_backbone(input_ids, attention_mask)
        else:
            with torch.no_grad():
                tokens = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        return self.proj(tokens), attention_mask.bool()

    @torch.no_grad()
    def _cached_backbone(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Include the complete padded row and mask: left padding changes BERT's
        # absolute positions. A changed instruction/mask must never reuse a row.
        precision = _autocast_dtype(input_ids.device.type)
        backbone_dtype = next(self.backbone.parameters()).dtype
        keys = [
            (str(input_ids.device), backbone_dtype, precision, ids, mask)
            for ids, mask in zip(
                map(tuple, input_ids.cpu().tolist()),
                map(tuple, attention_mask.cpu().tolist()),
                strict=True,
            )
        ]
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
            hidden = self.backbone(
                input_ids=input_ids[indices],
                attention_mask=attention_mask[indices],
            ).last_hidden_state.cpu()
            for key, row in zip(missing, hidden, strict=True):
                value = row.clone()
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
        return torch.stack([values[key] for key in keys]).to(input_ids.device)


class IdentityHead(nn.Module):
    """Project reference-free person features to normalized identity embeddings."""

    def __init__(self, dim: int, identity_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, identity_dim)

    def forward(self, person: Tensor) -> Tensor:
        return F.normalize(self.proj(person).float(), dim=-1)
