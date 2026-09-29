"""Shared encoders, query text preparation, and identity projection."""

import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.methods.common.data import sample_selection_texts

SUBJECT_MARKERS = {1: "[S1]", 2: "[S2]"}


def encode_query_text(
    samples: list[dict],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """Encode a batch with equal Subject counts (one query during retrieval).

    Return selection tokens [B,S,L,D], change tokens [B,L,D], their masks,
    Subject IDs, and every Subject-marker position. Text gradients stay enabled.
    """
    b, s = len(samples), len(samples[0]["subjects"])
    selection_texts = [
        text for sample in samples for text in sample_selection_texts(sample)
    ]
    selection = tokenizer(selection_texts, padding=True, return_tensors="pt")
    selections, selection_mask = text_encoder(
        selection["input_ids"].to(device), selection["attention_mask"].to(device)
    )

    changes = []
    for sample in samples:
        text = sample["final_change"]
        for subject in sample["subjects"]:
            subject_id = int(subject["subject_id"])
            text = re.sub(
                rf"\bSubject\s+{subject_id}\b", SUBJECT_MARKERS[subject_id], text
            )
        changes.append(text)
    encoded = tokenizer(changes, padding=True, return_tensors="pt")
    change_ids = encoded["input_ids"].to(device)
    change, change_mask = text_encoder(change_ids, encoded["attention_mask"].to(device))

    subject_pos = torch.empty(b, s, dtype=torch.long, device=device)
    subject_token_mask = torch.zeros(
        b, s, change_ids.shape[1], dtype=torch.bool, device=device
    )
    for n, sample in enumerate(samples):
        for j, subject in enumerate(sample["subjects"]):
            marker = SUBJECT_MARKERS[int(subject["subject_id"])]
            marker_id = tokenizer.convert_tokens_to_ids(marker)
            positions = (change_ids[n] == marker_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError(f"{marker} must be one tokenizer token and occur")
            subject_pos[n, j] = positions[0]
            subject_token_mask[n, j, positions] = True

    return {
        "selections": selections.reshape(b, s, *selections.shape[1:]),
        "selection_mask": selection_mask.reshape(b, s, -1),
        "change": change,
        "change_mask": change_mask,
        "subject_pos": subject_pos,
        "subject_token_mask": subject_token_mask,
        "subject_ids": torch.tensor(
            [[int(row["subject_id"]) for row in x["subjects"]] for x in samples],
            device=device,
        ),
    }


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
