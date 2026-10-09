"""Frozen backbone features; trainable heads live only in the retrieval model."""

import re

import torch
import torch.nn.functional as F
from torch import nn

MENTION = re.compile(r"\bSubject\s+(\d+)\b", re.IGNORECASE)
DEFINITION = re.compile(r"\bSubject\s+(\d+)\s+as\s+", re.IGNORECASE)


def parse_subjects(description, change):
    """Infer ordered roles from text alone, never from annotation cardinalities."""
    definitions = list(DEFINITION.finditer(description))
    ids = [int(m[1]) for m in definitions]
    if not ids or len(ids) != len(set(ids)) or min(ids) < 1:
        raise ValueError("selection must define distinct positive Subject IDs")
    selections = []
    for i, match in enumerate(definitions):
        end = definitions[i + 1].start() if i + 1 < len(ids) else len(description)
        value = re.sub(r"\s+and\s*$", "", description[match.end() : end]).strip()
        if not value:
            raise ValueError("empty Subject description")
        selections.append(value)
    mentions = [(int(m[1]), m.start(), m.end()) for m in MENTION.finditer(change)]
    if not {sid for sid, _, _ in mentions} <= set(ids):
        raise ValueError("change mentions an undefined Subject")
    return ids, selections, mentions


class FrozenEncoder(nn.Module):
    def train(self, mode=True):
        super().train(False)
        return self


class ImageEncoder(FrozenEncoder):
    """DINO patch tokens without CLS/registers, plus the raw CLS for source reuse."""

    def __init__(self, backbone, dim):
        super().__init__()
        if dim != backbone.config.hidden_size:
            raise ValueError("cache only raw frozen DINO features")
        self.backbone = backbone.requires_grad_(False).eval()
        self.patch_size = backbone.config.patch_size
        self.num_register_tokens = getattr(backbone.config, "num_register_tokens", 0)

    @torch.no_grad()
    def forward(self, images):
        tokens = self.backbone(pixel_values=images).last_hidden_state
        hw = (images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size)
        return tokens[:, 1 + self.num_register_tokens :], tokens[:, 0], hw


class CLIPFeatures(FrozenEncoder):
    """Hugging Face CLIPModel: aligned pooled image, raw hidden visual/text tokens.

    Only pooled EOS/CLS projections were contrastively aligned by CLIP. Hidden
    patch/text tokens are contextual features with separately learned task heads.
    """

    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        if not tokenizer.is_fast:
            raise ValueError("CLIP requires the fast tokenizer for offset mapping")

    @torch.no_grad()
    def image(self, pixels):
        output = self.backbone.vision_model(pixel_values=pixels)
        pooled = F.normalize(
            self.backbone.visual_projection(output.pooler_output), dim=-1
        )
        # last_hidden_state is pre-post_layernorm; only pooled CLS is post-LN.
        tokens = output.last_hidden_state
        return pooled.float(), tokens.float()

    @torch.no_grad()
    def text(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if not ids:
            raise ValueError("empty text")
        window = self.backbone.config.text_config.max_position_embeddings - 2
        device = next(self.backbone.parameters()).device
        features = []
        for start in range(0, len(ids), window):
            chunk = ids[start : start + window]
            inputs = torch.tensor(
                [[self.tokenizer.bos_token_id, *chunk, self.tokenizer.eos_token_id]],
                device=device,
            )
            output = self.backbone.text_model(input_ids=inputs)
            features.append(
                output.last_hidden_state[0, 1 : 1 + len(chunk)].float().cpu()
            )
        return {"tokens": torch.cat(features), "offsets": offsets, "input_ids": ids}


class QueryTextCache:
    """Frozen CLIP features keyed by exact text; no trainable outputs are cached."""

    def __init__(self, features):
        self.features = features

    def batch(self, samples):
        parsed = [parse_subjects(x["final_desc"], x["final_change"]) for x in samples]
        b, s = len(samples), max(len(x[0]) for x in parsed)
        changes = [self.features[x["final_change"]] for x in samples]
        length = max(len(x["tokens"]) for x in changes)
        width = changes[0]["tokens"].shape[-1]
        sel_length = max(
            len(self.features[t]["tokens"]) for _, texts, _ in parsed for t in texts
        )
        selections = torch.zeros(b, s, sel_length, width)
        selection_mask = torch.zeros(b, s, sel_length, dtype=torch.bool)
        change = torch.zeros(b, length, width)
        change_mask = torch.zeros(b, length, dtype=torch.bool)
        mentions = torch.zeros(b, s, length, dtype=torch.bool)
        subject_ids = torch.zeros(b, s, dtype=torch.long)
        subject_mask = torch.zeros(b, s, dtype=torch.bool)
        for n, (ids, texts, spans) in enumerate(parsed):
            item = changes[n]
            size = len(item["tokens"])
            change[n, :size] = item["tokens"]
            change_mask[n, :size] = True
            for j, (sid, text) in enumerate(zip(ids, texts, strict=True)):
                tokens = self.features[text]["tokens"]
                selections[n, j, : len(tokens)] = tokens
                selection_mask[n, j, : len(tokens)] = True
                subject_ids[n, j] = sid
                subject_mask[n, j] = True
                for role, left, right in spans:
                    if role != sid:
                        continue
                    overlap = torch.tensor(
                        [a < right and z > left for a, z in item["offsets"]]
                    )
                    if not overlap.any():
                        raise ValueError("Subject span has no CLIP tokens")
                    mentions[n, j, :size] |= overlap
        return dict(
            selections=selections,
            selection_mask=selection_mask,
            change=change,
            change_mask=change_mask,
            subject_ids=subject_ids,
            subject_mask=subject_mask,
            subject_token_mask=mentions,
        )


class IdentityHead(nn.Module):
    def __init__(self, dim, identity_dim):
        super().__init__()
        self.proj = nn.Linear(dim, identity_dim)

    def forward(self, persons):
        return F.normalize(self.proj(persons).float(), dim=-1)
