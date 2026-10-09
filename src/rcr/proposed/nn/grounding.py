"""Categorical query grounding across background and all textual Subjects."""

import torch
from torch import nn

from rcr.proposed.nn.attention import geometry, local_scene


class SubjectGrounding(nn.Module):
    def __init__(self, clip_dim, text_dim, scene_dim, dim, heads):
        super().__init__()
        # Owned only by grounding: ranking never trains these projections.
        self.person = nn.Linear(clip_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.scene = nn.Linear(scene_dim, dim)
        self.box = nn.Linear(4, dim)
        self.selection_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.scene_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.subject_score = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.background = nn.Linear(dim, 1)

    def forward(self, visual, text, patch_hw):
        mask, subjects = visual["person_mask"], text["subject_mask"]
        b, s = subjects.shape
        k = mask.shape[1]
        person = self.person(visual["clip_pooled"]) + self.box(
            geometry(visual["boxes"])
        )
        local = local_scene(visual["scene"], visual["boxes"], patch_hw).mean(-2)
        person = person + self.scene(local)
        selections = self.text(text["selections"])
        sm = text["selection_mask"]
        # A neutral key for a padded Subject avoids all-masked attention.
        selections = torch.cat(
            (selections, selections.new_zeros(b, s, 1, selections.shape[-1])), -2
        )
        sm = torch.cat((sm, ~sm.any(-1, keepdim=True)), -1)
        if k:
            keys = selections.flatten(0, 1)
            query = person[:, None].expand(-1, s, -1, -1).flatten(0, 1)
            with torch.autocast(person.device.type, enabled=False):
                selected, _ = self.selection_attention(
                    query.float(),
                    keys.float(),
                    keys.float(),
                    key_padding_mask=~sm.flatten(0, 1),
                    need_weights=False,
                )
                scene = (
                    self.scene(visual["scene"].float())[:, None]
                    .expand(-1, s, -1, -1)
                    .flatten(0, 1)
                )
                context, _ = self.scene_attention(
                    selected, scene, scene, need_weights=False
                )
            evidence = torch.cat((query, selected, context), -1)
            foreground = self.subject_score(evidence).reshape(b, s, k)
        else:
            foreground = person.new_empty(b, s, 0)
        foreground = foreground.masked_fill(~subjects[:, :, None], -1e4)
        logits = torch.cat(
            (self.background(person).transpose(1, 2), foreground), 1
        ).float()
        # Padding is not background supervision or membership.
        membership = logits.softmax(1)[:, 1:] * mask[:, None] * subjects[:, :, None]
        return logits, membership
