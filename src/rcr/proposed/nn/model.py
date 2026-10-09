"""Grounding -> structured composition -> partial matching -> joint reasoning."""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from rcr.proposed.nn.binding import EvidenceBinding
from rcr.proposed.nn.composition import StructuredComposition
from rcr.proposed.nn.encoders import IdentityHead
from rcr.proposed.nn.grounding import SubjectGrounding
from rcr.proposed.nn.matching import IdentityMatching
from rcr.proposed.nn.reasoning import JointReasoner
from rcr.proposed.scores import identity_score

ARCHITECTURE_VERSION = "rcr-soft-partial-v2"


@dataclass
class QueryEncoding:
    logits: torch.Tensor
    identity: torch.Tensor
    membership: torch.Tensor
    subject_mask: torch.Tensor
    reference: torch.Tensor
    reference_mask: torch.Tensor
    prior: torch.Tensor
    members: torch.Tensor
    roles: torch.Tensor

    def repeat_candidates(self, count):
        def repeat(value):
            return value[:, None].expand(-1, count, *value.shape[1:]).flatten(0, 1)

        return QueryEncoding(
            **{f.name: repeat(getattr(self, f.name)) for f in fields(self)}
        )


def clean_visual(visual):
    """Prevent even NaN/Inf padding from entering linear projections/attention."""
    mask = visual["person_mask"]
    output = {**visual}
    for key in ("persons", "clip_pooled", "boxes"):
        output[key] = visual[key].masked_fill(~mask[..., None], 0)
    token_mask = visual["clip_token_mask"] & mask[..., None]
    output["clip_token_mask"] = token_mask
    output["clip_tokens"] = visual["clip_tokens"].masked_fill(~token_mask[..., None], 0)
    return output


class RCRModel(nn.Module):
    """Only trainable heads; every incoming backbone feature is frozen."""

    def __init__(
        self,
        person_dim,
        clip_dim,
        token_dim,
        text_dim,
        scene_dim,
        dim=256,
        identity_dim=128,
        num_heads=4,
        max_subjects=16,
        dropout=0.1,
        transport_tau=0.2,
        transport_iterations=512,
        transport_tolerance=1e-5,
        roi_size=2,
        roi_expansion=1.25,
    ):
        super().__init__()
        self.identity_head = IdentityHead(person_dim, identity_dim)
        self.grounding = SubjectGrounding(clip_dim, text_dim, scene_dim, dim, num_heads)
        self.composition = StructuredComposition(
            clip_dim, text_dim, dim, num_heads, max_subjects, dropout
        )
        self.matching = IdentityMatching(
            transport_tau, transport_iterations, transport_tolerance
        )
        self.binding = EvidenceBinding(
            token_dim, scene_dim, dim, num_heads, roi_size, roi_expansion
        )
        self.reasoner = JointReasoner(dim, num_heads, dropout)

    def encode_identity(self, persons):
        return self.identity_head(persons.detach())

    def encode_query(self, visual, text, patch_hw):
        visual = clean_visual({k: v.detach() for k, v in visual.items()})
        text = {k: v.detach() for k, v in text.items()}
        text["selections"] = text["selections"].masked_fill(
            ~text["selection_mask"][..., None], 0
        )
        text["change"] = text["change"].masked_fill(~text["change_mask"][..., None], 0)
        logits, membership = self.grounding(visual, text, patch_hw)
        identity = self.encode_identity(visual["persons"])
        # No shared trainable projection or retrieval gradient into G/P_id.
        a = membership.detach()
        reference, mask, prior, members, roles = self.composition(visual, text, a)
        return QueryEncoding(
            logits,
            identity,
            a,
            text["subject_mask"],
            reference,
            mask,
            prior,
            members,
            roles,
        )

    def score_target(self, query, target, patch_hw):
        target = clean_visual({k: v.detach() for k, v in target.items()})
        with torch.no_grad():
            identity = self.encode_identity(target["persons"])
        p, confidence, diagnostics = self.matching(
            query.identity, identity, query.membership, target["person_mask"]
        )
        evidence, null_mass = self.binding(query.members, target, p, patch_hw)
        condition = self.reasoner(query, evidence, p, null_mass)
        q_id = (p[..., 1:] * confidence).sum(-1)
        s_id = identity_score(q_id, query.membership, query.subject_mask)
        return {
            "scores": s_id + F.logsigmoid(condition),
            "identity_score": s_id,
            "condition_logits": condition,
            "transport": p,
            "null_mass": null_mass,
            "diagnostics": diagnostics,
        }

    def forward(self, query_visual, text, target_visual, patch_hw):
        query = self.encode_query(query_visual, text, patch_hw)
        return {
            "grounding_logits": query.logits,
            **self.score_target(query, target_visual, patch_hw),
        }
