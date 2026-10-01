"""Neural RCR model over precomputed visual and text features."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.methods.proposed.binding import EvidenceBinding
from rcr.methods.proposed.composition import (
    StructuredComposition,
    composed_query_mask,
    reference_key_bias,
)
from rcr.methods.proposed.encoders import IdentityHead
from rcr.methods.proposed.grounding import SubjectGrounding
from rcr.methods.proposed.reasoning import FineReasoner, TargetPersonBuilder


class RCRModel(nn.Module):
    """Ground, compose, bind target evidence, and score."""

    def __init__(
        self,
        dim: int,
        identity_dim: int,
        num_heads: int,
        max_subjects: int = 2,
        mlp_ratio: int = 4,
        geo_dim: int = 32,
        state_dim: int | None = None,
        coarse_beta: float = 0.3,
    ) -> None:
        super().__init__()

        state_dim = identity_dim if state_dim is None else state_dim
        if state_dim < 1 or not math.isfinite(coarse_beta) or coarse_beta < 0:
            raise ValueError(
                "state_dim must be positive and coarse_beta finite/nonnegative"
            )
        self.coarse_beta = float(coarse_beta)
        self.state_text_proj = nn.Linear(dim, state_dim)
        self.state_image_proj = nn.Linear(dim, state_dim)

        binding = EvidenceBinding(dim, num_heads, mlp_ratio, geo_dim)
        self.grounding = SubjectGrounding(binding, dim)
        self.identity_head = IdentityHead(dim, identity_dim)
        self.composition = StructuredComposition(
            dim, identity_dim, num_heads, max_subjects, mlp_ratio
        )
        self.target_builder = TargetPersonBuilder(dim, identity_dim)
        self.reasoner = FineReasoner(dim, num_heads, mlp_ratio)

    def encode_text_state(
        self, change: Tensor, change_mask: Tensor | None = None
    ) -> Tensor:
        """Pool final_change tokens only, project, and L2-normalize [B,Ds]."""
        if change_mask is None:
            pooled = change.mean(dim=-2)
        else:
            mask = change_mask.bool()
            tokens = change.masked_fill(~mask[..., None], 0)
            pooled = tokens.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return F.normalize(self.state_text_proj(pooled), dim=-1)

    def encode_image_state(self, global_features: Tensor) -> Tensor:
        """Project mean whole-image patch features and L2-normalize [...,Ds]."""
        return F.normalize(self.state_image_proj(global_features), dim=-1)

    def forward(
        self,
        query_scene: Tensor,
        query_persons: Tensor,
        query_boxes: Tensor,
        selections: Tensor,
        change: Tensor,
        subject_pos: Tensor,
        target_scene: Tensor,
        target_persons: Tensor,
        target_boxes: Tensor,
        patch_hw: tuple[int, int],
        selection_mask: Tensor | None = None,
        change_mask: Tensor | None = None,
        query_person_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        subject_mask: Tensor | None = None,
        subject_token_mask: Tensor | None = None,
        subject_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return grounding logits [B,S,Kq] and fine scores [B]."""

        logits, _, query, query_mask, prior = self.encode_query(
            query_scene,
            query_persons,
            query_boxes,
            selections,
            change,
            subject_pos,
            patch_hw,
            selection_mask,
            change_mask,
            query_person_mask,
            subject_mask,
            subject_token_mask,
            subject_ids,
        )
        score = self.score_target(
            query,
            query_mask,
            prior,
            target_scene,
            target_persons,
            target_boxes,
            patch_hw,
            target_mask,
        )
        return logits, score

    def encode_query(
        self,
        query_scene: Tensor,
        query_persons: Tensor,
        query_boxes: Tensor,
        selections: Tensor,
        change: Tensor,
        subject_pos: Tensor,
        patch_hw: tuple[int, int],
        selection_mask: Tensor | None = None,
        change_mask: Tensor | None = None,
        query_person_mask: Tensor | None = None,
        subject_mask: Tensor | None = None,
        subject_token_mask: Tensor | None = None,
        subject_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Ground and compose once, then reuse for every candidate target.

        Returns raw logits [B,S,K], identities [B,K,Di], composed query
        [B,1+L+S*K,D], valid-token mask, and additive membership prior.
        Training, forward(), and retrieval all use this same path.
        """
        query_identity = self.identity_head(query_persons)
        logits = self.grounding(
            query_scene,
            query_persons,
            query_boxes,
            selections,
            patch_hw,
            selection_mask,
        )
        composition_logits = logits
        if query_person_mask is not None:
            composition_logits = logits.masked_fill(
                ~query_person_mask[:, None].bool(), -torch.inf
            )

        query = self.composition(
            change,
            query_identity,
            composition_logits,
            subject_pos,
            change_mask,
            subject_mask,
            subject_token_mask,
            subject_ids,
        )
        query_mask = composed_query_mask(
            change, composition_logits, change_mask, subject_mask
        )
        prior = reference_key_bias(
            change, composition_logits, change_mask, subject_mask
        )
        return logits, query_identity, query, query_mask, prior

    def score_target(
        self,
        reference: Tensor,
        reference_mask: Tensor,
        reference_key_bias: Tensor,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        patch_hw: tuple[int, int],
        person_mask: Tensor | None = None,
    ) -> Tensor:
        """Score independent query-target pairs using shared target binding."""
        target_identity = self.identity_head(persons)
        evidence = self.grounding.binding(
            scene,
            persons,
            boxes,
            reference,
            patch_hw,
            reference_mask,
            reference_key_bias,
        )
        target = self.target_builder(evidence, target_identity, boxes)
        return self.reasoner(
            reference, target, person_mask, reference_mask, reference_key_bias
        )
