"""Neural RCR model over precomputed visual and text features."""

import torch
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
    ) -> None:
        super().__init__()

        binding = EvidenceBinding(dim, num_heads, mlp_ratio, geo_dim)
        self.grounding = SubjectGrounding(binding, dim)
        self.identity_head = IdentityHead(dim, identity_dim)
        self.composition = StructuredComposition(
            dim, identity_dim, num_heads, max_subjects, mlp_ratio
        )
        self.target_builder = TargetPersonBuilder(dim, identity_dim)
        self.reasoner = FineReasoner(dim, num_heads, mlp_ratio)

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
