"""RCR person identity/semantics over frozen, reference-free feature caches."""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.proposed.matching import (
    exclusive_membership,
    gather_matched,
    matching_settings,
    partial_identity_matching,
)
from rcr.proposed.nn.binding import EvidenceBinding
from rcr.proposed.nn.composition import (
    StructuredComposition,
    composed_query_mask,
    reference_key_bias,
)
from rcr.proposed.nn.encoders import IdentityHead
from rcr.proposed.nn.grounding import SubjectGrounding
from rcr.proposed.nn.reasoning import FineReasoner, TargetPersonBuilder
from rcr.proposed.scores import BINDING_MODES, binding_scores

ARCHITECTURE_VERSION = "person-id-sem-partial-match-v2"


@dataclass
class QueryEncoding:
    """All query evidence, computed once and reused for target candidates."""

    logits: Tensor
    identity: Tensor
    reference: Tensor
    mask: Tensor
    prior: Tensor
    composed: Tensor
    membership: Tensor
    subjects: Tensor

    def repeat_candidates(self, count: int) -> "QueryEncoding":
        """Expand [B,...] into query-major [B*C,...], preserving autograd."""
        if count < 1:
            raise ValueError("candidate count must be positive")

        def repeat(value):
            return (
                value[:, None]
                .expand(-1, count, *value.shape[1:])
                .reshape(value.shape[0] * count, *value.shape[1:])
            )

        return QueryEncoding(
            **{field.name: repeat(getattr(self, field.name)) for field in fields(self)}
        )


class RCRModel(nn.Module):
    """Separate person heads; shared representation is a retrained control only."""

    def __init__(
        self,
        dim: int,
        identity_dim: int,
        num_heads: int,
        max_subjects: int = 2,
        mlp_ratio: int = 2,
        geo_dim: int = 32,
        state_dim: int | None = None,
        coarse_beta: float = 0.4,
        input_dim: int | None = None,
        person_input_dim: int | None = None,
        dropout: float = 0.0,
        representation: str = "dual",
        binding_mode: str = "both",
        fine_matching: dict | None = None,
    ) -> None:
        super().__init__()
        if representation not in ("dual", "shared"):
            raise ValueError("representation must be dual or shared")
        if binding_mode not in BINDING_MODES:
            raise ValueError(f"unknown binding mode: {binding_mode}")
        if representation == "shared" and binding_mode != "none":
            raise ValueError("shared control requires binding_mode=none")
        input_dim = dim if input_dim is None else input_dim
        person_input_dim = input_dim if person_input_dim is None else person_input_dim
        if representation == "shared" and person_input_dim != input_dim:
            raise ValueError("shared control requires equal raw scene/person widths")
        state_dim = identity_dim if state_dim is None else state_dim
        if state_dim < 1 or not math.isfinite(coarse_beta) or coarse_beta < 0:
            raise ValueError("invalid state dimension or coarse beta")
        self.representation = representation
        self.binding_mode = binding_mode
        self.coarse_beta = float(coarse_beta)
        self.fine_matching = matching_settings(fine_matching)
        # Scene/state only in the dual model; shared control reuses it for crops.
        self.visual_proj = (
            nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim)
        )
        self.state_text_proj = nn.Linear(dim, state_dim)
        self.state_image_proj = nn.Linear(dim, state_dim)
        self.identity_head = IdentityHead(
            person_input_dim if representation == "dual" else dim, identity_dim
        )
        self.semantic_proj = (
            nn.Linear(person_input_dim, dim) if representation == "dual" else None
        )
        self.binding_log_scale = (
            nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
            if representation == "dual"
            else None
        )
        binding = EvidenceBinding(dim, num_heads, mlp_ratio, geo_dim, dropout)
        self.grounding = SubjectGrounding(binding, dim)
        self.composition = StructuredComposition(
            dim,
            dim if representation == "dual" else identity_dim,
            num_heads,
            max_subjects,
            mlp_ratio,
            dropout,
        )
        self.target_builder = TargetPersonBuilder(
            dim, identity_dim if representation == "shared" else None
        )
        self.reasoner = FineReasoner(dim, num_heads, mlp_ratio, dropout)
        self.null_person = nn.Parameter(torch.zeros(dim))

    @property
    def binding_scale(self) -> Tensor:
        if self.binding_log_scale is None:
            return self.state_text_proj.weight.new_zeros(())
        return F.softplus(self.binding_log_scale.float())

    def encode_identity(self, persons: Tensor) -> Tensor:
        if self.representation == "shared":
            persons = self.visual_proj(persons)
        return self.identity_head(persons)

    def encode_semantic(self, persons: Tensor) -> Tensor:
        if self.semantic_proj is None:
            return self.visual_proj(persons)
        return F.normalize(self.semantic_proj(persons).float(), dim=-1)

    def encode_text_state(
        self, change: Tensor, change_mask: Tensor | None = None
    ) -> Tensor:
        if change_mask is None:
            pooled = change.float().mean(dim=-2)
        else:
            mask = change_mask.bool()
            tokens = change.float().masked_fill(~mask[..., None], 0)
            pooled = tokens.sum(dim=-2) / mask.sum(-1, keepdim=True).clamp_min(1)
        return F.normalize(self.state_text_proj(pooled).float(), dim=-1)

    def encode_image_state(self, global_features: Tensor) -> Tensor:
        return F.normalize(
            self.state_image_proj(self.visual_proj(global_features)).float(), dim=-1
        )

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
        query = self.encode_query(
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
        return query.logits, self.score_target(
            query, target_scene, target_persons, target_boxes, patch_hw, target_mask
        )

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
    ) -> QueryEncoding:
        if query_person_mask is not None:
            query_persons = query_persons.masked_fill(~query_person_mask[..., None], 0)
            query_boxes = query_boxes.masked_fill(~query_person_mask[..., None], 0)
        semantic = self.encode_semantic(query_persons)
        identity = self.encode_identity(query_persons)
        logits = self.grounding(
            self.visual_proj(query_scene),
            semantic,
            query_boxes,
            selections,
            patch_hw,
            selection_mask,
        )
        subjects = (
            torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
            if subject_mask is None
            else subject_mask.bool()
        )
        membership = exclusive_membership(
            logits,
            subjects,
            query_person_mask,
            query_boxes,
            selection_threshold=self.fine_matching["selection_threshold"],
            duplicate_iou=self.fine_matching["duplicate_iou"],
        )
        # The fine reference contains only selected, uniquely owned people.
        # Detaching its weights prevents retrieval from hiding difficult members.
        member_logits = torch.logit(membership.clamp(1e-6, 1 - 1e-6)).masked_fill(
            membership == 0, -torch.inf
        )
        reference = self.composition(
            change,
            semantic if self.representation == "dual" else identity,
            member_logits,
            subject_pos,
            change_mask,
            subjects,
            subject_token_mask,
            subject_ids,
        )
        mask = composed_query_mask(change, member_logits, change_mask, subjects)
        prior = reference_key_bias(change, member_logits, change_mask, subjects)
        b, s, k = logits.shape
        composed = (
            F.normalize(
                reference[:, 1 + change.shape[1] :].float().reshape(b, s, k, -1), dim=-1
            )
            if k
            else reference.new_empty(b, s, 0, reference.shape[-1])
        )
        return QueryEncoding(
            logits,
            identity,
            reference,
            mask,
            prior,
            composed,
            membership,
            subjects,
        )

    def encode_grounded_identity(
        self,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        selections: Tensor,
        patch_hw: tuple[int, int],
        selection_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Coarse/mining path: no instruction composition or fine reasoning."""
        logits = self.grounding(
            self.visual_proj(scene),
            self.encode_semantic(persons),
            boxes,
            selections,
            patch_hw,
            selection_mask,
        )
        return logits, self.encode_identity(persons)

    def score_target(
        self,
        query: QueryEncoding,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        patch_hw: tuple[int, int],
        person_mask: Tensor | None = None,
        *,
        binding_mode: str | None = None,
        return_components: bool = False,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Shared identity matching -> aligned state/context -> null-aware score.

        Case labels, GT identities and GT cardinalities never enter this path.
        ``return_details`` is a diagnostic view of this very same forward pass.
        """
        if return_components and return_details:
            raise ValueError("choose return_components or return_details, not both")
        mode = self.binding_mode if binding_mode is None else binding_mode
        if mode not in BINDING_MODES:
            raise ValueError(f"unknown binding mode: {mode}")
        if self.representation == "shared" and mode != "none":
            raise ValueError("shared control has no separate semantic binding")
        if person_mask is not None:
            persons = persons.masked_fill(~person_mask[..., None], 0)
            boxes = boxes.masked_fill(~person_mask[..., None], 0)
        semantic = self.encode_semantic(persons)
        if self.representation == "dual":
            with torch.no_grad():
                target_identity = self.encode_identity(persons)
        else:
            target_identity = self.encode_identity(persons)
        matching = partial_identity_matching(
            query.identity,
            target_identity,
            query.membership,
            query.subjects,
            person_mask,
            boxes,
            identity_threshold=self.fine_matching["identity_threshold"],
            duplicate_iou=self.fine_matching["duplicate_iou"],
        )
        b, s, k = matching.membership.shape
        selected = matching.membership > 0
        matched = selected & matching.matched[:, None]

        def align(values):
            width = values.shape[-1]
            values = gather_matched(values, matching)
            values = values[:, None].expand(-1, s, -1, -1)
            return values.masked_fill(~matched[..., None], 0).reshape(b, s * k, width)

        aligned_boxes = align(boxes)
        evidence = self.grounding.binding(
            self.visual_proj(scene),
            align(semantic),
            aligned_boxes,
            query.reference,
            patch_hw,
            query.mask,
            query.prior,
        )
        # Null/unselected persons cannot acquire substitute evidence from scene.
        evidence = evidence.masked_fill(~matched.flatten(1)[..., None], 0)
        target = self.target_builder(
            evidence,
            aligned_boxes,
            align(target_identity) if self.representation == "shared" else None,
        )
        target = torch.where(
            matched.flatten(1)[..., None], target, self.null_person[None, None]
        )
        offset = query.reference.shape[1] - s * k
        context = self.reasoner(
            query.reference,
            target,
            selected.flatten(1),
            query.mask,
            query.prior,
            aligned_offset=offset,
            aligned_boxes=aligned_boxes,
            matched_mask=matched.flatten(1),
        ).float()
        binding = binding_scores(
            query.identity,
            target_identity,
            query.composed,
            semantic,
            query.membership,
            query.subjects,
            person_mask,
            matching=matching,
            missing_penalty=self.fine_matching["missing_penalty"],
        )
        gated_context = matching.coverage * context
        scale = self.binding_scale if self.representation == "dual" else 1.0
        components = {
            name: gated_context + scale * value
            for name, value in binding.items()
            if self.representation == "dual" or name == "none"
        }
        if return_details:
            return {
                "score": components[mode],
                "context": context,
                "gated_context": gated_context,
                "binding": binding[mode],
                "coverage": matching.coverage,
                "subject_coverage": matching.subject_coverage,
                "matched_indices": matching.indices,
                "membership": matching.membership,
            }
        if return_components:
            return components
        return components[mode]
