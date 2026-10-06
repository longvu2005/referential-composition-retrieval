"""RCR person identity/semantics over frozen, reference-free feature caches."""

import math
from dataclasses import dataclass, fields

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
from rcr.methods.proposed.person_binding import BINDING_MODES, binding_scores
from rcr.methods.proposed.reasoning import FineReasoner, TargetPersonBuilder

ARCHITECTURE_VERSION = "person-id-sem-v1"


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
        member_logits = logits
        if query_person_mask is not None:
            member_logits = logits.masked_fill(
                ~query_person_mask[:, None].bool(), -torch.inf
            )
        subjects = (
            torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
            if subject_mask is None
            else subject_mask.bool()
        )
        member_logits = member_logits.masked_fill(~subjects[:, :, None], -torch.inf)
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
            member_logits.float().sigmoid(),
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
    ) -> Tensor | dict[str, Tensor]:
        """Context + positive learned scale * same-person binding, in train/eval."""
        mode = self.binding_mode if binding_mode is None else binding_mode
        if mode not in BINDING_MODES:
            raise ValueError(f"unknown binding mode: {mode}")
        if self.representation == "shared" and mode != "none":
            raise ValueError("shared control has no separate semantic binding")
        if person_mask is not None:
            persons = persons.masked_fill(~person_mask[..., None], 0)
            boxes = boxes.masked_fill(~person_mask[..., None], 0)
        semantic = self.encode_semantic(persons)
        evidence = self.grounding.binding(
            self.visual_proj(scene),
            semantic,
            boxes,
            query.reference,
            patch_hw,
            query.mask,
            query.prior,
        )
        target_identity = None
        if self.representation == "shared":
            target_identity = self.encode_identity(persons)
        target = self.target_builder(evidence, boxes, target_identity)
        context = self.reasoner(
            query.reference, target, person_mask, query.mask, query.prior
        ).float()
        if self.representation == "shared":
            return {"none": context} if return_components else context
        with torch.no_grad():
            target_identity = self.encode_identity(persons)
        binding = binding_scores(
            query.identity,
            target_identity,
            query.composed,
            semantic,
            query.membership,
            query.subjects,
            person_mask,
        )
        if return_components:
            return {
                name: context + self.binding_scale * value
                for name, value in binding.items()
            }
        return context + self.binding_scale * binding[mode]
