"""RCR person identity/semantics over frozen, reference-free feature caches."""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rcr.proposed.contracts import ARCHITECTURE_VERSION
from rcr.proposed.matching import (
    PersonMatching,
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
from rcr.proposed.nn.encoders import SUBJECT_MARKERS, IdentityHead
from rcr.proposed.nn.grounding import SubjectGrounding
from rcr.proposed.nn.reasoning import FineReasoner, TargetPersonBuilder
from rcr.proposed.scores import fafa_binding_scores

DEFAULT_COARSE_BETA = 0.4


@dataclass
class QueryEncoding:
    """All query evidence, computed once and reused for target candidates."""

    logits: Tensor
    identity: Tensor
    reference: Tensor
    mask: Tensor
    prior: Tensor
    fafa: Tensor | None
    membership: Tensor
    subjects: Tensor

    def repeat_candidates(self, count: int) -> "QueryEncoding":
        """Expand [B,...] into query-major [B*C,...], preserving autograd."""
        if count < 1:
            raise ValueError("candidate count must be positive")

        def repeat(value):
            if value is None:
                return None
            expanded = value[:, None].expand(-1, count, *value.shape[1:])
            return expanded.reshape(value.shape[0] * count, *value.shape[1:])

        return QueryEncoding(
            **{field.name: repeat(getattr(self, field.name)) for field in fields(self)}
        )


class RCRModel(nn.Module):
    """Shared person features for identity, grounding and semantic composition."""

    architecture_version = ARCHITECTURE_VERSION

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        input_dim: int,
        person_input_dim: int,
        global_dim: int | None = None,
        max_subjects: int = len(SUBJECT_MARKERS),
    ) -> "RCRModel":
        """Create the model from the single YAML model section."""
        dim = config.get("dim", input_dim)
        return cls(
            dim=dim,
            input_dim=input_dim,
            person_input_dim=person_input_dim,
            global_dim=global_dim,
            max_subjects=max_subjects,
            **{
                key: value
                for key, value in config.items()
                if key not in {"text_model", "dim", "representation"}
            },
        )

    def __init__(
        self,
        dim: int,
        identity_dim: int,
        num_heads: int,
        max_subjects: int = 2,
        mlp_ratio: int = 2,
        geo_dim: int = 32,
        state_dim: int | None = None,
        global_dim: int | None = None,
        coarse_beta: float = DEFAULT_COARSE_BETA,
        input_dim: int | None = None,
        person_input_dim: int | None = None,
        dropout: float = 0.0,
        fine_matching: dict | None = None,
        fda_k: int = 6,
    ) -> None:
        super().__init__()
        input_dim = dim if input_dim is None else input_dim
        person_input_dim = input_dim if person_input_dim is None else person_input_dim
        self.global_dim = dim if global_dim is None else global_dim
        state_dim = self.global_dim if state_dim is None else state_dim
        if state_dim < 1 or not math.isfinite(coarse_beta) or coarse_beta < 0:
            raise ValueError("invalid state dimension or coarse beta")
        self.coarse_beta = float(coarse_beta)
        self.fine_matching = matching_settings(fine_matching)
        if type(fda_k) is not int or fda_k < 1:
            raise ValueError("fda_k must be a positive integer")
        self.fda_k = fda_k
        # Scene tokens have their own context projection; globals stay in CLIP space.
        self.visual_proj = (
            nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim)
        )
        self.state_text_proj = nn.Linear(self.global_dim, state_dim)
        self.state_image_proj = nn.Linear(self.global_dim, state_dim)
        for projection in (self.state_text_proj, self.state_image_proj):
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.person_proj = nn.Sequential(
            nn.Linear(person_input_dim, dim), nn.LayerNorm(dim)
        )
        self.identity_head = IdentityHead(dim, identity_dim)
        # A fixed reference scale: calibration owns the relative context weight.
        # Keep its state key/value so existing checkpoint inference stays exact.
        self.binding_log_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0))), requires_grad=False
        )
        binding = EvidenceBinding(dim, num_heads, mlp_ratio, geo_dim, dropout)
        self.grounding = SubjectGrounding(binding, dim)
        self.composition = StructuredComposition(
            dim,
            num_heads,
            max_subjects,
            mlp_ratio,
            dropout,
        )
        self.target_builder = TargetPersonBuilder(dim)
        self.reasoner = FineReasoner(dim, num_heads, mlp_ratio, dropout)
        self.null_person = nn.Parameter(torch.zeros(dim))

    @property
    def binding_scale(self) -> Tensor:
        return F.softplus(self.binding_log_scale.float())

    def encode_identity(self, persons: Tensor) -> Tensor:
        return self.identity_head(self.person_proj(persons))

    def encode_semantic(self, persons: Tensor) -> Tensor:
        # Shared person tokens for grounding/composition/context, not FDA scoring.
        return self.person_proj(persons)

    def encode_text_state(self, global_features: Tensor) -> Tensor:
        return F.normalize(self.state_text_proj(global_features).float(), dim=-1)

    def encode_image_state(self, global_features: Tensor) -> Tensor:
        return F.normalize(self.state_image_proj(global_features).float(), dim=-1)

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
        *,
        fafa_query: Tensor,
        fafa_targets: Tensor,
        pooling: str = "penalized",
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
            fafa_query=fafa_query,
        )
        return query.logits, self.score_target(
            query,
            target_scene,
            target_persons,
            target_boxes,
            patch_hw,
            target_mask,
            fafa_targets=fafa_targets,
            pooling=pooling,
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
        *,
        fafa_query: Tensor | None = None,
    ) -> QueryEncoding:
        if fafa_query is not None and (
            fafa_query.ndim != 4
            or fafa_query.shape[:3]
            != (query_persons.shape[0], selections.shape[1], query_persons.shape[1])
        ):
            raise ValueError("fafa_query must be [B,S,Kq,D_fafa]")
        if query_person_mask is not None:
            query_persons = query_persons.masked_fill(~query_person_mask[..., None], 0)
            query_boxes = query_boxes.masked_fill(~query_person_mask[..., None], 0)
        semantic = self.encode_semantic(query_persons)
        identity = self.identity_head(semantic)
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
            semantic,
            member_logits,
            subject_pos,
            change_mask,
            subjects,
            subject_token_mask,
            subject_ids,
        )
        mask = composed_query_mask(change, member_logits, change_mask, subjects)
        prior = reference_key_bias(change, member_logits, change_mask, subjects)
        return QueryEncoding(
            logits,
            identity,
            reference,
            mask,
            prior,
            None if fafa_query is None else fafa_query.detach(),
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
        """Encode the coarse identity branch without fine composition."""
        semantic = self.encode_semantic(persons)
        logits = self.grounding(
            self.visual_proj(scene),
            semantic,
            boxes,
            selections,
            patch_hw,
            selection_mask,
        )
        return logits, self.identity_head(semantic)

    def score_context(
        self,
        query: QueryEncoding,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        patch_hw: tuple[int, int],
        person_mask: Tensor | None = None,
    ) -> tuple[Tensor, PersonMatching]:
        """Learnable context path shared by training and fine inference.

        Selection/assignment stay prediction-based and detached. Training needs
        no native FAFA query/target tokens to supervise this component.
        """
        if person_mask is not None:
            persons = persons.masked_fill(~person_mask[..., None], 0)
            boxes = boxes.masked_fill(~person_mask[..., None], 0)
        semantic = self.encode_semantic(persons)
        # Identity fixes the assignment. Only aligned context receives gradients
        # through this path; crop identity has its own explicit objectives.
        with torch.no_grad():
            target_identity = self.identity_head(semantic)
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
        target = self.target_builder(evidence, aligned_boxes)
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
        return context, matching

    def score_target(
        self,
        query: QueryEncoding,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        patch_hw: tuple[int, int],
        person_mask: Tensor | None = None,
        *,
        fafa_targets: Tensor,
        return_details: bool = False,
        pooling: str = "penalized",
    ) -> Tensor | dict[str, Tensor]:
        """Shared identity matching -> aligned evidence/context -> null-aware score.

        Case labels, GT identities and GT cardinalities never enter this path.
        ``return_details`` is a diagnostic view of this very same forward pass.
        """
        if query.fafa is None:
            raise ValueError("fine retrieval requires native FAFA query features")
        context, matching = self.score_context(
            query, scene, persons, boxes, patch_hw, person_mask
        )
        if fafa_targets.ndim != 4 or fafa_targets.shape[:2] != persons.shape[:2]:
            raise ValueError("fafa_targets must be [B,Kt,L,D_fafa]")
        poolings = fafa_binding_scores(
            query.fafa,
            fafa_targets,
            matching=matching,
            k=self.fda_k,
            missing_penalty=self.fine_matching["missing_penalty"],
            pooling=pooling,
            return_poolings=True,
        )
        binding = poolings[pooling]
        gated_context = matching.coverage * context
        score = gated_context + self.binding_scale * binding
        if return_details:
            return {
                "score": score,
                "context": context,
                "gated_context": gated_context,
                "binding": binding,
                "coverage": matching.coverage,
                "subject_coverage": matching.subject_coverage,
                "matched_indices": matching.indices,
                "membership": matching.membership,
                **{f"binding_{key}": value for key, value in poolings.items()},
            }
        return score
