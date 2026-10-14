"""Case-free person selection and one shared partial identity assignment."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def matching_settings(config: dict | None = None) -> dict[str, float]:
    """Resolve architecture settings, shared by training and inference."""
    defaults = {
        "selection_threshold": 0.5,
        "identity_threshold": 0.2,
        "missing_penalty": 2.5,
        "duplicate_iou": 0.9,
    }
    config = {} if config is None else config
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"unknown fine_matching settings: {sorted(unknown)}")
    settings = {key: float(config.get(key, value)) for key, value in defaults.items()}
    if not all(math.isfinite(value) for value in settings.values()):
        raise ValueError("fine_matching settings must be finite")
    if not 0 < settings["selection_threshold"] <= 1:
        raise ValueError("selection_threshold must be in (0, 1]")
    if not -1 <= settings["identity_threshold"] <= 1:
        raise ValueError("identity_threshold must be in [-1, 1]")
    # Both active cosine terms lie in [-1, 1]. Null must be worse than -2.
    if settings["missing_penalty"] <= 2:
        raise ValueError("missing_penalty must exceed 2 for the two cosine terms")
    if not 0 < settings["duplicate_iou"] <= 1:
        raise ValueError("duplicate_iou must be in (0, 1]; 1 disables deduplication")
    return settings


@torch.no_grad()
def unique_box_mask(
    boxes: Tensor | None,
    mask: Tensor,
    *,
    threshold: float,
    priority: Tensor | None = None,
) -> Tensor:
    """Suppress near-identical boxes only; no GT identity/count is consulted.

    Query priority is grounding confidence; target priority is box area. Stable
    ties keep the first detection. This is an IoU heuristic, not an identity
    oracle: overlapping people should only be suppressed at a high threshold.
    """
    if boxes is None or threshold == 1 or boxes.shape[1] == 0:
        return mask.bool().clone()
    if boxes.shape != (*mask.shape, 4):
        raise ValueError("boxes must have shape [B,K,4] matching the person mask")
    coords = boxes.detach().float().cpu().numpy()
    valid = mask.detach().bool().cpu().numpy().copy()
    widths = np.maximum(coords[..., 2:] - coords[..., :2], 0)
    area = widths.prod(-1)
    priorities = area if priority is None else priority.detach().float().cpu().numpy()
    for n in range(len(coords)):
        candidates = np.flatnonzero(valid[n])
        order = candidates[np.argsort(-priorities[n, candidates], kind="stable")]
        while len(order):
            first, rest = order[0], order[1:]
            low = np.maximum(coords[n, first, :2], coords[n, rest, :2])
            high = np.minimum(coords[n, first, 2:], coords[n, rest, 2:])
            intersection = np.maximum(high - low, 0).prod(-1)
            union = area[n, first] + area[n, rest] - intersection
            iou = intersection / np.maximum(union, 1e-12)
            duplicate = iou > threshold
            valid[n, rest[duplicate]] = False
            order = rest[~duplicate]
    return torch.as_tensor(valid, device=mask.device, dtype=torch.bool)


@torch.no_grad()
def exclusive_membership(
    logits: Tensor,
    subjects: Tensor,
    person_mask: Tensor | None = None,
    boxes: Tensor | None = None,
    *,
    selection_threshold: float = 0.5,
    duplicate_iou: float = 0.9,
) -> Tensor:
    """Select a variable-sized set per Subject, with one owner per query person.

    Background has fixed logit zero. Softmax is over Subject/background, NEVER
    over people. Existing independent BCE supervises the raw grounding logits;
    the exclusive distribution conditions their odds on at most one owner.
    Fine weights and hard decisions are detached from retrieval loss. Coarse
    continues to use the original raw logits and sigmoid memberships.
    """
    b, s, k = logits.shape
    if s == 0:
        raise ValueError("a query must have at least one Subject slot")
    valid = (
        torch.ones(b, k, device=logits.device, dtype=torch.bool)
        if person_mask is None
        else person_mask.bool()
    )
    foreground = logits.detach().float().masked_fill(~subjects[:, :, None], -torch.inf)
    foreground = foreground.masked_fill(~valid[:, None], -torch.inf)
    probability = torch.cat((foreground.new_zeros(b, 1, k), foreground), 1).softmax(1)
    confidence = 1 - probability[:, 0]
    best, owner = probability[:, 1:].max(1)
    selected = valid & (confidence >= selection_threshold) & (best > probability[:, 0])
    selected = unique_box_mask(
        boxes, selected, threshold=duplicate_iou, priority=confidence
    )
    role = F.one_hot(owner, s).permute(0, 2, 1).bool()
    return (role & selected[:, None]).to(confidence.dtype) * confidence[:, None]


@dataclass
class PersonMatching:
    """One assignment per physical query detection, shared across Subject roles."""

    indices: Tensor  # [B,Kq], target index, -1 for null/unselected
    membership: Tensor  # [B,S,Kq], exclusive detached confidence
    subjects: Tensor  # [B,S], active text Subject slots
    subject_coverage: Tensor  # [B,S], missing query Subject also has coverage zero
    coverage: Tensor  # [B], mean coverage over active Subjects

    @property
    def matched(self) -> Tensor:
        return self.indices >= 0


@torch.no_grad()
def partial_identity_matching(
    query_identity: Tensor,
    target_identity: Tensor,
    membership: Tensor,
    subject_mask: Tensor,
    target_mask: Tensor | None = None,
    target_boxes: Tensor | None = None,
    *,
    identity_threshold: float = 0.2,
    duplicate_iou: float = 0.9,
) -> PersonMatching:
    """Globally maximize confidence-weighted identity utility with private nulls.

    Every target detection can be used once across ALL Subjects. Each query
    member has its own dummy column, so any number of members can be unmatched.
    Identity-threshold ties choose null. No semantic term enters this assignment.
    GPU tensors are transferred as batches, not once per candidate/person.
    """
    from scipy.optimize import linear_sum_assignment

    b, s, q = membership.shape
    t = target_identity.shape[1]
    active = subject_mask.bool()
    weights = membership.detach().float().masked_fill(~active[:, :, None], 0)
    # Also defend callers supplying overlapping soft memberships: one owner.
    confidence, owner = weights.max(1)
    weights = weights * F.one_hot(owner, s).permute(0, 2, 1)
    per_subject = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    importance = per_subject.sum(1)
    indices = torch.full((b, q), -1, device=membership.device, dtype=torch.long)
    valid = (
        torch.ones(b, t, device=membership.device, dtype=torch.bool)
        if target_mask is None
        else target_mask.bool()
    )
    valid = unique_box_mask(target_boxes, valid, threshold=duplicate_iou)
    if q and t:
        # Mask before normalization/einsum: NaN/Inf padding is not valid evidence.
        query = (
            query_identity.detach().float().masked_fill(~(confidence > 0)[..., None], 0)
        )
        target = target_identity.detach().float().masked_fill(~valid[..., None], 0)
        similarity = torch.einsum(
            "bqd,btd->bqt", F.normalize(query, dim=-1), F.normalize(target, dim=-1)
        )
        similarity_cpu = similarity.cpu().numpy()
        priority_cpu = importance.cpu().numpy()
        selected_cpu = (confidence > 0).cpu().numpy()
        valid_cpu = valid.cpu().numpy()
        assigned = np.full((b, q), -1, dtype=np.int64)
        for n in range(b):
            rows = np.flatnonzero(selected_cpu[n])
            columns = np.flatnonzero(valid_cpu[n])
            if not len(rows) or not len(columns):
                continue
            pair = similarity_cpu[n][np.ix_(rows, columns)]
            if not np.isfinite(pair).all():
                raise ValueError("non-finite identity evidence in fine matching")
            utility = priority_cpu[n, rows, None] * (pair - identity_threshold)
            # Forbid nonpositive real edges. Private nulls always remain feasible.
            utility = np.where(pair > identity_threshold, utility, -np.inf)
            null = np.full((len(rows), len(rows)), -np.inf)
            np.fill_diagonal(null, 0)
            row, column = linear_sum_assignment(
                np.concatenate((utility, null), axis=1), maximize=True
            )
            real = column < len(columns)
            assigned[n, rows[row[real]]] = columns[column[real]]
        indices = torch.as_tensor(assigned, device=membership.device)
    matched = indices >= 0
    subject_coverage = (per_subject * matched[:, None]).sum(-1)
    coverage = subject_coverage.masked_fill(~active, 0).sum(-1) / active.sum(
        -1
    ).clamp_min(1)
    return PersonMatching(indices, weights, active, subject_coverage, coverage)


def gather_matched(values: Tensor, matching: PersonMatching) -> Tensor:
    """Gather [B,Kt,D] into [B,Kq,D], with exact zero for null rows."""
    b, q = matching.indices.shape
    if values.shape[1] == 0:
        # Keep a zero autograd connection for batches with no target people.
        return values.sum(1, keepdim=True).expand(b, q, values.shape[-1]) * 0
    result = values.gather(
        1, matching.indices.clamp_min(0)[..., None].expand(-1, -1, values.shape[-1])
    )
    return result.masked_fill(~matching.matched[..., None], 0)
