"""Case-free person selection and one shared partial identity assignment."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rcr.proposed.contracts import DEFAULT_OWNER_MODE, OWNER_MODES


def _validate_owner_mode(owner_mode: str) -> None:
    if owner_mode not in OWNER_MODES:
        raise ValueError(f"fine_matching.owner_mode must be {' or '.join(OWNER_MODES)}")


def matching_settings(config: dict | None = None) -> dict[str, float | str]:
    """Resolve architecture settings, shared by training and inference."""
    defaults = {
        "background_logit": 0.0,
        "selection_threshold": 0.5,
        "identity_threshold": 0.2,
        "missing_penalty": 1.1,
        "duplicate_iou": 0.9,
    }
    config = {} if config is None else config
    unknown = set(config) - defaults.keys() - {"owner_mode"}
    if unknown:
        raise ValueError(f"unknown fine_matching settings: {sorted(unknown)}")
    settings = {key: float(config.get(key, value)) for key, value in defaults.items()}
    if not all(math.isfinite(value) for value in settings.values()):
        raise ValueError("fine_matching settings must be finite")
    if not 0 < settings["selection_threshold"] <= 1:
        raise ValueError("selection_threshold must be in (0, 1]")
    if not -1 <= settings["identity_threshold"] <= 1:
        raise ValueError("identity_threshold must be in [-1, 1]")
    # Native FDA is a mean of normalized dot products in [-1, 1].
    if settings["missing_penalty"] <= 1:
        raise ValueError("missing_penalty must exceed 1 for normalized FDA scores")
    if not 0 < settings["duplicate_iou"] <= 1:
        raise ValueError("duplicate_iou must be in (0, 1]; 1 disables deduplication")
    owner_mode = config.get("owner_mode", DEFAULT_OWNER_MODE)
    _validate_owner_mode(owner_mode)
    return {**settings, "owner_mode": owner_mode}


def _unique_boxes(coords, valid, threshold, priority=None):
    """Stable greedy suppression, batched on the assignment's CPU workspace."""
    coords = np.where(valid[..., None], coords, 0)
    widths = np.maximum(coords[..., 2:] - coords[..., :2], 0)
    area = widths.prod(-1)
    priority = area if priority is None else priority
    order = np.argsort(-np.where(valid, priority, -np.inf), axis=1, kind="stable")
    boxes = np.take_along_axis(coords, order[..., None], axis=1)
    area = np.take_along_axis(area, order, axis=1)
    keep = np.take_along_axis(valid, order, axis=1).copy()
    low = np.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    high = np.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(-1)
    union = area[:, :, None] + area[:, None, :] - intersection
    duplicate = intersection / np.maximum(union, 1e-12) > threshold
    # Fixed-size slices avoid one GPU synchronization per image and kept box.
    for index in range(coords.shape[1]):
        keep[:, index + 1 :] &= ~(
            keep[:, index, None] & duplicate[:, index, index + 1 :]
        )
    result = np.empty_like(keep)
    np.put_along_axis(result, order, keep, axis=1)
    return result


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
    if boxes is None or threshold == 1 or not mask.numel():
        return mask.bool().clone()
    if boxes.shape != (*mask.shape, 4):
        raise ValueError("boxes must have shape [B,K,4] matching the person mask")
    if mask.shape[1] <= 1:
        return mask.bool().clone()
    values = [boxes.detach().float(), mask[..., None].float()]
    if priority is not None:
        values.append(priority.detach()[..., None].float())
    packed = torch.cat(values, dim=-1).cpu().numpy()
    keep = _unique_boxes(
        packed[..., :4],
        packed[..., 4].astype(bool),
        threshold,
        packed[..., 5] if priority is not None else None,
    )
    return torch.from_numpy(keep).to(mask.device, non_blocking=True)


@torch.no_grad()
def exclusive_membership(
    logits: Tensor,
    subjects: Tensor,
    person_mask: Tensor | None = None,
    boxes: Tensor | None = None,
    *,
    background_logit: float = 0.0,
    selection_threshold: float = 0.5,
    duplicate_iou: float = 0.9,
    owner_mode: str = DEFAULT_OWNER_MODE,
) -> Tensor:
    """Select a variable-sized set per Subject, with one owner per query person.

    Softmax over Subjects and background sets the selection mask and confidence.
    The optional crop log-softmax ranks owners above the raw background logit.
    Fine decisions are detached. Coarse scoring uses sigmoid of the raw logits.
    """
    b, s, k = logits.shape
    if not math.isfinite(background_logit):
        raise ValueError("background_logit must be finite")
    if s == 0:
        raise ValueError("a query must have at least one Subject slot")
    _validate_owner_mode(owner_mode)
    valid = (
        torch.ones(b, k, device=logits.device, dtype=torch.bool)
        if person_mask is None
        else person_mask.bool()
    )
    foreground = logits.detach().float().masked_fill(~subjects[:, :, None], -torch.inf)
    foreground = foreground.masked_fill(~valid[:, None], -torch.inf)
    background = foreground.new_full((b, 1, k), background_logit)
    probability = torch.cat((background, foreground), 1).softmax(1)
    confidence = 1 - probability[:, 0]
    best, owner = probability[:, 1:].max(1)
    selected = valid & (confidence >= selection_threshold) & (best > probability[:, 0])
    selected = unique_box_mask(
        boxes, selected, threshold=duplicate_iou, priority=confidence
    )
    if owner_mode == "crop_log_softmax" and s > 1 and k > 1:
        # Include all valid crops in the denominator, before selection and dedup.
        has_logits = torch.isfinite(foreground).any(dim=-1, keepdim=True)
        owner_scores = F.log_softmax(foreground.masked_fill(~has_logits, 0), dim=-1)
        eligible = foreground > background_logit
        owner_scores = owner_scores.masked_fill(~eligible, -torch.inf)
        best_roles = eligible & (owner_scores == owner_scores.amax(1, keepdim=True))
        # Break normalized ties by raw logit, then by Subject order.
        normalized_owner = foreground.masked_fill(~best_roles, -torch.inf).argmax(1)
        # With one valid crop every eligible Subject has a normalized score of zero.
        owner = torch.where(valid.sum(-1, keepdim=True) > 1, normalized_owner, owner)
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
    indices_cpu: Tensor | None = None  # Reuse the assignment workspace for CPU gathers.
    context_flags_cpu: Tensor | None = None

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
    context_flags: Tensor | None = None,
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
    if context_flags is not None and (
        context_flags.ndim != 2 or len(context_flags) != b
    ):
        raise ValueError("context flags must be a [batch,tokens] matrix")
    context_flags_cpu = None
    valid = (
        torch.ones(b, t, device=membership.device, dtype=torch.bool)
        if target_mask is None
        else target_mask.bool()
    )
    if q and t:
        # Mask before normalization/einsum: NaN/Inf padding is not valid evidence.
        query = (
            query_identity.detach().float().masked_fill(~(confidence > 0)[..., None], 0)
        )
        target = target_identity.detach().float().masked_fill(~valid[..., None], 0)
        similarity = torch.einsum(
            "bqd,btd->bqt", F.normalize(query, dim=-1), F.normalize(target, dim=-1)
        )
        # Assignment and box suppression share one device-to-host transfer.
        parts = [similarity.float().flatten(1), importance, confidence, valid.float()]
        deduplicate = target_boxes is not None and duplicate_iou != 1 and t > 1
        if deduplicate:
            parts.append(target_boxes.detach().float().flatten(1))
        if context_flags is not None:
            parts.append(context_flags.detach().float())
        packed = torch.cat(parts, dim=1).cpu().numpy()
        boundaries = np.cumsum([q * t, q, q, t])
        similarity_cpu, priority_cpu, confidence_cpu, valid_cpu, auxiliary_cpu = (
            np.split(packed, boundaries, axis=1)
        )
        geometry_width = 4 * t if deduplicate else 0
        if context_flags is not None:
            context_flags_cpu = torch.from_numpy(
                auxiliary_cpu[:, geometry_width:]
            ).bool()
        similarity_cpu = similarity_cpu.reshape(b, q, t)
        selected_cpu = confidence_cpu > 0
        valid_cpu = valid_cpu.astype(bool)
        if deduplicate:
            valid_cpu = _unique_boxes(
                auxiliary_cpu[:, :geometry_width].reshape(b, t, 4),
                valid_cpu,
                duplicate_iou,
            )
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
        indices_cpu = torch.from_numpy(assigned)
    else:
        indices_cpu = torch.full((b, q), -1, dtype=torch.long)
    indices = indices_cpu.to(membership.device, non_blocking=True)
    matched = indices >= 0
    subject_coverage = (per_subject * matched[:, None]).sum(-1).clamp(0, 1)
    coverage = subject_coverage.masked_fill(~active, 0).sum(-1) / active.sum(
        -1
    ).clamp_min(1)
    coverage = coverage.clamp(0, 1)
    return PersonMatching(
        indices,
        weights,
        active,
        subject_coverage,
        coverage,
        indices_cpu,
        context_flags_cpu,
    )


def gather_matched(values: Tensor, indices: Tensor) -> Tensor:
    """Gather [B,Kt,D] into [B,Kq,D], with exact zero for null rows."""
    b, q = indices.shape
    if values.shape[1] == 0:
        # Keep a zero autograd connection for batches with no target people.
        return values.sum(1, keepdim=True).expand(b, q, values.shape[-1]) * 0
    result = values.gather(
        1, indices.clamp_min(0)[..., None].expand(-1, -1, values.shape[-1])
    )
    return result.masked_fill((indices < 0)[..., None], 0)
