"""Train grounding, crop identity, coarse identity, context and global state."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from rcr.common.tensor_checks import check_tensor, checked
from rcr.proposed.cache.native import NativeTargetBatch
from rcr.proposed.cache.store import SceneTargetBatch
from rcr.proposed.identity import select_identity_observations
from rcr.proposed.matching import gather_matched
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import (
    DEFAULT_CONTEXT_WEIGHT,
    DEFAULT_COVERAGE_WEIGHT,
    fafa_binding_scores,
    identity_scores_paired,
    linear_score_components,
)

FINE_COUNT_FIELDS = (
    "fine_unfiltered_queries",
    "fine_unfiltered_pairs",
    "fine_owner_pairs",
    "fine_matched_pairs",
    "fine_full_pairs",
    "fine_partial_pairs",
    "fine_active_queries",
    "fine_labeled_query_crops",
    "fine_multi_head_query_crops",
    "fine_labeled_target_matches",
    "fine_multi_head_target_matches",
)


def loss_settings(config: dict) -> dict:
    """Resolve one objective; reject stale keys and invalid weights before training."""
    defaults = {
        "grounding_weight": 1.0,
        "identity_weight": 0.1,
        "coarse_weight": 1.0,
        "context_weight": 1.0,
        "context_verify_members": False,
        "context_rejection_weight": 0.0,
        "rejection_loss_mode": "softplus",
        "context_swap_weight": 0.0,
        "context_temperature": 0.1,
        "state_weight": 1.0,
        "identity_temperature": 0.1,
        "coarse_temperature": 0.1,
        "state_temperature": 0.1,
    }
    if unknown := config.keys() - defaults.keys():
        raise ValueError(f"unknown loss settings: {', '.join(sorted(unknown))}")
    settings = {**defaults, **config}
    for key, value in settings.items():
        if key == "context_verify_members":
            if type(value) is not bool:
                raise ValueError("loss.context_verify_members must be a boolean")
            continue
        if key == "rejection_loss_mode":
            if type(value) is not str or value not in ("softplus", "hinge"):
                raise ValueError("loss.rejection_loss_mode must be softplus or hinge")
            continue
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value <= 0 if key.endswith("temperature") else value < 0)
        ):
            raise ValueError(
                f"invalid loss.{key}: weights must be nonnegative "
                "and temperatures positive"
            )
    if not any(
        value
        for key, value in settings.items()
        if key.endswith("weight")
        and key not in ("context_rejection_weight", "context_swap_weight")
    ):
        raise ValueError("at least one loss weight must be positive")
    return settings


def grounding_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Balance foreground/background per Subject, then Subjects and queries.

    Unknown/padded crops and Subjects with no aligned positive are masked by the
    caller. They contribute neither loss nor gradient.
    """
    logits = logits.float().masked_fill(~mask, 0)
    targets = targets.float().masked_fill(~mask, 0)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive, negative = mask & targets.bool(), mask & ~targets.bool()
    p, n = positive.sum(-1), negative.sum(-1)
    subject = (
        loss.masked_fill(~positive, 0).sum(-1) / p.clamp_min(1)
        + loss.masked_fill(~negative, 0).sum(-1) / n.clamp_min(1)
    ) / ((p > 0).long() + (n > 0)).clamp_min(1)
    valid_subject = mask.any(-1)
    per_query = subject.sum(-1) / valid_subject.sum(-1).clamp_min(1)
    return per_query.sum() / valid_subject.any(-1).sum().clamp_min(1)


def identity_loss(identity: Tensor, labels: Tensor, temperature=0.1) -> Tensor:
    """Supervised contrastive loss; anchors without a positive are ignored."""
    keep = labels != -1
    # Keep a fixed tensor shape. Boolean indexing CUDA tensors would first
    # synchronize to discover the number of known observations.
    identity = F.normalize(identity.float().masked_fill(~keep[..., None], 0), dim=-1)
    if len(identity) < 2:
        return identity.sum() * 0

    similarity = identity @ identity.T / temperature
    diagonal = torch.eye(len(identity), device=identity.device, dtype=torch.bool)
    valid = keep[:, None] & keep[None] & ~diagonal
    positive = labels[:, None].eq(labels[None]) & valid
    masked_similarity = similarity.masked_fill(~valid, -torch.inf)
    # Unknown/sole-known rows have no keys. A neutral denominator avoids NaN
    # gradients from logsumexp(-inf), while those rows still contribute zero.
    masked_similarity = masked_similarity.masked_fill(~valid.any(1, keepdim=True), 0)
    log_probability = masked_similarity - torch.logsumexp(
        masked_similarity, dim=1, keepdim=True
    )
    counts = positive.sum(1)
    valid = counts > 0
    per_anchor = -(
        log_probability.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1)
    )
    return per_anchor.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def _ranking_pairs(positive: Tensor, valid: Tensor) -> Tensor:
    return (positive & valid)[:, :, None] & (~positive & valid)[:, None, :]


@checked
def retrieval_loss(
    scores: Tensor,
    positive: Tensor,
    valid_mask: Tensor | None = None,
    *,
    pair_mask: Tensor | None = None,
):
    """Mean pairwise softplus per query, then mean over queries with both labels.

    Every positive must outrank every eligible negative. Positives never compete
    against one another. Invalid entries are removed before arithmetic, including
    the differentiable zero returned when no pair is available.
    """
    valid = (
        torch.ones_like(positive, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    scores = scores.float().masked_fill(~valid, 0)
    check_tensor(
        torch.isfinite(scores).all(), "ranking loss requires finite eligible scores"
    )
    pairs = _ranking_pairs(positive.bool(), valid)
    if pair_mask is not None:
        pairs &= pair_mask
    difference = scores[:, None, :] - scores[:, :, None]
    pair_loss = F.softplus(difference).masked_fill(~pairs, 0)
    counts = pairs.sum((1, 2))
    per_query = pair_loss.sum((1, 2)) / counts.clamp_min(1)
    return per_query.sum() / (counts > 0).sum().clamp_min(1)


@checked
def rejection_hinge_loss(
    base_scores: Tensor,
    context_contribution: Tensor,
    positive: Tensor,
    valid_mask: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Penalize context only when it worsens a Full Positive/wrong-ID ordering.

    ``base_scores`` is binding_scale * B + a_train * C; its entire margin is
    detached. ``context_contribution`` already includes b_train. Valid pairs,
    including zero hinges, use the same per-query reduction as retrieval_loss.
    """
    valid = valid_mask.bool()
    base = base_scores.detach().float().masked_fill(~valid, 0)
    context = context_contribution.float().masked_fill(~valid, 0)
    check_tensor(
        torch.isfinite(base).all() & torch.isfinite(context).all(),
        "rejection hinge requires finite eligible scores",
    )
    positive = positive.bool() & valid
    pairs = positive[:, :, None] & (valid & ~positive)[:, None]
    margin = (base[:, :, None] - base[:, None, :]).clamp_min(0)
    residual = context[:, None, :] - context[:, :, None]
    pair_loss = F.relu(residual - margin).masked_fill(~pairs, 0) / temperature
    counts = pairs.sum((1, 2))
    per_query = pair_loss.sum((1, 2)) / counts.clamp_min(1)
    return per_query.sum() / (counts > 0).sum().clamp_min(1)


def _context_scores(
    model,
    query,
    batch,
    patch_hw,
    batch_size,
    *,
    eligible=None,
    target_identity=None,
    positions_cpu=None,
    host_matching=None,
):
    """Score eligible pairs; transfer native features only for the current chunk."""
    queries, candidates = batch["positive_mask"].shape
    pairs = queries * candidates
    names = [
        "target_scene",
        "target_persons",
        "target_boxes",
        "target_mask",
    ]
    targets = tuple(
        batch[name]
        if isinstance(batch[name], SceneTargetBatch)
        else batch[name].flatten(0, 1)
        for name in names
    )
    native = batch["fafa_targets"]
    if isinstance(native, Tensor):
        native = native.flatten(0, 1)
    pooling = batch["fafa_pooling"]
    device = query.reference.device
    if positions_cpu is None:
        positions_cpu = (
            torch.arange(pairs)
            if eligible is None
            else eligible.detach().cpu().flatten().nonzero().flatten()
        )
    positions = positions_cpu.to(device, non_blocking=True)
    if "context_layout" in batch:
        last_column = batch["context_layout"]["widths"]
    else:
        masks_cpu = batch["target_mask"].detach().flatten(0, 1).cpu()
        last_column = (
            torch.where(masks_cpu, torch.arange(masks_cpu.shape[1]), -1).amax(1) + 1
            if masks_cpu.shape[1]
            else torch.zeros(pairs, dtype=torch.long)
        )
    identity = (
        None if target_identity is None else target_identity.detach().flatten(0, 1)
    )

    def score_pairs(indices, scene, persons, boxes, mask, assignment):
        selected = query.index_select(indices)
        padding = ~mask[..., None]
        semantic = model.encode_semantic(persons.masked_fill(padding, 0))
        return model.score_aligned_context(
            selected,
            scene,
            semantic,
            boxes.masked_fill(padding, 0),
            assignment,
            patch_hw,
        )

    context, coverage, binding, assignments = [], [], [], []
    for start in range(0, len(positions_cpu), batch_size):
        cpu_rows = positions_cpu[start : start + batch_size]
        rows = positions[start : start + batch_size]
        width = int(last_column[cpu_rows].max())
        chunk = []
        for name, value in zip(names, targets, strict=True):
            if isinstance(value, SceneTargetBatch):
                chunk.append(value.load(cpu_rows).to(device, non_blocking=True).float())
                continue
            value = value if name == "target_scene" else value[:, :width]
            index = cpu_rows if value.device.type == "cpu" else rows
            chunk.append(value.index_select(0, index).to(device, non_blocking=True))
        scene, persons, boxes, mask = chunk
        if isinstance(native, NativeTargetBatch):
            native_chunk = native.select(cpu_rows, width)
        else:
            index = cpu_rows if native.device.type == "cpu" else rows
            native_chunk = native[:, :width].index_select(0, index)
        indices = rows // candidates
        selected = query.index_select(indices)
        with torch.no_grad():
            projected = (
                model.encode_identity(persons.masked_fill(~mask[..., None], 0))
                if identity is None
                else identity[:, :width].index_select(0, rows)
            )
            matching = model.match_target(
                selected,
                projected,
                boxes,
                mask,
                **({"capture_context": True} if host_matching is not None else {}),
            )
        if host_matching is not None:
            if not host_matching:
                host_matching["assignments"] = torch.full(
                    (pairs, query.identity.shape[1]), -1, dtype=torch.long
                )
                host_matching["selected"] = torch.zeros(
                    query.membership.shape, dtype=torch.bool
                )
            host_matching["assignments"].index_copy_(0, cpu_rows, matching.indices_cpu)
            if matching.context_flags_cpu is not None:
                end = query.reference.shape[1]
                offset = end - query.membership.shape[1] * query.membership.shape[2]
                selected_cpu = matching.context_flags_cpu[:, offset:end].unflatten(
                    1, query.membership.shape[1:]
                )
                host_matching["selected"].index_copy_(
                    0, cpu_rows // candidates, selected_cpu
                )
        offset = fafa_binding_scores(
            selected.fafa,
            native_chunk
            if isinstance(native_chunk, NativeTargetBatch)
            else native_chunk.to(device, non_blocking=True),
            matching=matching,
            k=model.fda_k,
            missing_penalty=model.fine_matching["missing_penalty"],
            pooling=pooling,
        )
        inputs = (indices, scene, persons, boxes, mask, matching.indices)
        if len(positions_cpu) > batch_size and torch.is_grad_enabled():
            values = checkpoint(score_pairs, *inputs, use_reentrant=False)
        else:
            values = score_pairs(*inputs)
        context.append(values)
        coverage.append(matching.coverage)
        binding.append(offset)
        assignments.append(matching.indices)
    shape = (queries, candidates)
    if not context:
        zero = query.reference.sum() * 0
        return (
            zero.expand(shape),
            zero.detach().expand(shape),
            zero.detach().expand(shape),
            torch.full((*shape, query.identity.shape[1]), -1, device=device),
        )

    def scatter(chunks, fill=0):
        values = torch.cat(chunks)
        output = values.new_full((pairs, *values.shape[1:]), fill)
        return output.index_copy(0, positions, values).unflatten(0, shape)

    return (
        scatter(context),
        scatter(coverage),
        scatter(binding),
        scatter(assignments, -1),
    )


def _swap_loss(model, query, batch, patch_hw, context, assignments, host_matching=None):
    """Rank a verified edited condition below the original on the same assignment."""
    pair = batch.get("direction_swap")
    zero = context.sum() * 0
    if pair is None:
        return zero, zero.detach()

    rows, cols = pair["positions"].unbind(1)
    host = (
        host_matching
        if host_matching and pair["positions"].device.type == "cpu"
        else None
    )
    if host is not None:
        indices = host["assignments"].unflatten(0, batch["positive_mask"].shape)[
            rows, cols
        ]
        selected = host["selected"][rows]
    else:
        rows, cols = (value.to(query.reference.device) for value in (rows, cols))
        indices = assignments[rows, cols]
        selected = query.membership[rows] > 0
    query_roles, target_roles = (
        pair[key].to(indices.device) for key in ("query_roles", "target_roles")
    )
    assigned_role = target_roles.gather(
        2, indices.clamp_min(0)[:, None].expand_as(query_roles)
    )
    correct = selected & (indices >= 0)[:, None] & query_roles & assigned_role
    valid = correct.any(2).all(1)
    if not valid.any():
        return zero, zero.detach()

    rows, cols, indices = rows[valid], cols[valid], indices[valid]
    positions_cpu = (
        rows * batch["positive_mask"].shape[1] + cols if host is not None else None
    )
    device = query.reference.device
    rows, cols, indices = (
        value.to(device, non_blocking=True) for value in (rows, cols, indices)
    )
    positions = valid.nonzero().flatten().to(device, non_blocking=True)
    alternative = model.recondition_query(
        query.index_select(rows),
        batch["query_persons"][rows],
        batch["query_person_mask"][rows],
        **{
            key: value.index_select(0, positions) for key, value in pair["text"].items()
        },
    )
    persons = batch["target_persons"][rows, cols]
    boxes = batch["target_boxes"][rows, cols]
    padding = ~batch["target_mask"][rows, cols, :, None]
    scene = batch["target_scene"]
    if isinstance(scene, SceneTargetBatch):
        positions = (
            positions_cpu
            if positions_cpu is not None
            else (rows * batch["positive_mask"].shape[1] + cols).detach().cpu()
        )
        scene = scene.load(positions).to(persons.device, non_blocking=True).float()
    else:
        scene = scene[rows, cols]
    wrong = model.score_aligned_context(
        alternative,
        scene,
        model.encode_semantic(persons.masked_fill(padding, 0)),
        boxes.masked_fill(padding, 0),
        indices,
        patch_hw,
    )
    return F.softplus(wrong - context[rows, cols]).mean(), valid.sum() / len(context)


@torch.no_grad()
def _verified_context_pairs(query, batch, assignments, valid, pairs):
    """Filter by equal verified ID sets for every active Subject."""
    query_ids = batch["query_identity_labels"]
    annotations = torch.stack(
        (
            batch["context_target_identity_labels"],
            batch["context_target_multi_head_crops"],
        ),
        dim=-1,
    )
    assigned = gather_matched(
        annotations.flatten(0, 1), assignments.flatten(0, 1)
    ).unflatten(0, valid.shape)
    labeled_target = valid[:, :, None] & (assignments >= 0) & (assigned[..., 0] >= 0)
    correct_id = labeled_target & (query_ids[:, None] == assigned[..., 0])

    active = query.subjects.bool()
    selected = (query.membership > 0) & active[..., None]
    labeled_query = selected.any(1) & (query_ids >= 0)
    owners = selected & batch["grounding_targets"].bool()
    has_owners = (owners.any(-1) | ~active).all(-1) & active.any(-1)
    owner_pairs = pairs & has_owners[:, None, None]

    # Use the first crop slot of each ID as its representative, even when a
    # different detection of that person was the crop that matched.
    same_id = query_ids[:, :, None] == query_ids[:, None, :]
    first = ~same_id.tril(-1).any(-1) & (query_ids >= 0)
    representative = same_id & first[:, None, :]
    verified = owners[:, None] & correct_id[:, :, None]
    members = (verified[..., None] & representative[:, None, None]).any(-2)
    counts = members.sum(-1)
    supported = ((counts > 0) | ~active[:, None]).all(-1)
    matched_pairs = owner_pairs & supported[:, :, None] & supported[:, None, :]

    member_sets = members.flatten(2)
    equal = (member_sets[:, :, None] == member_sets[:, None, :]).all(-1)
    verified_pairs = matched_pairs & equal
    complete = (
        (counts == batch["context_required_member_counts"][:, None]) | ~active[:, None]
    ).all(-1)
    full_pairs = verified_pairs & complete[:, :, None] & complete[:, None, :]
    diagnostics = {
        "fine_unfiltered_queries": pairs.any((1, 2)),
        "fine_unfiltered_pairs": pairs.sum((1, 2)),
        "fine_owner_pairs": owner_pairs.sum((1, 2)),
        "fine_matched_pairs": matched_pairs.sum((1, 2)),
        "fine_full_pairs": full_pairs.sum((1, 2)),
        "fine_partial_pairs": (verified_pairs & ~full_pairs).sum((1, 2)),
        "fine_active_queries": verified_pairs.any((1, 2)),
        "fine_labeled_query_crops": labeled_query.sum(1),
        "fine_multi_head_query_crops": (
            labeled_query & batch["context_query_multi_head_crops"]
        ).sum(1),
        "fine_labeled_target_matches": labeled_target.sum((1, 2)),
        "fine_multi_head_target_matches": (
            labeled_target & assigned[..., 1].bool()
        ).sum((1, 2)),
    }
    return verified_pairs, torch.stack(
        [diagnostics[key] for key in FINE_COUNT_FIELDS], dim=1
    )


def _fda_pair_error(binding: Tensor, pairs: Tensor) -> Tensor:
    """Share of eligible (positive, negative) pairs that FDA alone ranks wrong or ties.

    This is the headroom the context scorer can correct. It uses the same
    eligible pairs as the context ranking loss and carries no gradient.
    """
    wrong = (binding[:, None, :] >= binding[:, :, None]) & pairs
    return (wrong.sum() / pairs.sum().clamp_min(1)).detach()


@checked
def compute_loss(
    model: RCRModel,
    batch: dict[str, Tensor | dict | str | NativeTargetBatch],
    patch_hw: tuple[int, int],
    *,
    grounding_weight: float,
    identity_weight: float,
    coarse_weight: float,
    context_weight: float,
    state_weight: float,
    identity_temperature: float,
    coarse_temperature: float,
    state_temperature: float,
    context_rejection_weight: float = 0.0,
    rejection_loss_mode: str = "softplus",
    context_verify_members: bool = False,
    context_swap_weight: float = 0.0,
    context_temperature: float = 0.1,
    context_score_weight: float = DEFAULT_CONTEXT_WEIGHT,
    coverage_score_weight: float = DEFAULT_COVERAGE_WEIGHT,
    context_batch_size: int = 64,
):
    """Use only train labels in objectives; selection/assignment stay label-free."""
    if (
        context_weight
        and not {"fafa_query", "fafa_targets", "fafa_pooling"} <= batch.keys()
    ):
        raise ValueError(
            "context ranking requires native FAFA tokens in the batch; "
            "build it with fafa_pooling"
        )
    audit_context = "context_target_identity_labels" in batch
    if context_weight and context_verify_members and not audit_context:
        raise ValueError("verified context requires context member annotations")
    if context_weight:
        query = model.encode_query(
            batch["query_scene"],
            batch["query_persons"],
            batch["query_boxes"],
            batch["selections"],
            batch["change"],
            batch["subject_pos"],
            patch_hw,
            selection_mask=batch.get("selection_mask"),
            change_mask=batch.get("change_mask"),
            query_person_mask=batch.get("query_person_mask"),
            subject_mask=batch.get("subject_mask"),
            subject_token_mask=batch.get("subject_token_mask"),
            subject_ids=batch.get("subject_ids"),
            fafa_query=batch["fafa_query"],
        )
        logits, query_identity = query.logits, query.identity
    else:
        # Reuse the coarse query path; warmup needs no composition or fine matching.
        mask = batch["query_person_mask"][..., None]
        logits, query_identity = model.encode_grounded_identity(
            batch["query_scene"],
            batch["query_persons"].masked_fill(~mask, 0),
            batch["query_boxes"].masked_fill(~mask, 0),
            batch["selections"],
            patch_hw,
            selection_mask=batch.get("selection_mask"),
        )
    positive = batch["positive_mask"]
    identity_positive = batch["identity_positive_mask"]
    check_tensor(
        ~(positive & ~identity_positive).any(),
        "every Full Positive must also be an identity positive",
    )
    zero = query_identity.new_zeros(())
    ground, identity, coarse, context, state = [zero] * 5
    semantic_context_used = None
    semantic = rejection = swap = zero
    context_active = rejection_active = swap_active = fda_error = zero.detach()
    fine_counts = None
    hard_count = random_count = 0
    target_identity = None
    if identity_weight or coarse_weight:
        persons = batch["target_persons"]
        if "target_layout" in batch:
            positions = batch["target_layout"]["positions"].to(
                persons.device, non_blocking=True
            )
            projected_identity = model.encode_identity(
                persons.flatten(0, 2).index_select(0, positions)
            )
            target_identity = (
                projected_identity.new_zeros(
                    (batch["target_mask"].numel(), projected_identity.shape[-1])
                )
                .index_copy(0, positions, projected_identity)
                .reshape(*batch["target_mask"].shape, projected_identity.shape[-1])
            )
        else:
            target_identity = model.encode_identity(
                persons.masked_fill(~batch["target_mask"][..., None], 0)
            )

    if grounding_weight:
        subjects = batch.get("subject_mask")
        targets = batch["grounding_targets"]
        known = batch["query_person_mask"] & (batch["query_identity_labels"] >= 0)
        mask = known[:, None] & targets.any(-1, keepdim=True)
        if subjects is not None:
            mask = mask & subjects[:, :, None]
        ground = grounding_loss(logits, targets, mask)

    if identity_weight:
        layout = batch.get("identity_layout")
        layout = layout if layout is not None and layout.current(batch) else None
        source = layout.host if layout is not None else batch
        query_keep, target_keep = (
            source["query_identity_mask"],
            source["target_identity_mask"],
        )
        labels = [
            source["query_identity_labels"][query_keep],
            source["target_identity_labels"][target_keep],
        ]

        def take(values, indices):
            return values.index_select(0, indices.to(values.device, non_blocking=True))

        features = (
            [
                take(
                    query_identity.flatten(0, 1),
                    layout.positions["query_identity_mask"],
                ),
                take(
                    target_identity.flatten(0, 2),
                    layout.positions["target_identity_mask"],
                ),
            ]
            if layout is not None
            else [query_identity[query_keep], target_identity[target_keep]]
        )
        if "identity_sampling" in batch:
            hard, extra = select_identity_observations(
                query_identity, target_identity, batch, batch["identity_sampling"]
            )
            target_flat = target_identity.flatten(0, 2)
            # Index selection is detached; these gathers preserve head/projection
            # gradients. A crop present in targets needs no extra identity encode.
            features.append(take(target_flat, hard))
            target_labels = torch.full_like(source["target_identity_labels"], -1)
            target_labels.view(-1)[source["supplemental_target_indices"]] = source[
                "supplemental_target_labels"
            ]
            labels.append(target_labels.flatten()[hard])
            reused = source["extra_identity_target_indices"][extra]
            in_target = reused >= 0
            features.append(take(target_flat, reused[in_target]))
            if extra.numel():
                extra_labels = source["extra_identity_labels"][extra]
                labels.extend((extra_labels[in_target], extra_labels[~in_target]))
                outside = extra[~in_target]
                if outside.numel():
                    features.append(
                        model.encode_identity(
                            take(batch["extra_identity_persons"], outside)
                        )
                    )
                else:
                    features.append(target_flat[:0])
            hard_count, random_count = hard.numel(), extra.numel()
        elif source.get("extra_identity_labels", labels[0][:0]).numel():
            labels.append(source["extra_identity_labels"])
            features.append(model.encode_identity(batch["extra_identity_persons"]))
            random_count = batch["extra_identity_labels"].numel()
        identity = identity_loss(
            torch.cat(features),
            torch.cat(labels).to(query_identity.device, non_blocking=True),
            identity_temperature,
        )

    if coarse_weight:
        # Reuse the exact inference identity scorer without an unnecessary BxB
        # cross-query gallery. No state/relation label enters this identity task.
        with torch.autocast(query_identity.device.type, enabled=False):
            coarse_scores = identity_scores_paired(
                query_identity,
                logits.float(),
                target_identity,
                batch["target_mask"],
                query_mask=batch["query_person_mask"],
                subject_mask=batch.get("subject_mask"),
            )
        valid = batch["query_person_mask"].any(-1)[:, None] & batch["target_mask"].any(
            -1
        )
        if "subject_mask" in batch:
            valid = valid & batch["subject_mask"].any(-1)[:, None]
        coarse = (
            retrieval_loss(coarse_scores / coarse_temperature, identity_positive, valid)
            + query_identity.sum() * 0
        )

    if context_weight:
        eligible = identity_positive
        if context_rejection_weight:
            eligible = eligible | batch["hard_identity_negative_mask"]
        host_matching = (
            {} if context_swap_weight and "direction_swap" in batch else None
        )
        raw_context, coverage, binding, assignments = _context_scores(
            model,
            query,
            batch,
            patch_hw,
            context_batch_size,
            eligible=eligible,
            target_identity=target_identity,
            host_matching=host_matching,
            positions_cpu=(
                batch["context_layout"][
                    "rejection_rows" if context_rejection_weight else "identity_rows"
                ]
                if "context_layout" in batch
                else None
            ),
        )
        binding_scale = model.binding_scale
        scores = linear_score_components(
            binding,
            raw_context,
            coverage,
            binding_scale,
            context_score_weight,
            coverage_score_weight,
        )
        valid = identity_positive & (coverage > 0)
        pairs = _ranking_pairs(positive.bool(), valid)
        pair_options = {}
        if audit_context:
            verified_pairs, fine_counts = _verified_context_pairs(
                query, batch, assignments, valid, pairs
            )
            if context_verify_members:
                pairs = verified_pairs
                pair_options["pair_mask"] = pairs
        if "semantic_context_mask" in batch:
            semantic_context_used = (
                pairs.any(1) & batch["semantic_context_mask"]
            ).sum()
        semantic = retrieval_loss(
            scores["context_contribution"] / context_temperature,
            positive,
            valid,
            **pair_options,
        )
        context = semantic
        fda_error = _fda_pair_error(binding, pairs)
        context_active = pairs.any((1, 2)).float().mean()
        if context_rejection_weight:
            hard = batch["hard_identity_negative_mask"].bool()
            check_tensor(
                ~(hard & identity_positive).any(),
                "rejection negatives must lack required identities",
            )
            rejection_valid = (positive | hard) & (coverage > 0)
            if rejection_loss_mode == "softplus":
                rejection = retrieval_loss(
                    scores["fine_scores"] / context_temperature,
                    positive,
                    rejection_valid,
                )
            elif rejection_loss_mode == "hinge":
                with torch.no_grad():
                    base = (
                        binding_scale.float() * binding.float()
                        + scores["coverage_bonus"]
                    )
                rejection = rejection_hinge_loss(
                    base,
                    scores["context_contribution"],
                    positive,
                    rejection_valid,
                    context_temperature,
                )
            else:
                raise ValueError("loss.rejection_loss_mode must be softplus or hinge")
            rejection_active = (
                ((positive & rejection_valid).any(1) & (hard & rejection_valid).any(1))
                .float()
                .mean()
            )
            context = semantic + context_rejection_weight * rejection

        if context_swap_weight:
            swap, swap_active = _swap_loss(
                model, query, batch, patch_hw, raw_context, assignments, host_matching
            )
            context = context + context_swap_weight * swap

    if state_weight:
        text_state = model.encode_text_state(batch["change_global"])
        image_state = model.encode_image_state(batch["target_global"])
        state_scores = torch.einsum("bd,bcd->bc", text_state, image_state)
        state = retrieval_loss(
            state_scores / state_temperature, positive, identity_positive
        )

    total = (
        grounding_weight * ground
        + identity_weight * identity
        + coarse_weight * coarse
        + context_weight * context
        + state_weight * state
    )
    return total, {
        "grounding": ground.detach(),
        "identity": identity.detach(),
        "coarse": coarse.detach(),
        "context": context.detach(),
        "context_semantic": semantic.detach(),
        "context_rejection": rejection.detach(),
        "context_swap": swap.detach(),
        "context_fda_error": fda_error,
        "state": state.detach(),
        "context_active_fraction": context_active.detach(),
        "context_rejection_active_fraction": rejection_active.detach(),
        "context_swap_active_fraction": swap_active.detach(),
        "identity_hard_crops": hard_count,
        "identity_random_crops": random_count,
        "identity_supplemental_crops": hard_count + random_count,
        **(
            {"semantic_context_used": semantic_context_used}
            if semantic_context_used is not None
            else {}
        ),
        **({"fine_pair_counts": fine_counts} if fine_counts is not None else {}),
    }
