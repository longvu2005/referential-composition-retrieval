"""Train grounding, crop identity, coarse identity, context and global state."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from rcr.proposed.identity import select_identity_observations
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import fafa_binding_scores, identity_scores


def loss_settings(config: dict) -> dict:
    """Resolve one objective; reject stale keys and invalid weights before training."""
    defaults = {
        "grounding_weight": 1.0,
        "identity_weight": 0.1,
        "coarse_weight": 1.0,
        "context_weight": 1.0,
        "context_rejection_weight": 0.0,
        "context_residual_scale": 0.0,
        "state_weight": 1.0,
        "identity_temperature": 0.1,
        "coarse_temperature": 0.1,
        "state_temperature": 0.1,
    }
    if unknown := config.keys() - defaults.keys():
        raise ValueError(f"unknown loss settings: {', '.join(sorted(unknown))}")
    settings = {**defaults, **config}
    for key, value in settings.items():
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
        if key.endswith("weight") and key != "context_rejection_weight"
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
    identity = F.normalize(identity[keep].float(), dim=-1)
    labels = labels[keep]
    if len(identity) < 2:
        return identity.sum() * 0

    similarity = identity @ identity.T / temperature
    diagonal = torch.eye(len(identity), device=identity.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None]) & ~diagonal
    masked_similarity = similarity.masked_fill(diagonal, -torch.inf)
    log_probability = masked_similarity - torch.logsumexp(
        masked_similarity, dim=1, keepdim=True
    )
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return identity.sum() * 0
    per_anchor = -(
        log_probability.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1)
    )
    return per_anchor[valid].mean()


def retrieval_loss(scores: Tensor, positive: Tensor, valid_mask: Tensor | None = None):
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
    if not torch.isfinite(scores).all():
        raise ValueError("ranking loss requires finite eligible scores")
    positive = positive.bool() & valid
    pairs = positive[:, :, None] & (valid & ~positive)[:, None]
    difference = scores[:, None, :] - scores[:, :, None]
    pair_loss = F.softplus(difference).masked_fill(~pairs, 0)
    counts = pairs.sum((1, 2))
    per_query = pair_loss.sum((1, 2)) / counts.clamp_min(1)
    return per_query.sum() / (counts > 0).sum().clamp_min(1)


def _context_scores(model, query, batch, patch_hw, batch_size, residual_scale=0.0):
    """Score bounded context chunks before computing the full ranking loss.

    With ``residual_scale > 0`` the ranking score is ``scale * B + coverage * context``.
    ``B`` is the frozen native FDA binding pooled over the same detached assignment
    that gates the context path. It is computed without autograd, so it is a constant
    offset: the loss gradient reaches only the context modules, and context is trained
    to correct pairs that FDA alone ranks wrongly.

    Returns ``(scores, coverage, binding)``; ``binding`` is zero when disabled.
    """
    queries, candidates = batch["positive_mask"].shape
    pairs = queries * candidates
    names = ["target_scene", "target_persons", "target_boxes", "target_mask"]
    residual = residual_scale > 0
    if residual:
        names.append("fafa_targets")
    targets = tuple(batch[name].flatten(0, 1) for name in names)
    pooling = batch["fafa_pooling"] if residual else None

    def score_pairs(indices, scene, persons, boxes, mask, fafa_targets=None):
        selected = query.index_select(indices)
        raw, matching = model.score_context(
            selected, scene, persons, boxes, patch_hw, mask
        )
        score = matching.coverage * raw
        binding = torch.zeros_like(score)
        if fafa_targets is not None:
            binding = fafa_binding_scores(
                selected.fafa,
                fafa_targets,
                matching=matching,
                k=model.fda_k,
                missing_penalty=model.fine_matching["missing_penalty"],
                pooling=pooling,
            )
            score = score + residual_scale * binding
        return score, matching.coverage, binding

    scores, coverage, binding = [], [], []
    for start in range(0, pairs, batch_size):
        stop = min(start + batch_size, pairs)
        indices = torch.arange(start, stop, device=query.reference.device) // candidates
        inputs = (indices, *(value[start:stop] for value in targets))
        if pairs > batch_size and torch.is_grad_enabled():
            values, covered, offset = checkpoint(
                score_pairs, *inputs, use_reentrant=False
            )
        else:
            values, covered, offset = score_pairs(*inputs)
        scores.append(values)
        coverage.append(covered)
        binding.append(offset)
    shape = (queries, candidates)
    return (
        torch.cat(scores).view(shape),
        torch.cat(coverage).view(shape),
        torch.cat(binding).view(shape),
    )


def _fda_pair_error(binding: Tensor, positive: Tensor, valid: Tensor) -> Tensor:
    """Share of eligible (positive, negative) pairs that FDA alone ranks wrong or ties.

    This is the headroom the residual context can correct. It uses the same
    eligible pairs as the context ranking loss and carries no gradient.
    """
    positive = positive.bool() & valid
    pairs = positive[:, :, None] & (valid & ~positive)[:, None]
    wrong = (binding[:, None, :] >= binding[:, :, None]) & pairs
    return (wrong.sum() / pairs.sum().clamp_min(1)).detach()


def compute_loss(
    model: RCRModel,
    batch: dict[str, Tensor | dict],
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
    context_residual_scale: float = 0.0,
    context_batch_size: int = 64,
):
    """Use only train labels in objectives; selection/assignment stay label-free."""
    residual = bool(context_weight) and context_residual_scale > 0
    if residual and not {"fafa_query", "fafa_targets", "fafa_pooling"} <= batch.keys():
        raise ValueError(
            "loss.context_residual_scale requires native FAFA tokens in the batch; "
            "build it with fafa_pooling"
        )
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
            fafa_query=batch["fafa_query"] if residual else None,
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
    if (positive & ~identity_positive).any():
        raise ValueError("every Full Positive must also be an identity positive")
    batch_size = len(positive)
    zero = query_identity.new_zeros(())
    ground, identity, coarse, context, state = [zero] * 5
    semantic = rejection = zero
    context_active = rejection_active = fda_error = zero.detach()
    hard_count = random_count = 0
    target_identity = None
    if identity_weight or coarse_weight:
        target_identity = model.encode_identity(
            batch["target_persons"].masked_fill(~batch["target_mask"][..., None], 0)
        )

    if grounding_weight:
        mask = batch["query_person_mask"][:, None].expand_as(logits)
        if "subject_mask" in batch:
            mask = mask & batch["subject_mask"][:, :, None]
        mask = mask & (batch["query_identity_labels"][:, None] >= 0)
        mask = mask & batch["grounding_targets"].any(-1, keepdim=True)
        ground = grounding_loss(logits, batch["grounding_targets"], mask)

    if identity_weight:
        query_keep, target_keep = (
            batch["query_identity_mask"],
            batch["target_identity_mask"],
        )
        labels = [
            batch["query_identity_labels"][query_keep],
            batch["target_identity_labels"][target_keep],
        ]
        features = [query_identity[query_keep], target_identity[target_keep]]
        if "identity_sampling" in batch:
            hard, extra = select_identity_observations(
                query_identity, target_identity, batch, batch["identity_sampling"]
            )
            target_flat = target_identity.flatten(0, 2)
            # Index selection is detached; these gathers preserve head/projection
            # gradients. A crop present in targets needs no extra identity encode.
            features.append(target_flat[hard])
            target_labels = torch.full_like(batch["target_identity_labels"], -1)
            target_labels.view(-1)[batch["supplemental_target_indices"]] = batch[
                "supplemental_target_labels"
            ]
            labels.append(target_labels.flatten()[hard])
            reused = batch["extra_identity_target_indices"][extra]
            in_target = reused >= 0
            features.append(target_flat[reused[in_target]])
            if extra.numel():
                extra_labels = batch["extra_identity_labels"][extra]
                labels.extend((extra_labels[in_target], extra_labels[~in_target]))
                outside = extra[~in_target]
                if outside.numel():
                    features.append(
                        model.encode_identity(batch["extra_identity_persons"][outside])
                    )
                else:
                    features.append(target_flat[:0])
            hard_count, random_count = hard.numel(), extra.numel()
        elif batch.get("extra_identity_labels", labels[0][:0]).numel():
            labels.append(batch["extra_identity_labels"])
            features.append(model.encode_identity(batch["extra_identity_persons"]))
            random_count = batch["extra_identity_labels"].numel()
        identity = identity_loss(
            torch.cat(features), torch.cat(labels), identity_temperature
        )

    if coarse_weight:
        # Reuse the exact inference identity scorer without an unnecessary BxB
        # cross-query gallery. No state/relation label enters this identity task.
        with torch.autocast(query_identity.device.type, enabled=False):
            coarse_scores = torch.stack(
                [
                    identity_scores(
                        query_identity[i],
                        logits[i].float(),
                        target_identity[i],
                        batch["target_mask"][i],
                        query_mask=batch["query_person_mask"][i],
                        subject_mask=None
                        if "subject_mask" not in batch
                        else batch["subject_mask"][i],
                    )
                    for i in range(batch_size)
                ]
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
        scores, coverage, binding = _context_scores(
            model,
            query,
            batch,
            patch_hw,
            context_batch_size,
            context_residual_scale if residual else 0.0,
        )
        valid = identity_positive & (coverage > 0)
        context = retrieval_loss(scores, positive, valid)
        semantic = context
        if residual:
            fda_error = _fda_pair_error(binding, positive, valid)
        context_active = (
            ((positive & valid).any(1) & (~positive & valid).any(1)).float().mean()
        )
        if context_rejection_weight:
            hard = batch["hard_identity_negative_mask"].bool()
            if (hard & identity_positive).any():
                raise ValueError("rejection negatives must lack required identities")
            rejection_valid = (positive | hard) & (coverage > 0)
            rejection = retrieval_loss(scores, positive, rejection_valid)
            rejection_active = (
                ((positive & rejection_valid).any(1) & (hard & rejection_valid).any(1))
                .float()
                .mean()
            )
            context = semantic + context_rejection_weight * rejection

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
        "context_fda_error": fda_error,
        "state": state.detach(),
        "context_active_fraction": context_active.detach(),
        "context_rejection_active_fraction": rejection_active.detach(),
        "identity_hard_crops": hard_count,
        "identity_random_crops": random_count,
        "identity_supplemental_crops": hard_count + random_count,
    }
