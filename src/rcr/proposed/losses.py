"""Train grounding, crop identity, coarse identity, context and global state."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import identity_scores


def loss_settings(config: dict) -> dict:
    """Resolve one objective; reject stale keys and invalid weights before training."""
    defaults = {
        "grounding_weight": 1.0,
        "identity_weight": 0.1,
        "coarse_weight": 1.0,
        "context_weight": 1.0,
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
    if not any(value for key, value in settings.items() if key.endswith("weight")):
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


def compute_loss(
    model: RCRModel,
    batch: dict[str, Tensor],
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
):
    """Use only train labels in objectives; selection/assignment stay label-free."""
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
    )
    positive = batch["positive_mask"]
    identity_positive = batch["identity_positive_mask"]
    if (positive & ~identity_positive).any():
        raise ValueError("every Full Positive must also be an identity positive")
    batch_size, candidates = positive.shape
    zero = query.identity.new_zeros(())
    ground, identity, coarse, context, state = [zero] * 5
    context_active = zero.detach()
    target_identity = None
    if identity_weight or coarse_weight:
        target_identity = model.encode_identity(
            batch["target_persons"].masked_fill(~batch["target_mask"][..., None], 0)
        )

    if grounding_weight:
        mask = batch["query_person_mask"][:, None].expand_as(query.logits)
        if "subject_mask" in batch:
            mask = mask & batch["subject_mask"][:, :, None]
        mask = mask & (batch["query_identity_labels"][:, None] >= 0)
        mask = mask & batch["grounding_targets"].any(-1, keepdim=True)
        ground = grounding_loss(query.logits, batch["grounding_targets"], mask)

    if identity_weight:
        query_keep, target_keep = (
            batch["query_identity_mask"],
            batch["target_identity_mask"],
        )
        labels = [
            batch["query_identity_labels"][query_keep],
            batch["target_identity_labels"][target_keep],
        ]
        features = [query.identity[query_keep], target_identity[target_keep]]
        if batch.get("extra_identity_labels", labels[0][:0]).numel():
            labels.append(batch["extra_identity_labels"])
            features.append(model.encode_identity(batch["extra_identity_persons"]))
        identity = identity_loss(
            torch.cat(features), torch.cat(labels), identity_temperature
        )

    if coarse_weight:
        # Reuse the exact inference identity scorer without an unnecessary BxB
        # cross-query gallery. No state/relation label enters this identity task.
        with torch.autocast(query.identity.device.type, enabled=False):
            coarse_scores = torch.stack(
                [
                    identity_scores(
                        query.identity[i],
                        query.logits[i].float(),
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
            + query.identity.sum() * 0
        )

    if context_weight:
        raw_context, matching = model.score_context(
            query.repeat_candidates(candidates),
            batch["target_scene"].flatten(0, 1),
            batch["target_persons"].flatten(0, 1),
            batch["target_boxes"].flatten(0, 1),
            patch_hw,
            batch["target_mask"].flatten(0, 1),
        )
        scores = (matching.coverage * raw_context).reshape(batch_size, candidates)
        valid = identity_positive & (
            matching.coverage.reshape(batch_size, candidates) > 0
        )
        context = retrieval_loss(scores, positive, valid)
        context_active = (
            ((positive & valid).any(1) & (~positive & valid).any(1)).float().mean()
        )

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
        "state": state.detach(),
        "context_active_fraction": context_active.detach(),
    }
