"""Losses for Subject grounding, identity, fine retrieval and global state."""

import torch
import torch.nn.functional as F
from torch import Tensor

from rcr.proposed.nn.model import RCRModel


def grounding_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    logits = logits.float().masked_fill(~mask, 0)
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


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
    """Pairwise softplus ranking over every positive-negative pair."""
    scores = scores.float()
    valid = torch.ones_like(positive) if valid_mask is None else valid_mask.bool()
    positive = positive.bool() & valid
    pairs = positive[:, :, None] & (valid & ~positive)[:, None]
    if not pairs.any():
        return scores.sum() * 0
    scores = scores.masked_fill(~valid, 0)
    difference = scores[:, None, :] - scores[:, :, None]
    return F.softplus(difference[pairs]).mean()


def compute_loss(
    model: RCRModel,
    batch: dict[str, Tensor],
    patch_hw: tuple[int, int],
    grounding_weight=1.0,
    identity_weight=0.1,
    retrieval_weight=1.0,
    identity_temperature=0.1,
    state_weight=1.0,
    state_temperature=0.1,
):
    """Compute the complete objective described in ``docs/proposed_method.md``."""
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

    batch_size, candidates = batch["target_scene"].shape[:2]
    target_mask = batch["target_mask"].flatten(0, 1)
    scores = model.score_target(
        query.repeat_candidates(candidates),
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        patch_hw,
        target_mask,
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
    ).reshape(batch_size, candidates)

    grounding_mask = batch["query_person_mask"][:, None].expand_as(query.logits)
    if "subject_mask" in batch:
        grounding_mask = grounding_mask & batch["subject_mask"][:, :, None]
    grounding_mask = grounding_mask & (batch["query_identity_labels"][:, None] >= 0)
    grounding_mask = grounding_mask & batch["grounding_targets"].any(-1, keepdim=True)
    ground = grounding_loss(query.logits, batch["grounding_targets"], grounding_mask)

    query_keep = batch["query_identity_mask"]
    target_keep = batch["target_identity_mask"]
    labels = torch.cat(
        (
            batch["query_identity_labels"][query_keep],
            batch["target_identity_labels"][target_keep],
        )
    )
    extra_labels = batch.get("extra_identity_labels", labels[:0])
    labels = torch.cat((labels, extra_labels))
    identity = scores.new_zeros(())
    if identity_weight:
        features = [
            query.identity[query_keep],
            model.encode_identity(batch["target_persons"][target_keep]),
        ]
        if extra_labels.numel():
            features.append(model.encode_identity(batch["extra_identity_persons"]))
        identity = identity_loss(torch.cat(features), labels, identity_temperature)

    retrieval = retrieval_loss(scores, batch["positive_mask"])

    state = scores.new_zeros(())
    if state_weight:
        text_state = model.encode_text_state(batch["change"], batch["change_mask"])
        image_state = model.encode_image_state(batch["target_scene"].mean(-2))
        state_scores = torch.einsum("bd,bcd->bc", text_state, image_state)
        state = retrieval_loss(
            state_scores / state_temperature,
            batch["positive_mask"],
            batch["state_mask"],
        )

    total = (
        grounding_weight * ground
        + identity_weight * identity
        + retrieval_weight * retrieval
        + state_weight * state
    )
    parts = {
        "grounding": ground.detach(),
        "identity": identity.detach(),
        "retrieval": retrieval.detach(),
        "state": state.detach(),
    }
    return total, parts
