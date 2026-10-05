"""Training step for the proposed RCR model."""

import math

import torch
from torch import Tensor

from rcr.methods.proposed.losses import grounding_loss, identity_loss, retrieval_loss
from rcr.methods.proposed.model import RCRModel


def compute_loss(
    model: RCRModel,
    batch: dict[str, Tensor],
    patch_hw: tuple[int, int],
    grounding_weight: float = 1.0,
    identity_weight: float = 0.1,
    retrieval_weight: float = 1.0,
    identity_temperature: float = 0.1,
    state_weight: float = 1.0,
    state_temperature: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute grounding, identity, fine ranking and global state ranking losses."""

    if not math.isfinite(state_temperature) or state_temperature <= 0:
        raise ValueError("state_temperature must be finite and positive")
    if not math.isfinite(state_weight) or state_weight < 0:
        raise ValueError("state_weight must be finite and nonnegative")

    query_person_mask = batch.get("query_person_mask")
    subject_mask = batch.get("subject_mask")
    logits, query_identity, query, query_mask, prior = model.encode_query(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        batch["change"],
        batch["subject_pos"],
        patch_hw,
        selection_mask=batch.get("selection_mask"),
        change_mask=batch.get("change_mask"),
        query_person_mask=query_person_mask,
        subject_mask=subject_mask,
        subject_token_mask=batch.get("subject_token_mask"),
        subject_ids=batch.get("subject_ids"),
    )

    # Score B queries against C candidates as B*C independent pairs.
    b, c = batch["target_scene"].shape[:2]

    query = query[:, None].expand(-1, c, -1, -1).reshape(b * c, *query.shape[1:])
    query_mask = query_mask[:, None].expand(-1, c, -1).reshape(b * c, -1)
    prior = prior[:, None].expand(-1, c, -1).reshape(b * c, -1)

    target_mask = batch.get("target_mask")
    if target_mask is not None:
        target_mask = target_mask.flatten(0, 1)

    scores = model.score_target(
        query,
        query_mask,
        prior,
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        patch_hw,
        target_mask,
    ).reshape(b, c)

    person_mask = torch.ones_like(logits, dtype=torch.bool)
    if query_person_mask is not None:
        person_mask &= query_person_mask[:, None].bool()
    if subject_mask is not None:
        person_mask &= subject_mask[:, :, None].bool()
    # Unmatched detections are unknown, not annotated negative people.
    person_mask &= batch["query_identity_labels"][:, None] >= 0
    # A missed GT person is unknown, not evidence that every detected person
    # is a negative for that Subject. Skip its grounding row entirely.
    person_mask &= batch["grounding_targets"].bool().any(dim=-1, keepdim=True)

    loss_ground = grounding_loss(logits, batch["grounding_targets"], person_mask)
    query_labels = batch["query_identity_labels"]
    target_labels = batch["target_identity_labels"]
    query_keep = query_labels >= 0
    if query_person_mask is not None:
        query_keep &= query_person_mask.bool()
    if "query_identity_mask" in batch:
        query_keep &= batch["query_identity_mask"].bool()

    # Share the query/target label vocabulary and learn from positive images.
    # Retrieval negatives, padding, unknown IDs and duplicate crops are excluded.
    target_keep = (target_labels >= 0) & batch["positive_mask"][:, :, None].bool()
    if "target_mask" in batch:
        target_keep &= batch["target_mask"].bool()
    if "candidate_mask" in batch:
        target_keep &= batch["candidate_mask"][:, :, None].bool()
    if "target_identity_mask" in batch:
        target_keep &= batch["target_identity_mask"].bool()
    target_identity = model.identity_head(batch["target_persons"][target_keep])
    loss_identity = identity_loss(
        torch.cat((query_identity[query_keep], target_identity), dim=0),
        torch.cat((query_labels[query_keep], target_labels[target_keep]), dim=0),
        temperature=identity_temperature,
    )
    loss_retrieval = retrieval_loss(
        scores,
        batch["positive_mask"],
        valid_mask=batch.get("candidate_mask"),
    )

    # State sees only the condition. Compare positive/negative images that both
    # contain all required identities; wrong-ID images have unknown state labels.
    loss_state = scores.new_zeros(())
    state_pairs = scores.new_zeros((), dtype=torch.long)
    state_queries = scores.new_zeros((), dtype=torch.long)
    if state_weight != 0:
        if "state_mask" not in batch:
            raise ValueError("state loss requires a GT identity-based state_mask")
        state_mask = batch["state_mask"].bool()
        if state_mask.shape != batch["positive_mask"].shape:
            raise ValueError("state_mask must match candidate/positive shape")
        if "candidate_mask" in batch:
            state_mask = state_mask & batch["candidate_mask"].bool()
        positives = (batch["positive_mask"] & state_mask).sum(dim=-1)
        negatives = (~batch["positive_mask"] & state_mask).sum(dim=-1)
        state_pairs = (positives * negatives).sum()
        state_queries = ((positives > 0) & (negatives > 0)).sum()
        if state_pairs > 0:
            z_text = model.encode_text_state(batch["change"], batch.get("change_mask"))
            z_image = model.encode_image_state(batch["target_scene"].mean(dim=-2))
            state_scores = (z_text[:, None] * z_image).sum(dim=-1)
            loss_state = retrieval_loss(
                state_scores / state_temperature,
                batch["positive_mask"],
                valid_mask=state_mask,
            )

    loss = (
        grounding_weight * loss_ground
        + identity_weight * loss_identity
        + retrieval_weight * loss_retrieval
        + state_weight * loss_state
    )
    predicted = (logits.detach() >= 0) & person_mask
    expected = batch["grounding_targets"].bool() & person_mask
    active_subjects = (
        torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        if subject_mask is None
        else subject_mask.bool()
    )
    return loss, {
        "grounding": loss_ground.detach(),
        "identity": loss_identity.detach(),
        "retrieval": loss_retrieval.detach(),
        "state": loss_state.detach(),
        "grounding_tp": (predicted & expected).sum(),
        "grounding_fp": (predicted & ~expected).sum(),
        "grounding_fn": (~predicted & expected).sum(),
        "grounding_subjects": active_subjects.sum(),
        "grounding_supervised_subjects": person_mask.any(dim=-1).sum(),
        "grounding_complete_subjects": (
            batch.get("grounding_complete", torch.zeros_like(active_subjects))
            & active_subjects
        ).sum(),
        "state_pairs": state_pairs,
        "state_active_queries": state_queries,
        "queries": scores.new_tensor(b, dtype=torch.long),
    }
