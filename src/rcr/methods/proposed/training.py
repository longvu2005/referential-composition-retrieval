"""Training step for the proposed RCR model."""

import torch
from torch import Tensor

from rcr.methods.proposed.composition import composed_query_mask, reference_key_bias
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
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute masked grounding, identity and pairwise fine ranking losses."""

    query_identity = model.identity_head(batch["query_persons"])

    logits = model.grounding(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        patch_hw,
        batch.get("selection_mask"),
    )

    query_person_mask = batch.get("query_person_mask")
    composition_logits = logits
    if query_person_mask is not None:
        composition_logits = logits.masked_fill(
            ~query_person_mask[:, None].bool(), -torch.inf
        )

    subject_mask = batch.get("subject_mask")
    query = model.composition(
        batch["change"],
        query_identity,
        composition_logits,
        batch["subject_pos"],
        batch.get("change_mask"),
        subject_mask,
        batch.get("subject_token_mask"),
        batch.get("subject_ids"),
    )

    b, c = batch["target_scene"].shape[:2]

    def flat(x: Tensor) -> Tensor:
        return x.reshape(b * c, *x.shape[2:])

    query_mask = composed_query_mask(
        batch["change"],
        composition_logits,
        batch.get("change_mask"),
        subject_mask,
    )
    prior = reference_key_bias(
        batch["change"], composition_logits, batch.get("change_mask"), subject_mask
    )
    query = query[:, None].expand(-1, c, -1, -1).reshape(b * c, *query.shape[1:])
    query_mask = query_mask[:, None].expand(-1, c, -1).reshape(b * c, -1)
    prior = prior[:, None].expand(-1, c, -1).reshape(b * c, -1)

    target_mask = batch.get("target_mask")
    if target_mask is not None:
        target_mask = flat(target_mask)

    scores = model.score_target(
        query,
        query_mask,
        prior,
        flat(batch["target_scene"]),
        flat(batch["target_persons"]),
        flat(batch["target_boxes"]),
        patch_hw,
        target_mask,
    ).reshape(b, c)

    person_mask = torch.ones_like(logits, dtype=torch.bool)
    if query_person_mask is not None:
        person_mask &= query_person_mask[:, None].bool()
    if subject_mask is not None:
        person_mask &= subject_mask[:, :, None].bool()

    loss_ground = grounding_loss(logits, batch["grounding_targets"], person_mask)
    loss_identity = identity_loss(
        query_identity,
        batch["query_identity_labels"],
        temperature=identity_temperature,
    )
    loss_retrieval = retrieval_loss(
        scores,
        batch["positive_mask"],
        valid_mask=batch.get("candidate_mask"),
    )

    loss = (
        grounding_weight * loss_ground
        + identity_weight * loss_identity
        + retrieval_weight * loss_retrieval
    )
    return loss, {
        "grounding": loss_ground.detach(),
        "identity": loss_identity.detach(),
        "retrieval": loss_retrieval.detach(),
    }
