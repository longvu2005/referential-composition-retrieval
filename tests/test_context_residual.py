"""CPU workflow tests for context residual."""

import torch

from rcr.proposed import train as training
from rcr.proposed.losses import _context_scores, retrieval_loss
from tests.helpers import _only_loss


def _encode(model, batch):
    return model.encode_query(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        batch["change"],
        batch["subject_pos"],
        (2, 2),
        selection_mask=batch.get("selection_mask"),
        change_mask=batch.get("change_mask"),
        query_person_mask=batch.get("query_person_mask"),
        subject_mask=batch.get("subject_mask"),
        subject_token_mask=batch.get("subject_token_mask"),
        subject_ids=batch.get("subject_ids"),
        fafa_query=batch["fafa_query"],
    )


def test_training_score_and_loss_equal_inference_formula(objective_batch):
    model, batch = objective_batch
    query = _encode(model, batch)
    weight, temperature = 0.2, 0.15
    scores, coverage, binding = _context_scores(model, query, batch, (2, 2), 64, weight)
    details = model.score_target(
        query.repeat_candidates(3),
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        (2, 2),
        batch["target_mask"].flatten(0, 1),
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
        pooling=batch["fafa_pooling"],
        context_weight=weight,
        return_details=True,
    )
    expected = model.binding_scale * binding + weight * (
        coverage * details["context"].view_as(scores).tanh() - (1 - coverage)
    )
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores.flatten(), details["score"], rtol=0, atol=0)
    settings = {**_only_loss("context"), "context_temperature": temperature}
    total, parts = training.compute_loss(
        model, batch, (2, 2), **settings, context_score_weight=weight
    )
    valid = batch["identity_positive_mask"] & (coverage > 0)
    torch.testing.assert_close(
        total, retrieval_loss(expected / temperature, batch["positive_mask"], valid)
    )
    assert not binding.requires_grad
    assert parts["context_fda_error"].item() == 1


def test_model_forward_uses_the_requested_residual_weight(objective_batch):
    model, batch = objective_batch
    query = _encode(model, batch)
    weight = 0.27
    target = {
        "target_scene": batch["target_scene"][:, 0],
        "target_persons": batch["target_persons"][:, 0],
        "target_boxes": batch["target_boxes"][:, 0],
        "fafa_targets": batch["fafa_targets"][:, 0],
    }
    _, actual = model(
        **{
            key: batch[key]
            for key in (
                "query_scene",
                "query_persons",
                "query_boxes",
                "selections",
                "change",
                "subject_pos",
                "selection_mask",
                "change_mask",
                "query_person_mask",
                "subject_mask",
                "subject_token_mask",
                "subject_ids",
                "fafa_query",
            )
            if key in batch
        },
        **target,
        target_mask=batch["target_mask"][:, 0],
        patch_hw=(2, 2),
        pooling=batch["fafa_pooling"],
        context_weight=weight,
    )
    expected = model.score_target(
        query,
        target["target_scene"],
        target["target_persons"],
        target["target_boxes"],
        (2, 2),
        batch["target_mask"][:, 0],
        fafa_targets=target["fafa_targets"],
        pooling=batch["fafa_pooling"],
        context_weight=weight,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
