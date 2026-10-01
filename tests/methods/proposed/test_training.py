import pytest
import torch

from rcr.methods.proposed.losses import grounding_loss, identity_loss
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.training import compute_loss


def _boxes(*shape: int) -> torch.Tensor:
    boxes = torch.rand(*shape, 4)
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:] * (1 - boxes[..., :2])
    return boxes


def _batch() -> dict[str, torch.Tensor]:
    b, c, d = 2, 3, 8
    return {
        "query_scene": torch.randn(b, 6, d),
        "query_persons": torch.randn(b, 3, d),
        "query_boxes": _boxes(b, 3),
        "query_identity_labels": torch.tensor([[0, 1, -1], [0, 2, -1]]),
        "query_person_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "selections": torch.randn(b, 2, 4, d),
        "selection_mask": torch.ones(b, 2, 4, dtype=torch.bool),
        "grounding_targets": torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ),
        "change": torch.randn(b, 5, d),
        "change_mask": torch.ones(b, 5, dtype=torch.bool),
        "subject_pos": torch.tensor([[0, 3], [1, 4]]),
        "target_scene": torch.randn(b, c, 6, d),
        "target_persons": torch.randn(b, c, 4, d),
        "target_boxes": _boxes(b, c, 4),
        "target_mask": torch.ones(b, c, 4, dtype=torch.bool),
        "target_identity_labels": torch.tensor(
            [
                [[0, 1, -1, -1], [4, 5, -1, -1], [0, 1, -1, -1]],
                [[4, 5, -1, -1], [0, 2, -1, -1], [4, 5, -1, -1]],
            ]
        ),
        "positive_mask": torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        "candidate_mask": torch.ones(b, c, dtype=torch.bool),
    }


def test_compute_loss_and_backward() -> None:
    torch.manual_seed(0)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    batch = _batch()

    loss, parts = compute_loss(model, batch, patch_hw=(2, 3))

    assert loss.ndim == 0
    assert set(parts) == {"grounding", "identity", "retrieval", "state"}
    assert torch.isfinite(loss)

    loss.backward()
    assert model.grounding.score.weight.grad is not None
    assert model.identity_head.proj.weight.grad is not None
    assert model.composition.identity_proj.weight.grad is not None
    assert model.reasoner.score[-1].weight.grad is not None
    assert model.state_text_proj.weight.grad.norm() > 0
    assert model.state_image_proj.weight.grad.norm() > 0


def test_optimizer_step_changes_model() -> None:
    torch.manual_seed(1)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _batch()

    before = model.grounding.score.weight.detach().clone()
    loss, _ = compute_loss(model, batch, patch_hw=(2, 3))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.grounding.score.weight.detach())


def test_padding_has_no_effect_on_loss_or_gradients() -> None:
    torch.manual_seed(9)
    model = RCRModel(dim=8, identity_dim=6, num_heads=2).eval()
    batch = _batch()
    batch["subject_mask"] = torch.tensor([[True, False], [True, True]])
    batch["candidate_mask"] = torch.tensor([[True, True, False], [True, True, True]])
    batch["selections"].requires_grad_()
    batch["target_persons"].requires_grad_()
    expected, expected_parts = compute_loss(model, batch, (2, 3))
    expected.backward()
    assert batch["selections"].grad[0, 1].count_nonzero() == 0
    assert batch["target_persons"].grad[0, 2].count_nonzero() == 0

    altered = {key: value.detach().clone() for key, value in batch.items()}
    altered["selections"][0, 1] = 1000
    altered["grounding_targets"][0, 1] = 1
    altered["subject_pos"][0, 1] = 4
    altered["target_persons"][0, 2] = 1000
    altered["positive_mask"][0, 2] = False
    actual, actual_parts = compute_loss(model, altered, (2, 3))
    torch.testing.assert_close(actual, expected)
    for key in expected_parts:
        torch.testing.assert_close(actual_parts[key], expected_parts[key])


@pytest.mark.parametrize("zero_length", [True, False])
def test_empty_target_detections_have_finite_loss_and_gradients(zero_length):
    batch = _batch()
    if zero_length:
        for key in (
            "target_persons",
            "target_boxes",
            "target_mask",
            "target_identity_labels",
        ):
            batch[key] = batch[key][:, :, :0]
    else:
        batch["target_mask"][0, 0] = False
    model = RCRModel(dim=8, identity_dim=6, num_heads=2)
    loss, _ = compute_loss(model, batch, (2, 3))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_zero_query_people_and_all_zero_target_people() -> None:
    batch = _batch()
    for key in ("query_persons", "query_boxes"):
        batch[key] = batch[key][:, :0]
    for key in ("query_person_mask", "query_identity_labels", "grounding_targets"):
        batch[key] = batch[key][..., :0]
    for key in (
        "target_persons",
        "target_boxes",
        "target_mask",
        "target_identity_labels",
    ):
        batch[key] = batch[key][:, :, :0]
    model = RCRModel(8, 6, 2)
    loss, parts = compute_loss(model, batch, (2, 3))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(x) for x in parts.values())
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_positive_targets_supply_identity_pairs_and_gradients() -> None:
    torch.manual_seed(42)
    model = RCRModel(8, 6, 2).eval()
    batch = _batch()
    batch["query_identity_labels"] = torch.tensor([[0, 1, -1], [2, 3, -1]])
    batch["target_identity_labels"][1, 1] = torch.tensor([2, 3, -1, -1])
    # Exclude an extra positive candidate and a known-ID padded person.
    batch["candidate_mask"][0, 2] = False
    batch["target_identity_labels"][0, 0, 3] = 0
    batch["target_mask"][0, 0, 3] = False
    batch["query_persons"].requires_grad_()
    batch["target_persons"].requires_grad_()
    q = model.identity_head(batch["query_persons"])
    assert identity_loss(q, batch["query_identity_labels"]).item() == 0

    loss, parts = compute_loss(
        model,
        batch,
        (2, 3),
        grounding_weight=0,
        identity_weight=1,
        retrieval_weight=0,
        state_weight=0,
    )
    # Only four query observations and their four positive target observations.
    expected_features = torch.cat(
        (
            batch["query_persons"][:, :2].reshape(4, 8),
            batch["target_persons"][0, 0, :2],
            batch["target_persons"][1, 1, :2],
        )
    )
    expected = identity_loss(
        model.identity_head(expected_features), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    )
    torch.testing.assert_close(parts["identity"], expected)
    assert parts["identity"] > 0
    loss.backward()
    assert model.identity_head.proj.weight.grad.norm() > 0
    assert (batch["query_persons"].grad[:, :2].norm(dim=-1) > 0).all()
    assert batch["query_persons"].grad[:, 2:].count_nonzero() == 0
    grad = batch["target_persons"].grad
    assert (grad[0, 0, :2].norm(dim=-1) > 0).all()
    assert (grad[1, 1, :2].norm(dim=-1) > 0).all()
    assert grad[~batch["positive_mask"]].count_nonzero() == 0
    assert grad[0, 2].count_nonzero() == 0
    assert grad[:, :, 2:].count_nonzero() == 0


def test_identity_observation_masks_exclude_duplicate_crops() -> None:
    model = RCRModel(8, 6, 2).eval()
    batch = _batch()
    batch["query_identity_mask"] = torch.zeros_like(batch["query_person_mask"])
    batch["query_identity_mask"][0, :2] = True
    batch["target_identity_mask"] = torch.zeros_like(batch["target_mask"])
    _, parts = compute_loss(model, batch, (2, 3))
    # Two distinct query IDs, with all duplicate observations excluded: no pairs.
    assert parts["identity"].item() == 0


def test_all_unknown_identities_have_zero_identity_loss() -> None:
    model = RCRModel(8, 6, 2)
    batch = _batch()
    batch["query_identity_labels"].fill_(-1)
    batch["target_identity_labels"].fill_(-1)
    loss, parts = compute_loss(model, batch, (2, 3))
    assert parts["identity"].item() == 0
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_grounding_skips_subject_whose_gt_person_was_not_detected() -> None:
    model = RCRModel(8, 6, 2).eval()
    batch = _batch()
    batch["grounding_targets"][0, 0] = 0
    batch["selections"].requires_grad_()

    logits = model.grounding(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        batch["selections"],
        (2, 3),
        batch["selection_mask"],
    )
    valid = batch["query_person_mask"][:, None] & batch["grounding_targets"].bool().any(
        -1, keepdim=True
    )
    expected = grounding_loss(logits, batch["grounding_targets"], valid)

    loss, parts = compute_loss(
        model, batch, (2, 3), identity_weight=0, retrieval_weight=0, state_weight=0
    )
    torch.testing.assert_close(parts["grounding"], expected)
    loss.backward()
    assert batch["selections"].grad[0, 0].count_nonzero() == 0
    assert batch["selections"].grad[0, 1].count_nonzero() > 0


def test_state_loss_trains_text_and_image_but_ignores_candidate_padding() -> None:
    torch.manual_seed(71)
    model = RCRModel(8, 6, 2).eval()
    batch = _batch()
    batch["change"].requires_grad_()
    batch["target_scene"].requires_grad_()
    batch["change_mask"][0, -1] = False
    batch["candidate_mask"][0, 2] = False
    options = dict(grounding_weight=0, identity_weight=0, retrieval_weight=0)
    loss, parts = compute_loss(model, batch, (2, 3), **options)
    torch.testing.assert_close(loss.detach(), parts["state"])
    loss.backward()
    assert model.state_text_proj.weight.grad.norm() > 0
    assert model.state_image_proj.weight.grad.norm() > 0
    assert batch["change"].grad[0, -1].count_nonzero() == 0
    assert batch["change"].grad[0, :-1].norm() > 0
    assert batch["target_scene"].grad[0, 2].count_nonzero() == 0
    assert batch["target_scene"].grad[0, :2].norm() > 0
    altered = {key: value.detach().clone() for key, value in batch.items()}
    altered["change"][0, -1] = 1000
    altered["target_scene"][0, 2] = -1000
    altered["positive_mask"][0, 2] = False
    actual, _ = compute_loss(model, altered, (2, 3), **options)
    torch.testing.assert_close(actual, loss)


def test_state_loss_can_be_disabled_and_temperature_must_be_positive() -> None:
    model = RCRModel(8, 6, 2)
    _, parts = compute_loss(model, _batch(), (2, 3), state_weight=0)
    assert parts["state"] == 0
    with pytest.raises(ValueError, match="state_temperature"):
        compute_loss(model, _batch(), (2, 3), state_temperature=0)
