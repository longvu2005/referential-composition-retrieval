"""Retrieval batches align query/positive crops; gallery extras are separate."""

import copy

import pytest
import torch

from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.losses import compute_loss
from rcr.proposed.nn.model import RCRModel
from tests.proposed import test_batch as fixtures


class _RecordingSupervision(fixtures._Supervision):
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.calls = []

    def load_ids(self, image_ids, max_people):
        self.calls.append(list(image_ids))
        assert set(image_ids) <= self.allowed, "negative-only image requested GT"
        return super().load_ids(image_ids, max_people)


class _RecordingCache(fixtures._Cache):
    def load_groups(self, *groups):
        self.feature_groups = [indices.tolist() for indices in groups]
        return super().load_groups(*groups)

    def native_targets(self, indices, max_people):
        self.native_target_rows = indices.tolist()
        return super().native_targets(indices, max_people)


@pytest.mark.parametrize("seed_only", [False, True])
def test_negative_only_images_need_no_gt_but_keep_all_features(seed_only):
    sample = dict(fixtures._samples()[0])
    if seed_only:
        sample["target_image_id"] = sample.pop("positive_image_ids")[0]
    cache = _RecordingCache()
    supervision = _RecordingSupervision({"q1", "p1"})
    batch = prepare_visual_batch(
        [sample],
        [["n1", "p2", "p1"]],
        cache,
        supervision=supervision,
        state_image_ids=[{"p1", "n1"}],
    )

    # p2 has known crop identities, but is negative for this query.
    assert supervision.calls == [["q1"], ["p1"]]
    assert cache.feature_groups == [[0], [3, 4, 2]]
    assert cache.native_target_rows == [3, 4, 2]
    assert batch["target_persons"].shape == (1, 3, 3, 4)
    assert batch["fafa_targets"].shape[:3] == (1, 3, 3)
    assert batch["positive_mask"].tolist() == [[False, False, True]]
    assert batch["target_identity_labels"].tolist() == [
        [[-1, -1, -1], [-1, -1, -1], [1, 0, -1]]
    ]
    # State still supervises an ID-compatible negative without crop GT.
    assert batch["state_mask"].tolist() == [[True, False, True]]


def test_positive_labels_follow_each_query_and_deduplicate_crop_observations():
    supervision = _RecordingSupervision({"q1", "q2", "p1", "p2"})
    batch = prepare_visual_batch(
        fixtures._samples(),
        [["p2", "p1", "p1"], ["p1", "p2", "n2"]],
        fixtures._Cache(),
        supervision=supervision,
    )

    assert supervision.calls == [["q1", "q2"], ["p1", "p1", "p2"]]
    # A positive for another query still has unknown crop labels in this row.
    assert batch["target_identity_labels"].tolist() == [
        [[-1, -1, -1], [1, 0, -1], [1, 0, -1]],
        [[-1, -1, -1], [2, 3, -1], [-1, -1, -1]],
    ]
    assert batch["target_identity_mask"].tolist() == [
        [[False, False, False], [True, True, False], [False, False, False]],
        [[False, False, False], [True, True, False], [False, False, False]],
    ]


def test_skipping_negative_crop_labels_preserves_losses_and_model_gradients():
    torch.manual_seed(17)
    cache = fixtures._Cache()
    batch = build_batch(
        fixtures._samples(),
        [["p2", "p1", "n1"], ["p2", "n2", "p1"]],
        cache,
        fixtures._Tokenizer(),
        fixtures._TextEncoder(),
        "cpu",
        supervision=_RecordingSupervision({"q1", "q2", "p1", "p2"}),
        state_image_ids=[{"p1", "n1"}, {"p2", "n2"}],
    )
    batch = {key: value.detach() for key, value in batch.items()}
    legacy = {
        **batch,
        "target_identity_labels": batch["target_identity_labels"].clone(),
    }
    # Restore known negative crop labels, as the previous eager path did.
    legacy["target_identity_labels"][0, 0] = batch["target_identity_labels"][1, 0]
    legacy["target_identity_labels"][1, 2] = batch["target_identity_labels"][0, 1]

    model = RCRModel(dim=4, identity_dim=4, num_heads=2)
    legacy_model = copy.deepcopy(model)
    loss, parts = compute_loss(model, batch, cache.patch_hw)
    legacy_loss, legacy_parts = compute_loss(legacy_model, legacy, cache.patch_hw)
    assert torch.isfinite(loss)
    assert parts["state_pairs"] > 0
    torch.testing.assert_close(loss, legacy_loss)
    for key in parts:
        torch.testing.assert_close(parts[key], legacy_parts[key])

    loss.backward()
    legacy_loss.backward()
    for (name, parameter), (legacy_name, legacy_parameter) in zip(
        model.named_parameters(), legacy_model.named_parameters(), strict=True
    ):
        assert name == legacy_name
        if parameter.grad is None:
            assert legacy_parameter.grad is None, name
        else:
            assert legacy_parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, legacy_parameter.grad)
