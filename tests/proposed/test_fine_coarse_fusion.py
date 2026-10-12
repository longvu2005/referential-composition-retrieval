import pytest
import torch

from rcr.proposed.ranking import (
    fuse_fine_coarse_scores,
    retrieval_settings,
)


def test_weight_zero_exactly_preserves_historical_fine_scores() -> None:
    fine = torch.tensor([0.2, 0.9, -0.3])
    coarse = torch.tensor([1.0, float("-inf"), -2.0])

    fused = fuse_fine_coarse_scores(fine, coarse, 0.0)

    assert fused.data_ptr() == fine.data_ptr()
    torch.testing.assert_close(fused, fine)


def test_positive_weight_can_restore_strong_coarse_evidence() -> None:
    # Fine alone prefers index 0, coarse alone prefers index 2. A sufficiently
    # large fusion weight should restore the coarse-preferred candidate.
    fine = torch.tensor([3.0, 2.0, 1.0])
    coarse = torch.tensor([-1.0, 0.0, 1.0])

    fused = fuse_fine_coarse_scores(fine, coarse, 2.0)

    assert fused.argsort(descending=True).tolist() == [2, 1, 0]


def test_nonfinite_coarse_candidate_cannot_be_promoted_by_fine() -> None:
    fine = torch.tensor([0.1, 10.0])
    coarse = torch.tensor([0.5, float("-inf")])

    fused = fuse_fine_coarse_scores(fine, coarse, 1.0)

    assert torch.isfinite(fused[0])
    assert torch.isneginf(fused[1])
    assert fused.argsort(descending=True).tolist() == [0, 1]


def test_constant_branches_are_safe() -> None:
    fine = torch.tensor([2.0, 2.0, 2.0])
    coarse = torch.tensor([1.0, 1.0, 1.0])

    fused = fuse_fine_coarse_scores(fine, coarse, 0.5)

    torch.testing.assert_close(fused, torch.zeros(3))


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="fine_coarse_weight"):
        fuse_fine_coarse_scores(torch.ones(2), torch.ones(2), -0.1)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("weight", [0.0, 0.4])
def test_nonfinite_fine_scores_never_become_rankings(invalid, weight):
    with pytest.raises(ValueError, match="fine scores must be finite"):
        fuse_fine_coarse_scores(
            torch.tensor([0.5, invalid]), torch.tensor([1.0, -torch.inf]), weight
        )


def test_old_configs_default_to_fine_only() -> None:
    settings = retrieval_settings(
        {"coarse_mode": "identity_state", "coarse_beta": None}, default_beta=0.4
    )

    assert settings["coarse_beta"] == 0.4
    assert settings["fine_coarse_weight"] == 0.0
