"""Learning-rate endpoint and validation contracts."""

import pytest

from rcr.proposed.optimization import epoch_learning_rates


def test_cosine_rates_decay_to_floor_without_resetting_text_ratio():
    cfg = {"lr": 1e-4, "text_lr": 1e-5, "schedule": "cosine", "min_lr_ratio": 0.1}
    rates = [epoch_learning_rates(cfg, epoch, 25) for epoch in range(1, 26)]
    assert rates[0] == (1e-4, 1e-5)
    assert rates[-1] == pytest.approx((1e-5, 1e-6))
    assert all(a[0] >= b[0] for a, b in zip(rates, rates[1:], strict=False))
    assert all(model / text == pytest.approx(10) for model, text in rates)


def test_constant_and_single_epoch_keep_base_rates():
    cfg = {"lr": 1e-4, "text_lr": 1e-5}
    assert epoch_learning_rates(cfg, 25, 25) == (1e-4, 1e-5)
    assert epoch_learning_rates({**cfg, "schedule": "cosine"}, 1, 1) == (1e-4, 1e-5)


@pytest.mark.parametrize(
    "setting",
    [
        {"schedule": "unknown"},
        {"min_lr_ratio": 0},
        {"min_lr_ratio": float("nan")},
        {"min_lr_ratio": 1.1},
        {"lr": -1},
        {"text_lr": float("inf")},
    ],
)
def test_invalid_learning_rate_settings(setting):
    with pytest.raises(ValueError, match="optimizer"):
        epoch_learning_rates({"lr": 1e-4, "text_lr": 1e-5, **setting}, 1, 25)
