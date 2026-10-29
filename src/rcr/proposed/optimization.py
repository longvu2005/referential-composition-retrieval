"""Epoch learning rates, independent of context and mining warmup."""

import math


def epoch_learning_rates(config: dict, epoch: int, epochs: int) -> tuple[float, float]:
    """Return rates actually used this epoch; cosine reaches its floor last.

    Constant is the default. Both AdamW groups share a decay factor, retaining
    the model/text LR ratio. A one-epoch run uses the configured base rates.
    """
    schedule = config.get("schedule", "constant")
    if schedule not in ("constant", "cosine"):
        raise ValueError("optimizer.schedule must be constant or cosine")
    minimum = config.get("min_lr_ratio", 0.1)
    if (
        type(minimum) not in (int, float)
        or not math.isfinite(minimum)
        or not 0 < minimum <= 1
    ):
        raise ValueError("optimizer.min_lr_ratio must be in (0, 1]")
    rates = config["lr"], config["text_lr"]
    if any(
        type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0
        for rate in rates
    ):
        raise ValueError("optimizer.lr and optimizer.text_lr must be nonnegative")
    if not 1 <= epoch <= epochs:
        raise ValueError("learning-rate epoch must be in [1, train.epochs]")
    factor = 1.0
    if schedule == "cosine" and epochs > 1:
        progress = (epoch - 1) / (epochs - 1)
        factor = minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return tuple(rate * factor for rate in rates)
