"""Plain YAML loading and explicit dotted overrides; no config inheritance."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

from rcr.proposed.contracts import CALIBRATION_PARAMETERS


def load_config(path: str | Path) -> dict:
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return cfg


def save_config(path: str | Path, cfg: dict) -> None:
    """Write an effective YAML configuration atomically, preserving key order."""
    from rcr.common.io import atomic_write_text

    atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False))


def override_config(cfg: dict, overrides: dict) -> dict:
    """Override known dotted keys without mutating the base experiment."""
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        parts, target = key.split("."), cfg
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"unknown config key: {key}")
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise ValueError(f"unknown config key: {key}")
        target[parts[-1]] = value
    return cfg


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        if key in overrides:
            raise ValueError(f"duplicate override: {key}")
        try:
            overrides[key] = yaml.safe_load(value)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML for override {key}: {error}") from error
    return overrides


def calibration_settings(settings: dict) -> dict:
    """Validate coefficient grids without importing model code."""
    if not isinstance(settings, dict):
        raise ValueError("evaluation.calibration must be a mapping")
    if unknown := settings.keys() - CALIBRATION_PARAMETERS.keys():
        raise ValueError(f"unknown calibration settings: {', '.join(sorted(unknown))}")
    result = {}
    for key in CALIBRATION_PARAMETERS:
        weights = settings.get(key)
        if weights is not None and (
            not isinstance(weights, list)
            or not weights
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in weights
            )
            or len(set(weights)) != len(weights)
        ):
            raise ValueError(f"{key} must be distinct finite nonnegative weights")
        result[key] = None if weights is None else list(weights)
    return result


def checkpoint_settings(settings: dict) -> dict:
    """Validate checkpoint cadence independently of the numerical experiment."""
    if not isinstance(settings, dict) or settings.keys() - {"every_steps"}:
        raise ValueError("checkpointing accepts only every_steps")
    value = settings.get("every_steps", 50)
    if type(value) is not int or value < 1:
        raise ValueError("checkpointing.every_steps must be a positive integer")
    return {"every_steps": value}


def validate_training_schedule(cfg: dict) -> None:
    """Reject no-op training, bad seeds and invalid intervals before any writes."""
    for key in ("epochs", "batch_size"):
        value = cfg["train"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    seed = cfg["train"].get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("train.seed must be a non-negative integer")
    deterministic = cfg.get("runtime", {}).get("deterministic", True)
    if type(deterministic) is not bool:
        raise ValueError("runtime.deterministic must be true or false")
    context_batch_size = cfg["train"].get("context_batch_size", 64)
    if type(context_batch_size) is not int or context_batch_size < 1:
        raise ValueError("train.context_batch_size must be a positive integer")
    prefetch = cfg["train"].get("prefetch_batches", 0)
    if type(prefetch) is not int or prefetch not in (0, 1):
        raise ValueError("train.prefetch_batches must be 0 or 1")
    candidates = cfg["train"].get("candidates", 2)
    if type(candidates) is not int or candidates < 2:
        raise ValueError("train.candidates must be an integer >= 2")
    warmup = cfg["train"].get("context_warmup_epochs", 0)
    if type(warmup) is not int or not 0 <= warmup < cfg["train"]["epochs"]:
        raise ValueError(
            "train.context_warmup_epochs must be an integer >= 0 and < train.epochs"
        )
    minimum = cfg.get("optimizer", {}).get("min_lr_ratio", 0.1)
    if type(minimum) not in (int, float) or not 0 <= minimum < 1:
        raise ValueError("optimizer.min_lr_ratio must be in [0, 1)")
    evaluation = cfg.get("evaluation", {})
    checkpoint_settings(cfg.get("checkpointing", {}))
    calibration_settings(evaluation.get("calibration", {}))
    if evaluation.get("enabled", False):
        value = evaluation["every_epochs"]
        if type(value) is not int or value < 1:
            raise ValueError("evaluation.every_epochs must be a positive integer")
