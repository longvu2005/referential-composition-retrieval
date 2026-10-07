"""Plain YAML loading and explicit dotted overrides; no config inheritance."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """Apply existing dotted keys without mutating the base experiment."""
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


def validate_training_schedule(cfg: dict) -> None:
    """Reject no-op training and invalid validation intervals before any writes."""
    for key in ("epochs", "batch_size"):
        value = cfg["train"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    evaluation = cfg.get("evaluation", {})
    if evaluation.get("enabled", False):
        value = evaluation["every_epochs"]
        if type(value) is not int or value < 1:
            raise ValueError("evaluation.every_epochs must be a positive integer")
