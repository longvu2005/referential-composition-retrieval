"""Explicit model selection shared by training and checkpoint-based retrieval."""

from importlib import import_module
from pathlib import Path

from rcr.common.io import sha256_file
from rcr.evaluation.provenance import digest_json
from rcr.proposed.nn.model import RCRModel

DEFAULT_MODEL_CLASS = "rcr.proposed.nn.model:RCRModel"


def model_metadata(cfg: dict) -> dict:
    return {
        "variant": cfg.get("variant", "proposed"),
        "model_class": cfg.get("model_class", DEFAULT_MODEL_CLASS),
        **({"experiment": cfg["experiment"]} if "experiment" in cfg else {}),
    }


def create_model(cfg: dict, **dimensions) -> RCRModel:
    """Default to the original class; an experiment can name one subclass."""
    path = cfg.get("model_class", DEFAULT_MODEL_CLASS)
    module, separator, name = path.partition(":")
    if not separator:
        raise ValueError("model_class must be module:Class")
    cls = getattr(import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, RCRModel):
        raise ValueError("model_class must be an RCRModel subclass")
    return cls.from_config(cfg["model"], **dimensions)


def validate_checkpoint_model(cfg: dict, checkpoint: dict) -> None:
    """Legacy checkpoints are Proposed; explicit variant mismatches are errors."""
    saved = model_metadata(checkpoint["config"])
    metadata = checkpoint.get(
        "model_metadata",
        {"variant": "proposed", "model_class": DEFAULT_MODEL_CLASS},
    )
    if metadata != saved:
        raise ValueError("checkpoint model metadata disagrees with its configuration")
    for key in ("variant", "model_class"):
        if key in cfg and cfg[key] != saved[key]:
            raise ValueError(f"checkpoint {key} differs from the requested model")


def source_fingerprints(cfg: dict) -> dict[str, str]:
    """Hash optional experiment packages from disk for exact resume integrity."""
    result = {}
    for name in cfg.get("source_packages", []):
        package = Path(import_module(name).__file__).resolve().parent
        files = {
            str(path.relative_to(package)): sha256_file(path)
            for path in sorted(package.rglob("*.py"))
            if not path.name.startswith(".")
        }
        result[name] = digest_json(files)
    return result
