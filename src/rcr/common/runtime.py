"""Runtime and evaluation shared by proposed, CLIP and FAFA experiments."""

import json
from copy import deepcopy

import torch

from rcr.common.data import load_rcr_data, split_samples
from rcr.common.io import output_directory, write_json
from rcr.evaluation.evaluate import evaluate_retrieval_output


def override_config(cfg: dict, overrides: dict) -> dict:
    """Apply flat dotted keys without mutating the base experiment."""
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        parts, target = key.split("."), cfg
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown config key: {key}")
        target[parts[-1]] = value
    return cfg


def resolve_device(cfg):
    name = cfg["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def evaluate_run(cfg, *, data=None, samples=None, output=None, max_queries=None):
    """Evaluate in-memory or saved rankings through the same official protocol."""
    directory = output_directory(cfg)
    if data is None:
        data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    if samples is None:
        samples = split_samples(data, cfg["split"])
        if max_queries is not None:
            samples = samples[:max_queries]
    if output is None:
        output = torch.load(
            directory / "rankings.pt", map_location="cpu", weights_only=True
        )
    result = evaluate_retrieval_output(
        data, samples, output, cfg.get("candidate_ks", (500,)), split=cfg["split"]
    )
    write_json(directory / "metrics.json", result)
    print(
        f"{cfg.get('mode', cfg['method'])}/{cfg['split']}: "
        f"{json.dumps(result['overall'])}",
        flush=True,
    )
    return result
