"""Evaluate saved or in-memory results with the shared benchmark protocol."""

import json

import torch

from rcr.common.data import load_rcr_data, split_fingerprint, split_samples
from rcr.common.io import output_directory, write_json
from rcr.evaluation.evaluate import evaluate_retrieval_output


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
        metadata_path = directory / "run.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fingerprint = metadata.get("split_sha256")
            if fingerprint is not None and fingerprint != split_fingerprint(
                data, cfg["split"]
            ):
                raise ValueError(
                    "benchmark text/labels changed since retrieval; "
                    "retrieve again before evaluating"
                )
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
