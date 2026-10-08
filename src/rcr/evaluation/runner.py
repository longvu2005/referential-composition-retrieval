"""Evaluate saved or in-memory results with the shared benchmark protocol."""

import json

import torch

from rcr.common.data import load_rcr_data, split_fingerprint, split_samples
from rcr.common.io import output_directory, preserve_file, sha256_file, write_json
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.provenance import BENCHMARK_FIELDS


def evaluate_run(cfg, *, data=None, samples=None, output=None, max_queries=None):
    """Evaluate in-memory or saved rankings through the same official protocol."""
    directory = output_directory(cfg)
    if data is None:
        data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    if samples is None:
        samples = split_samples(data, cfg["split"])
        if max_queries is not None:
            samples = samples[:max_queries]
    metadata_path = directory / "run.json"
    metadata = None
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
    if output is None:
        output = torch.load(
            directory / "rankings.pt", map_location="cpu", weights_only=True
        )
    result = evaluate_retrieval_output(
        data, samples, output, cfg.get("candidate_ks", (500,)), split=cfg["split"]
    )
    if metadata is not None:
        result["provenance"] = {
            "run_sha256": sha256_file(metadata_path),
            **{key: metadata[key] for key in BENCHMARK_FIELDS if key in metadata},
        }
    preserve_file(directory / "metrics.json")
    write_json(directory / "metrics.json", result)
    print(
        f"{cfg.get('mode', cfg['method'])}/{cfg['split']}: "
        f"{json.dumps(result['overall'])}",
        flush=True,
    )
    return result
