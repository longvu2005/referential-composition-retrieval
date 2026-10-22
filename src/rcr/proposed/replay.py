"""Replay a complete saved Coarse Top-M on CPU, without model weights or caches."""

import argparse
import json
from pathlib import Path

import torch

from rcr.common.io import save_results, sha256_file
from rcr.evaluation.provenance import runtime_metadata
from rcr.proposed.ranking import (
    SCORE_EPS,
    SCORE_SCHEMA_VERSION,
    fine_score_components,
    fuse_fine_coarse_scores,
    rerank_topm,
)


def _validate_artifact(output: dict) -> None:
    if output.get("score_schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError(
            "offline replay requires score_schema_version=1; retrieve again"
        )
    required = {
        "sample_ids",
        "gallery_ids",
        "coarse_rankings",
        "coarse_topm",
        "coarse_topm_scores",
        "fafa_binding",
        "context",
        "coverage",
        "binding_scale",
        "score_config",
    }
    if missing := required - output.keys():
        raise ValueError(
            f"offline replay is missing fields: {', '.join(sorted(missing))}"
        )
    config = output["score_config"]
    if not isinstance(config, dict):
        raise ValueError("offline replay has an incompatible score_config")
    if (
        config.get("eps") != SCORE_EPS
        or config.get("std_unbiased") is not False
        or config.get("normalization") != "per_query_coarse_topm_population"
        or config.get("normalization_dtype") != "float32"
        or config.get("normalization_device") != "cpu"
        or config.get("score_alignment") != "coarse_topm"
        or config.get("binding_scale") != output["binding_scale"]
        or not {
            "fine_score_mode",
            "context_weight",
            "coarse_beta",
            "fine_coarse_weight",
            "top_m",
        }
        <= config.keys()
    ):
        raise ValueError("offline replay has an incompatible score_config")
    n, g = len(output["sample_ids"]), len(output["gallery_ids"])
    if (
        not n
        or g < 2
        or len(set(output["sample_ids"])) != n
        or len(set(output["gallery_ids"])) != g
    ):
        raise ValueError("offline replay requires non-empty unique sample/gallery IDs")
    top, coarse = output["coarse_topm"], output["coarse_rankings"]
    for key, value in (("coarse_topm", top), ("coarse_rankings", coarse)):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{key} must be a 2D integer tensor")
    if top.shape != (n, min(config["top_m"], g - 1)) or coarse.shape != (n, g - 1):
        raise ValueError(
            "offline replay requires the complete Coarse ranking and Top-M"
        )
    if not torch.equal(top, coarse[:, : top.shape[1]]):
        raise ValueError("coarse_topm must be the prefix of coarse_rankings")
    for row in coarse:
        if row.min() < 0 or row.max() >= g or row.unique().numel() != g - 1:
            raise ValueError(
                "coarse_rankings contains invalid or repeated gallery indices"
            )
    for key in ("coarse_topm_scores", "fafa_binding", "context", "coverage"):
        value = output[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != top.shape
            or value.dtype != torch.float32
        ):
            raise ValueError(f"{key} must be float32 with shape coarse_topm")


@torch.inference_mode()
def replay_rankings(
    output: dict,
    *,
    fine_score_mode: str | None = None,
    context_weight: float | None = None,
    fine_coarse_weight: float | None = None,
) -> dict:
    """Reweight only saved candidates; never infer Coarse scores from ranks."""
    _validate_artifact(output)
    config = dict(output["score_config"])
    for key, value in (
        ("fine_score_mode", fine_score_mode),
        ("context_weight", context_weight),
        ("fine_coarse_weight", fine_coarse_weight),
    ):
        if value is not None:
            config[key] = value
    rows = []
    for i, top in enumerate(output["coarse_topm"].cpu()):
        scored = fine_score_components(
            output["fafa_binding"][i].cpu(),
            output["context"][i].cpu(),
            output["coverage"][i].cpu(),
            output["binding_scale"],
            config["fine_score_mode"],
            config["context_weight"],
        )
        fused = fuse_fine_coarse_scores(
            scored["fine_scores"],
            output["coarse_topm_scores"][i].cpu(),
            config["fine_coarse_weight"],
        )
        order = output["coarse_rankings"][i].cpu()
        rows.append(
            {
                **scored,
                "fused_topm_scores": fused,
                "fine_rankings": rerank_topm(order, top, scored["fine_scores"]).to(
                    torch.int32
                ),
                "rankings": rerank_topm(order, top, fused).to(torch.int32),
            }
        )
    return {
        **output,
        "score_config": config,
        **{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("legacy_raw", "raw_gamma", "context_zscore", "both_zscore")
    )
    parser.add_argument("--context-weight", type=float)
    parser.add_argument("--fine-coarse-weight", type=float)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.rankings.parent.resolve():
        parser.error("use a separate output directory to retain the source artifact")
    saved = torch.load(args.rankings, map_location="cpu", weights_only=True)
    output = replay_rankings(
        saved,
        fine_score_mode=args.mode,
        context_weight=args.context_weight,
        fine_coarse_weight=args.fine_coarse_weight,
    )
    metadata_path = args.rankings.parent / "run.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    retrieval = {
        **metadata.get("retrieval", {}),
        **{
            key: output["score_config"][key]
            for key in (
                "fine_score_mode",
                "context_weight",
                "coarse_beta",
                "fine_coarse_weight",
                "top_m",
            )
        },
    }
    save_results(
        args.output_dir,
        output,
        {
            **metadata,
            **runtime_metadata(torch.device("cpu")),
            "score_schema_version": output["score_schema_version"],
            "score_config": output["score_config"],
            "binding_scale": output["binding_scale"],
            "fine_score_mode": output["score_config"]["fine_score_mode"],
            "context_weight": output["score_config"]["context_weight"],
            "retrieval": retrieval,
            "replay": {
                "source": str(args.rankings),
                "source_sha256": sha256_file(args.rankings),
                "scope": "same_coarse_topm",
            },
        },
    )
    print(f"Saved {args.output_dir / 'rankings.pt'}")


if __name__ == "__main__":
    main()
