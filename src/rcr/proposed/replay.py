"""Rebuild fine rankings from saved Top-M scores on CPU."""

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch

from rcr.common.io import save_results, sha256_file
from rcr.evaluation.provenance import digest_json, runtime_metadata
from rcr.proposed.ranking import (
    SCORE_EPS,
    SCORE_SCHEMA_VERSION,
    fine_score_components,
    rerank_topm,
)
from rcr.proposed.scores import FINE_SCORE_MODE, POOLING_MODES

SHORTLIST_KEYS = ("coarse_beta", "top_m")
BINDING_KEYS = tuple(f"binding_{mode}" for mode in POOLING_MODES)


def _validate_settings(config: dict) -> None:
    if config.get("pooling") not in POOLING_MODES:
        raise ValueError("unknown FAFA pooling")
    if type(config.get("top_m")) is not int or config["top_m"] < 1:
        raise ValueError("top_m must be a positive integer")
    for key in ("coarse_beta", "context_weight", "binding_scale"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config.get("fine_score_mode") != FINE_SCORE_MODE:
        raise ValueError("unknown fine_score_mode")


def validate_artifact(output: dict) -> None:
    if not isinstance(output, dict) or any(
        type(output.get(key)) is not int or output[key] != SCORE_SCHEMA_VERSION
        for key in ("score_schema_version", "schema_version")
    ):
        raise ValueError(
            "offline replay requires "
            f"score_schema_version={SCORE_SCHEMA_VERSION} and "
            f"schema_version={SCORE_SCHEMA_VERSION}"
        )
    required = {
        "sample_ids",
        "gallery_ids",
        "query_indices",
        "fafa_binding",
        *BINDING_KEYS,
        "context",
        "coverage",
        "binding_scale",
        "score_config",
        "state_trained",
        "fine_matching",
        "rankings",
        "coarse_rankings",
        "coarse_topm",
        "provenance",
    }
    if missing := required - output.keys():
        raise ValueError(
            f"offline replay is missing fields: {', '.join(sorted(missing))}"
        )
    config = output["score_config"]
    if not isinstance(config, dict) or any(
        config.get(key) != value
        for key, value in {
            "eps": SCORE_EPS,
            "std_unbiased": False,
            "normalization": "none",
            "normalization_dtype": "float32",
            "normalization_device": "cpu",
            "binding_scale": output["binding_scale"],
        }.items()
    ):
        raise ValueError("offline replay has an incompatible score_config")
    if type(output["state_trained"]) is not bool:
        raise ValueError("artifact must record whether the state branch is trained")
    if not isinstance(output["fine_matching"], dict) or not isinstance(
        output["provenance"], dict
    ):
        raise ValueError("fine matching and provenance must be metadata dictionaries")
    for key in ("sample_ids", "gallery_ids"):
        fingerprint = output["provenance"].get(f"{key}_sha256")
        if fingerprint is not None and fingerprint != digest_json(output[key]):
            raise ValueError(f"{key} order changed since retrieval")
    _validate_settings(config)
    n, g = len(output["sample_ids"]), len(output["gallery_ids"])
    if (
        not n
        or g < 2
        or len(set(output["sample_ids"])) != n
        or len(set(output["gallery_ids"])) != g
    ):
        raise ValueError("offline replay requires non-empty unique sample/gallery IDs")
    m = min(config["top_m"], g - 1)
    indices = output["query_indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.shape != (n,)
        or indices.min() < 0
        or indices.max() >= g
    ):
        raise ValueError("query_indices must identify one gallery image per query")
    for key in ("fafa_binding", "context", "coverage", *BINDING_KEYS):
        value = output[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n, m)
            or value.dtype != torch.float32
        ):
            raise ValueError(f"{key} must be float32 aligned with coarse_topm")
        if value.device.type != "cpu":
            raise ValueError("offline replay requires CPU score tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains NaN/Inf")
    if not torch.equal(output["fafa_binding"], output[f"binding_{config['pooling']}"]):
        raise ValueError("fafa_binding disagrees with its recorded pooling")
    for key in ("rankings", "coarse_rankings", "coarse_topm"):
        value = output[key]
        width = m if key == "coarse_topm" else g - 1
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype not in (torch.int32, torch.int64)
            or value.shape != (n, width)
        ):
            raise ValueError(f"{key} must contain the complete ranking/Top-M")
        for i, row in enumerate(value):
            if (
                row.min() < 0
                or row.max() >= g
                or row.unique().numel() != width
                or (row == indices[i]).any()
            ):
                raise ValueError(
                    f"{key} contains self, invalid or repeated gallery indices"
                )
    if not torch.equal(output["coarse_topm"], output["coarse_rankings"][:, :m]):
        raise ValueError("coarse_topm must be the prefix of coarse_rankings")


@torch.inference_mode()
def replay_rankings(
    output: dict,
    *,
    context_weight: float | None = None,
    coarse_beta: float | None = None,
    top_m: int | None = None,
    pooling: str | None = None,
) -> dict:
    """Recreate the fine and FAFA rankings of one Top-M artifact.

    Pooling and context weight can be varied on the saved shortlist evidence. The
    coarse beta and Top-M define that shortlist, so another value needs retrieval.
    """
    validate_artifact(output)
    recorded = output["score_config"]
    for key, value in (("coarse_beta", coarse_beta), ("top_m", top_m)):
        if value is not None and value != recorded[key]:
            raise ValueError(
                f"{key} changes the Top-M shortlist; retrieve with the new value"
            )
    config = {
        **recorded,
        **{
            key: value
            for key, value in (("pooling", pooling), ("context_weight", context_weight))
            if value is not None
        },
    }
    _validate_settings(config)
    binding = output[f"binding_{config['pooling']}"]
    order, top = output["coarse_rankings"], output["coarse_topm"]
    scored = [
        fine_score_components(
            binding[i],
            output["context"][i],
            output["coverage"][i],
            output["binding_scale"],
            config["fine_score_mode"],
            config["context_weight"],
        )
        for i in range(len(top))
    ]
    fine_rankings = torch.stack(
        [
            rerank_topm(o, t, row["fine_scores"])
            for o, t, row in zip(order, top, scored, strict=True)
        ]
    )
    metadata = deepcopy(output["provenance"])
    if any(config[key] != recorded[key] for key in ("pooling", "context_weight")):
        metadata["fine_score_ablation"] = True
    retrieval = {
        **metadata.get("retrieval", {}),
        **{
            key: config[key]
            for key in (
                "pooling",
                "fine_score_mode",
                "context_weight",
                "coarse_beta",
                "top_m",
            )
        },
    }
    metadata.update(
        score_config=config,
        retrieval=retrieval,
        fine_score_mode=config["fine_score_mode"],
        context_weight=config["context_weight"],
    )
    if isinstance(metadata.get("config"), dict):
        metadata["config"]["retrieval"] = retrieval
        for key in (
            "selected_checkpoint_sha256",
            "selected_validation_sha256",
            "selected_retrieval",
        ):
            metadata["config"].pop(key, None)
    return {
        **output,
        **{key: torch.stack([row[key] for row in scored]) for key in scored[0]},
        "score_config": config,
        "fafa_binding": binding,
        "provenance": metadata,
        "fafa_rankings": torch.stack(
            [rerank_topm(o, t, b) for o, t, b in zip(order, top, binding, strict=True)]
        ),
        "fine_rankings": fine_rankings,
        "rankings": fine_rankings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-weight", type=float)
    parser.add_argument("--coarse-beta", type=float)
    parser.add_argument("--top-m", type=int)
    parser.add_argument("--pooling", choices=POOLING_MODES)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.rankings.parent.resolve():
        parser.error("use a separate output directory to retain the source artifact")
    try:
        output = replay_rankings(
            torch.load(args.rankings, map_location="cpu", weights_only=True),
            context_weight=args.context_weight,
            coarse_beta=args.coarse_beta,
            top_m=args.top_m,
            pooling=args.pooling,
        )
    except ValueError as error:
        parser.error(str(error))
    metadata = {
        **output["provenance"],
        **runtime_metadata(torch.device("cpu")),
        "replay": {
            "source": str(args.rankings),
            "source_sha256": sha256_file(args.rankings),
        },
    }
    output["provenance"] = metadata
    save_results(args.output_dir, output, metadata)
    print(f"Saved {args.output_dir / 'rankings.pt'}")


if __name__ == "__main__":
    main()
