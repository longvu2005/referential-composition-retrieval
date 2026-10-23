"""Replay saved scores on CPU, including new shortlists from a full tuning export."""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch

from rcr.common.io import save_results, sha256_file
from rcr.evaluation.provenance import runtime_metadata
from rcr.proposed.ranking import (
    SCORE_EPS,
    SCORE_SCHEMA_VERSION,
    TUNING_CONVENTIONS,
    fine_score_components,
    fuse_fine_coarse_scores,
    rerank_topm,
)
from rcr.proposed.scores import combine_scores


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
    if type(config["top_m"]) is not int or config["top_m"] < 1:
        raise ValueError("top_m must be a positive integer")
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


def _validate_tuning(output: dict) -> dict:
    """Reject incomplete/full-gallery exports before changing any shortlist."""
    tuning = output["tuning_scores"]
    if not isinstance(tuning, dict) or any(
        tuning.get(key) != value for key, value in TUNING_CONVENTIONS.items()
    ):
        raise ValueError("incompatible full-gallery tuning_scores contract")
    required = {
        "query_indices",
        "identity_scores",
        "state_scores",
        "state_trained",
        "fixed_fine_matching",
        "fafa_binding",
        "context",
        "coverage",
    }
    if missing := required - tuning.keys():
        raise ValueError(
            f"tuning_scores is missing fields: {', '.join(sorted(missing))}"
        )
    n, g = len(output["sample_ids"]), len(output["gallery_ids"])
    indices = tuning.get("query_indices")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.shape != (n,)
        or indices.min() < 0
        or indices.max() >= g
    ):
        raise ValueError(
            "tuning query_indices must identify one gallery image per query"
        )
    if (output["coarse_rankings"].cpu() == indices.cpu()[:, None]).any():
        raise ValueError("tuning query_indices disagree with self-excluded rankings")
    if type(tuning.get("state_trained")) is not bool:
        raise ValueError(
            "tuning_scores must record whether the state branch is trained"
        )
    if not isinstance(tuning.get("fixed_fine_matching"), dict):
        raise ValueError("tuning_scores must record the fixed fine matching settings")
    if not isinstance(tuning.get("provenance", {}), dict):
        raise ValueError("tuning provenance must be a metadata dictionary")
    fields = ["identity_scores", "fafa_binding", "context", "coverage"]
    if tuning["state_trained"]:
        fields.append("state_scores")
    elif tuning.get("state_scores") is not None:
        raise ValueError("untrained state scores must not be exported")
    for key in fields:
        value = tuning.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n, g)
            or value.dtype != torch.float32
        ):
            raise ValueError(
                f"tuning {key} must be float32 aligned with the full gallery"
            )
    top = output["coarse_topm"].long().cpu()
    for key in ("fafa_binding", "context", "coverage"):
        value = tuning[key].cpu()
        if not torch.isfinite(value).all():
            raise ValueError(f"tuning {key} contains NaN/Inf")
        if not torch.equal(value.gather(1, top), output[key].cpu()):
            raise ValueError(f"tuning {key} is misaligned with saved Coarse Top-M")
        if value[torch.arange(n), indices.long().cpu()].count_nonzero():
            raise ValueError("self-excluded fine scores must use the zero sentinel")
    return tuning


@torch.inference_mode()
def replay_rankings(
    output: dict,
    *,
    fine_score_mode: str | None = None,
    context_weight: float | None = None,
    fine_coarse_weight: float | None = None,
    coarse_beta: float | None = None,
    top_m: int | None = None,
) -> dict:
    """Use full-gallery components for new beta/Top-M; retain schema-1 replay."""
    _validate_artifact(output)
    config = dict(output["score_config"])
    for key, value in (
        ("fine_score_mode", fine_score_mode),
        ("context_weight", context_weight),
        ("fine_coarse_weight", fine_coarse_weight),
        ("coarse_beta", coarse_beta),
        ("top_m", top_m),
    ):
        if value is not None:
            config[key] = value
    if type(config["top_m"]) is not int or config["top_m"] < 1:
        raise ValueError("top_m must be a positive integer")
    if not math.isfinite(config["coarse_beta"]) or config["coarse_beta"] < 0:
        raise ValueError("coarse_beta must be finite and nonnegative")
    tuning = _validate_tuning(output) if "tuning_scores" in output else None
    changed = any(
        config[key] != output["score_config"][key] for key in ("coarse_beta", "top_m")
    )
    if changed and tuning is None:
        raise ValueError(
            "changing coarse_beta/top_m requires a full-gallery --save-tuning export; "
            "saved Top-M scores cannot recover unseen candidates"
        )
    if (
        tuning is not None
        and config["coarse_beta"] != 0
        and not tuning["state_trained"]
    ):
        raise ValueError("checkpoint state branch is untrained; use coarse_beta=0")
    rows = []
    for i, saved_top in enumerate(output["coarse_topm"].cpu()):
        if tuning is None:
            top = saved_top.long()
            order = output["coarse_rankings"][i].cpu()
            coarse_topm = output["coarse_topm_scores"][i].cpu()
            raw = {
                key: output[key][i].cpu()
                for key in ("fafa_binding", "context", "coverage")
            }
        else:
            query_index = tuning["query_indices"][i].item()
            state = tuning["state_scores"]
            coarse = combine_scores(
                tuning["identity_scores"][i].cpu(),
                None if state is None else state[i].cpu(),
                beta=config["coarse_beta"],
                exclude_index=query_index,
            )
            order = torch.argsort(coarse, descending=True, stable=True)
            order = order[order != query_index]
            top = order[: config["top_m"]]
            coarse_topm = coarse[top]
            raw = {
                key: tuning[key][i].cpu()[top]
                for key in ("fafa_binding", "context", "coverage")
            }
        scored = fine_score_components(
            raw["fafa_binding"],
            raw["context"],
            raw["coverage"],
            output["binding_scale"],
            config["fine_score_mode"],
            config["context_weight"],
        )
        fused = fuse_fine_coarse_scores(
            scored["fine_scores"],
            coarse_topm,
            config["fine_coarse_weight"],
        )
        rows.append(
            {
                **raw,
                **scored,
                "coarse_rankings": order.to(torch.int32),
                "coarse_topm": top.to(torch.int32),
                "coarse_topm_scores": coarse_topm,
                "fused_topm_scores": fused,
                "fafa_rankings": rerank_topm(order, top, raw["fafa_binding"]).to(
                    torch.int32
                ),
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
    parser.add_argument("--coarse-beta", type=float)
    parser.add_argument("--top-m", type=int)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.rankings.parent.resolve():
        parser.error("use a separate output directory to retain the source artifact")
    saved = torch.load(args.rankings, map_location="cpu", weights_only=True)
    output = replay_rankings(
        saved,
        fine_score_mode=args.mode,
        context_weight=args.context_weight,
        fine_coarse_weight=args.fine_coarse_weight,
        coarse_beta=args.coarse_beta,
        top_m=args.top_m,
    )
    metadata_path = args.rankings.parent / "run.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    embedded = output.get("tuning_scores", {}).get("provenance", {})
    for key, value in embedded.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"run.json disagrees with embedded tuning provenance: {key}"
            )
    metadata = {**embedded, **metadata}
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
    metadata = deepcopy(metadata)
    if isinstance(metadata.get("config"), dict):
        metadata["config"]["retrieval"] = retrieval
        for key in (
            "selected_checkpoint_sha256",
            "selected_validation_sha256",
            "selected_retrieval",
        ):
            metadata["config"].pop(key, None)
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
                "scope": (
                    "full_gallery" if "tuning_scores" in output else "same_coarse_topm"
                ),
            },
        },
    )
    print(f"Saved {args.output_dir / 'rankings.pt'}")


if __name__ == "__main__":
    main()
