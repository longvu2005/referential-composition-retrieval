"""Rebuild rankings from schema-1 full-gallery raw scores on CPU."""

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
    _score_statistics,
    _zscore_1d,
    fine_score_components,
    fuse_fine_coarse_scores,
    rerank_topm,
)
from rcr.proposed.scores import combine_scores


def _validate_settings(config: dict, state_trained: bool) -> None:
    if type(config.get("top_m")) is not int or config["top_m"] < 1:
        raise ValueError("top_m must be a positive integer")
    for key in ("coarse_beta", "fine_coarse_weight", "context_weight", "binding_scale"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config.get("fine_score_mode") not in (
        "legacy_raw",
        "context_zscore",
        "raw_gamma",
        "both_zscore",
    ):
        raise ValueError("unknown fine_score_mode")
    if not state_trained and config["coarse_beta"] != 0:
        raise ValueError("checkpoint state branch is untrained; use coarse_beta=0")


def _validate_artifact(output: dict) -> None:
    if not isinstance(output, dict) or any(
        type(output.get(key)) is not int or output[key] != SCORE_SCHEMA_VERSION
        for key in ("score_schema_version", "schema_version")
    ):
        raise ValueError(
            "offline replay requires score_schema_version=1 and schema_version=1"
        )
    required = {
        "sample_ids",
        "gallery_ids",
        "query_indices",
        "identity_scores",
        "state_scores",
        "fafa_binding",
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
            "normalization": "per_query_coarse_topm_population",
            "normalization_dtype": "float32",
            "normalization_device": "cpu",
            "binding_scale": output["binding_scale"],
        }.items()
    ):
        raise ValueError("offline replay has an incompatible score_config")
    if config.get("score_alignment") != "gallery_ids":
        raise ValueError(
            "offline replay requires full-gallery raw scores; retrieve validation again"
        )
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
    _validate_settings(config, output["state_trained"])
    n, g = len(output["sample_ids"]), len(output["gallery_ids"])
    if (
        not n
        or g < 2
        or len(set(output["sample_ids"])) != n
        or len(set(output["gallery_ids"])) != g
    ):
        raise ValueError("offline replay requires non-empty unique sample/gallery IDs")
    indices = output["query_indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.shape != (n,)
        or indices.min() < 0
        or indices.max() >= g
    ):
        raise ValueError("query_indices must identify one gallery image per query")
    for key in (
        "identity_scores",
        "state_scores",
        "fafa_binding",
        "context",
        "coverage",
    ):
        value = output[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n, g)
            or value.dtype != torch.float32
        ):
            raise ValueError(f"{key} must be float32 aligned with the full gallery")
        if value.device.type != "cpu":
            raise ValueError("offline replay requires CPU score tensors")
        if key in ("fafa_binding", "context", "coverage"):
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")
            if value[torch.arange(n), indices.long()].count_nonzero():
                raise ValueError("self-excluded fine scores must use the zero sentinel")
    for key in ("rankings", "coarse_rankings", "coarse_topm"):
        value = output[key]
        width = min(config["top_m"], g - 1) if key == "coarse_topm" else g - 1
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
    if not torch.equal(
        output["coarse_topm"],
        output["coarse_rankings"][:, : min(config["top_m"], g - 1)],
    ):
        raise ValueError("coarse_topm must be the prefix of coarse_rankings")


@torch.inference_mode()
def _replay_rankings(output: dict, config: dict, cache: dict) -> dict:
    """Replay validated scores, retaining only the current beta/shortlist/mode.

    Calibration groups beta, Top-M, context weight, then fusion weight so this
    bounded cache reuses coarse sorting, gathered evidence, population statistics,
    fine diagnostics and fusion z-scores across the grid.
    """
    beta = config["coarse_beta"]
    top_m = min(config["top_m"], len(output["gallery_ids"]) - 1)
    if cache.get("source") is not output or cache.get("beta") != beta:
        coarse_rows, orders = [], []
        for i, query_index in enumerate(output["query_indices"].tolist()):
            coarse = combine_scores(
                output["identity_scores"][i],
                output["state_scores"][i],
                beta=beta,
                exclude_index=query_index,
            )
            order = torch.argsort(coarse, descending=True, stable=True)
            coarse_rows.append(coarse)
            orders.append(order[order != query_index].to(torch.int32))
        cache.clear()
        cache.update(
            source=output,
            beta=beta,
            coarse=torch.stack(coarse_rows),
            order=torch.stack(orders),
        )
    order = cache["order"]
    if cache.get("top_m") != top_m:
        top = order[:, :top_m]
        raw = {
            key: output[key].gather(1, top.long())
            for key in ("fafa_binding", "context", "coverage")
        }
        cache.update(
            top_m=top_m,
            top=top,
            raw=raw,
            coarse_topm=cache["coarse"].gather(1, top.long()),
            context_statistics=[
                _score_statistics(c * s)
                for c, s in zip(raw["coverage"], raw["context"], strict=True)
            ],
            fafa_rankings=torch.stack(
                [
                    rerank_topm(o, t, b)
                    for o, t, b in zip(order, top, raw["fafa_binding"], strict=True)
                ]
            ),
        )
        cache.pop("fine_key", None)
    top, coarse_topm = cache["top"], cache["coarse_topm"]
    fine_key = config["fine_score_mode"], config["context_weight"]
    if cache.get("fine_key") != fine_key:
        scored = [
            fine_score_components(
                cache["raw"]["fafa_binding"][i],
                cache["raw"]["context"][i],
                cache["raw"]["coverage"][i],
                output["binding_scale"],
                *fine_key,
                context_statistics=statistics,
            )
            for i, statistics in enumerate(cache["context_statistics"])
        ]
        cache.update(
            fine_key=fine_key,
            scored={
                key: torch.stack([row[key] for row in scored]) for key in scored[0]
            },
            fine_rankings=torch.stack(
                [
                    rerank_topm(o, t, row["fine_scores"])
                    for o, t, row in zip(order, top, scored, strict=True)
                ]
            ),
        )
        cache.pop("normalized_scores", None)
    fine = cache["scored"]["fine_scores"]
    if config["fine_coarse_weight"] > 0 and "normalized_scores" not in cache:
        if "coarse_z" not in cache or cache.get("coarse_z_top_m") != top_m:
            cache["coarse_z"] = [
                _zscore_1d(row[torch.isfinite(row)]) for row in coarse_topm
            ]
            cache["coarse_z_top_m"] = top_m
        cache["normalized_scores"] = [
            (_zscore_1d(f[torch.isfinite(c)]), z)
            for f, c, z in zip(fine, coarse_topm, cache["coarse_z"], strict=True)
        ]
    normalized = cache.get("normalized_scores")
    fused = torch.stack(
        [
            fuse_fine_coarse_scores(
                f,
                c,
                config["fine_coarse_weight"],
                normalized_scores=None if normalized is None else normalized[i],
            )
            for i, (f, c) in enumerate(zip(fine, coarse_topm, strict=True))
        ]
    )
    metadata = deepcopy(output["provenance"])
    retrieval = {
        **metadata.get("retrieval", {}),
        **{
            key: config[key]
            for key in (
                "fine_score_mode",
                "context_weight",
                "coarse_beta",
                "fine_coarse_weight",
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
        "score_config": config,
        "provenance": metadata,
        "coarse_rankings": order,
        "coarse_topm": top,
        "coarse_topm_scores": coarse_topm,
        **cache["scored"],
        "fafa_rankings": cache["fafa_rankings"],
        "fine_rankings": cache["fine_rankings"],
        "fused_topm_scores": fused,
        "rankings": torch.stack(
            [rerank_topm(o, t, f) for o, t, f in zip(order, top, fused, strict=True)]
        ),
    }


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
    """Recreate all rankings and diagnostics from the one full-gallery artifact."""
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
    _validate_settings(config, output["state_trained"])
    return _replay_rankings(output, config, {})


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
    output = replay_rankings(
        torch.load(args.rankings, map_location="cpu", weights_only=True),
        fine_score_mode=args.mode,
        context_weight=args.context_weight,
        fine_coarse_weight=args.fine_coarse_weight,
        coarse_beta=args.coarse_beta,
        top_m=args.top_m,
    )
    metadata = {
        **output["provenance"],
        **runtime_metadata(torch.device("cpu")),
        "replay": {
            "source": str(args.rankings),
            "source_sha256": sha256_file(args.rankings),
            "scope": "full_gallery",
        },
    }
    output["provenance"] = metadata
    save_results(args.output_dir, output, metadata)
    print(f"Saved {args.output_dir / 'rankings.pt'}")


if __name__ == "__main__":
    main()
