"""Calibrate validation rankings and freeze the selected scoring contract."""

import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from rcr.common.config import calibration_settings, override_config, save_config
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import save_results, sha256_file, write_json, write_summary
from rcr.evaluation.evaluate import (
    TopMSelectionEvaluator,
    candidate_cutoffs,
    evaluate_retrieval_output,
    selection_metrics,
)
from rcr.evaluation.provenance import digest_json
from rcr.evaluation.runner import save_evaluation
from rcr.proposed.contracts import (
    CALIBRATED_RETRIEVAL_KEYS,
    CALIBRATION_PARAMETERS,
    SELECTED_RETRIEVAL_KEYS,
)
from rcr.proposed.matching import matching_settings
from rcr.proposed.ranking import normalize_fusion_rows
from rcr.proposed.replay import replay_rankings, validate_artifact
from rcr.proposed.scores import linear_score_components

SETTINGS = {"rankings", "output_dir", "selection_metric", *CALIBRATION_PARAMETERS}
SELECTION_METRICS = ("checkpoint_score", "full_map", "macro_full_map")


@dataclass
class CalibrationResult:
    output: dict
    metrics: dict
    rows: list[dict]
    best: dict
    metric: str


@torch.inference_mode()
def calibrate_rankings(
    output: dict,
    data,
    samples,
    cfg: dict,
    *,
    coverage_weights: list | None = None,
    context_weights: list | None = None,
    fine_coarse_weights: list | None = None,
    metric: str = "checkpoint_score",
) -> CalibrationResult:
    """Search the complete coefficient product on CPU with fixed Top-M evidence."""
    started = perf_counter()
    if metric not in SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be checkpoint_score, full_map or macro_full_map"
        )
    if output["provenance"].get("fine_score_ablation"):
        raise ValueError(
            "calibration requires the checkpoint pooling and fine score mode; "
            "use the official validation export"
        )
    if (
        output["sample_ids"] != [row["sample_id"] for row in split_samples(data, "val")]
        or output["sample_ids"] != [row["sample_id"] for row in samples]
        or output["gallery_ids"] != split_image_ids(data, "val")
    ):
        raise ValueError("calibration requires the complete validation split")
    settings = calibration_settings(cfg.get("evaluation", {}).get("calibration", {}))
    settings = calibration_settings(
        {
            **settings,
            **{
                key: value
                for key, value in (
                    ("coverage_weights", coverage_weights),
                    ("context_weights", context_weights),
                    ("fine_coarse_weights", fine_coarse_weights),
                )
                if value is not None
            },
        }
    )
    grid = {
        parameter: settings[key]
        if settings[key] is not None
        else [output["score_config"][parameter]]
        for key, parameter in CALIBRATION_PARAMETERS.items()
    }
    validate_artifact(output)
    reference = evaluate_retrieval_output(
        data, samples, output, candidate_cutoffs(cfg), split="val"
    )
    evaluator = TopMSelectionEvaluator(samples, output)
    support = torch.isfinite(output["coarse_topm_scores"])
    coarse_z = normalize_fusion_rows(output["coarse_topm_scores"], support)
    fingerprint = split_fingerprint(data, "val")
    rows = []
    best = None
    for coverage, context in product(grid["coverage_weight"], grid["context_weight"]):
        fine = linear_score_components(
            output["fafa_binding"],
            output["context"],
            output["coverage"],
            output["binding_scale"],
            context_weight=context,
            coverage_weight=coverage,
        )["fine_scores"]
        fine_z = normalize_fusion_rows(fine, support)
        for weight in grid["fine_coarse_weight"]:
            fused = fine if weight == 0 else fine_z + weight * coarse_z
            if not torch.isfinite(fused[support]).all():
                raise ValueError("fine/coarse fusion overflowed float32")
            order = torch.argsort(fused, dim=1, descending=True, stable=True)
            result = evaluator.evaluate(order)
            row = {
                "index": len(rows),
                "split": "val",
                **{key: output["score_config"][key] for key in SELECTED_RETRIEVAL_KEYS},
                "coverage_weight": coverage,
                "context_weight": context,
                "fine_coarse_weight": weight,
                "validation_sha256": fingerprint,
                "num_queries": result["overall"]["num_queries"],
                **{
                    key: value
                    for key, value in reference["overall"].items()
                    if key.startswith(("candidate_recall_", "candidate_hit_"))
                },
                **selection_metrics(result),
                **{
                    f"{case.lower()}_full_map": value["full_map"]
                    for case, value in result["by_case"].items()
                },
            }
            if not math.isfinite(row[metric]):
                raise ValueError(f"calibration requires finite validation {metric}")
            rows.append(row)
            if best is None or row[metric] > best[metric]:
                best = row
    best_output = replay_rankings(
        output, **{key: best[key] for key in CALIBRATED_RETRIEVAL_KEYS}
    )
    best_metrics = evaluate_retrieval_output(
        data, samples, best_output, candidate_cutoffs(cfg), split="val"
    )
    selected_metrics = selection_metrics(best_metrics)
    if any(best[key] != value for key, value in selected_metrics.items()):
        raise RuntimeError(
            "cached calibration scores disagree with official evaluation"
        )
    best_metrics["overall"].update(selected_metrics)
    elapsed = perf_counter() - started
    best_output["timings_seconds"] = {
        **output.get("timings_seconds", {}),
        "calibration": elapsed,
    }
    best_output["provenance"]["timings_seconds"] = dict(best_output["timings_seconds"])
    tqdm.write(f"calibration: {len(rows)} candidates in {elapsed:.2f}s (CPU)")
    return CalibrationResult(best_output, best_metrics, rows, best, metric)


def write_calibration(root: Path, selection: CalibrationResult, **details) -> None:
    """Save every candidate and the deterministic validation winner."""
    write_summary(root, selection.rows)
    metric = selection.metric
    write_json(
        root / "selection.json",
        {
            **selection.best,
            "metric": metric,
            "value": selection.best[metric],
            "reference_value": selection.rows[0][metric],
            "improvement": selection.best[metric] - selection.rows[0][metric],
            "tie_break": "first artifact, then (a, b, eta) in configured grid order",
            "num_candidates": len(selection.rows),
            **details,
        },
    )


def save_selection(
    root: Path, selection: CalibrationResult, cfg: dict, **details
) -> dict:
    """Publish a checkpoint-bound configuration and its already selected validation."""
    output = selection.output
    provenance = output["provenance"]
    selected = override_config(
        cfg,
        {
            "output.dir": str(root / "selected"),
            **{
                f"retrieval.{key}": output["score_config"][key]
                for key in SELECTED_RETRIEVAL_KEYS
            },
        },
    )
    selected["split"] = "val"
    selected["selected_checkpoint_sha256"] = provenance["checkpoint_sha256"]
    selected["selected_validation_sha256"] = provenance["validation_sha256"]
    selected["selected_retrieval"] = {
        key: selected["retrieval"][key] for key in SELECTED_RETRIEVAL_KEYS
    }
    selected["model"]["fine_matching"] = matching_settings(output["fine_matching"])
    selected["selected_background_logit"] = selected["model"]["fine_matching"][
        "background_logit"
    ]
    metadata = {
        **provenance,
        "config": selected,
        "retrieval": selected["retrieval"],
        "selection": {
            "selection_split": "val",
            "selection_metric": selection.metric,
            **details,
        },
    }
    output = {**output, "provenance": metadata}
    save_results(root / "selected/val", output, metadata)
    save_evaluation(selected, selection.metrics, metadata=metadata)
    write_calibration(
        root, selection, checkpoint_sha256=provenance["checkpoint_sha256"], **details
    )
    save_config(root / "selected.yaml", selected)
    return selected


def _load_candidate(path: Path) -> dict:
    """Read one validation artifact and check it against its embedded contract."""
    source = torch.load(path, map_location="cpu", weights_only=True)
    validate_artifact(source)
    provenance = source["provenance"]
    if provenance.get("split") != "val" or provenance.get("query_subset") is not False:
        raise ValueError("offline calibration requires the complete validation split")
    cfg = deepcopy(provenance.get("config"))
    if (
        not isinstance(cfg, dict)
        or cfg.get("method") != "proposed"
        or cfg.get("split") != "val"
    ):
        raise ValueError("rankings.pt must embed its proposed validation config")
    digest = provenance.get("checkpoint_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(letter not in "0123456789abcdef" for letter in digest)
    ):
        raise ValueError("rankings.pt is missing its checkpoint fingerprint")
    if cfg.get("selected_checkpoint_sha256", digest) != digest:
        raise ValueError(
            "checkpoint fingerprint differs from the validation-selected config"
        )
    score_config = source["score_config"]
    if (
        cfg.get("checkpoint") != provenance.get("checkpoint")
        or provenance.get("score_config") != score_config
        or any(
            cfg["retrieval"].get(key) != score_config[key]
            for key in SELECTED_RETRIEVAL_KEYS
        )
        or provenance.get("binding_scale") != source["binding_scale"]
        or provenance.get("fine_matching") != source["fine_matching"]
    ):
        raise ValueError(
            "embedded provenance disagrees with the saved scoring contract"
        )
    # Matching settings remain checkpoint-authoritative.
    cfg.setdefault("model", {})["fine_matching"] = matching_settings(
        source["fine_matching"]
    )
    return {
        "path": path,
        "source": source,
        "cfg": cfg,
        "checkpoint_sha256": digest,
        "source_sha256": sha256_file(path),
    }


def _check_validation(candidate: dict, data, samples, fingerprint: str) -> None:
    source, provenance = candidate["source"], candidate["source"]["provenance"]
    if (
        fingerprint != provenance.get("validation_sha256")
        or fingerprint != provenance.get("split_sha256")
        or candidate["cfg"].get("selected_validation_sha256", fingerprint)
        != fingerprint
        or provenance.get("num_queries") != len(samples)
        or provenance.get("num_gallery") != len(source["gallery_ids"])
        or source["sample_ids"] != [sample["sample_id"] for sample in samples]
        or source["gallery_ids"] != split_image_ids(data, "val")
        or provenance.get("sample_ids_sha256") != digest_json(source["sample_ids"])
        or provenance.get("gallery_ids_sha256") != digest_json(source["gallery_ids"])
        or [source["gallery_ids"][i] for i in source["query_indices"].tolist()]
        != [sample["query_image_id"] for sample in samples]
    ):
        raise ValueError(
            "validation text/labels/ordered gallery changed since retrieval"
        )


def _read_settings(settings: dict) -> tuple[list[Path], Path, str, dict]:
    if unknown := settings.keys() - SETTINGS:
        raise ValueError(f"unknown calibration settings: {', '.join(sorted(unknown))}")
    if missing := {"rankings", "output_dir"} - settings.keys():
        raise ValueError(f"missing calibration settings: {', '.join(sorted(missing))}")
    rankings = settings["rankings"]
    if (
        not isinstance(rankings, list)
        or not rankings
        or any(not isinstance(path, str) for path in rankings)
        or len(set(rankings)) != len(rankings)
    ):
        raise ValueError("rankings must be a non-empty list of distinct artifact paths")
    metric = settings.get("selection_metric", "checkpoint_score")
    if metric not in SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be checkpoint_score, full_map or macro_full_map"
        )
    grids = calibration_settings(
        {key: settings.get(key) for key in CALIBRATION_PARAMETERS}
    )
    root = Path(settings["output_dir"])
    paths = [Path(path) for path in rankings]
    if any(
        root.resolve() in (path.resolve(), path.parent.resolve())
        or (root / "selected/val/rankings.pt").resolve() == path.resolve()
        for path in paths
    ):
        raise ValueError(
            "use a separate calibration output root to retain the source artifact"
        )
    return paths, root, metric, grids


def calibrate(settings: dict) -> list[dict]:
    """Select validation coefficients without loading a checkpoint or cache.

    Every artifact keeps the shortlist it was retrieved with. Settings that change
    the shortlist are compared by retrieving once per value.
    """
    paths, root, metric, grids = _read_settings(settings)
    candidates = [_load_candidate(path) for path in paths]
    if len({candidate["checkpoint_sha256"] for candidate in candidates}) != 1:
        raise ValueError("validation artifacts must come from one checkpoint")
    cfg = candidates[0]["cfg"]
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, "val")
    fingerprint = split_fingerprint(data, "val")
    rows = []
    winner = winner_cfg = None
    for candidate in tqdm(candidates, desc="validation selection"):
        _check_validation(candidate, data, samples, fingerprint)
        selection = calibrate_rankings(
            candidate["source"],
            data,
            samples,
            candidate["cfg"],
            **grids,
            metric=metric,
        )
        offset = len(rows)
        enriched = [
            {
                **row,
                "index": offset + row["index"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "score_source": str(candidate["path"]),
                "score_source_sha256": candidate["source_sha256"],
            }
            for row in selection.rows
        ]
        rows.extend(enriched)
        if winner is None or selection.best[metric] > winner.best[metric]:
            winner = CalibrationResult(
                selection.output,
                selection.metrics,
                rows,
                enriched[selection.best["index"]],
                metric,
            )
            winner_cfg = candidate["cfg"]
    save_selection(
        root,
        winner,
        winner_cfg,
        source=winner.best["score_source"],
        source_sha256=winner.best["score_source_sha256"],
    )
    print(
        f"Selected {metric}={winner.best[metric]:.6f}; "
        f"first artifact={rows[0][metric]:.6f}",
        flush=True,
    )
    return rows
