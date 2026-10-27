"""Select coarse/fine fusion weights on validation and freeze them for test."""

import json
import math
from itertools import product
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from rcr.common.config import load_config, override_config
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import (
    atomic_write_text,
    output_directory,
    save_results,
    sha256_file,
    write_json,
    write_summary,
)
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.provenance import runtime_metadata
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA
from rcr.proposed.ranking import retrieval_settings
from rcr.proposed.replay import replay_rankings
from rcr.proposed.retrieve import retrieve_weight_grid


def calibrate(settings: dict) -> list[dict]:
    """Select validation scoring settings using inference or saved raw scores."""
    unknown = settings.keys() - {
        "model_config",
        "rankings",
        "output_dir",
        "coarse_beta",
        "fine_coarse_weight",
        "context_weight",
        "top_m",
    }
    if unknown:
        raise ValueError(f"unknown calibration settings: {', '.join(sorted(unknown))}")
    if settings.get("rankings") is not None:
        return _calibrate_saved(settings)
    if "context_weight" in settings or "top_m" in settings:
        raise ValueError(
            "context_weight/top_m grids require saved full-gallery rankings"
        )
    grids = [settings[key] for key in ("coarse_beta", "fine_coarse_weight")]
    for values in grids:
        if (
            not isinstance(values, list)
            or not values
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values
            )
        ):
            raise ValueError(
                "calibration weights must be non-empty finite nonnegative lists"
            )
    weights = list(product(*grids))
    cfg = load_config(settings["model_config"])
    if cfg["method"] != "proposed":
        raise ValueError("calibration requires method=proposed")
    cfg["retrieval"] = retrieval_settings(
        cfg["retrieval"], cfg["model"].get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
        raise ValueError("candidate_ks cannot exceed top_m")
    root = Path(settings["output_dir"])
    cfg = override_config(cfg, {"output.dir": str(root), "split": "val"})
    cfg.pop("selected_checkpoint_sha256", None)
    cfg.pop("selected_validation_sha256", None)
    cfg.pop("selected_retrieval", None)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        root / "calibration.yaml", yaml.safe_dump(settings, sort_keys=False)
    )
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    outputs = retrieve_weight_grid(cfg, weights)
    rows = []
    for index, (output, (beta, fine_weight)) in enumerate(
        zip(outputs, weights, strict=True)
    ):
        run_cfg = override_config(
            cfg,
            {
                "retrieval.coarse_beta": beta,
                "retrieval.fine_coarse_weight": fine_weight,
                "output.dir": str(root / f"grid_{index:03d}"),
            },
        )
        metrics = evaluate_run(run_cfg, data=data, output=output)
        metadata = json.loads((output_directory(run_cfg) / "run.json").read_text())
        rows.append(
            {
                "index": index,
                "split": "val",
                "coarse_beta": beta,
                "fine_coarse_weight": fine_weight,
                "fine_score_mode": cfg["retrieval"]["fine_score_mode"],
                "context_weight": cfg["retrieval"]["context_weight"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "validation_sha256": metadata["validation_sha256"],
                **metrics["overall"],
                **{
                    f"{case.lower()}_{key}": value
                    for case, values in metrics["by_case"].items()
                    for key, value in values.items()
                },
            }
        )
    write_summary(root, rows)
    _write_selection(cfg, root, rows)
    return rows


def _write_selection(cfg: dict, root: Path, rows: list[dict]) -> tuple[dict, dict]:
    """Freeze one complete scoring contract for either inference or CPU replay."""
    if any(not math.isfinite(row["full_map"]) for row in rows):
        raise ValueError("calibration requires finite validation Full-mAP")
    best = max(rows, key=lambda row: row["full_map"])
    selected = override_config(
        cfg,
        {
            **{
                f"retrieval.{key}": best.get(key, cfg["retrieval"][key])
                for key in (
                    "coarse_beta",
                    "fine_coarse_weight",
                    "fine_score_mode",
                    "context_weight",
                    "top_m",
                )
            },
            "output.dir": str(root / "selected"),
        },
    )
    for key in ("checkpoint_sha256", "validation_sha256"):
        selected[f"selected_{key}"] = best[key]
    selected["selected_retrieval"] = {
        key: selected["retrieval"][key]
        for key in (
            "fine_score_mode",
            "context_weight",
            "coarse_beta",
            "fine_coarse_weight",
            "top_m",
        )
    }
    atomic_write_text(root / "selected.yaml", yaml.safe_dump(selected, sort_keys=False))
    write_json(
        root / "selection.json",
        {
            **best,
            "metric": "full_map",
            "value": best["full_map"],
            "tie_break": "first configuration in YAML grid order",
        },
    )
    return selected, best


def _saved_grid(settings: dict, key: str, default=None, *, integer=False) -> list:
    values = settings.get(key, default)
    if (
        not isinstance(values, list)
        or not values
        or any(
            (type(value) is not int or value < 1)
            if integer
            else (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            )
            for value in values
        )
    ):
        raise ValueError(f"{key} must be a non-empty list of valid tuning values")
    return values


def _candidate_ks(cfg: dict, top_m: int, gallery_size: int) -> list[int]:
    maximum = min(top_m, gallery_size - 1)
    return [k for k in cfg["candidate_ks"] if k <= maximum] or [maximum]


def _calibrate_saved(settings: dict) -> list[dict]:
    """Select weights from a complete validation export; never load a checkpoint."""
    cfg = load_config(settings["model_config"])
    if cfg.get("method") != "proposed":
        raise ValueError("calibration requires method=proposed")
    cfg["retrieval"] = retrieval_settings(
        cfg["retrieval"], cfg["model"].get("coarse_beta", DEFAULT_COARSE_BETA)
    )
    source_path = Path(settings["rankings"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if "tuning_scores" not in source:
        raise ValueError(
            "offline calibration requires a full-gallery --save-tuning export"
        )
    # Validate the complete score contract once before allocating any run outputs.
    replay_rankings(source)
    provenance = source["tuning_scores"].get("provenance", {})
    if provenance.get("split") != "val" or provenance.get("query_subset") is not False:
        raise ValueError("offline calibration requires the complete validation split")
    digest = provenance.get("checkpoint_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(letter not in "0123456789abcdef" for letter in digest)
    ):
        raise ValueError("tuning export is missing its checkpoint fingerprint")
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, "val")
    fingerprint = split_fingerprint(data, "val")
    if (
        fingerprint != provenance.get("validation_sha256")
        or fingerprint != provenance.get("split_sha256")
        or provenance.get("num_queries") != len(samples)
        or provenance.get("num_gallery") != len(source["gallery_ids"])
        or source["sample_ids"] != [sample["sample_id"] for sample in samples]
        or source["gallery_ids"] != split_image_ids(data, "val")
    ):
        raise ValueError(
            "validation text/labels/ordered gallery changed since tuning export"
        )
    keys = ("coarse_beta", "fine_coarse_weight", "context_weight", "top_m")
    grids = [
        _saved_grid(settings, "coarse_beta"),
        _saved_grid(settings, "fine_coarse_weight"),
        _saved_grid(settings, "context_weight", [cfg["retrieval"]["context_weight"]]),
        _saved_grid(settings, "top_m", [cfg["retrieval"]["top_m"]], integer=True),
    ]
    if not source["tuning_scores"]["state_trained"] and any(
        beta != 0 for beta in grids[0]
    ):
        raise ValueError("checkpoint state branch is untrained; use a beta=0 grid")
    root = Path(settings["output_dir"])
    if (
        root.resolve() in (source_path.resolve(), source_path.parent.resolve())
        or (root / "selected/val/rankings.pt").resolve() == source_path.resolve()
    ):
        raise ValueError(
            "use a separate calibration output root to retain the source export"
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        root / "calibration.yaml", yaml.safe_dump(settings, sort_keys=False)
    )
    rows, best_output = [], None
    best_full_map = -math.inf
    weights = list(product(*grids))
    source_sha256 = sha256_file(source_path)
    for values in tqdm(weights, desc="offline calibration"):
        parameters = dict(zip(keys, values, strict=True))
        output = replay_rankings(
            source, fine_score_mode=cfg["retrieval"]["fine_score_mode"], **parameters
        )
        metrics = evaluate_retrieval_output(
            data,
            samples,
            output,
            candidate_ks=_candidate_ks(
                cfg, parameters["top_m"], len(source["gallery_ids"])
            ),
            split="val",
        )
        row = {
            "index": len(rows),
            "split": "val",
            **parameters,
            "fine_score_mode": cfg["retrieval"]["fine_score_mode"],
            "checkpoint_sha256": digest,
            "validation_sha256": fingerprint,
            "score_source_sha256": source_sha256,
            **metrics["overall"],
            **{
                f"{case.lower()}_{key}": value
                for case, case_metrics in metrics["by_case"].items()
                for key, value in case_metrics.items()
            },
        }
        if row["full_map"] > best_full_map:
            best_output = output
            best_full_map = row["full_map"]
        rows.append(row)
    write_summary(root, rows)
    cfg = override_config(cfg, {"split": "val"})
    best = max(rows, key=lambda row: row["full_map"])
    cfg["candidate_ks"] = _candidate_ks(cfg, best["top_m"], len(source["gallery_ids"]))
    selected, best = _write_selection(cfg, root, rows)
    directory = root / "selected" / "val"
    save_results(
        directory,
        best_output,
        {
            **provenance,
            **runtime_metadata(torch.device("cpu")),
            "method": "proposed",
            "config": selected,
            "retrieval": selected["retrieval"],
            "checkpoint": selected["checkpoint"],
            "score_schema_version": best_output["score_schema_version"],
            "score_config": best_output["score_config"],
            "binding_scale": best_output["binding_scale"],
            "fine_score_mode": selected["retrieval"]["fine_score_mode"],
            "context_weight": selected["retrieval"]["context_weight"],
            "replay": {
                "source": str(source_path),
                "source_sha256": source_sha256,
                "scope": "full_gallery",
                "selection_split": "val",
            },
        },
    )
    # Reuse the official evaluation path to attach normal report provenance.
    evaluate_run(selected, data=data, samples=samples, output=best_output)
    return rows
