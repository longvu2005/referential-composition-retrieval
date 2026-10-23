"""Select coarse/fine fusion weights on validation and freeze them for test."""

import json
import math
from itertools import product
from pathlib import Path

import yaml

from rcr.common.config import load_config, override_config
from rcr.common.data import load_rcr_data
from rcr.common.io import (
    atomic_write_text,
    output_directory,
    preserve_artifacts,
    preserve_file,
    write_json,
    write_summary,
)
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.nn.model import DEFAULT_COARSE_BETA
from rcr.proposed.ranking import retrieval_settings
from rcr.proposed.retrieve import retrieve_weight_grid


def calibrate(settings: dict) -> list[dict]:
    """Calibrate only the two score weights; architecture and data stay fixed."""
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
    preserve_file(root / "calibration.yaml")
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
    if any(not math.isfinite(row["full_map"]) for row in rows):
        raise ValueError("calibration requires finite validation Full-mAP")
    best = max(rows, key=lambda row: row["full_map"])
    selected = override_config(
        cfg,
        {
            "retrieval.coarse_beta": best["coarse_beta"],
            "retrieval.fine_coarse_weight": best["fine_coarse_weight"],
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
    preserve_artifacts(root, ("selection.json", "selected.yaml"))
    atomic_write_text(root / "selected.yaml", yaml.safe_dump(selected, sort_keys=False))
    write_json(
        root / "selection.json",
        {
            **best,
            "metric": "full_map",
            "value": best["full_map"],
            "tie_break": "first pair in YAML grid order",
        },
    )
    return rows
