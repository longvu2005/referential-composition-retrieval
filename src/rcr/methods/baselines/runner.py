"""Small experiment runner: retrieve, select fusion on val, evaluate frozen test."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.methods.baselines.clip import (
    MODES,
    canonical_mode,
    iter_clip_scores,
    prepare_clip_inputs,
    write_clip_scores,
)
from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.common.results import (
    image_signature,
    output_directory,
    save_results,
    scores_to_rankings,
    sha256_file,
)


def run_retrieval(cfg: dict, *, max_queries: int | None = None) -> dict:
    """Shared retrieval entry point for the single-run and experiment CLIs."""
    cfg = copy.deepcopy(cfg)
    _validate_runtime(cfg)
    cfg["output"]["dir"] = str(output_directory(cfg))
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, cfg["split"])
    full_num_queries = len(samples)
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        samples = samples[:max_queries]
    if not samples:
        raise ValueError("selected query split is empty")
    gallery_ids = split_image_ids(data, cfg["split"])
    device = _device(cfg)
    print(
        f"{cfg['method']}: {len(samples)} queries / "
        f"{len(gallery_ids)} gallery images / {device}",
        flush=True,
    )
    start = perf_counter()
    if cfg["method"] == "clip":
        from rcr.methods.baselines.clip import retrieve_clip

        scores, details = retrieve_clip(data, samples, gallery_ids, cfg, device)
    elif cfg["method"] == "fafa":
        from rcr.methods.baselines.fafa import retrieve_fafa

        scores, details = retrieve_fafa(data, samples, gallery_ids, cfg, device)
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    output = scores_to_rankings(samples, gallery_ids, scores)
    save_run(
        cfg,
        data,
        output,
        details,
        perf_counter() - start,
        query_subset=len(samples) != full_num_queries,
    )
    return output


def save_run(cfg, data, output, details, elapsed_seconds, *, query_subset=False):
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "clip",
        "scipy",
        "timm",
        "eva-decord",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    save_results(
        cfg["output"]["dir"],
        output,
        {
            "method": cfg["method"],
            "mode": cfg.get("mode"),
            "config": cfg,
            "dataset_version": data.manifest.get("version"),
            "num_queries": len(output["sample_ids"]),
            "num_gallery": len(output["gallery_ids"]),
            "query_subset": query_subset,
            "higher_is_better": True,
            "elapsed_seconds": elapsed_seconds,
            "python": sys.version,
            "packages": versions,
            **details,
        },
    )
    print(f"Saved {cfg['output']['dir']}/rankings.pt", flush=True)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _device(cfg):
    name = cfg["runtime"]["device"]
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )


def _validate_runtime(cfg):
    for key, value in cfg["runtime"].items():
        if key.endswith("batch_size") and int(value) < 1:
            raise ValueError(f"runtime.{key} must be positive")
    if int(cfg["runtime"].get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers must be non-negative")


def _grid(cfg):
    tuning = cfg["tuning"]
    grid = sorted(set(float(x) for x in tuning["image_weights"]))
    if not grid or not np.isfinite(grid).all() or min(grid) < 0 or max(grid) > 1:
        raise ValueError("tuning.image_weights must be a non-empty grid in [0, 1]")
    if tuning["metric"] not in ("full_map", "full_r1", "full_r5", "full_r10"):
        raise ValueError("tuning.metric must be full_map or full_r1/5/10")
    return grid


def tuning_context(data, cfg, device):
    """Only val inputs/labels enter selection provenance; no test labels."""
    ids = split_image_ids(data, "val")
    context = {
        "version": "clip-fusion-val-v1",
        "model": cfg["model"]["name"],
        "checkpoint_sha256": sha256_file(cfg["model"]["checkpoint"]),
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
        "text_field": cfg.get("text_field", "final_instruction"),
        "score_normalization": cfg.get("fusion", {}).get(
            "score_normalization", "zscore"
        ),
        "epsilon": float(cfg.get("fusion", {}).get("epsilon", 1e-6)),
        "metric": cfg["tuning"]["metric"],
        "image_weights": _grid(cfg),
        "val_samples": split_samples(data, "val"),
        "val_images": image_signature(data, ids),
        "val_heads": {i: data.gt_head_boxes_by_image.get(i, []) for i in ids},
    }
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tune_fusion(data, samples, gallery_ids, inputs, cfg, mode):
    """Evaluate every weight with the official evaluator, in bounded batches."""
    if [s["sample_id"] for s in samples] != data.splits["val"]:
        raise ValueError("fusion selection requires the complete validation split")
    trials = []
    batch_size = int(cfg["runtime"]["score_batch_size"])
    for iw in _grid(cfg):
        fusion = {**cfg.get("fusion", {}), "image_weight": iw, "text_weight": 1.0 - iw}
        total = 0.0
        for start, batch in iter_clip_scores(inputs, mode, fusion, batch_size):
            subset = samples[start : start + len(batch)]
            output = scores_to_rankings(subset, gallery_ids, batch)
            result = evaluate_retrieval_output(data, subset, output, split="val")
            total += result["overall"][cfg["tuning"]["metric"]] * len(subset)
        trials.append(
            {"image_weight": iw, "text_weight": 1.0 - iw, "value": total / len(samples)}
        )
    # Exact ties: prefer balanced fusion, then smaller image weight.
    best = max(
        trials,
        key=lambda r: (r["value"], -abs(r["image_weight"] - 0.5), -r["image_weight"]),
    )
    return {"selected": dict(best), "trials": trials}


def _evaluate(data, samples, output, cfg):
    result = evaluate_retrieval_output(data, samples, output, split=cfg["split"])
    _write_json(Path(cfg["output"]["dir"]) / "metrics.json", result)
    print(
        f"{cfg.get('mode', cfg['method'])}/{cfg['split']}: {result['overall']}",
        flush=True,
    )
    return result


def run_experiment(cfg, *, modes=None, splits=("val", "test")):
    """Default: tune val, freeze, then evaluate test. Test-only loads selection."""
    cfg = copy.deepcopy(cfg)
    splits = list(dict.fromkeys(splits))
    if not splits or any(s not in ("val", "test") for s in splits):
        raise ValueError("experiment splits must be val and/or test")
    # Always val before test, even if the CLI requested test val.
    splits = [s for s in ("val", "test") if s in splits]
    _validate_runtime(cfg)
    if cfg["method"] == "clip":
        modes = list(dict.fromkeys(canonical_mode(m) for m in (modes or MODES)))
    elif cfg["method"] == "fafa":
        if modes:
            raise ValueError("--modes is only for CLIP")
        modes = [None]
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    directories = [
        output_directory({**cfg, "mode": m, "split": s}).resolve()
        for s in ("val", "test")
        for m in modes
    ]
    if len(set(directories)) != len(directories):
        raise ValueError(
            "output.dir must distinguish every mode/split; use {mode}/{split}"
        )
    data = load_rcr_data(**cfg["data"])
    for split in splits:
        if not data.splits[split] or len(split_image_ids(data, split)) < 2:
            raise ValueError(
                f"{split}: need non-empty queries and at least two gallery images"
            )
    device = _device(cfg)
    fusion_modes = [m for m in modes if m in ("early_fusion", "late_fusion")]
    selection = None
    tuning_path = None
    if fusion_modes:
        if not data.splits["val"]:
            raise ValueError("fusion selection requires a non-empty validation split")
        context = tuning_context(data, cfg, device)
        tuning_path = Path(cfg["tuning"]["output"])
        if "val" in splits:
            selection = {
                "split": "val",
                "num_queries": len(data.splits["val"]),
                "metric": cfg["tuning"]["metric"],
                "context_sha256": context,
                "tie_break": "closest to 0.5, then smaller image weight",
                "modes": {},
            }
        else:
            if not tuning_path.is_file():
                raise FileNotFoundError(
                    f"Run --splits val first: missing {tuning_path}"
                )
            selection = json.loads(tuning_path.read_text(encoding="utf-8"))
            if (
                selection.get("split") != "val"
                or selection.get("context_sha256") != context
            ):
                raise ValueError(
                    "Saved fusion selection is stale; run --splits val again"
                )
            if not all(m in selection["modes"] for m in fusion_modes):
                raise ValueError(
                    "Saved selection lacks requested fusion modes; run --splits val"
                )
    rows = []
    for split in splits:
        samples = split_samples(data, split)
        gallery_ids = split_image_ids(data, split)
        split_cfg = {**cfg, "split": split}
        started = perf_counter()
        if cfg["method"] == "clip":
            inputs, details = prepare_clip_inputs(
                data, samples, gallery_ids, split_cfg, device, modes
            )
            preparation_seconds = perf_counter() - started
            if split == "val" and fusion_modes:
                for mode in fusion_modes:
                    selection["modes"][mode] = tune_fusion(
                        data, samples, gallery_ids, inputs, cfg, mode
                    )
                # Published before touching test inputs.
                _write_json(tuning_path, selection)
        for mode in modes:
            run_cfg = copy.deepcopy(split_cfg)
            if mode is not None:
                run_cfg["mode"] = mode
            run_cfg["output"]["dir"] = str(output_directory(run_cfg))
            started = perf_counter()
            if cfg["method"] == "clip":
                run_details = {
                    **details,
                    "shared_preparation_seconds": preparation_seconds,
                }
                if mode in fusion_modes:
                    best = selection["modes"][mode]["selected"]
                    run_cfg["fusion"] = {
                        **cfg.get("fusion", {}),
                        "image_weight": best["image_weight"],
                        "text_weight": best["text_weight"],
                    }
                    run_details["fusion_selection"] = {
                        "split": "val",
                        "metric": selection["metric"],
                        "val_value": best["value"],
                        "context_sha256": selection["context_sha256"],
                        "file": str(tuning_path),
                    }
                scores = write_clip_scores(inputs, run_cfg)
                output = scores_to_rankings(samples, gallery_ids, scores)
                save_run(run_cfg, data, output, run_details, perf_counter() - started)
            else:
                output = run_retrieval(run_cfg)
            metrics = _evaluate(data, samples, output, run_cfg)
            rows.append(
                {
                    "method": mode or cfg["method"],
                    "split": split,
                    "image_weight": run_cfg["fusion"]["image_weight"]
                    if mode in fusion_modes
                    else "",
                    "text_weight": run_cfg["fusion"]["text_weight"]
                    if mode in fusion_modes
                    else "",
                    **metrics["overall"],
                }
            )
    summary = Path(cfg.get("summary", f"runs/{cfg['method']}/summary.csv"))
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary}", flush=True)
    return rows
