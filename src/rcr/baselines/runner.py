"""Small experiment runner: retrieve, select fusion on val, evaluate frozen test."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rcr.baselines.clip import (
    MODES,
    canonical_mode,
    iter_clip_scores,
    prepare_clip_inputs,
    write_clip_scores,
)
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import (
    image_signature,
    output_directory,
    preserve_file,
    save_results,
    scores_to_rankings,
    sha256_file,
    write_json,
    write_summary,
)
from rcr.common.runtime import resolve_device
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.provenance import (
    benchmark_metadata,
    digest_json,
    runtime_metadata,
)
from rcr.evaluation.runner import evaluate_run


def run_retrieval(
    cfg: dict, *, max_queries: int | None = None, data=None, selection_metadata=None
) -> dict:
    """Shared retrieval entry point for the single-run and experiment CLIs."""
    cfg = copy.deepcopy(cfg)
    if data is None:
        data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, cfg["split"])
    full_num_queries = len(samples)
    if max_queries is not None:
        samples = samples[:max_queries]
    if not samples:
        raise ValueError("selected query split is empty")
    gallery_ids = split_image_ids(data, cfg["split"])
    device = resolve_device(cfg)
    print(
        f"{cfg['method']}: {len(samples)} queries / "
        f"{len(gallery_ids)} gallery images / {device}",
        flush=True,
    )
    start = perf_counter()
    if cfg["method"] == "clip":
        from rcr.baselines.clip import retrieve_clip

        scores, details = retrieve_clip(data, samples, gallery_ids, cfg, device)
    elif cfg["method"] == "fafa":
        from rcr.baselines.fafa import retrieve_fafa

        scores, details = retrieve_fafa(data, samples, gallery_ids, cfg, device)
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    if selection_metadata is not None:
        details["adapter_selection"] = selection_metadata
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
    save_results(
        output_directory(cfg),
        output,
        {
            "method": cfg["method"],
            "mode": cfg.get("mode"),
            "config": cfg,
            **benchmark_metadata(data, cfg["split"], output),
            **runtime_metadata(resolve_device(cfg)),
            "query_subset": query_subset,
            "higher_is_better": True,
            "elapsed_seconds": elapsed_seconds,
            **details,
        },
    )
    print(f"Saved {output_directory(cfg)}/rankings.pt", flush=True)


def _grid(cfg):
    tuning = cfg["tuning"]
    grid = sorted(set(float(x) for x in tuning["image_weights"]))
    if not grid or not np.isfinite(grid).all() or min(grid) < 0 or max(grid) > 1:
        raise ValueError("tuning.image_weights must be a non-empty grid in [0, 1]")
    if tuning["metric"] != "full_map":
        raise ValueError("tuning.metric must be full_map for the RCR benchmark")
    return grid


def tuning_context(data, cfg, device):
    """Only val inputs/labels enter selection provenance; no test labels."""
    ids = split_image_ids(data, "val")
    context = {
        "version": "clip-fusion-val-v2-self-excluded",
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
    by_id = {image_id: i for i, image_id in enumerate(gallery_ids)}
    self_indices = [by_id[s["query_image_id"]] for s in samples]
    trials = []
    batch_size = int(cfg["runtime"]["score_batch_size"])
    for iw in _grid(cfg):
        fusion = {**cfg.get("fusion", {}), "image_weight": iw, "text_weight": 1.0 - iw}
        total = 0.0
        for start, batch in iter_clip_scores(
            inputs, mode, fusion, batch_size, self_indices=self_indices
        ):
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


def fafa_context(data, cfg, device):
    """Lock the manually chosen adapter settings after complete validation.

    No automatic FAFA search is performed. Re-running val freezes the supplied
    configuration; test only checks it, without selecting from test metrics.
    """
    return digest_json(
        {
            "version": "fafa-rcr-val-v1",
            "val_sha256": split_fingerprint(data, "val"),
            "source": cfg["source"],
            "checkpoint": cfg["checkpoint"],
            "localization": cfg["localization"],
            "setmatch": cfg["setmatch"],
            "gallery_feature_dtype": cfg["runtime"]["gallery_feature_dtype"],
            "device_type": device.type,
            "weight_hashes": {
                "fafa": sha256_file(cfg["checkpoint"]["path"]),
                "detector": sha256_file(cfg["localization"]["detector"]["checkpoint"]),
                "selector": sha256_file(
                    cfg["localization"]["query_selector"]["checkpoint"]
                ),
                "runtime_assets": sha256_file(
                    cfg["checkpoint"]["runtime_assets_marker"]
                ),
            },
        }
    )


def run_experiment(cfg, *, modes=None, splits=("val",)):
    """Default: validation only. Explicit test loads validation-frozen settings."""
    cfg = copy.deepcopy(cfg)
    splits = list(dict.fromkeys(splits))
    if not splits or any(s not in ("val", "test") for s in splits):
        raise ValueError("experiment splits must be val and/or test")
    # Always val before test, even if the CLI requested test val.
    splits = [s for s in ("val", "test") if s in splits]
    if cfg["method"] == "clip":
        modes = list(dict.fromkeys(canonical_mode(m) for m in (modes or MODES)))
    elif cfg["method"] == "fafa":
        if modes:
            raise ValueError("--modes is only for CLIP")
        modes = [None]
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    data = load_rcr_data(**cfg["data"])
    for split in splits:
        if not data.splits[split] or len(split_image_ids(data, split)) < 2:
            raise ValueError(
                f"{split}: need non-empty queries and at least two gallery images"
            )
    device = resolve_device(cfg)
    adapter_selection, adapter_path = None, None
    if cfg["method"] == "fafa":
        if not data.splits["val"]:
            raise ValueError("FAFA benchmark runs require complete validation first")
        context = fafa_context(data, cfg, device)
        adapter_path = Path(cfg["output"]["dir"]) / "protocol.json"
        if "val" in splits:
            adapter_selection = {
                "split": "val",
                "metric": "full_map",
                "context_sha256": context,
                "num_queries": len(data.splits["val"]),
                "validation_sha256": split_fingerprint(data, "val"),
                "policy": "manually fixed parameters evaluated on validation",
            }
        else:
            if not adapter_path.is_file():
                raise FileNotFoundError(
                    f"Run --splits val first: missing {adapter_path}"
                )
            adapter_selection = json.loads(adapter_path.read_text(encoding="utf-8"))
            if (
                adapter_selection.get("split") != "val"
                or adapter_selection.get("context_sha256") != context
            ):
                raise ValueError("Saved FAFA configuration is stale; run --splits val")
    fusion_modes = [m for m in modes if m in ("early_fusion", "late_fusion")]
    selection = None
    tuning_path = None
    if fusion_modes:
        if not data.splits["val"]:
            raise ValueError("fusion selection requires a non-empty validation split")
        context = tuning_context(data, cfg, device)
        tuning_path = Path(cfg["output"]["dir"]) / "tuning.json"
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
    summary = Path(cfg["output"]["dir"]) / "summary.csv"
    preserve_file(summary)
    summary.unlink(missing_ok=True)
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
                preserve_file(tuning_path)
                write_json(tuning_path, selection)
        for mode in modes:
            run_cfg = copy.deepcopy(split_cfg)
            if mode is not None:
                run_cfg["mode"] = mode
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
                        "image_weight": best["image_weight"],
                        "text_weight": best["text_weight"],
                        "file": str(tuning_path),
                    }
                by_id = {image_id: i for i, image_id in enumerate(gallery_ids)}
                self_indices = [by_id[s["query_image_id"]] for s in samples]
                scores = write_clip_scores(inputs, run_cfg, self_indices=self_indices)
                output = scores_to_rankings(samples, gallery_ids, scores)
                save_run(run_cfg, data, output, run_details, perf_counter() - started)
            else:
                output = run_retrieval(
                    run_cfg, data=data, selection_metadata=adapter_selection
                )
            metrics = evaluate_run(run_cfg, data=data, samples=samples, output=output)
            if cfg["method"] == "fafa" and split == "val":
                preserve_file(adapter_path)
                write_json(
                    adapter_path,
                    {
                        **adapter_selection,
                        "val_full_map": metrics["overall"]["full_map"],
                        "config": run_cfg,
                    },
                )
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
    write_summary(cfg["output"]["dir"], rows)
    return rows
