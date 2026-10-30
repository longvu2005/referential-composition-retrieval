"""Select pooling and score weights from complete validation rankings."""

import math
from copy import deepcopy
from itertools import chain, product
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from tqdm import tqdm

from rcr.common.config import override_config, save_config
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import (
    save_results,
    sha256_file,
    write_json,
    write_summary,
)
from rcr.evaluation.evaluate import (
    RECALL_KS,
    _aggregate_cases,
    _aggregate_overall,
    evaluate_retrieval_output,
    identity_positive_ids,
)
from rcr.evaluation.provenance import digest_json, runtime_metadata
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.contracts import SELECTED_RETRIEVAL_KEYS
from rcr.proposed.ranking import FINE_SCORE_MODES
from rcr.proposed.replay import _replay_rankings, _validate_artifact
from rcr.proposed.scores import POOLING_MODES


class _CalibrationEvaluator:
    """Reuse labels and unchanged branches for validated CPU replay outputs."""

    def __init__(self, data, samples, source, candidate_ks):
        gallery = source["gallery_ids"]
        evaluate_retrieval_output(
            data,
            samples,
            {key: source[key] for key in ("sample_ids", "gallery_ids", "rankings")},
            candidate_ks=candidate_ks,
            split="val",
        )
        by_id = {image_id: i for i, image_id in enumerate(gallery)}
        identities = {
            image_id: {
                box["identity_id"]
                for box in data.gt_head_boxes_by_image.get(image_id, [])
            }
            for image_id in gallery
        }
        self.targets = [
            {
                "id": [
                    by_id[i] for i in identity_positive_ids(row, gallery, identities)
                ],
                "full": [by_id[i] for i in set(row["positive_image_ids"])],
            }
            for row in samples
        ]
        self.samples = samples
        self.gallery_size = len(gallery)
        self.candidate_ks = tuple(candidate_ks)
        self._cache = {}

    def _branch(self, field, ranking):
        cached = self._cache.get(field)
        if cached is not None and cached[0] is ranking:
            return cached[1]
        # Positive ranks suffice for AP/R@K; avoid decoding every gallery ID.
        inverse = np.full(
            (len(self.samples), self.gallery_size),
            self.gallery_size + 1,
            dtype=np.int32,
        )
        inverse[np.arange(len(self.samples))[:, None], ranking.numpy()] = np.arange(
            1, self.gallery_size
        )
        rows = []
        for n, (sample, targets) in enumerate(
            zip(self.samples, self.targets, strict=True)
        ):
            row = {"case_type": sample["case_type"]}
            for prefix, indices in targets.items():
                ranks = sorted(inverse[n, indices].tolist())
                # Python accumulation retains the official evaluator's AP arithmetic.
                precision_sum = 0.0
                for hit, rank in enumerate(ranks, 1):
                    precision_sum += hit / rank
                row[f"{prefix}_ap"] = precision_sum / len(ranks)
                for k in RECALL_KS:
                    row[f"{prefix}_r{k}"] = float(ranks[0] <= k)
                if field == "coarse_rankings" and prefix == "full":
                    for k in self.candidate_ks:
                        row[f"candidate_recall_{k}"] = sum(
                            rank <= k for rank in ranks
                        ) / len(ranks)
                        row[f"candidate_hit_{k}"] = float(ranks[0] <= k)
            rows.append(row)
        result = {
            "overall": _aggregate_overall(
                rows, self.candidate_ks if field == "coarse_rankings" else ()
            ),
            "by_case": _aggregate_cases(rows),
        }
        self._cache[field] = ranking, result
        return result

    def __call__(self, output):
        final = self._branch("rankings", output["rankings"])
        result = {
            "overall": dict(final["overall"]),
            "by_case": {case: dict(row) for case, row in final["by_case"].items()},
        }
        for field, prefix in (
            ("coarse_rankings", "coarse"),
            ("fine_rankings", "fine"),
            ("fafa_rankings", "fafa"),
        ):
            branch = self._branch(field, output[field])
            if prefix == "coarse":
                result["overall"].update(
                    (key, value)
                    for key, value in branch["overall"].items()
                    if key.startswith("candidate_")
                )
            result["overall"].update(
                (f"{prefix}_{key}", value)
                for key, value in branch["overall"].items()
                if key != "num_queries" and not key.startswith("candidate_")
            )
            for case, row in branch["by_case"].items():
                result["by_case"][case].update(
                    (f"{prefix}_{key}", value)
                    for key, value in row.items()
                    if key != "num_queries"
                )
        return result


def _write_selection(
    cfg: dict, root: Path, rows: list[dict], *, selection_metric="checkpoint_score"
) -> dict:
    """Freeze one complete scoring contract for test inference."""
    best = max(rows, key=lambda row: row[selection_metric])
    selected = override_config(
        cfg,
        {
            **{f"retrieval.{key}": best[key] for key in SELECTED_RETRIEVAL_KEYS},
            "output.dir": str(root / "selected"),
        },
    )
    for key in ("checkpoint_sha256", "validation_sha256"):
        selected[f"selected_{key}"] = best[key]
    selected["selected_retrieval"] = {
        key: selected["retrieval"][key] for key in SELECTED_RETRIEVAL_KEYS
    }
    save_config(root / "selected.yaml", selected)
    write_json(
        root / "selection.json",
        {
            **best,
            "metric": selection_metric,
            "value": best[selection_metric],
            "baseline_value": rows[0][selection_metric],
            "improvement": best[selection_metric] - rows[0][selection_metric],
            "tie_break": "first configuration in summary.csv order; source first",
        },
    )
    return selected


def _grid(settings: dict, key: str, *, integer=False) -> list:
    values = settings.get(key)
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
    return list(dict.fromkeys(values))


def calibrate(settings: dict) -> list[dict]:
    """Read only rankings.pt and annotations; never load a checkpoint or cache."""
    unknown = settings.keys() - {
        "rankings",
        "output_dir",
        "coarse_beta",
        "fine_coarse_weight",
        "context_weight",
        "top_m",
        "pooling",
        "selection_metric",
    }
    if unknown:
        raise ValueError(f"unknown calibration settings: {', '.join(sorted(unknown))}")
    if missing := {"rankings", "output_dir"} - settings.keys():
        raise ValueError(f"missing calibration settings: {', '.join(sorted(missing))}")
    selection_metric = settings.get("selection_metric", "checkpoint_score")
    if selection_metric not in ("checkpoint_score", "full_map", "macro_full_map"):
        raise ValueError(
            "selection_metric must be checkpoint_score, full_map or macro_full_map"
        )
    # Fusion weight varies fastest so normalization and sorting are shared.
    keys = ("coarse_beta", "top_m", "context_weight", "fine_coarse_weight")
    grids = [_grid(settings, key, integer=key == "top_m") for key in keys]
    source_path = Path(settings["rankings"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    _validate_artifact(source)
    if source["score_config"]["fine_score_mode"] not in FINE_SCORE_MODES:
        raise ValueError("calibration requires an inference-supported fine_score_mode")
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
    score_config = {"pooling": "penalized", **source["score_config"]}
    cfg["retrieval"].setdefault("pooling", "penalized")
    poolings = settings.get("pooling", [score_config["pooling"]])
    if (
        not isinstance(poolings, list)
        or not poolings
        or any(mode not in POOLING_MODES for mode in poolings)
    ):
        raise ValueError("pooling must be a non-empty list of supported modes")
    if any(
        mode != score_config["pooling"] and f"binding_{mode}" not in source
        for mode in poolings
    ):
        raise ValueError(
            "changing pooling requires saved binding scores for that mode; "
            "retrieve validation again with the updated scorer"
        )
    keys = (*keys[:2], "pooling", *keys[2:])
    grids = [*grids[:2], list(dict.fromkeys(poolings)), *grids[2:]]
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
    if (
        cfg.get("checkpoint") != provenance.get("checkpoint")
        or provenance.get("score_config") != source["score_config"]
        or any(
            cfg["retrieval"].get(key) != score_config[key]
            for key in (*keys, "fine_score_mode")
        )
        or provenance.get("binding_scale") != source["binding_scale"]
        or provenance.get("fine_matching") != source["fine_matching"]
    ):
        raise ValueError(
            "embedded provenance disagrees with the saved scoring contract"
        )
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, "val")
    fingerprint = split_fingerprint(data, "val")
    if (
        fingerprint != provenance.get("validation_sha256")
        or fingerprint != provenance.get("split_sha256")
        or cfg.get("selected_validation_sha256", fingerprint) != fingerprint
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
    if not source["state_trained"] and any(beta != 0 for beta in grids[0]):
        raise ValueError("checkpoint state branch is untrained; use a beta=0 grid")
    root = Path(settings["output_dir"])
    if (
        root.resolve() in (source_path.resolve(), source_path.parent.resolve())
        or (root / "selected/val/rankings.pt").resolve() == source_path.resolve()
    ):
        raise ValueError(
            "use a separate calibration output root to retain the source artifact"
        )
    evaluator = _CalibrationEvaluator(data, samples, source, cfg["candidate_ks"])
    root.mkdir(parents=True, exist_ok=True)
    rows, best_output, cache = [], None, {}
    best_value = -math.inf
    source_sha256 = sha256_file(source_path)
    baseline = tuple(score_config[key] for key in keys)
    baseline_in_grid = all(
        value in grid for value, grid in zip(baseline, grids, strict=True)
    )
    configurations = chain(
        [baseline], (values for values in product(*grids) if values != baseline)
    )
    for values in tqdm(
        configurations,
        total=1 + math.prod(map(len, grids)) - int(baseline_in_grid),
        desc="offline calibration",
    ):
        parameters = dict(zip(keys, values, strict=True))
        output = _replay_rankings(
            source, {**source["score_config"], **parameters}, cache
        )
        metrics = evaluator(output)
        macro = fmean(row["full_map"] for row in metrics["by_case"].values())
        row = {
            "index": len(rows),
            "split": "val",
            **parameters,
            "fine_score_mode": source["score_config"]["fine_score_mode"],
            "checkpoint_sha256": digest,
            "validation_sha256": fingerprint,
            "score_source_sha256": source_sha256,
            **metrics["overall"],
            "macro_full_map": macro,
            "checkpoint_score": 0.5 * (metrics["overall"]["full_map"] + macro),
            **{
                f"{case.lower()}_{key}": value
                for case, values in metrics["by_case"].items()
                for key, value in values.items()
            },
        }
        if not math.isfinite(row[selection_metric]):
            raise ValueError(
                f"calibration requires finite validation {selection_metric}"
            )
        if row[selection_metric] > best_value:
            best_output, best_value = output, row[selection_metric]
        rows.append(row)
    write_summary(root, rows)
    selected = _write_selection(cfg, root, rows, selection_metric=selection_metric)
    metadata = {
        **best_output["provenance"],
        **runtime_metadata(torch.device("cpu")),
        "config": selected,
        "retrieval": selected["retrieval"],
        "replay": {
            "source": str(source_path),
            "source_sha256": source_sha256,
            "scope": "full_gallery",
            "selection_split": "val",
            "selection_metric": selection_metric,
        },
    }
    best_output["provenance"] = metadata
    save_results(root / "selected/val", best_output, metadata)
    evaluate_run(selected, data=data, samples=samples, output=best_output)
    print(
        f"Selected {selection_metric}={best_value:.6f}; "
        f"source baseline={rows[0][selection_metric]:.6f}",
        flush=True,
    )
    return rows
