"""Select the validation result that is frozen for test inference."""

import math
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

from rcr.common.config import override_config, save_config
from rcr.common.data import (
    load_rcr_data,
    split_fingerprint,
    split_image_ids,
    split_samples,
)
from rcr.common.io import save_results, sha256_file, write_json, write_summary
from rcr.evaluation.evaluate import (
    candidate_cutoffs,
    evaluate_retrieval_output,
    selection_metrics,
)
from rcr.evaluation.provenance import digest_json
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.contracts import SELECTED_RETRIEVAL_KEYS
from rcr.proposed.matching import matching_settings
from rcr.proposed.replay import validate_artifact

SETTINGS = {"rankings", "output_dir", "selection_metric"}
SELECTION_METRICS = ("checkpoint_score", "full_map", "macro_full_map")


def _load_candidate(path: Path) -> dict:
    """Read one validation artifact and check it against its embedded contract."""
    source = torch.load(path, map_location="cpu", weights_only=True)
    validate_artifact(source)
    provenance = source["provenance"]
    if provenance.get("fine_score_ablation"):
        raise ValueError(
            "calibration requires the trained residual weight and pooling; "
            "use the official validation export"
        )
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


def _row(index: int, candidate: dict, result: dict, fingerprint: str) -> dict:
    score_config = candidate["source"]["score_config"]
    return {
        "index": index,
        "split": "val",
        **{key: score_config[key] for key in SELECTED_RETRIEVAL_KEYS},
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "validation_sha256": fingerprint,
        "score_source": str(candidate["path"]),
        "score_source_sha256": candidate["source_sha256"],
        **result["overall"],
        **selection_metrics(result),
        **{
            f"{case.lower()}_{key}": value
            for case, values in result["by_case"].items()
            for key, value in values.items()
        },
    }


def _write_selection(
    candidate: dict, root: Path, rows: list[dict], best: dict, metric: str
) -> dict:
    """Freeze one complete scoring contract for test inference."""
    selected = override_config(candidate["cfg"], {"output.dir": str(root / "selected")})
    selected["selected_checkpoint_sha256"] = best["checkpoint_sha256"]
    selected["selected_validation_sha256"] = best["validation_sha256"]
    selected["selected_retrieval"] = {
        key: selected["retrieval"][key] for key in SELECTED_RETRIEVAL_KEYS
    }
    selected["selected_background_logit"] = matching_settings(
        selected["model"]["fine_matching"]
    )["background_logit"]
    save_config(root / "selected.yaml", selected)
    write_json(
        root / "selection.json",
        {
            **best,
            "metric": metric,
            "value": best[metric],
            "baseline_value": rows[0][metric],
            "improvement": best[metric] - rows[0][metric],
            "tie_break": "first artifact in the configured rankings order",
        },
    )
    return selected


def _read_settings(settings: dict) -> tuple[list[Path], Path, str]:
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
    return paths, root, metric


def calibrate(settings: dict) -> list[dict]:
    """Pick the best saved validation artifact; no checkpoint or cache is loaded.

    Every artifact keeps the shortlist it was retrieved with. Settings that change
    the shortlist are compared by retrieving once per value.
    """
    paths, root, metric = _read_settings(settings)
    candidates = [_load_candidate(path) for path in paths]
    if len({candidate["checkpoint_sha256"] for candidate in candidates}) != 1:
        raise ValueError("validation artifacts must come from one checkpoint")
    cfg = candidates[0]["cfg"]
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, "val")
    fingerprint = split_fingerprint(data, "val")
    rows = []
    for index, candidate in enumerate(tqdm(candidates, desc="validation selection")):
        _check_validation(candidate, data, samples, fingerprint)
        result = evaluate_retrieval_output(
            data,
            samples,
            candidate["source"],
            candidate_cutoffs(candidate["cfg"]),
            split="val",
        )
        row = _row(index, candidate, result, fingerprint)
        if not math.isfinite(row[metric]):
            raise ValueError(f"calibration requires finite validation {metric}")
        rows.append(row)
    best = max(rows, key=lambda row: row[metric])
    best_candidate = candidates[best["index"]]
    root.mkdir(parents=True, exist_ok=True)
    write_summary(root, rows)
    selected = _write_selection(best_candidate, root, rows, best, metric)
    output = best_candidate["source"]
    metadata = {
        **output["provenance"],
        "config": selected,
        "retrieval": selected["retrieval"],
        "selection": {
            "source": str(best_candidate["path"]),
            "source_sha256": best_candidate["source_sha256"],
            "selection_split": "val",
            "selection_metric": metric,
        },
    }
    output["provenance"] = metadata
    save_results(root / "selected/val", output, metadata)
    evaluate_run(selected, data=data, samples=samples, output=output)
    print(
        f"Selected {metric}={best[metric]:.6f}; first artifact={rows[0][metric]:.6f}",
        flush=True,
    )
    return rows
