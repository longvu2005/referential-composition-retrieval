"""Select pooling and score weights from complete validation rankings."""

import math
from copy import deepcopy
from itertools import product
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
from rcr.common.io import (
    save_results,
    sha256_file,
    write_json,
    write_summary,
)
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.provenance import digest_json, runtime_metadata
from rcr.evaluation.runner import evaluate_run
from rcr.proposed.contracts import SELECTED_RETRIEVAL_KEYS
from rcr.proposed.ranking import FINE_SCORE_MODES
from rcr.proposed.replay import _replay_rankings, _validate_artifact
from rcr.proposed.scores import POOLING_MODES


def _write_selection(cfg: dict, root: Path, rows: list[dict]) -> dict:
    """Freeze one complete scoring contract for test inference."""
    best = max(rows, key=lambda row: row["full_map"])
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
            "metric": "full_map",
            "value": best["full_map"],
            "tie_break": "first configuration in summary.csv order",
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
    return values


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
    }
    if unknown:
        raise ValueError(f"unknown calibration settings: {', '.join(sorted(unknown))}")
    if missing := {"rankings", "output_dir"} - settings.keys():
        raise ValueError(f"missing calibration settings: {', '.join(sorted(missing))}")
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
    grids = [*grids[:2], poolings, *grids[2:]]
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
    root.mkdir(parents=True, exist_ok=True)
    rows, best_output, cache = [], None, {}
    best_full_map = -math.inf
    source_sha256 = sha256_file(source_path)
    for values in tqdm(
        product(*grids), total=math.prod(map(len, grids)), desc="offline calibration"
    ):
        parameters = dict(zip(keys, values, strict=True))
        output = _replay_rankings(
            source, {**source["score_config"], **parameters}, cache
        )
        metrics = evaluate_retrieval_output(
            data, samples, output, candidate_ks=cfg["candidate_ks"], split="val"
        )
        row = {
            "index": len(rows),
            "split": "val",
            **parameters,
            "fine_score_mode": source["score_config"]["fine_score_mode"],
            "checkpoint_sha256": digest,
            "validation_sha256": fingerprint,
            "score_source_sha256": source_sha256,
            **metrics["overall"],
            **{
                f"{case.lower()}_{key}": value
                for case, values in metrics["by_case"].items()
                for key, value in values.items()
            },
        }
        if not math.isfinite(row["full_map"]):
            raise ValueError("calibration requires finite validation Full-mAP")
        if row["full_map"] > best_full_map:
            best_output, best_full_map = output, row["full_map"]
        rows.append(row)
    write_summary(root, rows)
    selected = _write_selection(cfg, root, rows)
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
        },
    }
    best_output["provenance"] = metadata
    save_results(root / "selected/val", best_output, metadata)
    evaluate_run(selected, data=data, samples=samples, output=best_output)
    return rows
