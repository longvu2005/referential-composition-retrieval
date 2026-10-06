"""Explicit architecture/binding experiments and validation-only calibration."""

import json
import math
import re
from itertools import product
from pathlib import Path

import yaml

from rcr.methods.common.data import load_rcr_data
from rcr.methods.common.experiment import evaluate_run, override_config
from rcr.methods.common.results import output_directory, write_json, write_summary
from rcr.methods.proposed.inference import retrieve_experiments
from rcr.methods.proposed.runner import run_experiment


def expand_experiments(suite: dict) -> list[dict]:
    base = yaml.safe_load(Path(suite["base_config"]).read_text(encoding="utf-8"))
    base = override_config(base, suite.get("overrides", {}))
    stage = suite["stage"]
    if base["method"] != "proposed" or stage not in ("train", "inference"):
        raise ValueError("suite requires proposed and stage=train|inference")
    experiments, names = [], set()
    for item in suite["experiments"]:
        grid = item.get("sweep", {})
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("sweep values must be non-empty lists")
        for values in product(*grid.values()):
            swept = dict(zip(grid, values, strict=True))
            changes = {**item.get("overrides", {}), **swept}
            if stage == "inference" and any(
                not (key.startswith("retrieval.") or key == "candidate_ks")
                for key in changes
            ):
                raise ValueError("inference variants only change retrieval settings")
            if stage == "train" and any(
                key.split(".")[0]
                not in (
                    "model",
                    "train",
                    "optimizer",
                    "loss",
                    "retrieval",
                    "candidate_ks",
                )
                and key not in ("data.cache", "person_encoder.backend")
                for key in changes
            ):
                raise ValueError("training variants must share benchmark data/protocol")
            name = item["name"] + "".join(
                f"__{key.replace('.', '-')}-{value}" for key, value in swept.items()
            )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
                raise ValueError(f"duplicate or invalid experiment name: {name}")
            names.add(name)
            cfg = override_config(base, changes)
            if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
                raise ValueError("candidate_ks cannot exceed top_m")
            root = Path(suite["output_dir"]) / name
            cfg["output"]["dir"] = str(root)
            if stage == "train":
                if not cfg["evaluation"]["enabled"]:
                    raise ValueError("training variants require validation")
                cfg["checkpoint"] = str(root / "best.pt")
                cfg.pop("selected_checkpoint_sha256", None)
                cfg.pop("selected_validation_sha256", None)
                cfg["wandb"]["name"] = name
            experiments.append(
                {"name": name, "group": item["name"], "sweep": swept, "config": cfg}
            )
    if not experiments:
        raise ValueError("suite requires at least one experiment")
    return experiments


def _write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _result_row(experiment: dict, cfg: dict, metrics: dict, stage: str) -> dict:
    metadata = json.loads(
        (output_directory(cfg) / "run.json").read_text(encoding="utf-8")
    )
    settings = metadata["retrieval"]
    return {
        "experiment": experiment["name"],
        "group": experiment["group"],
        "stage": stage,
        "split": cfg["split"],
        "checkpoint": cfg["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "validation_sha256": metadata["validation_sha256"],
        "seed": metadata["checkpoint_seed"],
        "sweep": json.dumps(experiment["sweep"], sort_keys=True),
        "representation": metadata["representation"],
        "person_backend": (metadata.get("encoder_metadata") or {})
        .get("person_encoder", {})
        .get("backend"),
        "binding_mode": metadata["binding_mode"],
        "binding_scale": metadata["binding_scale"],
        **{
            key: settings[key]
            for key in (
                "coarse_mode",
                "coarse_beta",
                "coarse_normalization",
                "fine_coarse_weight",
                "rerank",
                "top_m",
            )
        },
        **metrics["overall"],
        **{
            f"{case.lower()}_{key}": value
            for case, row in metrics["by_case"].items()
            for key, value in row.items()
        },
    }


def _select_validation_result(rows: list[dict], selection: dict) -> dict:
    candidates = [
        row
        for row in rows
        if row["split"] == "val" and selection.get("group") in (None, row["group"])
    ]
    metric = selection["metric"]
    if not candidates or any(
        metric not in row or not math.isfinite(row[metric]) for row in candidates
    ):
        raise ValueError("selection requires finite validation metrics")
    return max(candidates, key=lambda row: row[metric])  # First YAML entry wins ties.


def run_ablation(suite: dict, entrypoint: Path, *, splits=("val",)) -> list[dict]:
    splits = [split for split in ("val", "test") if split in splits]
    selection = suite.get("select")
    if not splits:
        raise ValueError("suite evaluates complete val/test splits only")
    if selection and (suite["stage"] != "inference" or "val" not in splits):
        raise ValueError("selection requires inference on val; test never selects")
    experiments = expand_experiments(suite)
    root = Path(suite["output_dir"])
    _write_config(root / "suite.yaml", suite)
    for filename in ("summary.csv", "selection.json", "selected.yaml"):
        (root / filename).unlink(missing_ok=True)
    rows = []
    if suite["stage"] == "train":
        for experiment in experiments:
            cfg = experiment["config"]
            _write_config(Path(cfg["output"]["dir"]) / "config.yaml", cfg)
            results = run_experiment(cfg, entrypoint, splits=splits, train=True)
            for result in results:
                run_cfg = {**cfg, "split": result["split"]}
                metrics = json.loads(
                    (output_directory(run_cfg) / "metrics.json").read_text()
                )
                rows.append(_result_row(experiment, run_cfg, metrics, "train"))
            write_summary(root, rows)
        return rows

    base = experiments[0]["config"]
    data = load_rcr_data(base["data"]["final_dir"], base["data"]["image_root"])

    def infer(group, split):
        configs = {e["name"]: {**e["config"], "split": split} for e in group}
        outputs = retrieve_experiments(configs)
        for experiment in group:
            name = experiment["name"]
            _write_config(
                Path(configs[name]["output"]["dir"]) / "config.yaml", configs[name]
            )
            metrics = evaluate_run(configs[name], data=data, output=outputs[name])
            rows.append(_result_row(experiment, configs[name], metrics, "inference"))
        write_summary(root, rows)

    for split in ["val"] if selection else splits:
        infer(experiments, split)
    if selection:
        best = _select_validation_result(rows, selection)
        chosen = next(e for e in experiments if e["name"] == best["experiment"])
        cfg = override_config(
            chosen["config"], {"output.dir": str(root / "selected"), "split": "val"}
        )
        cfg["selected_checkpoint_sha256"] = best["checkpoint_sha256"]
        cfg["selected_validation_sha256"] = best["validation_sha256"]
        _write_config(root / "selected.yaml", cfg)
        write_json(
            root / "selection.json",
            {
                "split": "val",
                "metric": selection["metric"],
                "value": best[selection["metric"]],
                "experiment": best["experiment"],
                "sweep": chosen["sweep"],
                "coarse_beta": best["coarse_beta"],
                "fine_coarse_weight": best["fine_coarse_weight"],
                "checkpoint_sha256": best["checkpoint_sha256"],
                "validation_sha256": best["validation_sha256"],
                "tie_break": "first in YAML order",
            },
        )
        if "test" in splits:
            infer([{**chosen, "config": cfg}], "test")
    return rows
