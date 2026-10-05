"""Small YAML-driven ablation runner: explicit overrides, optional Cartesian sweeps."""

import json
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
    """One method YAML + flat overrides. No recursive config inheritance."""
    base = yaml.safe_load(Path(suite["base_config"]).read_text(encoding="utf-8"))
    base = override_config(base, suite.get("overrides", {}))
    stage = suite["stage"]
    if base["method"] != "proposed" or stage not in ("inference", "train"):
        raise ValueError("ablations require method=proposed and stage=inference|train")
    experiments, names = [], set()
    for item in suite["experiments"]:
        group = item["name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group):
            raise ValueError(
                f"experiment name must be a simple directory name: {group}"
            )
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
                raise ValueError(
                    "inference ablations only change retrieval or candidate_ks"
                )
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
                for key in changes
            ):
                raise ValueError(
                    "training variants must share data, cache and evaluation protocol"
                )
            name = group + "".join(
                f"__{key.replace('.', '-')}-{value}" for key, value in swept.items()
            )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
                raise ValueError(
                    f"duplicate or invalid expanded experiment name: {name}"
                )
            names.add(name)
            cfg = override_config(base, changes)
            if max(cfg["candidate_ks"]) > cfg["retrieval"]["top_m"]:
                raise ValueError(f"{name}: candidate_ks cannot exceed retrieval.top_m")
            root = Path(suite["output_dir"]) / name
            cfg["output"]["dir"] = str(root)
            if stage == "train":
                if not cfg["evaluation"]["enabled"]:
                    raise ValueError(
                        "training ablations require validation to select best.pt"
                    )
                cfg["checkpoint"] = str(root / "best.pt")
                cfg.pop("selected_checkpoint_sha256", None)
                cfg.pop("selected_validation_sha256", None)
                cfg["wandb"]["name"] = name
            experiments.append(
                {"name": name, "group": group, "sweep": swept, "config": cfg}
            )
    if not experiments:
        raise ValueError("ablation suite requires at least one experiment")
    return experiments


def _write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _result_row(experiment: dict, cfg: dict, overall: dict, stage: str) -> dict:
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
        **overall,
    }


def run_ablation(suite: dict, entrypoint: Path, *, splits=("val",)) -> list[dict]:
    """Evaluate fixed-checkpoint variants together, or train independent variants.

    Selection is optional and val-only. A selected sweep evaluates just its
    winner on test; use selected.yaml for a later standalone test invocation.
    """
    splits = [split for split in ("val", "test") if split in splits]
    if not splits:
        raise ValueError("ablations evaluate val/test only")
    experiments = expand_experiments(suite)
    selection = suite.get("select")
    if selection and (
        suite["stage"] != "inference"
        or "val" not in splits
        or selection["group"] not in {e["group"] for e in experiments}
    ):
        raise ValueError(
            "select requires inference on val and an existing experiment group"
        )
    root = Path(suite["output_dir"])
    _write_config(root / "suite.yaml", suite)
    for filename in ("summary.csv", "selection.json", "selected.yaml"):
        (root / filename).unlink(missing_ok=True)
    for experiment in experiments:
        _write_config(
            Path(experiment["config"]["output"]["dir"]) / "config.yaml",
            experiment["config"],
        )
    rows = []

    if suite["stage"] == "train":
        for experiment in experiments:
            cfg = experiment["config"]
            results = run_experiment(cfg, entrypoint, splits=splits, train=True)
            for result in results:
                run_cfg = {**cfg, "split": result["split"]}
                metrics = json.loads(
                    (output_directory(run_cfg) / "metrics.json").read_text()
                )
                rows.append(
                    _result_row(experiment, run_cfg, metrics["overall"], "train")
                )
    else:
        base = experiments[0]["config"]
        data = load_rcr_data(base["data"]["final_dir"], base["data"]["image_root"])

        def infer(group, split):
            configs = {e["name"]: {**e["config"], "split": split} for e in group}
            outputs = retrieve_experiments(configs)
            for experiment in group:
                name = experiment["name"]
                print(f"Evaluating {name}/{split}", flush=True)
                metrics = evaluate_run(configs[name], data=data, output=outputs[name])
                rows.append(
                    _result_row(
                        experiment, configs[name], metrics["overall"], "inference"
                    )
                )

        for split in ["val"] if selection else splits:
            infer(experiments, split)
        if selection:
            candidates = [row for row in rows if row["group"] == selection["group"]]
            metric = selection["metric"]
            # Ties keep the first experiment in the explicit YAML sweep order.
            best = max(candidates, key=lambda row: row[metric])
            chosen = next(e for e in experiments if e["name"] == best["experiment"])
            cfg = override_config(
                chosen["config"],
                {
                    "retrieval.coarse_beta": best["coarse_beta"],
                    "output.dir": str(root / "selected"),
                    "split": "val",
                },
            )
            cfg["selected_checkpoint_sha256"] = best["checkpoint_sha256"]
            cfg["selected_validation_sha256"] = best["validation_sha256"]
            if "test" in splits:
                infer([{**chosen, "config": cfg}], "test")
            _write_config(root / "selected.yaml", cfg)
            write_json(
                root / "selection.json",
                {
                    "split": "val",
                    "metric": metric,
                    "value": best[metric],
                    "experiment": best["experiment"],
                    "group": best["group"],
                    "coarse_beta": best["coarse_beta"],
                    "fine_coarse_weight": best["fine_coarse_weight"],
                    "checkpoint": best["checkpoint"],
                    "checkpoint_sha256": best["checkpoint_sha256"],
                    "validation_sha256": best["validation_sha256"],
                    "tie_break": "first in YAML sweep order",
                    "config": str(root / "selected.yaml"),
                },
            )
    write_summary(root, rows)
    return rows
