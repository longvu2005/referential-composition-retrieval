"""Small stage runner and explicit shortlist/full-gallery comparison suites."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rcr.common.config import load_config, override_config
from rcr.common.io import output_directory, write_summary


def run_experiment(
    cfg, entrypoint, *, splits=("val", "test"), build=False, train=False
):
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    if train:
        if (
            not cfg["evaluation"]["enabled"]
            or cfg["train"]["epochs"] <= cfg["train"]["warmup_epochs"]
        ):
            raise ValueError(
                "run --train requires main training and validation to select best.pt"
            )
        cfg["checkpoint"] = str(root / "best.pt")
    path = root / "run_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    def stage(command, *options):
        subprocess.run(
            [sys.executable, str(entrypoint), command, "--config", str(path), *options],
            check=True,
        )

    if build:
        stage("build-cache")
    if train:
        stage("train")
    rows = []
    for split in splits:
        stage("retrieve", "--splits", split)
        stage("evaluate", "--splits", split)
        metrics = json.loads(
            (output_directory({**cfg, "split": split}) / "metrics.json").read_text()
        )
        rows.append(
            {
                "method": "proposed",
                "split": split,
                "checkpoint": cfg["checkpoint"],
                **metrics["overall"],
            }
        )
    write_summary(root, rows)
    return rows


def run_ablation(suite, entrypoint, *, splits=("val",)):
    """Compare fixed retrieval policies; no implicit model or fusion calibration."""
    base = load_config(suite["base_config"])
    root = Path(suite["output_dir"])
    rows = []
    for item in suite["experiments"]:
        if any(
            not key.startswith("retrieval.") and key != "candidate_ks"
            for key in item["overrides"]
        ):
            raise ValueError("v2 suites compare retrieval policies only")
        cfg = override_config(base, item["overrides"])
        cfg["output"]["dir"] = str(root / item["name"])
        for result in run_experiment(cfg, entrypoint, splits=splits):
            rows.append({"experiment": item["name"], **result})
    write_summary(root, rows)
    return rows
