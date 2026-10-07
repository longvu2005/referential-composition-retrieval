"""Coordinate proposed stages in separate processes to release accelerator memory."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rcr.methods.common.results import output_directory, write_summary


def run_experiment(
    cfg, entrypoint, *, splits=("val", "test"), build=False, train=False
):
    if build and not train:
        raise ValueError(
            "run --build-cache requires --train; use build-cache to prepare/reuse "
            "caches separately"
        )
    if train and not cfg["evaluation"]["enabled"]:
        raise ValueError("run --train requires validation to select best.pt")
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    if train:
        cfg["checkpoint"] = str(root / "best.pt")
        cfg.pop("selected_checkpoint_sha256", None)
        cfg.pop("selected_validation_sha256", None)
    path = root / "run_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (root / "summary.csv").unlink(missing_ok=True)

    def stage(command, *options):
        print(f"Running {command}", flush=True)
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
        directory = output_directory({**cfg, "split": split})
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
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
