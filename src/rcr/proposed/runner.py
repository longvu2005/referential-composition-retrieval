"""Sequence cache, train, retrieval and evaluation stages."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rcr.common.config import validate_training_schedule
from rcr.common.io import (
    atomic_write_text,
    output_directory,
    preserve_file,
    write_summary,
)


def run_experiment(
    cfg, entrypoint, *, splits=("val",), build=False, train=False, save_tuning=False
):
    if train:
        validate_training_schedule(cfg)
    if train and not cfg["evaluation"]["enabled"]:
        raise ValueError("run --train requires validation to select best.pt")
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    if train:
        cfg["checkpoint"] = str(root / "best.pt")
        cfg.pop("selected_checkpoint_sha256", None)
        cfg.pop("selected_validation_sha256", None)
        cfg.pop("selected_retrieval", None)
    path = root / "run_config.yaml"
    preserve_file(path)
    atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False))

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
        options = ["--save-tuning"] if save_tuning else []
        stage("retrieve", "--splits", split, *options)
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
