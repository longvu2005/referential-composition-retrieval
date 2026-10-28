"""Retrieve and evaluate one split in separate processes to release GPU memory."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rcr.common.io import (
    atomic_write_text,
    output_directory,
    write_summary,
)


def run_experiment(cfg, entrypoint):
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_config.yaml"
    atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False))

    def stage(command, *options):
        print(f"Running {command}", flush=True)
        subprocess.run(
            [sys.executable, str(entrypoint), command, "--config", str(path), *options],
            check=True,
        )

    stage("retrieve")
    stage("evaluate")
    directory = output_directory(cfg)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        {
            "method": "proposed",
            "split": cfg["split"],
            "checkpoint": cfg["checkpoint"],
            **metrics["overall"],
        }
    ]
    write_summary(root, rows)
    return rows
