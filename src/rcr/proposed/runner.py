"""Retrieve and evaluate one split in separate processes to release GPU memory."""

import json
import subprocess
import sys
from pathlib import Path

from rcr.common.config import save_config
from rcr.common.io import output_directory, write_summary


def run_experiment(cfg, entrypoint):
    root = Path(cfg["output"]["dir"])
    path = root / "run_config.yaml"
    save_config(path, cfg)

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
