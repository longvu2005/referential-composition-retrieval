"""Keep the Bash workflow aligned with the configured research output paths."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rcr.common.config import load_config


@pytest.mark.parametrize("split", ["val", "test"])
def test_proposed_workflow_uses_configured_run_and_selection_paths(tmp_path, split):
    root = Path(__file__).resolve().parents[1]
    proposed = load_config(root / "configs/proposed.yaml")
    calibration = load_config(root / "configs/calibration.yaml")
    run_dir = Path(proposed["output"]["dir"])
    calibration_dir = Path(calibration["output_dir"])
    assert Path(proposed["checkpoint"]) == run_dir / "best.pt"
    assert Path(calibration["rankings"]) == run_dir / "val/rankings.pt"

    # Record real shell dispatch without loading encoders or writing run outputs.
    recorder = tmp_path / "python-recorder"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RCR_COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    recorder.chmod(0o755)
    command_log = tmp_path / "commands.jsonl"
    subprocess.run(
        ["bash", str(root / "scripts/methods/proposed.bash"), split],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(recorder),
            "RCR_COMMAND_LOG": str(command_log),
        },
        check=True,
    )
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    if split == "val":
        assert [command[1] for command in commands] == [
            "prepare",
            "build-cache",
            "train",
            "run",
            "calibrate",
        ]
        assert commands[3] == [
            "tools/run.py",
            "run",
            "--config",
            str(run_dir / "config.yaml"),
            "--split",
            "val",
        ]
        assert commands[4] == [
            "tools/run.py",
            "calibrate",
            "--config",
            "configs/calibration.yaml",
        ]
    else:
        assert commands == [
            [
                "tools/run.py",
                "run",
                "--config",
                str(calibration_dir / "selected.yaml"),
                "--split",
                "test",
            ]
        ]
