"""Reproduction recipes locate the repo, freeze test settings and stop on errors."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def recorder(tmp_path):
    # A Python path with spaces exercises quoting without downloading any weights.
    command = tmp_path / "record python"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RCR_RECIPE_LOG'], 'a') as out:\n"
        "    out.write(json.dumps([os.getcwd(), sys.argv[1:]]) + '\\n')\n"
        "sys.exit(int(os.environ.get('RCR_RECIPE_EXIT', '0')))\n"
    )
    command.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    return {**os.environ, "PYTHON": str(command), "RCR_RECIPE_LOG": str(log)}, log


@pytest.mark.parametrize("method", ["clip", "fafa", "proposed"])
@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("python_source", ["kernel", "path"])
def test_method_recipe_uses_validation_then_frozen_test(
    tmp_path, recorder, method, split, python_source
):
    root = Path(__file__).resolve().parents[2]
    env, log = recorder
    if python_source == "path":
        (tmp_path / "python").symlink_to(env.pop("PYTHON"))
        env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    subprocess.run(
        ["bash", str(root / f"scripts/methods/{method}.bash"), split],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(cwd == str(root) for cwd, _ in calls)
    commands = [args for _, args in calls]
    assert all(args[0] == "tools/run.py" for args in commands)
    if split == "test":
        assert len(commands) == 1
        assert commands[0][1] == "run"
        assert commands[0][-2:] == ["--splits", "test"]
        assert "--train" not in commands[0] and "--prepare" not in commands[0]
        expected = (
            "runs/proposed-v2/run_config.yaml"
            if method == "proposed"
            else f"configs/{method}.yaml"
        )
        assert commands[0][3] == expected
    elif method == "proposed":
        assert [args[1] for args in commands] == ["run"]
        assert all(
            flag in commands[0] for flag in ("--prepare", "--build-cache", "--train")
        )
        assert all(args[-2:] == ["--splits", "val"] for args in commands)
        assert commands[0][3] == "configs/proposed.yaml"
    else:
        assert [args[1] for args in commands] == ["prepare", "run"]
        assert commands[-1][-2:] == ["--splits", "val"]


def test_failed_stage_stops_workflow(tmp_path, recorder):
    root = Path(__file__).resolve().parents[2]
    env, log = recorder
    result = subprocess.run(
        ["bash", str(root / "scripts/methods/proposed.bash"), "val"],
        cwd=tmp_path,
        env={**env, "RCR_RECIPE_EXIT": "9"},
        capture_output=True,
    )
    assert result.returncode == 9
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("profile", ["proposed", "clip", "fafa"])
def test_setup_installs_into_notebook_python(tmp_path, recorder, profile):
    root = Path(__file__).resolve().parents[2]
    env, log = recorder
    subprocess.run(
        ["bash", str(root / "scripts/setup.bash"), profile],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        [
            str(root),
            [
                "-m", "pip", "install", "--no-cache-dir", "-r",
                "requirements/bootstrap.txt",
            ],
        ],
        [
            str(root),
            [
                "-m", "pip", "install", "--no-cache-dir", "--no-build-isolation", "-r",
                f"requirements/{profile}.txt",
            ],
        ],
    ]
