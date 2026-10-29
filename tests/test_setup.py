"""Exercise setup dispatch without downloading the model dependency stacks."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def setup_project(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copy2(root / "scripts/setup.bash", tmp_path / "scripts/setup.bash")
    (tmp_path / "bin").mkdir()
    # Keep interpreter discovery deterministic, independent of the test host.
    (tmp_path / "bin/bash").symlink_to(shutil.which("bash"))
    (tmp_path / "bin/dirname").symlink_to(shutil.which("dirname"))
    return tmp_path


def fake_python(root, name, version):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, shutil, sys\n"
        f"version = {version!r}\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['RCR_SETUP_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps({\n"
        "        'python': sys.argv[0], 'args': args,\n"
        "        'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "        'pythonhome': os.environ.get('PYTHONHOME'),\n"
        "        'no_user_site': os.environ.get('PYTHONNOUSERSITE'),\n"
        "    }) + '\\n')\n"
        "if args[:2] == ['-I', '-c']:\n"
        "    print(version)\n"
        "elif args[:3] == ['-I', '-m', 'venv']:\n"
        "    target = pathlib.Path(args[-1]) / 'bin/python'\n"
        "    target.parent.mkdir(parents=True)\n"
        "    shutil.copy2(sys.argv[0], target)\n"
        "elif args[:3] == ['-I', '-m', 'pip']:\n"
        "    if os.environ.get('RCR_FAIL_PIP'):\n"
        "        sys.exit(17)\n"
        "else:\n"
        "    sys.exit('Unexpected interpreter arguments: ' + repr(args))\n"
    )
    path.chmod(0o755)
    return path


def run_setup(root, **overrides):
    env = {key: value for key, value in os.environ.items() if key != "PYTHON"}
    env.update(
        PATH=str(root / "bin"),
        RCR_SETUP_LOG=str(root / "commands.jsonl"),
        **overrides,
    )
    return subprocess.run(
        [str(root / "bin/bash"), str(root / "scripts/setup.bash"), "proposed"],
        env=env,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def commands(root):
    return [
        json.loads(line) for line in (root / "commands.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("version", ["3.10", "3.13", "4.0"])
def test_explicit_unsupported_python_fails_before_creating_venv(setup_project, version):
    root = setup_project
    fake_python(root, "bin/python3.11", "3.11")
    requested = fake_python(root, "bin/requested", version)
    result = run_setup(root, PYTHON=str(requested))
    assert result.returncode != 0
    assert "Python 3.11 or 3.12 is required" in result.stderr
    assert not (root / ".venv-proposed").exists()
    assert len(commands(root)) == 1  # Never silently fall back from explicit PYTHON.


@pytest.mark.parametrize("selected", ["3.11", "3.12"])
def test_auto_selects_supported_python_and_isolates_install(setup_project, selected):
    root = setup_project
    fake_python(root, f"bin/python{selected}", selected)
    fake_python(root, "bin/python3", "3.13")
    result = run_setup(root, PYTHONPATH="host-hooks", PYTHONHOME="invalid-host-home")
    assert result.returncode == 0, result.stderr
    assert "Ready:" in result.stdout
    calls = commands(root)
    assert calls[0]["python"].endswith(f"python{selected}")
    assert calls[1]["args"] == ["-I", "-m", "venv", "--without-pip", ".venv-proposed"]
    assert calls[2]["args"][3:6] == ["--python", ".venv-proposed/bin/python", "install"]
    assert calls[2]["args"][-1] == "requirements/bootstrap.txt"
    assert calls[3]["args"][-1] == "requirements/proposed.txt"
    assert "--no-build-isolation" in calls[3]["args"]
    for call in calls:
        assert call["pythonpath"] is None
        assert call["pythonhome"] is None
        assert call["no_user_site"] == "1"


def test_auto_prefers_311_when_both_supported_versions_exist(setup_project):
    root = setup_project
    fake_python(root, "bin/python3.11", "3.11")
    fake_python(root, "bin/python3.12", "3.12")
    assert run_setup(root).returncode == 0
    assert commands(root)[0]["python"].endswith("python3.11")


def test_auto_accepts_compatible_python3(setup_project):
    root = setup_project
    fake_python(root, "bin/python3", "3.12")
    assert run_setup(root).returncode == 0


def test_auto_with_only_python313_does_not_start_pip(setup_project):
    root = setup_project
    fake_python(root, "bin/python3", "3.13")
    result = run_setup(root)
    assert result.returncode != 0
    assert "Python 3.11 or 3.12 is required" in result.stderr
    assert not (root / ".venv-proposed").exists()
    assert all(call["args"][:2] == ["-I", "-c"] for call in commands(root))


@pytest.mark.parametrize("existing", ["3.13", "3.11", None])
def test_incompatible_or_broken_venv_is_preserved(setup_project, existing):
    root = setup_project
    fake_python(root, "bin/python3.12", "3.12")
    venv = root / ".venv-proposed"
    venv.mkdir()
    marker = venv / "preserve-me"
    marker.write_text("existing environment")
    if existing:
        fake_python(root, ".venv-proposed/bin/python", existing)
    result = run_setup(root)
    assert result.returncode != 0
    assert "Move .venv-proposed aside" in result.stderr
    assert marker.read_text() == "existing environment"
    assert all(call["args"][:2] == ["-I", "-c"] for call in commands(root))


def test_setup_repairs_package_install_without_recreating_compatible_venv(
    setup_project,
):
    root = setup_project
    fake_python(root, "bin/python3.12", "3.12")
    fake_python(root, ".venv-proposed/bin/python", "3.12")
    result = run_setup(root)
    assert result.returncode == 0, result.stderr
    calls = commands(root)
    assert len(calls) == 4  # Two version probes, bootstrap, method install.
    assert all("venv" not in call["args"] for call in calls)
    assert calls[-1]["args"][-1] == "requirements/proposed.txt"


def test_pip_failure_does_not_report_ready(setup_project):
    root = setup_project
    fake_python(root, "bin/python3.12", "3.12")
    result = run_setup(root, RCR_FAIL_PIP="1")
    assert result.returncode == 17
    assert "Ready:" not in result.stdout
    assert len(commands(root)) == 3  # Do not continue after bootstrap fails.
