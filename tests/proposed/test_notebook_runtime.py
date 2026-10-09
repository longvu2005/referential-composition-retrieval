"""FAFA extraction stays in the caller's notebook runtime."""

import sys
from pathlib import Path

import yaml

from rcr.proposed.cache.build import run_person_worker


def test_fafa_worker_uses_current_python_and_preserves_config(monkeypatch):
    cfg = {
        "person_encoder": {"backend": "fafa", "python": "/stale/venv/python"},
        "data": {"cache": "cache with spaces"},
    }
    calls = []

    def run(command, *, check):
        assert check and command[0] == sys.executable
        assert Path(command[1]).name == "cache_fafa.py"
        assert Path(command[1]).parent.name == "tools"
        assert yaml.safe_load(Path(command[3]).read_text()) == cfg
        calls.append(command[-2:])

    monkeypatch.setattr("rcr.proposed.cache.build.subprocess.run", run)
    run_person_worker(cfg, prepare=True, force=True)
    assert calls == [["--prepare", "--force"]]
