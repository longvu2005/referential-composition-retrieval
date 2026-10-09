"""FAFA extraction and policy suites preserve the caller's notebook setup."""

import sys
from pathlib import Path

import yaml

from rcr.common.config import load_config
from rcr.proposed import experiments
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


def test_policy_suite_reuses_saved_cache_paths_and_legacy_flag(tmp_path, monkeypatch):
    suite = load_config("configs/ablations/shortlist.yaml")
    cfg = load_config("configs/proposed.yaml")
    cfg["data"]["dino_cache"] = "/kaggle/input/known-dino-cache"
    cfg["data"]["cache"] = "cache/notebook-fafa"
    cfg["data"]["clip_cache"] = "cache/notebook-clip"
    cfg["cache"]["allow_legacy_dino"] = True
    cfg["checkpoint"] = "runs/proposed-v2/selected-best.pt"
    saved = tmp_path / suite["base_config"]
    saved.parent.mkdir(parents=True)
    saved.write_text(yaml.safe_dump(cfg))
    before = saved.read_bytes()
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(current, entrypoint, *, splits):
        calls.append(current)
        assert splits == ("val",)
        return [{"split": "val", "full_map": 0.5}]

    monkeypatch.setattr(experiments, "run_experiment", run)
    rows = experiments.run_ablation(suite, Path("tools/run.py"), splits=("val",))
    assert len(rows) == len(calls) == 3
    assert {current["retrieval"]["mode"] for current in calls} == {
        "coarse", "shortlist", "full"
    }
    for current in calls:
        assert current["data"] == cfg["data"]
        assert current["cache"] == cfg["cache"]
        assert current["checkpoint"] == cfg["checkpoint"]
    assert saved.read_bytes() == before
