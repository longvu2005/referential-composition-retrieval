"""Reject malformed experiment input before model imports or stage side effects."""

import subprocess
import sys
from pathlib import Path

import pytest

from rcr.common.config import load_config, override_config, parse_overrides
from scripts import run as cli


def test_overrides_are_typed_and_leave_original_unchanged():
    cfg = {"train": {"epochs": 10}, "runtime": {"device": "auto"}}
    changed = override_config(cfg, parse_overrides(["train.epochs=2"]))
    assert changed["train"]["epochs"] == 2
    assert cfg["train"]["epochs"] == 10
    for key in ("missing", "train.typo", "train.epochs.typo"):
        with pytest.raises(ValueError, match="unknown config key"):
            override_config(cfg, {key: 1})


@pytest.mark.parametrize("text", ["", "[]", "train: ["])
def test_invalid_yaml_configs_have_actionable_errors(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="YAML"):
        load_config(path)


@pytest.mark.parametrize(
    "items", [["train.epochs"], ["=2"], ["train..epochs=2"], ["x=1", "x=2"], ["x=["]]
)
def test_malformed_cli_overrides_are_rejected(items):
    with pytest.raises(ValueError):
        parse_overrides(items)


def test_unknown_method_rejected_before_loading_assets(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("method: typo\n")
    with pytest.raises(SystemExit) as error:
        cli.main(["prepare", "--config", str(path)])
    assert error.value.code == 2
    assert "method must be" in capsys.readouterr().err


def test_help_and_config_parsing_do_not_import_model_runtime():
    # Run in a fresh interpreter: other tests legitimately import Torch first.
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.path[:0] = ['src', '.']
from scripts.run import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
heavy = {'torch', 'transformers', 'scipy', 'rcr.evaluation.evaluate'}
assert not heavy & sys.modules.keys()
""",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
