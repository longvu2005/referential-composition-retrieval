"""Reject malformed experiment input before model imports or stage side effects."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rcr.common.config import load_config, override_config, parse_overrides
from tools import run as cli


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
from tools.run import main
for args in (['--help'], ['run', '--help'], ['build-cache', '--help']):
    try:
        main(args)
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


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["run", "--split", "test"], "--split val"),
        (["retrieve", "--split", "train"], "--split val"),
    ],
)
def test_tuning_export_rejects_nonvalidation_before_stages(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([*arguments, "--config", "configs/proposed.yaml", "--save-tuning"])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_tuning_export_rejects_baseline_method_before_stages(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["retrieve", "--config", "configs/fafa.yaml", "--save-tuning"])
    assert error.value.code == 2
    assert "only available for proposed" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["proposed", "clip", "fafa"])
@pytest.mark.parametrize("split_flag,expected", [(None, "test"), ("val", "val")])
def test_run_honors_yaml_overrides_and_split_precedence(
    monkeypatch, method, split_flag, expected
):
    """Dispatch one split without changing the selected checkpoint or YAML file."""
    name = "rcr.proposed.runner" if method == "proposed" else "rcr.baselines.runner"
    runner = ModuleType(name)
    calls = []
    runner.run_experiment = lambda cfg, *args, **kwargs: calls.append(
        (cfg, args, kwargs)
    )
    monkeypatch.setitem(sys.modules, name, runner)
    path = Path(f"configs/{method}.yaml")
    original = path.read_bytes()
    arguments = [
        "run",
        "--config",
        str(path),
        "--set",
        "split=test",
        "output.dir=runs/custom",
    ]
    if method == "proposed":
        arguments.append("checkpoint=runs/frozen/best.pt")
    if split_flag:
        arguments += ["--split", split_flag]
    cli.main(arguments)
    assert len(calls) == 1
    cfg, _, options = calls[0]
    assert cfg["split"] == expected
    assert cfg["output"]["dir"] == "runs/custom"
    if method == "proposed":
        assert cfg["checkpoint"] == "runs/frozen/best.pt"
        assert options == {"save_tuning": False}
    else:
        assert options == {"splits": [expected]}
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["run", "--set", "split=train"], "invalid split"),
        (["retrieve", "--set", "split=typo"], "invalid split"),
        (["train", "--set", "train.epochs=0"], "positive integer"),
        (["run", "--set", "retrieval.top_m=1"], "candidate_ks"),
        (["run", "--set", "split=test", "--save-tuning"], "--split val"),
    ],
)
def test_invalid_experiments_fail_before_model_imports(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([*arguments, "--config", "configs/proposed.yaml"])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
