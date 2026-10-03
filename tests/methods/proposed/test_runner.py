"""Exercise stage coordination with real cache/train/retrieval/evaluation code."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from tests.methods.proposed.test_build_cache import (
    Backbone as ImageBackbone,
)
from tests.methods.proposed.test_build_cache import (
    Detector,
    DetectorProcessor,
)
from tests.methods.proposed.test_retrieval import _sample
from tests.methods.proposed.test_train_cli import Backbone as TextBackbone
from tests.methods.proposed.test_train_cli import Tokenizer
from tools.methods import (
    build_cache,
    evaluate,
    retrieve_proposed,
    run_proposed,
    train_proposed,
)


@pytest.fixture
def experiment(tmp_path):
    final = tmp_path / "final"
    (final / "splits").mkdir(parents=True)
    images = tmp_path / "images"
    samples, gallery, heads = [], [], []
    for split in ("train", "val", "test"):
        for name in ("q", "a", "n"):
            image_id = f"{split}_{name}"
            relative = f"{split}/{image_id}.png"
            path = images / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.new("RGB", (10, 10), (30, 60, 90)).save(path)
            gallery.append({"image_id": image_id, "path": relative})
            heads.append(
                {
                    "image_id": image_id,
                    "identity_id": "p1" if name != "n" else "p2",
                    "x": 2,
                    "y": 2,
                    "width": 2,
                    "height": 2,
                }
            )
        sample = {
            **_sample(split),
            "query_image_id": f"{split}_q",
            "target_image_id": f"{split}_a",
            "positive_image_ids": [f"{split}_a"],
        }
        samples.append(sample)
        (final / "splits" / f"{split}.txt").write_text(split + "\n")
    for filename, rows in [
        ("samples", samples),
        ("images", gallery),
        ("gallery", gallery),
        ("head_boxes", heads),
    ]:
        (final / f"{filename}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (final / "manifest.json").write_text('{"version":"tiny"}')
    data = {
        "final_dir": str(final),
        "image_root": str(images),
        "cache": str(tmp_path / "cache"),
    }
    cfg_root = Path("configs/methods/proposed")
    configs = {
        name: yaml.safe_load((cfg_root / f"{name}.yaml").read_text())
        for name in ("build_cache", "train", "retrieve")
    }
    for cfg in configs.values():
        cfg["data"] = copy.deepcopy(data)
    configs["build_cache"]["image_encoder"].update(
        model="tiny-image", scene_size=[8, 8], person_size=[8, 4]
    )
    configs["build_cache"]["detector"]["model"] = "tiny-detector"
    configs["build_cache"]["device"] = "cpu"
    train = configs["train"]
    train["model"].update(
        text_model="tiny-text",
        identity_dim=6,
        state_dim=4,
        num_heads=2,
        mlp_ratio=2,
        geo_dim=4,
    )
    train["train"].update(epochs=1, batch_size=1, candidates=2, device="cpu")
    train["evaluation"].update(
        every_epochs=1,
        train_max_queries=0,
        top_m=2,
        fine_batch_size=1,
        identity_batch_size=2,
        coarse_batch_size=2,
        candidate_ks=[1, 2],
    )
    train["wandb"]["enabled"] = False
    train["output"]["dir"] = str(tmp_path / "run")
    retrieve = configs["retrieve"]
    retrieve["retrieval"].update(
        top_m=2,
        fine_batch_size=1,
        identity_batch_size=2,
        coarse_batch_size=2,
        device="cpu",
    )
    retrieve["candidate_ks"] = [1, 2]
    retrieve["checkpoint"] = str(tmp_path / "run/best.pt")
    retrieve["output"]["dir"] = str(tmp_path / "run/{split}")
    paths = {}
    for name, cfg in configs.items():
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(cfg))
    args = run_proposed.make_parser().parse_args(
        [
            "--config",
            str(paths["retrieve"]),
            "--train-config",
            str(paths["train"]),
            "--cache-config",
            str(paths["build_cache"]),
            "--build-cache",
            "--train",
        ]
    )
    return args, configs, paths


@pytest.fixture
def local_stages(monkeypatch):
    class PixelProcessor:
        def __call__(self, images, **kwargs):
            rows = images if isinstance(images, list) else [images]
            width, height = rows[0].size
            return {"pixel_values": torch.ones(len(rows), 3, height, width)}

    def factory(fn):
        return SimpleNamespace(from_pretrained=lambda *args, **kwargs: fn())

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=factory(Tokenizer),
            AutoModel=SimpleNamespace(
                from_pretrained=lambda name: (
                    ImageBackbone() if name == "tiny-image" else TextBackbone()
                )
            ),
            AutoImageProcessor=factory(PixelProcessor),
            AutoProcessor=factory(DetectorProcessor),
            AutoModelForZeroShotObjectDetection=factory(Detector),
        ),
    )
    calls = []
    modules = {
        "build_cache.py": build_cache,
        "train_proposed.py": train_proposed,
        "retrieve_proposed.py": retrieve_proposed,
        "evaluate.py": evaluate,
    }

    def execute(command, *, cwd, check):
        assert command[0] == sys.executable and cwd == run_proposed.ROOT and check
        name = Path(command[1]).name
        cfg = yaml.safe_load(Path(command[3]).read_text())
        calls.append((name, cfg))
        if name == "evaluate.py":
            # New retrieval must have removed any prior metrics before evaluation.
            assert not (Path(cfg["output"]["dir"]) / "metrics.json").exists()
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", command[1:])
            modules[name].main()

    monkeypatch.setattr(run_proposed.subprocess, "run", execute)
    return calls


def test_complete_pipeline_then_evaluate_without_rebuild(
    experiment, local_stages, monkeypatch
):
    args, configs, paths = experiment
    original_configs = {name: path.read_bytes() for name, path in paths.items()}
    rows = run_proposed.run_pipeline(args)
    assert [name for name, _ in local_stages] == [
        "build_cache.py",
        "train_proposed.py",
        "retrieve_proposed.py",
        "evaluate.py",
        "retrieve_proposed.py",
        "evaluate.py",
    ]
    assert [row["split"] for row in rows] == ["val", "test"]
    assert all(row["num_queries"] == 1 for row in rows)
    run = Path(configs["train"]["output"]["dir"])
    best = torch.load(run / "best.pt", weights_only=True)
    index = Path(configs["train"]["data"]["cache"]) / "index.pt"
    cache = torch.load(index, weights_only=True)
    assert best["cache_id"] == cache["cache_id"]
    assert best["best_epoch"] == 1 and best["best_full_map"] is not None
    for split in ("val", "test"):
        cfg = yaml.safe_load((run / f"runner_configs/{split}.yaml").read_text())
        assert cfg["checkpoint"] == str(run / "best.pt")
        saved = torch.load(run / split / "rankings.pt", weights_only=True)
        assert saved["sample_ids"] == [split]
        assert saved["gallery_ids"] == [f"{split}_{x}" for x in ("q", "a", "n")]
        assert set(saved["rankings"][0].tolist()) == {1, 2}
        metadata = json.loads((run / split / "run.json").read_text())
        assert metadata["cache_id"] == best["cache_id"]
        assert metadata["checkpoint_epoch"] == 1
    assert len((run / "summary.csv").read_text().splitlines()) == 3
    assert all(
        path.read_bytes() == original_configs[name] for name, path in paths.items()
    )
    before = index.stat().st_mtime_ns, (run / "best.pt").read_bytes()
    local_stages.clear()
    args.build_cache = args.train = False
    args.splits = ["test"]
    run_proposed.run_pipeline(args)
    assert [name for name, _ in local_stages] == ["retrieve_proposed.py", "evaluate.py"]
    assert (index.stat().st_mtime_ns, (run / "best.pt").read_bytes()) == before
    assert len((run / "summary.csv").read_text().splitlines()) == 2

    # A rebuilt cache must be rejected before any backbone download/load.
    cache["cache_id"] = "rebuilt"
    torch.save(cache, index)
    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel,
        "from_pretrained",
        lambda *a: pytest.fail("loaded backbone before cache check"),
    )
    monkeypatch.setattr(
        sys, "argv", ["retrieve", "--config", str(run / "runner_configs/test.yaml")]
    )
    with pytest.raises(ValueError, match="differs from the training cache"):
        retrieve_proposed.main()


@pytest.mark.parametrize("failure", ["build_cache.py", "train_proposed.py"])
def test_failed_stage_never_evaluates_old_checkpoint(
    experiment, local_stages, monkeypatch, failure
):
    args, configs, _ = experiment
    run = Path(configs["train"]["output"]["dir"])
    run.mkdir()
    (run / "best.pt").write_bytes(b"old checkpoint")
    (run / "summary.csv").write_text("old summary")
    execute = run_proposed.subprocess.run

    def fail(command, **kwargs):
        if Path(command[1]).name == failure:
            raise subprocess.CalledProcessError(1, command)
        return execute(command, **kwargs)

    monkeypatch.setattr(run_proposed.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_proposed.run_pipeline(args)
    assert not (run / "summary.csv").exists()
    assert not any(name == "retrieve_proposed.py" for name, _ in local_stages)


@pytest.mark.parametrize(
    "invalid", ["data", "disabled_val", "top_m", "build_without_train"]
)
def test_invalid_pipeline_fails_before_any_stage(experiment, local_stages, invalid):
    args, configs, paths = experiment
    if invalid == "data":
        configs["train"]["data"]["cache"] += "-other"
    elif invalid == "disabled_val":
        configs["train"]["evaluation"]["enabled"] = False
    elif invalid == "top_m":
        configs["train"]["evaluation"]["top_m"] = 1
    else:
        args.train = False
    paths["train"].write_text(yaml.safe_dump(configs["train"]))
    with pytest.raises(ValueError):
        run_proposed.run_pipeline(args)
    assert not local_stages


def test_path_overrides_are_shared_by_all_stages(experiment, local_stages, tmp_path):
    args, _, _ = experiment
    args.cache = str(tmp_path / "new cache")
    args.output_dir = str(tmp_path / "new run")
    rows = run_proposed.run_pipeline(args)
    assert all(cfg["data"]["cache"] == args.cache for _, cfg in local_stages)
    assert all(
        row["checkpoint"] == str(Path(args.output_dir) / "best.pt") for row in rows
    )
    assert (Path(args.cache) / "index.pt").is_file()
    assert (Path(args.output_dir) / "test/metrics.json").is_file()


@pytest.mark.parametrize(
    "script,entry",
    [("build_cache.bash", "build_cache.py"), ("run_proposed.bash", "run_proposed.py")],
)
def test_shell_wrapper_environment_and_quoted_arguments(tmp_path, script, entry):
    interpreter = tmp_path / "python with spaces"
    log = tmp_path / "called.json"
    interpreter.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\n"
        'Path(os.environ["CALL_LOG"]).write_text(\n'
        "    json.dumps([os.getcwd(), sys.argv[1:]]))\n"
    )
    interpreter.chmod(0o755)
    env = {**os.environ, "PROPOSED_PYTHON": str(interpreter), "CALL_LOG": str(log)}
    subprocess.run(
        [
            "bash",
            str(run_proposed.ROOT / "scripts" / script),
            "--config",
            "config with spaces.yaml",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    cwd, args = json.loads(log.read_text())
    assert cwd == str(run_proposed.ROOT)
    assert args == [f"tools/methods/{entry}", "--config", "config with spaces.yaml"]
