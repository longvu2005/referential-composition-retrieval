"""Exercise stage coordination with real cache/train/retrieval/evaluation code."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from rcr.methods.proposed import runner
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
from tools.methods import run as cli


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
    cfg = yaml.safe_load(Path("configs/methods/proposed.yaml").read_text())
    cfg["data"] = data
    cfg["runtime"]["device"] = "cpu"
    cfg["image_encoder"].update(
        model="tiny-image", scene_size=[8, 8], person_size=[8, 4]
    )
    cfg["detector"]["model"] = "tiny-detector"
    cfg["model"].update(
        text_model="tiny-text",
        identity_dim=6,
        state_dim=4,
        num_heads=2,
        mlp_ratio=2,
        geo_dim=4,
    )
    cfg["train"].update(epochs=1, batch_size=1, candidates=2)
    cfg["evaluation"].update(every_epochs=1, train_max_queries=0)
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=2
    )
    cfg["candidate_ks"] = [1, 2]
    cfg["wandb"]["enabled"] = False
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["checkpoint"] = str(tmp_path / "run/best.pt")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


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

    def execute(command, *, check):
        assert command[0] == sys.executable and check
        name = command[2]
        cfg = yaml.safe_load(Path(command[4]).read_text())
        calls.append((name, cfg))
        cli.main(command[2:])

    monkeypatch.setattr(runner.subprocess, "run", execute)
    return calls


def test_complete_pipeline_then_evaluate_without_rebuild(experiment, local_stages):
    cfg, path = experiment
    original = path.read_bytes()
    rows = cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    assert [name for name, _ in local_stages] == [
        "build-cache",
        "train",
        "retrieve",
        "evaluate",
        "retrieve",
        "evaluate",
    ]
    assert [row["split"] for row in rows] == ["val", "test"]
    assert all(row["num_queries"] == 1 for row in rows)
    root = Path(cfg["output"]["dir"])
    best = torch.load(root / "best.pt", weights_only=True)
    index = Path(cfg["data"]["cache"]) / "index.pt"
    cache = torch.load(index, weights_only=True)
    assert best["cache_id"] == cache["cache_id"]
    assert best["best_epoch"] == 1
    for split in ("val", "test"):
        saved = torch.load(root / split / "rankings.pt", weights_only=True)
        assert saved["sample_ids"] == [split]
        assert saved["gallery_ids"] == [f"{split}_{x}" for x in ("q", "a", "n")]
        assert set(saved["rankings"][0].tolist()) == {1, 2}
        metadata = json.loads((root / split / "run.json").read_text())
        assert metadata["cache_id"] == best["cache_id"]
        assert metadata["checkpoint_epoch"] == 1
    assert path.read_bytes() == original
    before = index.read_bytes(), (root / "best.pt").read_bytes()
    local_stages.clear()
    cli.main(["run", "--config", str(path), "--splits", "test"])
    assert [name for name, _ in local_stages] == ["retrieve", "evaluate"]
    assert (index.read_bytes(), (root / "best.pt").read_bytes()) == before
    assert len((root / "summary.csv").read_text().splitlines()) == 2


@pytest.mark.parametrize("failure", ["build-cache", "train"])
def test_failed_stage_never_evaluates_old_checkpoint(
    experiment, local_stages, monkeypatch, failure
):
    cfg, path = experiment
    root = Path(cfg["output"]["dir"])
    root.mkdir()
    (root / "best.pt").write_bytes(b"old checkpoint")
    (root / "summary.csv").write_text("old summary")
    execute = runner.subprocess.run

    def fail(command, **kwargs):
        if command[2] == failure:
            raise subprocess.CalledProcessError(1, command)
        return execute(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    assert not (root / "summary.csv").exists()
    assert not any(name == "retrieve" for name, _ in local_stages)


@pytest.mark.parametrize("invalid", ["disabled_val", "top_m", "build_without_train"])
def test_invalid_pipeline_fails_before_any_stage(experiment, local_stages, invalid):
    cfg, path = experiment
    options = ["--build-cache", "--train"]
    if invalid == "disabled_val":
        cfg["evaluation"]["enabled"] = False
    elif invalid == "top_m":
        cfg["retrieval"]["top_m"] = 1
    else:
        options.remove("--train")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises((ValueError, SystemExit)):
        cli.main(["run", "--config", str(path), *options])
    assert not local_stages


def test_overrides_are_shared_and_original_yaml_is_unchanged(
    experiment, local_stages, tmp_path
):
    _, path = experiment
    before = path.read_bytes()
    cache, output = tmp_path / "new cache", tmp_path / "new run"
    rows = cli.main(
        [
            "run",
            "--config",
            str(path),
            "--build-cache",
            "--train",
            "--set",
            f"data.cache={cache}",
            f"output.dir={output}",
        ]
    )
    assert all(cfg["data"]["cache"] == str(cache) for _, cfg in local_stages)
    assert all(row["checkpoint"] == str(output / "best.pt") for row in rows)
    assert (cache / "index.pt").is_file()
    assert (output / "test/metrics.json").is_file()
    assert path.read_bytes() == before


def test_rebuilt_cache_is_rejected_before_model_load(
    experiment, local_stages, monkeypatch
):
    cfg, path = experiment
    cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    index = Path(cfg["data"]["cache"]) / "index.pt"
    cache = torch.load(index, weights_only=True)
    cache["cache_id"] = "rebuilt"
    torch.save(cache, index)
    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel,
        "from_pretrained",
        lambda *args: pytest.fail("loaded model before cache check"),
    )
    with pytest.raises(ValueError, match="differs from the training cache"):
        cli.main(["retrieve", "--config", str(path), "--splits", "test"])
