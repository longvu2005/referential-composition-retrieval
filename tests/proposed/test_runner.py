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

from rcr.proposed import experiments as runner
from rcr.proposed import train as training_cli
from tests.proposed.test_build_cache import (
    Backbone as ImageBackbone,
)
from tests.proposed.test_build_cache import (
    Detector,
    DetectorProcessor,
)
from tests.proposed.test_retrieval import _sample
from tests.proposed.test_train_cli import Backbone as TextBackbone
from tests.proposed.test_train_cli import Tokenizer
from tools import run as cli


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
                    # Same ID, different condition: meaningful state supervision.
                    "identity_id": "p1",
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
    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["data"] = data
    cfg["person_encoder"]["backend"] = "dino"
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
    local_stages.clear()
    cli.main(["run", "--config", str(path), "--build-cache", "--splits", "test"])
    assert [name for name, _ in local_stages] == ["build-cache", "retrieve", "evaluate"]
    assert (index.read_bytes(), (root / "best.pt").read_bytes()) == before


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


@pytest.mark.parametrize("invalid", ["disabled_val", "top_m"])
def test_invalid_pipeline_fails_before_any_stage(experiment, local_stages, invalid):
    cfg, path = experiment
    options = ["--build-cache", "--train"]
    if invalid == "disabled_val":
        cfg["evaluation"]["enabled"] = False
    elif invalid == "top_m":
        cfg["retrieval"]["top_m"] = 1
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


def test_training_mining_schedule_and_train_only_pool(
    experiment, local_stages, monkeypatch
):
    cfg, path = experiment
    cfg["train"]["epochs"] = 4
    cfg["train"]["sampling"].update(
        identity_fraction=0.0,
        hard_fraction=1.0,
        warmup_epochs=1,
        refresh_every_epochs=2,
        pool_size=2,
    )
    path.write_text(yaml.safe_dump(cfg))
    calls = []
    original = training_cli.mine_hard_negatives

    def mine(samples, *args, **kwargs):
        assert [row["sample_id"] for row in samples] == ["train"]
        assert kwargs["gallery_ids"] == ["train_q", "train_a", "train_n"]
        pools = original(samples, *args, **kwargs)
        assert pools == {"train": ["train_n"]}
        calls.append(pools)
        return pools

    monkeypatch.setattr(training_cli, "mine_hard_negatives", mine)
    cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    assert len(calls) == 2
    output = Path(cfg["output"]["dir"])
    pools = json.loads((output / "hard_negatives.json").read_text())
    assert pools["model_epoch"] == 3 and pools["split"] == "train"
    history = [
        json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()
    ]
    assert [row["epoch/sampled_hard_fraction"] for row in history] == [0, 1, 1, 1]
    assert all(row["epoch/state_pairs"] == 1 for row in history)
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["state_supervised_pairs"] == 4


def test_training_rejects_using_state_without_its_loss(experiment, local_stages):
    cfg, _ = experiment
    cfg["loss"]["state_weight"] = 0
    with pytest.raises(ValueError, match="state_weight=0"):
        training_cli.train(cfg)


def test_training_rejects_wrong_person_backend_without_native_assets(
    experiment, local_stages
):
    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    cfg["person_encoder"].update(backend="fafa", fafa_config="missing-fafa.yaml")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="cache person encoder differs"):
        cli.main(["train", "--config", str(path)])


@pytest.mark.parametrize("mismatch", ["width", "metadata"])
def test_retrieval_rejects_checkpoint_cache_mismatch(
    experiment, local_stages, mismatch
):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    if mismatch == "width":
        checkpoint["person_input_dim"] += 1
    else:
        checkpoint["encoder_metadata"]["person_encoder"] = {"backend": "fafa"}
    torch.save(checkpoint, cfg["checkpoint"])
    with pytest.raises(ValueError, match="differ.*training"):
        cli.main(["retrieve", "--config", str(path), "--splits", "val"])


def test_validation_fingerprint_never_depends_on_test_labels(experiment):
    from rcr.common.data import load_rcr_data, split_fingerprint

    cfg, _ = experiment
    data = load_rcr_data(cfg["data"]["final_dir"])
    before = split_fingerprint(data, "val")
    data.samples_by_id["test"]["positive_image_ids"] = ["test_n"]
    data.samples_by_id["test"]["final_change"] = "changed test"
    assert split_fingerprint(data, "val") == before
    data.samples_by_id["val"]["final_change"] += " outdoors"
    assert split_fingerprint(data, "val") != before


def test_inference_rejects_checkpoint_with_untrained_state(
    experiment, local_stages, monkeypatch
):
    cfg, path = experiment
    cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    checkpoint["state_supervised_pairs"] = 0
    torch.save(checkpoint, cfg["checkpoint"])
    monkeypatch.setattr(
        sys.modules["transformers"].AutoModel,
        "from_pretrained",
        lambda *args: pytest.fail("loaded untrained state model"),
    )
    with pytest.raises(ValueError, match="state branch is untrained"):
        cli.main(["retrieve", "--config", str(path)])


@pytest.mark.parametrize(
    "override", ["train.epochs=0", "train.batch_size=0", "evaluation.every_epochs=0"]
)
def test_invalid_training_schedule_never_prepares_assets(
    experiment, local_stages, monkeypatch, override
):
    from rcr.proposed.cache import build

    _, path = experiment
    monkeypatch.setattr(
        build, "run_person_worker", lambda *a, **k: pytest.fail("prepared assets")
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "run",
                "--config",
                str(path),
                "--prepare",
                "--build-cache",
                "--train",
                "--set",
                override,
            ]
        )
    assert not local_stages
