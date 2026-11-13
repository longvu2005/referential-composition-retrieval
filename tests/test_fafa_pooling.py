"""CPU workflow tests for fafa pooling."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from rcr.baselines import FAFA_POOLINGS
from rcr.common.io import output_directory, write_jsonl
from rcr.evaluation.report import load_run


@pytest.fixture
def frozen_fafa(tmp_path, monkeypatch):
    from rcr.baselines import fafa, fafa_detector

    final = tmp_path / "final"
    images = tmp_path / "images"
    samples, image_rows, heads = [], [], []
    for split, prefix, ids in [("val", "v", [10, 20]), ("test", "t", [30, 40])]:
        samples.append(
            {
                "sample_id": prefix,
                "case_type": "DUAL",
                "query_image_id": prefix + "q",
                "target_image_id": prefix + "p",
                "positive_image_ids": [prefix + "p"],
                "subjects": [
                    {"subject_id": s + 1, "identity_ids": [str(i)]}
                    for s, i in enumerate(ids)
                ],
                "final_desc": (
                    "Identify Subject 1 as the man and Subject 2 as the woman"
                ),
                "final_change": "Subject 1 is seated and Subject 2 is standing",
                "final_instruction": "public instruction",
            }
        )
        for suffix in ["q", "p", "n"]:
            iid = prefix + suffix
            relative = f"{split}/{iid}.png"
            path = images / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
            image_rows.append({"image_id": iid, "path": relative})
            for j, identity in enumerate(ids[:1] if suffix == "n" else ids):
                heads.append(
                    {
                        "box_id": f"{iid}-{j}",
                        "image_id": iid,
                        "identity_id": str(identity),
                        "x": 4 * j,
                        "y": 0,
                        "width": 4,
                        "height": 8,
                    }
                )
    write_jsonl(final / "samples.jsonl", samples)
    write_jsonl(final / "images.jsonl", image_rows)
    write_jsonl(
        final / "gallery.jsonl", [{"image_id": x["image_id"]} for x in image_rows]
    )
    write_jsonl(final / "head_boxes.jsonl", heads)
    (final / "splits").mkdir()
    for split, text in [("train", ""), ("val", "v\n"), ("test", "t\n")]:
        (final / "splits" / f"{split}.txt").write_text(text)
    (final / "manifest.json").write_text(json.dumps({"version": "fixture"}))
    cfg = yaml.safe_load(Path("configs/fafa_pooling.yaml").read_text())
    cfg["data"] = {"final_dir": str(final), "image_root": str(images)}
    cfg["runtime"]["device"] = "cpu"
    cfg["cache"]["dir"] = str(tmp_path / "cache")
    cfg["output"]["dir"] = str(tmp_path / "pooling")
    for section, key in [
        ("checkpoint", "path"),
        ("checkpoint", "runtime_assets_marker"),
    ]:
        path = tmp_path / key
        path.write_text(key)
        cfg[section][key] = str(path)
    selector = tmp_path / "selector"
    selector.write_text("selector")
    cfg["localization"]["query_selector"]["checkpoint"] = str(selector)
    calls = {"encode": 0, "select": 0}
    monkeypatch.setattr(fafa, "official_source", lambda *a: tmp_path)
    monkeypatch.setattr(fafa, "assets_ready", lambda *a: True)
    monkeypatch.setattr(
        fafa,
        "load_fafa",
        lambda *a: (
            None,
            None,
            None,
            {
                "missing_keys": [],
                "unexpected_keys": [],
            },
        ),
    )

    def detections(data, ids, *args):
        boxes = [
            [
                {"box": [4 * j, 0, 4 * (j + 1), 8], "score": 1.0}
                for j in range(1 if iid.endswith("n") else 2)
            ]
            for iid in ids
        ]
        return boxes, "shared-detections"

    monkeypatch.setattr(fafa, "detect_gallery", detections)
    monkeypatch.setattr(
        fafa_detector,
        "load_detection_cache",
        lambda *a: SimpleNamespace(cache_id="shared-detections"),
    )

    def components(data, public_samples, *args):
        calls["select"] += 1
        grouped = []
        for sample in public_samples:
            assert "case_type" not in sample and "positive_image_ids" not in sample
            assert all(set(s) == {"subject_id"} for s in sample["subjects"])
            grouped.append(
                [
                    (
                        str(data.image_path(sample["query_image_id"])),
                        [4 * j, 0, 4 * (j + 1), 8],
                        "condition",
                        s["subject_id"],
                    )
                    for j, s in enumerate(sample["subjects"])
                ]
            )
        return grouped, {"uses_gt_boxes_or_identities": False}

    monkeypatch.setattr(fafa, "build_query_components", components)

    def encode(model, processors, preprocess, items, runtime, device, *, query, path):
        calls["encode"] += 1
        values = []
        for filename, box, _caption in items:
            axis = int(box[0] > 0)
            vector = np.zeros(4, dtype=np.float32)
            if query or Path(filename).stem.endswith("q"):
                vector[axis] = 1
            else:
                value = 0.9 if Path(filename).stem.endswith("n") else 0.6
                vector[axis], vector[2] = value, np.sqrt(1 - value**2)
            values.append(vector)
        values = np.asarray(values)
        if not query:
            values = np.repeat(values[:, None], 2, axis=1)
        np.save(path, values)
        return values

    monkeypatch.setattr(fafa, "encode_crops", encode)
    return cfg, calls


def test_real_runner_shares_native_cache_preserves_fafa_and_locks_test(frozen_fafa):
    from rcr.baselines.runner import run_experiment
    from rcr.evaluation.runner import evaluate_run

    cfg, calls = frozen_fafa
    with pytest.raises(FileNotFoundError, match="Run --split val first"):
        run_experiment(cfg, splits=["test"])
    baseline = copy.deepcopy(cfg)
    baseline["method"] = "fafa"
    baseline["output"]["dir"] += "_original"
    run_experiment(baseline)
    original_path = Path(baseline["output"]["dir"]) / "val" / "scores.npy"
    original_bytes = original_path.read_bytes()
    assert calls == {"encode": 2, "select": 1}
    run_experiment(cfg)
    assert calls == {"encode": 2, "select": 1}
    assert original_path.read_bytes() == original_bytes
    baseline_run = json.loads((original_path.parent / "run.json").read_text())
    for mode in FAFA_POOLINGS:
        directory = output_directory({**cfg, "mode": mode, "split": "val"})
        run = json.loads((directory / "run.json").read_text())
        assert run["rcr_training"] is False
        assert run["checkpoint_load"]["model_loaded_this_run"] is False
        for key in ["gallery_cache", "query_cache", "detector_cache_id"]:
            assert run[key] == baseline_run[key]
        load_run(f"fafa_{mode}", directory, "val")
    evaluated = evaluate_run(cfg)
    assert evaluated["matched_mean"]["overall"]["full_map"] == 0.5
    assert evaluated["subject_mean"]["overall"]["full_map"] == 1.0
    lock = Path(cfg["output"]["dir"]) / "protocol.json"
    assert set(json.loads(lock.read_text())["val_full_map"]) == set(FAFA_POOLINGS)
    run_experiment(cfg, splits=["test"])
    for mode in FAFA_POOLINGS:
        directory = output_directory({**cfg, "mode": mode, "split": "test"})
        load_run(f"fafa_{mode}", directory, "test")
    changed = copy.deepcopy(cfg)
    changed["setmatch"]["unmatched_score"] = -1.1
    with pytest.raises(ValueError, match="stale"):
        run_experiment(changed, splits=["test"])
