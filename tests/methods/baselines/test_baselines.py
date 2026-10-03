"""Real loader/scoring/cache/CLI/evaluation paths with small deterministic encoders.

Model downloads and accelerator inference are separate from these CPU contract
tests. Native FAFA FDA and Hungarian semantics are checked numerically here.
"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from rcr.methods.baselines import clip, fafa, fafa_adapter
from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.common.results import output_directory
from tools.methods import evaluate, retrieve_baseline


def preprocess(image):
    return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255


class TinyCLIP(torch.nn.Module):
    text_projection = torch.eye(3)

    def encode_image(self, images):
        return images.mean((2, 3))

    def encode_text(self, tokens):
        return tokens.float()


def tiny_clip(checkpoint, device):
    module = SimpleNamespace(
        tokenize=lambda texts, truncate=False: torch.tensor(
            [[0.0, 1.0, 0.0]] * len(texts)
        )
    )
    return module, TinyCLIP().to(device), preprocess


class TinyFAFA(torch.nn.Module):
    def extract_target_features(self, images, mode):
        vectors = F.normalize(images.mean((2, 3)), dim=-1)
        return vectors[:, None].repeat(1, 2, 1), None

    def extract_features(self, samples):
        vector = samples["image"].mean((2, 3)) + torch.tensor([0.0, 1.0, 0.0])
        return SimpleNamespace(multimodal_embeds=F.normalize(vector, dim=-1))


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    final = tmp_path / "final"
    (final / "splits").mkdir(parents=True)
    images_root = tmp_path / "images"
    rows = []
    for image_id, color, split in [
        ("q", (255, 0, 0), "test"),
        ("a", (255, 255, 0), "test"),
        ("b", (0, 0, 255), "test"),
        ("left", (0, 255, 0), "leftover"),
    ]:
        path = images_root / split / f"{image_id}.png"
        path.parent.mkdir(exist_ok=True, parents=True)
        Image.new("RGB", (8, 8), color).save(path)
        rows.append({"image_id": image_id, "path": f"{split}/{image_id}.png"})
    samples = []
    for index, case in enumerate(["INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL"]):
        subjects = [
            {
                "subject_id": 1,
                "identity_ids": ["p1", "p2"] if case == "GROUP" else ["p1"],
            }
        ]
        desc = "Identify Subject 1 as the person in red"
        change = "then retrieve target images where Subject 1 is wearing green"
        if case in ("DUAL", "RELATIONAL"):
            subjects.append({"subject_id": 2, "identity_ids": ["p2"]})
            desc += " and Subject 2 as the other person in red"
            change += (
                " and Subject 2 is standing" if case == "DUAL" else " next to Subject 2"
            )
        samples.append(
            {
                "sample_id": f"s{index}",
                "case_type": case,
                "query_image_id": "q",
                "target_image_id": "a",
                "positive_image_ids": ["a", "left"],
                "subjects": subjects,
                "final_desc": desc,
                "final_change": change,
                "final_instruction": desc + "; " + change,
            }
        )
    heads = [
        {"image_id": image_id, "identity_id": pid}
        for image_id in ["q", "a"]
        for pid in ["p1", "p2"]
    ]
    for name, records in [
        ("samples", samples),
        ("images", rows),
        ("gallery", [{"image_id": r["image_id"]} for r in rows]),
        ("head_boxes", heads),
    ]:
        (final / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    (final / "manifest.json").write_text(json.dumps({"version": "test"}))
    for split in ["train", "val", "test"]:
        (final / "splits" / f"{split}.txt").write_text(
            "\n".join(s["sample_id"] for s in samples) if split == "test" else ""
        )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test model identity")
    marker = tmp_path / "marker.json"
    marker.write_text("{}")
    cfg = yaml.safe_load(Path("configs/methods/baselines/clip.yaml").read_text())
    cfg.update(
        split="test", data={"final_dir": str(final), "image_root": str(images_root)}
    )
    cfg["model"]["checkpoint"] = str(weights)
    cfg["runtime"].update(
        device="cpu", image_batch_size=2, text_batch_size=2, score_batch_size=2
    )
    cfg["cache"]["dir"] = str(tmp_path / "clip_cache")
    cfg["output"]["dir"] = str(tmp_path / "runs" / "{method}" / "{mode}" / "{split}")
    monkeypatch.setattr(clip, "load_clip", tiny_clip)
    monkeypatch.setattr(fafa_adapter, "load_clip", tiny_clip)
    return cfg, weights, marker


@pytest.mark.parametrize("mode", ["image", "text", "early_fusion", "late_fusion"])
def test_clip_retrieval_to_official_evaluation(benchmark, tmp_path, monkeypatch, mode):
    cfg, _, _ = benchmark
    cfg["mode"] = mode
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["retrieve", "--config", str(config)])
    retrieve_baseline.main()
    directory = output_directory(cfg)
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert saved["gallery_ids"] == ["q", "a", "b"]
    assert saved["rankings"].tolist() == [[1, 2]] * 4
    assert "coarse_topm" not in saved
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config)])
    evaluate.main()
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["overall"]["full_r1"] == 1
    assert metrics["overall"]["num_queries"] == 4
    assert set(metrics["by_case"]) == {"INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL"}


def test_clip_cache_shared_across_modes_and_invalidated_by_image(
    benchmark, monkeypatch
):
    cfg, _, _ = benchmark
    cfg["output"]["dir"] = str(output_directory(cfg))
    data = load_rcr_data(**cfg["data"])
    samples, gallery_ids = split_samples(data, "test"), split_image_ids(data, "test")
    _, first = clip.retrieve_clip(data, samples, gallery_ids, cfg, torch.device("cpu"))
    original_encode = clip.encode_images
    monkeypatch.setattr(
        clip, "encode_images", lambda *a, **k: pytest.fail("gallery cache not reused")
    )
    cfg["mode"] = "image"
    _, second = clip.retrieve_clip(data, samples, gallery_ids, cfg, torch.device("cpu"))
    assert first["gallery_cache"] == second["gallery_cache"]
    monkeypatch.setattr(clip, "encode_images", original_encode)
    Image.new("RGB", (9, 9), (0, 255, 0)).save(data.image_path("a"))
    # Keep crop tensors the same size; the image input fingerprint still changes.
    monkeypatch.setattr(
        clip,
        "load_clip",
        lambda c, d: (*tiny_clip(c, d)[:2], lambda im: preprocess(im.resize((8, 8)))),
    )
    _, third = clip.retrieve_clip(data, samples, gallery_ids, cfg, torch.device("cpu"))
    assert third["gallery_cache"] != first["gallery_cache"]


def test_predicted_membership_is_disjoint_and_supports_groups():
    similarity = np.array([[0.9, 0.88, 0.1, 0.2], [0.1, 0.2, 0.95, 0.92]])
    settings = dict(threshold=0.8, margin=0.05, max_members=3)
    assert fafa_adapter.select_subject_members(
        similarity, allow_multiple=True, **settings
    ) == [[0, 1], [2, 3]]
    assert fafa_adapter.select_subject_members(
        similarity, allow_multiple=False, **settings
    ) == [[0], [2]]


def test_selection_does_not_read_gold_identity_count_or_positives(benchmark):
    cfg, _, _ = benchmark
    data = load_rcr_data(**cfg["data"])
    samples, gallery = split_samples(data, "test"), split_image_ids(data, "test")
    localization = yaml.safe_load(
        Path("configs/methods/baselines/fafa.yaml").read_text()
    )["localization"]
    candidates = [
        [
            {"box": [0, 0, 4, 8], "score": 0.9, "fallback": False},
            {"box": [4, 0, 8, 8], "score": 0.8, "fallback": False},
        ]
    ] * 3
    first, stats = fafa_adapter.build_query_components(
        data, samples, gallery, candidates, localization, torch.device("cpu")
    )
    changed = copy.deepcopy(samples)
    for sample in changed:
        sample["target_image_id"] = "b"
        sample["positive_image_ids"] = ["b"]
        for subject in sample["subjects"]:
            subject["identity_ids"] = [f"unknown-{i}" for i in range(7)]
    second, changed_stats = fafa_adapter.build_query_components(
        data, changed, gallery, candidates, localization, torch.device("cpu")
    )
    assert first == second
    assert stats == changed_stats


def test_fafa_native_fda_and_setmatch_semantics():
    query = torch.tensor([[1.0, 0.0]])
    target = torch.tensor(
        [[[0.9, 0.0], [0.7, 0.0], [0.1, 0.0]], [[0.3, 0.0], [0.2, 0.0], [0.1, 0.0]]]
    )
    assert torch.allclose(
        fafa.fda_scores(query, target, 2), torch.tensor([[0.8, 0.25]])
    )
    assert torch.allclose(
        fafa.fda_scores(query, target, 2, use_soft=False), torch.tensor([[0.9, 0.3]])
    )
    # Max-sum chooses the diagonal (1.4 > 1.3), even though its minimum is lower.
    assert fafa_adapter.setmatch_score(np.array([[0.9, 0.65], [0.65, 0.5]])) == 0.5
    assert fafa_adapter.setmatch_score(np.array([[0.9], [0.2]])) == -1
    assert fafa_adapter.setmatch_score(np.empty((2, 0))) == -1


def test_fafa_retrieval_to_official_evaluation(benchmark, tmp_path, monkeypatch):
    clip_cfg, weights, marker = benchmark
    cfg = yaml.safe_load(Path("configs/methods/baselines/fafa.yaml").read_text())
    cfg.update(split="test", data=clip_cfg["data"])
    cfg["output"] = {"dir": str(tmp_path / "runs" / "fafa" / "{split}")}
    cfg["cache"]["dir"] = str(tmp_path / "fafa_cache")
    cfg["checkpoint"].update(path=str(weights), runtime_assets_marker=str(marker))
    cfg["localization"]["query_selector"]["checkpoint"] = str(weights)
    cfg["runtime"].update(device="cpu", num_workers=0, gallery_feature_dtype="float32")
    monkeypatch.setattr(fafa, "official_source", lambda *a, **k: tmp_path)
    monkeypatch.setattr(fafa, "assets_ready", lambda cfg: True)
    candidates = [
        [
            {"box": [0, 0, 4, 8], "score": 0.9, "fallback": False},
            {"box": [4, 0, 8, 8], "score": 0.8, "fallback": False},
        ]
    ] * 3
    monkeypatch.setattr(
        fafa, "detect_gallery", lambda *a, **k: (candidates, tmp_path / "detections")
    )
    monkeypatch.setattr(
        fafa,
        "load_fafa",
        lambda *a, **k: (
            TinyFAFA(),
            {"eval": str},
            preprocess,
            {"missing_keys": [], "unexpected_keys": []},
        ),
    )
    config = tmp_path / "fafa.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["retrieve", "--config", str(config)])
    retrieve_baseline.main()
    directory = output_directory(cfg)
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert saved["rankings"].tolist() == [[1, 2]] * 4
    run = json.loads((directory / "run.json").read_text())
    assert not run["selector"]["uses_gt_boxes_or_identities"]
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config)])
    evaluate.main()
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["overall"]["full_map"] == 1

    from rcr.methods.baselines.runner import run_experiment

    cfg["summary"] = str(tmp_path / "fafa_summary.csv")
    rows = run_experiment(cfg, splits=["test"])
    assert rows[0]["method"] == "fafa" and rows[0]["full_map"] == 1


def test_query_subset_requires_matching_evaluation_selection(
    benchmark, tmp_path, monkeypatch
):
    cfg, _, _ = benchmark
    config = tmp_path / "subset.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        sys, "argv", ["retrieve", "--config", str(config), "--max-queries", "2"]
    )
    retrieve_baseline.main()
    directory = output_directory(cfg)
    run = json.loads((directory / "run.json").read_text())
    assert run["query_subset"] and run["num_queries"] == 2
    assert run["num_gallery"] == 3
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config)])
    with pytest.raises(ValueError, match="sample_ids"):
        evaluate.main()
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--config", str(config), "--max-queries", "2"]
    )
    evaluate.main()
    assert (
        json.loads((directory / "metrics.json").read_text())["overall"]["num_queries"]
        == 2
    )


def test_native_clip_library_load_and_rankings(benchmark, tmp_path, monkeypatch):
    """Exercise actual pinned CLIP loading/tokenization with a tiny random network."""
    pytest.importorskip("clip")
    from clip.model import CLIP

    # This is a compatibility test, not a pretrained-quality experiment.
    torch.manual_seed(11)
    native_model = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    ).eval()
    checkpoint = tmp_path / "tiny-native-clip.pt"
    # Released OpenAI checkpoints are TorchScript archives. Match that format;
    # the pinned upstream loader's raw-state_dict fallback does not rewind its
    # file handle after a failed JIT load on newer PyTorch.
    traced = torch.jit.trace(
        native_model,
        (torch.zeros(1, 3, 32, 32), torch.zeros(1, 77, dtype=torch.int64)),
        check_trace=False,
    )
    traced.save(str(checkpoint))
    import clip as native_clip

    def load_native(path, device):
        model, transform = native_clip.load(str(path), device=device, jit=False)
        return native_clip, model, transform

    monkeypatch.setattr(clip, "load_clip", load_native)
    cfg, _, _ = benchmark
    cfg["model"]["checkpoint"] = str(checkpoint)
    cfg["output"]["dir"] = str(output_directory(cfg))
    data = load_rcr_data(**cfg["data"])
    samples, gallery = split_samples(data, "test"), split_image_ids(data, "test")
    scores, details = clip.retrieve_clip(
        data, samples, gallery, cfg, torch.device("cpu")
    )
    assert scores.shape == (4, 3) and np.isfinite(scores).all()
    from rcr.evaluation.evaluate import evaluate_retrieval_output
    from rcr.methods.common.results import scores_to_rankings

    output = scores_to_rankings(samples, gallery, scores)
    metrics = evaluate_retrieval_output(data, samples, output, split="test")
    assert 0 <= metrics["overall"]["full_map"] <= 1
    assert details["truncated_text_queries"] >= 0


@pytest.fixture
def fusion_benchmark(benchmark):
    """Val wants text (green); test wants image (red), so their optima differ."""
    cfg, _, _ = benchmark
    final = Path(cfg["data"]["final_dir"])
    root = Path(cfg["data"]["image_root"])
    samples = [
        json.loads(line) for line in (final / "samples.jsonl").read_text().splitlines()
    ]
    for sample in samples:
        sample["positive_image_ids"] = ["a"]
    Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "test/a.png")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(root / "test/b.png")
    (root / "val").mkdir()
    val_samples = []
    for i, original in enumerate(samples[:3]):  # Uneven final score batch.
        row = copy.deepcopy(original)
        row.update(
            sample_id=f"v{i}",
            query_image_id="vq",
            target_image_id="vb",
            positive_image_ids=["vb"],
        )
        val_samples.append(row)
    samples += val_samples
    (final / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples)
    )
    (final / "splits/val.txt").write_text(
        "\n".join(s["sample_id"] for s in val_samples)
    )
    for name in ["images", "gallery", "head_boxes"]:
        rows = [
            json.loads(line)
            for line in (final / f"{name}.jsonl").read_text().splitlines()
        ]
        if name == "images":
            rows += [
                {"image_id": f"v{i}", "path": f"val/v{i}.png"} for i in ["q", "a", "b"]
            ]
        elif name == "gallery":
            rows += [{"image_id": f"v{i}"} for i in ["q", "a", "b"]]
        else:
            rows += [
                {"image_id": i, "identity_id": p}
                for i in ["b", "vq", "va", "vb"]
                for p in ["p1", "p2"]
            ]
        (final / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    for i in ["q", "a", "b"]:
        (root / f"val/v{i}.png").write_bytes((root / f"test/{i}.png").read_bytes())
    cfg["tuning"].update(
        image_weights=[0, 0.25, 0.5, 0.75, 1], output=str(final.parent / "tuning.json")
    )
    cfg["summary"] = str(final.parent / "summary.csv")
    return cfg


def test_runner_freezes_val_weights_before_test(fusion_benchmark, monkeypatch):
    from rcr.methods.baselines import runner

    cfg = fusion_benchmark
    original = clip.load_clip
    loads = []

    def tracked_load(*args):
        loads.append(1)
        return original(*args)

    monkeypatch.setattr(clip, "load_clip", tracked_load)
    rows = runner.run_experiment(cfg, splits=["test", "val"])
    assert (
        len(rows) == 8 and len(loads) == 2
    )  # One encoder load per split, not per weight.
    selection = json.loads(Path(cfg["tuning"]["output"]).read_text())
    assert selection["split"] == "val" and selection["num_queries"] == 3
    for mode in ["early_fusion", "late_fusion"]:
        trial = selection["modes"][mode]
        assert len(trial["trials"]) == 5
        assert trial["selected"]["image_weight"] == 0.25
        val = next(r for r in rows if r["method"] == mode and r["split"] == "val")
        test = next(r for r in rows if r["method"] == mode and r["split"] == "test")
        assert val["full_r1"] == 1 and test["full_r1"] == 0
        assert val["image_weight"] == test["image_weight"] == 0.25
        directory = output_directory({**cfg, "mode": mode, "split": "test"})
        metadata = json.loads((directory / "run.json").read_text())
        assert metadata["fusion_selection"]["split"] == "val"
        assert metadata["config"]["fusion"]["image_weight"] == 0.25
    assert len(Path(cfg["summary"]).read_text().splitlines()) == 9
    monkeypatch.setattr(
        clip, "load_clip", lambda *a: pytest.fail("warm cache loaded model")
    )
    monkeypatch.setattr(
        runner, "tune_fusion", lambda *a: pytest.fail("test retuned weights")
    )
    again = runner.run_experiment(cfg, splits=["test"])
    assert len(again) == 4
    assert all(r["image_weight"] == 0.25 for r in again if "fusion" in r["method"])


def test_test_labels_cannot_change_selection_context(fusion_benchmark):
    from rcr.methods.baselines.runner import tuning_context

    cfg = fusion_benchmark
    data = load_rcr_data(**cfg["data"])
    before = tuning_context(data, cfg, torch.device("cpu"))
    for sample in data.samples:
        if sample["sample_id"] in data.splits["test"]:
            sample["positive_image_ids"] = ["b"]
            sample["target_image_id"] = "b"
    assert tuning_context(data, cfg, torch.device("cpu")) == before
    data.samples_by_id["v0"]["positive_image_ids"] = ["va"]
    assert tuning_context(data, cfg, torch.device("cpu")) != before


@pytest.mark.parametrize("change", ["checkpoint", "text", "val_label", "grid"])
def test_test_rejects_stale_selection(fusion_benchmark, change):
    from rcr.methods.baselines.runner import run_experiment

    cfg = fusion_benchmark
    run_experiment(cfg, splits=["val"])
    if change == "checkpoint":
        Path(cfg["model"]["checkpoint"]).write_bytes(b"changed")
    elif change == "text":
        cfg["text_field"] = "final_change"
    elif change == "grid":
        cfg["tuning"]["image_weights"] = [0, 1]
    else:
        path = Path(cfg["data"]["final_dir"]) / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["positive_image_ids"] = ["va"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="stale"):
        run_experiment(cfg, splits=["test"])


def test_test_requires_selection_and_val_rejects_subset(fusion_benchmark):
    from rcr.methods.baselines.runner import run_experiment, tune_fusion

    cfg = fusion_benchmark
    with pytest.raises(FileNotFoundError, match="val first"):
        run_experiment(cfg, splits=["test"])
    data = load_rcr_data(**cfg["data"])
    with pytest.raises(ValueError, match="complete validation"):
        tune_fusion(
            data,
            split_samples(data, "val")[:1],
            split_image_ids(data, "val"),
            {},
            cfg,
            "early_fusion",
        )


@pytest.mark.parametrize("mode", clip.MODES)
@pytest.mark.parametrize("iw", [0, 0.25, 0.5, 1])
def test_cached_scoring_matches_original_formula(mode, iw):
    torch.manual_seed(31)
    gallery = F.normalize(torch.randn(11, 8), dim=-1)
    qi = F.normalize(torch.randn(5, 8), dim=-1)
    qt = F.normalize(torch.randn(5, 8), dim=-1)
    inputs = {
        "image": (qi @ gallery.T).numpy(),
        "text": (qt @ gallery.T).numpy(),
        "image_features": qi.numpy(),
        "text_features": qt.numpy(),
    }
    fusion = {"image_weight": iw, "text_weight": 1 - iw}
    actual = np.concatenate(
        [x for _, x in clip.iter_clip_scores(inputs, mode, fusion, 2)]
    )
    expected = clip.score_features(gallery, qi, qt, mode, fusion).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_text_cache_reused_and_invalidated_by_text(benchmark, monkeypatch):
    cfg, _, _ = benchmark
    data = load_rcr_data(**cfg["data"])
    samples, ids = split_samples(data, "test"), split_image_ids(data, "test")
    cfg["mode"] = "clip_text"
    _, first = clip.prepare_clip_inputs(data, samples, ids, cfg, torch.device("cpu"))
    encode = clip.encode_texts
    monkeypatch.setattr(
        clip, "encode_images", lambda *a: pytest.fail("gallery recomputed")
    )
    monkeypatch.setattr(clip, "encode_texts", lambda *a: pytest.fail("text recomputed"))
    _, second = clip.prepare_clip_inputs(data, samples, ids, cfg, torch.device("cpu"))
    assert second["text_cache"] == first["text_cache"]
    monkeypatch.setattr(clip, "encode_texts", encode)
    samples[0]["final_instruction"] += " changed"
    _, third = clip.prepare_clip_inputs(data, samples, ids, cfg, torch.device("cpu"))
    assert third["text_cache"] != first["text_cache"]
    assert third["gallery_cache"] == first["gallery_cache"]


@pytest.mark.parametrize("checkpoint_format", ["raw", "jit"])
def test_native_checkpoint_loads_without_network(tmp_path, checkpoint_format):
    pytest.importorskip("clip")
    from clip.model import CLIP

    model = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    ).eval()
    path = tmp_path / "raw.pt"
    if checkpoint_format == "raw":
        torch.save(model.state_dict(), path)
    else:
        torch.jit.trace(
            model,
            (torch.zeros(1, 3, 32, 32), torch.zeros(1, 77, dtype=torch.int64)),
            check_trace=False,
        ).save(str(path))
    _, loaded, _ = clip.load_clip(path, torch.device("cpu"))
    images = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        expected = model.encode_image(images)
        actual = loaded.encode_image(images)
    # The official builder converts to fp16 before CPU .float(), as clip.load does.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_runner_cli(fusion_benchmark, tmp_path, monkeypatch):
    from tools.methods import run_baselines

    config = tmp_path / "runner.yaml"
    config.write_text(yaml.safe_dump(fusion_benchmark))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--config", str(config), "--modes", "clip_text", "--splits", "test"],
    )
    run_baselines.main()
    directory = output_directory(
        {**fusion_benchmark, "mode": "clip_text", "split": "test"}
    )
    assert (directory / "metrics.json").is_file()


def test_shell_runner_with_native_clip(fusion_benchmark, tmp_path):
    """Real subprocess, native tokenizer/encoders, no monkeypatch in the child."""
    import os
    import subprocess

    pytest.importorskip("clip")
    from clip.model import CLIP

    cfg = fusion_benchmark
    model = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    ).eval()
    torch.save(model.state_dict(), cfg["model"]["checkpoint"])
    config = tmp_path / "native.yaml"
    config.write_text(yaml.safe_dump(cfg))
    env = {
        **os.environ,
        "BASELINE_PYTHON": sys.executable,
        "PYTHONPATH": str(Path("src").resolve()),
        "OMP_NUM_THREADS": "1",
    }
    for splits in (["val", "test"], ["test"]):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_baselines.bash",
                "clip",
                "--config",
                str(config),
                "--splits",
                *splits,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    for mode in clip.MODES:
        for split in ("val", "test"):
            directory = output_directory({**cfg, "mode": mode, "split": split})
            metrics = json.loads((directory / "metrics.json").read_text())
            assert 0 <= metrics["overall"]["full_map"] <= 1
