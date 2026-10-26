"""Standalone retrieve/evaluate and calibration using a real tiny checkpoint."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rcr.common.data import split_fingerprint
from rcr.common.io import sha256_file, write_json
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.evaluation.report import ReportError, load_run
from rcr.evaluation.runner import evaluate_run
from rcr.proposed import calibration, retrieve
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.replay import replay_rankings
from rcr.proposed.train import _make_checkpoint
from tests.test_train import TextBackbone, TinyCache, Tokenizer, _sample


@pytest.fixture
def saved_pipeline(tmp_path, monkeypatch):
    torch.manual_seed(3)
    cache = TinyCache()
    samples = [_sample("train", "q", "a"), _sample("val", "vq", "va")]
    data = SimpleNamespace(
        samples=samples,
        samples_by_id={s["sample_id"]: s for s in samples},
        splits={"train": ["train"], "val": ["val"], "test": []},
        gallery_ids=cache.image_ids,
        images_by_id={
            i: {
                "image_id": i,
                "path": f"{'val' if i.startswith('v') else 'train'}/{i}.jpg",
            }
            for i in cache.image_ids
        },
        gt_head_boxes_by_image={i: [{"identity_id": "p1"}] for i in cache.image_ids},
    )
    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["model"].update(dim=8, identity_dim=6, state_dim=4, num_heads=2, geo_dim=4)
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=2
    )
    cfg["runtime"]["device"] = "cpu"
    cfg["candidate_ks"] = [1, 2]
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["checkpoint"] = str(tmp_path / "old.pt")
    model = RCRModel.from_config(cfg["model"], input_dim=8, person_input_dim=8)
    # Non-default learned scale must survive an old checkpoint round trip.
    model.binding_log_scale.data.fill_(-0.7)
    text = TextEncoder(TextBackbone(), dim=8)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["retrieval"].pop("fine_score_mode")
    old_cfg["retrieval"].pop("context_weight")
    checkpoint = _make_checkpoint(
        epoch=2,
        model=model,
        text_encoder=text,
        tokenizer=Tokenizer(),
        cfg=old_cfg,
        cache=cache,
        dim=8,
        scene_dim=8,
        person_dim=8,
        best={"epoch": 2, "full_map": 0.1, "macro_full_map": 0.1, "score": 0.1},
        state_pairs=1,
    )
    torch.save(checkpoint, cfg["checkpoint"])
    monkeypatch.setattr(retrieve, "load_rcr_data", lambda *args: data)
    monkeypatch.setattr(retrieve, "GalleryCache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(retrieve, "check_query_text_config", lambda *args: None)
    monkeypatch.setattr(retrieve, "check_cache_config", lambda *args: None)
    monkeypatch.setattr(
        retrieve,
        "load_text_modules",
        lambda *args, **kwargs: (Tokenizer(), TextEncoder(TextBackbone(), dim=8)),
    )
    monkeypatch.setattr(calibration, "load_rcr_data", lambda *args: data)
    return cfg, data, model.binding_scale.item()


def test_standalone_retrieve_save_evaluate_and_old_artifact_compatibility(
    saved_pipeline,
):
    cfg, data, scale = saved_pipeline
    output = retrieve.retrieve(cfg)
    directory = Path(cfg["output"]["dir"]) / "val"
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert saved["binding_scale"] == scale != 1.0
    assert torch.equal(replay_rankings(saved)["rankings"], output["rankings"])
    current_metrics = evaluate_run(cfg, data=data)
    # The unchanged evaluator continues to accept the pre-score-schema payload.
    old = {
        k: saved[k]
        for k in (
            "sample_ids",
            "gallery_ids",
            "rankings",
            "coarse_rankings",
            "coarse_topm",
            "fine_rankings",
            "fafa_rankings",
            "fafa_binding",
            "context",
            "coverage",
        )
    }
    assert evaluate_retrieval_output(
        data, [data.samples[1]], old, [1, 2], split="val"
    ) == evaluate_retrieval_output(data, [data.samples[1]], saved, [1, 2], split="val")
    assert current_metrics["overall"]["num_queries"] == 1
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["score_config"] == saved["score_config"]
    assert metadata["score_schema_version"] == 1
    assert metadata["fine_score_mode"] == "context_zscore"
    assert metadata["context_weight"] == cfg["retrieval"]["context_weight"]
    assert metadata["checkpoint_epoch"] == 2
    assert metadata["validation_sha256"] == split_fingerprint(data, "val")
    assert {
        "checkpoint_sha256",
        "dataset_sha256",
        "git_commit",
        "git_dirty",
    } <= metadata.keys()

    unselected_test = copy.deepcopy(cfg)
    unselected_test["split"] = "test"
    with pytest.raises(ValueError, match="validation-selected configuration"):
        retrieve.retrieve(unselected_test)

    incomplete_test = copy.deepcopy(unselected_test)
    incomplete_test.update(
        selected_checkpoint_sha256=metadata["checkpoint_sha256"],
        selected_validation_sha256=metadata["validation_sha256"],
        selected_retrieval={},
    )
    with pytest.raises(ValueError, match="retrieval contract is incomplete"):
        retrieve.retrieve(incomplete_test)


def test_dino_bert_checkpoint_is_rejected_before_loading_backbone(
    saved_pipeline, monkeypatch
):
    cfg, _, _ = saved_pipeline
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    checkpoint["architecture_version"] = "fafa-native-partial-match-v1"
    torch.save(checkpoint, cfg["checkpoint"])

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible checkpoints must fail before model loading")

    monkeypatch.setattr(retrieve, "load_text_modules", forbidden)
    with pytest.raises(ValueError, match="checkpoint architecture is incompatible"):
        retrieve.retrieve(cfg)


def test_report_rejects_test_without_complete_validation_selection(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    retrieve.retrieve(cfg)
    evaluate_run(cfg, data=data)
    source = Path(cfg["output"]["dir"]) / "val"
    run = json.loads((source / "run.json").read_text())
    metrics = json.loads((source / "metrics.json").read_text())
    run["split"] = "test"
    run["config"]["split"] = "test"
    metrics["provenance"]["split"] = "test"
    directory = tmp_path / "report-test"
    directory.mkdir()

    def publish():
        write_json(directory / "run.json", run)
        metrics["provenance"]["run_sha256"] = sha256_file(directory / "run.json")
        write_json(directory / "metrics.json", metrics)

    publish()
    with pytest.raises(ReportError, match="validation-selected configuration"):
        load_run("proposed", directory, "test")

    run["config"].update(
        selected_checkpoint_sha256=run["checkpoint_sha256"],
        selected_validation_sha256=run["validation_sha256"],
        selected_retrieval={},
    )
    publish()
    with pytest.raises(ReportError, match="validation-selected configuration"):
        load_run("proposed", directory, "test")

    keys = (
        "fine_score_mode",
        "context_weight",
        "coarse_beta",
        "fine_coarse_weight",
        "top_m",
    )
    run["config"]["selected_retrieval"] = {key: run["retrieval"][key] for key in keys}
    publish()
    assert load_run("proposed", directory, "test")["run"] == run


def test_calibration_freezes_full_scoring_config_and_retrieves_selected(
    saved_pipeline, tmp_path
):
    cfg, data, _ = saved_pipeline
    model_config = tmp_path / "proposed.yaml"
    model_config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "calibration"
    rows = calibration.calibrate(
        {
            "model_config": str(model_config),
            "output_dir": str(root),
            "coarse_beta": [0.0, 0.4],
            "fine_coarse_weight": [0.0, 0.4],
        }
    )
    assert len(rows) == 4
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["retrieval"]["fine_score_mode"] == "context_zscore"
    assert selected["retrieval"]["context_weight"] == cfg["retrieval"]["context_weight"]
    assert selected["selected_validation_sha256"] == split_fingerprint(data, "val")
    assert len(selected["selected_checkpoint_sha256"]) == 64
    best = max(rows, key=lambda row: row["full_map"])
    output = retrieve.retrieve(selected)
    evaluate_run(selected, data=data)
    expected = torch.load(
        root / f"grid_{best['index']:03d}" / "val/rankings.pt", weights_only=True
    )
    assert torch.equal(output["rankings"], expected["rankings"])

    for key, value in (
        ("context_weight", 0.2),
        ("fine_score_mode", "legacy_raw"),
        ("coarse_beta", 3.0),
        ("fine_coarse_weight", 3.0),
        ("top_m", 1),
    ):
        tampered = copy.deepcopy(selected)
        tampered["split"] = "test"
        tampered["retrieval"][key] = value
        with pytest.raises(ValueError, match="settings changed since val selection"):
            retrieve.retrieve(tampered)
    # Chunking is operational, and may change without selecting a new method.
    selected["retrieval"]["fine_batch_size"] = 2
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])
    del selected["selected_retrieval"]
    with pytest.raises(ValueError, match="predates the scoring contract"):
        retrieve.retrieve(selected)
