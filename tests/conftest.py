"""Shared CPU fixtures for training, retrieval and validation workflows."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rcr.proposed import calibration, retrieve
from rcr.proposed import train as training
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.train import _make_checkpoint
from tests.helpers import TextBackbone, TinyCache, Tokenizer, WandbRun, _sample


@pytest.fixture(autouse=True)
def _restore_determinism_switches():
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


@pytest.fixture
def tiny_training_run(tmp_path, monkeypatch):
    samples = [_sample("train", "q", "a"), _sample("val", "vq", "va")]
    cache = TinyCache()
    native_queries = cache.native_queries

    def public_queries(rows, max_people):
        for row in rows:
            assert {"case_type", "positive_image_ids", "target_image_id"}.isdisjoint(
                row
            )
            assert all(set(subject) == {"subject_id"} for subject in row["subjects"])
        return native_queries(rows, max_people)

    monkeypatch.setattr(cache, "native_queries", public_queries)
    wandb_run = WandbRun()
    wandb_init = {}

    def init_wandb(**kwargs):
        wandb_init.update(kwargs)
        return wandb_run

    data = SimpleNamespace(
        samples=samples,
        samples_by_id={row["sample_id"]: row for row in samples},
        splits={"train": ["train"], "val": ["val"], "test": []},
        gallery_ids=cache.image_ids,
        images_by_id={
            image_id: {
                "path": f"{'train' if image_id[0] != 'v' else 'val'}/{image_id}.jpg"
            }
            for image_id in cache.image_ids
        },
        gt_head_boxes_by_image={
            image_id: [
                {
                    "identity_id": "p1",
                    "x": 0.2,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                }
            ]
            for image_id in cache.image_ids
        },
        image_path=lambda image_id: tmp_path / f"{image_id}.jpg",
    )
    monkeypatch.setattr(training, "load_rcr_data", lambda *args: data)
    monkeypatch.setattr(training, "GalleryCache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(training, "check_query_text_config", lambda *args: None)
    monkeypatch.setattr(training, "check_cache_config", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPTokenizerFast=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()
            ),
            CLIPTextModelWithProjection=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: TextBackbone()
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init_wandb))

    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text(encoding="utf-8"))
    cfg["data"] = {
        "final_dir": "unused",
        "image_root": "unused",
        "scene_cache": "unused-scene",
        "detection_cache": "unused-detections",
        "fafa_image_cache": "unused-fafa-images",
        "fafa_query_cache": "unused-fafa-queries",
    }
    cfg["cache"]["supervision_cache"] = str(tmp_path / "supervision.json")
    cfg["runtime"]["device"] = "cpu"
    cfg["model"].update(dim=6, identity_dim=6, state_dim=4, num_heads=2, geo_dim=4)
    cfg["train"].update(
        epochs=1, context_warmup_epochs=0, batch_size=1, candidates=2, amp=False
    )
    # Each training test chooses whether to enable hard mining and rejection.
    cfg["train"]["sampling"]["semantic_hard_fraction"] = 0
    cfg["loss"]["context_rejection_weight"] = 0
    cfg["evaluation"]["every_epochs"] = 1
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=2
    )
    cfg["candidate_ks"] = [1, 2]
    cfg["output"]["dir"] = str(tmp_path / "run")
    return cfg, cache, wandb_run, wandb_init


@pytest.fixture
def objective_batch():
    from rcr.proposed.batch import build_batch

    torch.manual_seed(23)
    cache = TinyCache()

    supervision = SimpleNamespace(
        load_ids=lambda ids, width: [["p1"] * width for _ in ids]
    )
    model = training.RCRModel(
        dim=8,
        identity_dim=6,
        state_dim=4,
        num_heads=2,
        fine_matching={"selection_threshold": 0.0001, "identity_threshold": -1.0},
    ).eval()
    batch = build_batch(
        [_sample("train", "q", "a")],
        [["a", "n", "va"]],
        cache,
        Tokenizer(),
        TextEncoder(TextBackbone(), dim=8),
        "cpu",
        supervision=supervision,
        identity_image_ids=[{"a", "n"}],
        fafa_pooling="subject_mean",
    )
    assert "fafa_query" in batch and "fafa_targets" in batch
    batch = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["extra_identity_persons"] = torch.randn(2, 8)
    batch["extra_identity_labels"] = torch.tensor([1, 1])
    for name in ("target_scene", "target_persons", "target_global"):
        batch[name] = batch[name].detach().clone().requires_grad_()
    return model, batch


@pytest.fixture
def tiny_ranker():
    torch.manual_seed(19)
    cache = TinyCache()
    model = RCRModel(dim=8, identity_dim=6, num_heads=2, state_dim=4)
    text = TextEncoder(TextBackbone(), dim=8)
    samples = [_sample("train", "q", "a"), _sample("val", "vq", "va")]
    settings = dict(
        top_m=4,
        fine_batch_size=2,
        identity_batch_size=2,
        coarse_batch_size=2,
        coarse_beta=0.4,
        fine_score_mode="bounded_residual",
        context_weight=0.03,
    )
    return cache, model, text, samples, settings


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
    cfg["checkpoint"] = str(tmp_path / "checkpoint.pt")
    model = RCRModel.from_config(cfg["model"], input_dim=8, person_input_dim=8)
    text = TextEncoder(TextBackbone(), dim=8)
    checkpoint = _make_checkpoint(
        epoch=2,
        model=model,
        text_encoder=text,
        tokenizer=Tokenizer(),
        cfg=cfg,
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
