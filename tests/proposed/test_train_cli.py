"""Exercise the real train/retrieve loop with tiny local text/cache fixtures."""

import json
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from rcr.evaluation import runner as evaluate_proposed
from rcr.proposed import retrieve as retrieve_proposed
from rcr.proposed import train as train_proposed
from scripts import run as cli
from tests.proposed.test_retrieval import _Cache, _sample, _Tokenizer


class Tokenizer(_Tokenizer):
    def __len__(self):
        return 3

    def add_special_tokens(self, _tokens):
        pass

    def save_pretrained(self, path):
        path.mkdir()


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(3, 8)

    def resize_token_embeddings(self, _size):
        pass

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("enabled,train_queries", [(True, 1), (True, 0), (False, 0)])
def test_train_checkpoints_metrics_and_retrieve(
    monkeypatch, tmp_path, enabled, train_queries, prefetch
):
    samples = [
        _sample("train1"),
        _sample("train2"),
        {
            **_sample("val1"),
            "query_image_id": "val_q",
            "target_image_id": "val_a",
            "positive_image_ids": ["val_a", "left"],
        },
    ]
    cache = _Cache(["q", "val_n", "a", "val_q", "left", "b", "val_a"])
    image_splits = ["train", "val", "train", "val", "leftover", "train", "val"]
    data = SimpleNamespace(
        samples=samples,
        samples_by_id={row["sample_id"]: row for row in samples},
        splits={"train": ["train1", "train2"], "val": ["val1"]},
        gallery_ids=cache.image_ids,
        images_by_id={
            image_id: {"path": f"{split}/{image_id}.jpg"}
            for image_id, split in zip(cache.image_ids, image_splits, strict=True)
        },
        gt_head_boxes_by_image={
            "q": [{"identity_id": "p1"}],
            "a": [{"identity_id": "p1"}],
            "b": [{"identity_id": "p1"}],
            "val_q": [{"identity_id": "p1"}],
            "val_a": [{"identity_id": "p1"}],
        },
    )
    cache.cache_id = "tiny-cache"
    cache.validate_gallery = lambda ids: None
    for module in (train_proposed, retrieve_proposed, evaluate_proposed):
        monkeypatch.setattr(module, "load_rcr_data", lambda *args: data)
    for module in (train_proposed, retrieve_proposed):
        monkeypatch.setattr(module, "GalleryCache", lambda path, **kwargs: cache)
    real_retrieve = train_proposed.retrieve_rankings
    evaluation_calls = []
    real_sample = train_proposed.sample_candidates
    real_evaluate = train_proposed.evaluate_retrieval_output
    val_calls = []
    main_thread = threading.get_ident()

    def sample_train_gallery(samples, gallery_ids, *args, **kwargs):
        assert threading.get_ident() == main_thread
        assert gallery_ids == ["q", "a", "b"]
        return real_sample(samples, gallery_ids, *args, **kwargs)

    monkeypatch.setattr(train_proposed, "sample_candidates", sample_train_gallery)

    def evaluate_with_controlled_validation(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        if kwargs["split"] == "val":
            # A lower overall score can win when the minority case improves.
            individual, group = (0.9, 0.1) if not val_calls else (0.78, 0.78)
            val_calls.append(True)
            result["overall"]["full_map"] = 0.9 * individual + 0.1 * group
            result["by_case"] = {
                "INDIVIDUAL": {"num_queries": 90, "full_map": individual},
                "GROUP": {"num_queries": 10, "full_map": group},
            }
        return result

    monkeypatch.setattr(
        train_proposed, "evaluate_retrieval_output", evaluate_with_controlled_validation
    )

    def retrieve_after_training(
        samples, cache, tokenizer, text_encoder, model, *args, **kwargs
    ):
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(parameter.grad is None for parameter in text_encoder.parameters())
        assert not text_encoder.backbone.training
        assert not any(p.requires_grad for p in text_encoder.backbone.parameters())
        expected_gallery = (
            ["val_n", "val_q", "val_a"]
            if samples[0]["sample_id"] == "val1"
            else ["q", "a", "b"]
        )
        assert kwargs["gallery_ids"] == expected_gallery
        assert all("left" not in row["positive_image_ids"] for row in samples)
        evaluation_calls.append(expected_gallery)
        return real_retrieve(
            samples, cache, tokenizer, text_encoder, model, *args, **kwargs
        )

    monkeypatch.setattr(train_proposed, "retrieve_rankings", retrieve_after_training)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args: Tokenizer()),
            AutoModel=SimpleNamespace(from_pretrained=lambda *args: Backbone()),
        ),
    )
    logs, finished = [], []
    run = SimpleNamespace(
        log=logs.append, summary={}, finish=lambda: finished.append(True)
    )
    monkeypatch.setattr(train_proposed, "_wandb_run", lambda *args: run)

    cfg = {
        "method": "proposed",
        "runtime": {"device": "cpu"},
        "retrieval": {
            "top_m": 2,
            "fine_batch_size": 1,
            "identity_batch_size": 2,
            "coarse_batch_size": 2,
        },
        "candidate_ks": [1, 2],
        "data": {"final_dir": "unused", "image_root": "unused", "cache": "unused"},
        "model": {
            "text_model": "tiny",
            "dim": 6,
            "dropout": 0.1,
            "identity_dim": 6,
            "state_dim": 4,
            "coarse_beta": 0.37,
            "num_heads": 2,
            "mlp_ratio": 2,
            "geo_dim": 4,
        },
        "cache": {"prefetch_batches": prefetch, "lru_mib": 1},
        "train": {
            "epochs": 3,
            "batch_size": 2,
            "candidates": 2,
            "seed": 0,
            "amp": True,  # CPU runtime must fall back to FP32.
            "mining_batch_size": 2,
            "sampling": {
                "identity_fraction": 0.0,
                "hard_fraction": 1.0,
                "warmup_epochs": 1,
                "refresh_every_epochs": 2,
                "pool_size": 1,
            },
        },
        "optimizer": {"lr": 1e-4, "text_lr": 1e-5, "weight_decay": 0.01},
        "loss": {
            "grounding_weight": 1.0,
            "identity_weight": 0.1,
            "retrieval_weight": 1.0,
            "identity_temperature": 0.1,
            "state_weight": 0.5,
            "state_temperature": 0.2,
        },
        "evaluation": {
            "enabled": enabled,
            "every_epochs": 2,
            "train_max_queries": train_queries,
            "top_m": 2,
            "fine_batch_size": 1,
            "identity_batch_size": 2,
            "candidate_ks": [1, 2],
        },
        "wandb": {"log_every_steps": 2},
        "output": {"dir": str(tmp_path / "run")},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    cli.main(["train", *sys.argv[1:]])
    assert len(evaluation_calls) == (2 * (1 + bool(train_queries)) if enabled else 0)

    output = tmp_path / "run"
    last = torch.load(output / "last.pt", weights_only=True)
    assert last["epoch"] == 3
    assert last["scaler"] == {}  # Disabled on CPU.
    mined = json.loads((output / "hard_negatives.json").read_text())
    assert mined["model_epoch"] == 1 and mined["split"] == "train"
    assert mined["pools"] == {"train1": ["b"], "train2": ["b"]}
    assert not any(t.name.startswith("rcr-cache") for t in threading.enumerate())
    assert last["cache_id"] == cache.cache_id
    assert last["config"]["model"]["coarse_beta"] == 0.37
    assert last["model"]["state_text_proj.weight"].shape == (4, 6)
    assert last["model"]["visual_proj.weight"].shape == (6, 8)
    assert last["text_encoder"]["proj.weight"].shape == (6, 8)
    assert last["dim"] == 6 and last["input_dim"] == 8
    assert len(last["optimizer"]["param_groups"][1]["params"]) == 2
    assert "train/state_loss" in logs[0]
    assert finished == [True]
    assert [row["global_step"] for row in logs if "global_step" in row] == [1, 2]
    assert not (output / "evaluation" / "epoch_001").exists()
    epoch_logs = [row for row in logs if "epoch" in row]
    assert [row["epoch"] for row in epoch_logs] == [1, 2, 3]
    assert all("epoch/state_loss" in row for row in epoch_logs)
    if enabled:
        assert [row["epoch"] for row in epoch_logs if "val/full_map" in row] == [2, 3]
        values = []
        scores = []
        for epoch in (2, 3):
            folder = output / "evaluation" / f"epoch_{epoch:03d}"
            val = json.loads((folder / "val_metrics.json").read_text())
            assert set(val) == {"overall", "by_case"}
            values.append(val["overall"]["full_map"])
            scores.append(val["overall"]["checkpoint_score"])
            assert (folder / "train_metrics.json").exists() == bool(train_queries)
            assert ("train_eval/full_map" in epoch_logs[epoch - 1]) == bool(
                train_queries
            )
            assert "val/num_queries" not in epoch_logs[epoch - 1]
        best = torch.load(output / "best.pt", weights_only=True)
        assert values[1] < values[0] and scores[1] > scores[0]
        assert best["best_score"] == max(scores)
        assert best["best_full_map"] == values[1]
        assert best["best_macro_full_map"] == pytest.approx(0.78)
        assert best["epoch"] == 3
        assert run.summary["best_epoch"] == best["epoch"]
    else:
        assert not (output / "best.pt").exists()
        assert not (output / "evaluation").exists()
        assert all("val/full_map" not in row for row in epoch_logs)

    # Reload the saved checkpoint through the public retrieval and evaluation CLIs.
    cfg["checkpoint"] = str(output / ("best.pt" if enabled else "last.pt"))
    cfg["split"] = "val"
    config_path.write_text(yaml.safe_dump(cfg))
    cli.main(["retrieve", *sys.argv[1:]])
    saved = torch.load(output / "val/rankings.pt", weights_only=True)
    assert saved["sample_ids"] == ["val1"]
    assert saved["gallery_ids"] == ["val_n", "val_q", "val_a"]
    assert set(saved["rankings"][0].tolist()) == {0, 2}

    cfg["candidate_ks"] = [1, 2]
    config_path.write_text(yaml.safe_dump(cfg))
    cli.main(["evaluate", *sys.argv[1:]])
    metrics = json.loads((output / "val/metrics.json").read_text())
    assert metrics["overall"]["num_queries"] == 1


def test_retrieve_rejects_old_checkpoint_before_loading_backbones(
    monkeypatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "old.pt"
    torch.save({"model": {}, "config": {"model": {}}}, checkpoint_path)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        retrieve_proposed.retrieve(
            {
                "checkpoint": str(checkpoint_path),
                "runtime": {"device": "cpu"},
                "retrieval": {},
            }
        )
