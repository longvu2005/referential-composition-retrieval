"""CPU contracts for training, context warmup and validation selection."""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from rcr.proposed import train as training
from rcr.proposed.nn.encoders import (
    TextEncoder,
    load_text_encoder_state,
    text_encoder_state,
)


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 6

    def __len__(self):
        return 7

    def __call__(self, texts, padding=True, return_tensors=None, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        rows, offsets = [], []
        for text in texts:
            matches = list(re.finditer(r"\w+|[^\w\s]", text))
            rows.append(
                [
                    5,
                    *[
                        {"subject": 1, "1": 2, "2": 3}.get(m.group().lower(), 4)
                        for m in matches
                    ],
                    6,
                ]
            )
            offsets.append([(0, 0), *[m.span() for m in matches], (0, 0)])
        if kwargs.get("return_offsets_mapping"):
            return {"input_ids": rows, "offset_mapping": offsets}
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for n, row in enumerate(rows):
            ids[n, : len(row)] = torch.tensor(row)
            mask[n, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def save_pretrained(self, path):
        path.mkdir(parents=True)


class TextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8, projection_dim=8, max_position_embeddings=77, eos_token_id=2
        )
        self.embedding = nn.Embedding.from_pretrained(
            torch.randn(7, 8, generator=torch.Generator().manual_seed(41))
        )
        self.text_projection = nn.Identity()

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class CountingTextBackbone(TextBackbone):
    def __init__(self, vocabulary=3):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 8)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return super().forward(input_ids, attention_mask)


class TinyCache:
    def __init__(self):
        self.image_ids = ["q", "a", "n", "vq", "va", "vn"]
        self.by_id = {image_id: n for n, image_id in enumerate(self.image_ids)}
        generator = torch.Generator().manual_seed(7)
        self.scenes = torch.randn(6, 4, 8, generator=generator)
        self.persons = torch.randn(6, 1, 8, generator=generator)
        self.mask = torch.ones(6, 1, dtype=torch.bool)
        self.boxes = torch.tensor([0.0, 0.0, 1.0, 1.0]).expand(6, 1, 4)
        self.patch_hw = (2, 2)
        self.scene_dim = self.person_dim = self.global_dim = 8
        self.cache_id = "tiny"
        self.scene_cache_id = "tiny-scene"
        self.detection_cache_id = "tiny-detections"
        self.encoder_metadata = {
            "image_encoder": None,
            "person_encoder": None,
            "detector": None,
        }
        self.native = SimpleNamespace(metadata={"version": "tiny"})

    def validate_gallery(self, _ids):
        pass

    def validate_samples(self, _samples):
        pass

    def load(self, indices):
        return (
            self.scenes[indices],
            self.persons[indices],
            self.boxes[indices],
            self.mask[indices],
        )

    def load_groups(self, *groups):
        return tuple(self.load(indices) for indices in groups)

    def load_persons(self, indices):
        return self.persons[indices], self.mask[indices]

    def person_row(self, index):
        return self.persons[index, self.mask[index]]

    def load_global_features(self, indices):
        return self.scenes[indices].mean(1)

    def load_geometry(self, index):
        return {
            "boxes_pixel": self.boxes[index],
            "boxes_scene": self.boxes[index],
        }

    def native_queries(self, samples, max_people):
        return torch.ones(len(samples), 1, max_people, 5)

    def native_targets(self, indices, max_people):
        return torch.ones(len(indices), max_people, 3, 5)


class WandbRun:
    def __init__(self):
        self.metrics = []
        self.logs = []
        self.summary = {}
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, values):
        self.logs.append(values)

    def finish(self):
        self.finished = True


def _sample(sample_id, query, target):
    return {
        "sample_id": sample_id,
        "case_type": "INDIVIDUAL",
        "query_image_id": query,
        "target_image_id": target,
        "positive_image_ids": [target],
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
        "final_desc": "Identify Subject 1 as the person",
        "final_change": "then retrieve images where Subject 1 is standing",
        "final_instruction": "person standing",
    }


@pytest.mark.parametrize("existing", ["config.yaml", "best.pt", "last.pt"])
def test_training_rejects_existing_run_before_loading_data(
    tmp_path, monkeypatch, existing
):
    path = tmp_path / existing
    path.write_bytes(b"previous run")
    monkeypatch.setattr(
        training,
        "load_rcr_data",
        lambda *a: pytest.fail("must reject before data loading"),
    )
    cfg = {"train": {"epochs": 1, "batch_size": 1}, "output": {"dir": str(tmp_path)}}
    with pytest.raises(FileExistsError, match="new output.dir"):
        training.train(cfg)
    assert path.read_bytes() == b"previous run"


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
        "fafa_image_cache": "unused-person-images",
        "fafa_query_cache": "unused-person-queries",
    }
    cfg["cache"]["supervision_cache"] = str(tmp_path / "supervision.json")
    cfg["runtime"]["device"] = "cpu"
    cfg["model"].update(dim=6, identity_dim=6, state_dim=4, num_heads=2, geo_dim=4)
    cfg["train"].update(
        epochs=1, context_warmup_epochs=0, batch_size=1, candidates=2, amp=False
    )
    # Keep lifecycle contracts independent of the current experiment's LR ablation.
    cfg["optimizer"]["schedule"] = "constant"
    cfg["evaluation"]["every_epochs"] = 1
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=2
    )
    cfg["candidate_ks"] = [1, 2]
    cfg["output"]["dir"] = str(tmp_path / "run")
    return cfg, cache, wandb_run, wandb_init


@pytest.mark.parametrize(
    "warmup,epochs,every_epochs,context_weight,expected_weights,expected_evaluations",
    [
        (0, 1, 1, 1.0, [1.0], [1]),
        (1, 3, 1, 0.7, [0.0, 0.7, 0.7], [2, 3]),
        (2, 3, 10, 1.0, [0.0, 0.0, 1.0], [3]),
        (3, 4, 2, 1.0, [0.0, 0.0, 0.0, 1.0], [4]),
        (1, 2, 1, 0.0, [0.0, 0.0], [1, 2]),
        (None, 1, 1, 1.0, [1.0], [1]),
    ],
)
def test_training_warmup_and_validation_selected_checkpoint(
    tiny_training_run,
    monkeypatch,
    warmup,
    epochs,
    every_epochs,
    context_weight,
    expected_weights,
    expected_evaluations,
):
    cfg, cache, wandb_run, wandb_init = tiny_training_run
    cfg["train"]["epochs"] = epochs
    if warmup is None:
        del cfg["train"]["context_warmup_epochs"]
    else:
        cfg["train"]["context_warmup_epochs"] = warmup
    cfg["loss"]["context_weight"] = context_weight
    cfg["evaluation"]["every_epochs"] = every_epochs
    cfg["wandb"]["log_every_steps"] = 1
    original = deepcopy(cfg)
    weights, evaluated = [], []
    compute_loss, evaluate = training.compute_loss, training._evaluate

    def observe_loss(*args, **settings):
        weights.append(settings.copy())
        return compute_loss(*args, **settings)

    def observe_evaluation(*args, **kwargs):
        evaluated.append(args[-2])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "compute_loss", observe_loss)
    monkeypatch.setattr(training, "_evaluate", observe_evaluation)

    checkpoint = training.train(cfg)
    assert cfg == original
    assert [settings["context_weight"] for settings in weights] == expected_weights
    assert all(
        {k: v for k, v in settings.items() if k != "context_weight"}
        == {k: v for k, v in cfg["loss"].items() if k != "context_weight"}
        for settings in weights
    )
    assert evaluated == expected_evaluations
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert [row["context_weight"] for row in history] == expected_weights
    assert [row["phase"] for row in history] == [
        "warmup" if context_weight and weight == 0 else "joint"
        for weight in expected_weights
    ]
    assert all(
        row["train_context"] == row["train_context_active_fraction"] == 0
        for row in history
        if row["phase"] == "warmup"
    )
    assert [row["epoch"] for row in history if "val_full_map" in row] == evaluated
    best_epoch = max(
        (row for row in history if "val_checkpoint_score" in row),
        key=lambda row: row["val_checkpoint_score"],
    )["epoch"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint.name == "best.pt"
    assert saved["best_epoch"] == saved["epoch"] == best_epoch
    assert saved["config"]["loss"]["context_weight"] == context_weight
    last = torch.load(
        checkpoint.parent / "last.pt", map_location="cpu", weights_only=True
    )
    assert last["epoch"] == epochs
    assert last["best_epoch"] == best_epoch
    assert saved["config"]["checkpoint"] == str(checkpoint)
    assert saved["scene_cache_id"] == cache.scene_cache_id
    assert saved["detection_cache_id"] == cache.detection_cache_id
    assert saved["state_supervised_pairs"] > 0
    assert saved["config"]["retrieval"]["fine_score_mode"] == "context_zscore"
    assert (
        saved["config"]["retrieval"]["context_weight"]
        == cfg["retrieval"]["context_weight"]
    )
    assert "optimizer" not in saved and "scaler" not in saved
    assert set(saved["text_encoder"]) == {"projection", "text_policy"}

    tokenizer = Tokenizer()
    restored = TextEncoder(TextBackbone(), dim=6)
    load_text_encoder_state(restored, tokenizer, saved["text_encoder"])
    torch.testing.assert_close(
        restored.proj.weight, saved["text_encoder"]["projection"]["weight"]
    )

    assert wandb_init["project"] == "referential-composition-retrieval"
    assert "data" not in wandb_init["config"]
    assert wandb_init["config"]["cache"]["id"] == "tiny"
    assert len(wandb_run.metrics) == 5
    assert wandb_run.logs[0]["global_step"] == 1
    assert "train/loss" in wandb_run.logs[0]
    assert wandb_run.logs[-1]["epoch"] == epochs
    assert "val/checkpoint_score" in wandb_run.logs[-1]
    step_logs = [row for row in wandb_run.logs if "global_step" in row]
    assert [row["train/context_weight"] for row in step_logs] == expected_weights
    assert [
        row["epoch/context_weight"] for row in wandb_run.logs if "epoch" in row
    ] == expected_weights
    assert wandb_run.summary["best_epoch"] == best_epoch
    assert wandb_run.finished


def test_warmup_keeps_optimizer_and_independent_hard_mining_schedule(
    tiny_training_run, monkeypatch
):
    cfg, _, _, _ = tiny_training_run
    cfg["train"].update(epochs=4, context_warmup_epochs=2)
    cfg["train"]["sampling"].update(
        identity_fraction=0, hard_fraction=1, warmup_epochs=1, refresh_every_epochs=2
    )
    cfg["evaluation"]["enabled"] = False
    cfg["wandb"]["enabled"] = False
    optimizers, mined_after_steps = [], []
    adamw, compute_loss = torch.optim.AdamW, training.compute_loss
    completed_steps = 0

    def create_optimizer(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def observe_loss(model, *args, **kwargs):
        nonlocal completed_steps
        if completed_steps:
            state = optimizers[0].state[model.person_proj[0].weight]
            assert state["step"].item() == completed_steps
        completed_steps += 1
        return compute_loss(model, *args, **kwargs)

    def mine(*args, **kwargs):
        assert kwargs["gallery_ids"] == ["q", "a", "n"]
        mined_after_steps.append(completed_steps)
        return {"train": ["n"]}

    monkeypatch.setattr(torch.optim, "AdamW", create_optimizer)
    monkeypatch.setattr(training, "compute_loss", observe_loss)
    monkeypatch.setattr(training, "mine_hard_negatives", mine)
    checkpoint = training.train(cfg)
    assert checkpoint.name == "last.pt"
    assert not (checkpoint.parent / "best.pt").exists()
    assert len(optimizers) == 1
    assert completed_steps == 4
    assert mined_after_steps == [1, 3]  # Epochs 2 and 4, despite context warmup=2.
    assert [group["lr"] for group in optimizers[0].param_groups] == [
        cfg["optimizer"]["lr"],
        cfg["optimizer"]["text_lr"],
    ]


def test_disabled_wandb_needs_no_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    cfg = {"wandb": {"enabled": False}}
    assert training._wandb_run(cfg, tmp_path, None, 1, 1) is None


def test_cosine_training_logs_rates_used_and_mines_identity_quota(
    tiny_training_run, monkeypatch
):
    cfg, _, wandb_run, _ = tiny_training_run
    cfg["train"].update(epochs=3, candidates=2)
    cfg["optimizer"].update(schedule="cosine", min_lr_ratio=0.1)
    cfg["train"]["sampling"].update(
        identity_fraction=1,
        identity_hard_fraction=1,
        hard_fraction=0,
        warmup_epochs=1,
        refresh_every_epochs=2,
    )
    cfg["evaluation"]["enabled"] = False
    cfg["wandb"]["log_every_steps"] = 1
    seen_rates, mined = [], []
    adamw = torch.optim.AdamW

    def create_optimizer(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        step = optimizer.step

        def observe_step(*args, **kwargs):
            seen_rates.append([group["lr"] for group in optimizer.param_groups])
            return step(*args, **kwargs)

        optimizer.step = observe_step
        return optimizer

    def mine(*args, **kwargs):
        mined.append(len(seen_rates))
        return {"train": ["n"]}

    monkeypatch.setattr(torch.optim, "AdamW", create_optimizer)
    monkeypatch.setattr(training, "mine_hard_negatives", mine)
    checkpoint = training.train(cfg)
    history = [
        json.loads(line)
        for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()
    ]
    assert mined == [1]  # Same-ID mining works with general hard_fraction=0.
    expected = [(1e-4, 1e-5), (5.5e-5, 5.5e-6), (1e-5, 1e-6)]
    for used, row, pair in zip(seen_rates, history, expected, strict=True):
        assert used == pytest.approx(pair)
        assert [row["model_lr"], row["text_lr"]] == pytest.approx(pair)
    assert [r["mining_active"] for r in history] == [False, True, True]
    assert all("train_sampled_identity_hard" in r for r in history)
    step_logs = [row for row in wandb_run.logs if "global_step" in row]
    assert [r["train/model_lr"] for r in step_logs] == pytest.approx(
        [p[0] for p in expected]
    )


def test_text_cache_reuses_right_padded_rows_without_saving_frozen_weights():
    backbone = CountingTextBackbone(vocabulary=10_000)
    encoder = TextEncoder(backbone, dim=8, cache_mib=1)
    encoder(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
    output, mask, pooled = encoder(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )
    assert backbone.calls == 2
    assert len(encoder._hidden) == 2
    assert output.shape == (2, 3, 8)
    assert mask.tolist() == [[True, True, False], [True, True, True]]

    assert pooled.shape == (2, 8)
    state = text_encoder_state(encoder, Tokenizer())
    assert set(state) == {"projection", "text_policy"}
    assert state["text_policy"] == {"context_length": 77, "subjects": "native_spans"}


@pytest.fixture
def objective_batch(monkeypatch):
    from rcr.proposed.batch import build_batch

    torch.manual_seed(23)
    cache = TinyCache()

    def no_fda(*args, **kwargs):
        pytest.fail("training must not load native FAFA query/target tokens or run FDA")

    monkeypatch.setattr(cache, "native_queries", no_fda)
    monkeypatch.setattr(cache, "native_targets", no_fda)
    monkeypatch.setattr("rcr.proposed.nn.model.fafa_binding_scores", no_fda)
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
    )
    assert "fafa_query" not in batch and "fafa_targets" not in batch
    batch = {key: value.detach() for key, value in batch.items()}
    batch["extra_identity_persons"] = torch.randn(2, 8)
    batch["extra_identity_labels"] = torch.tensor([1, 1])
    for name in ("target_scene", "target_persons", "target_global"):
        batch[name] = batch[name].detach().clone().requires_grad_()
    return model, batch


def _only_loss(name):
    settings = training.loss_settings({})
    for key in settings:
        if key.endswith("weight"):
            settings[key] = float(key == f"{name}_weight")
    return settings


def _has_gradient(module):
    return any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
    )


@pytest.mark.parametrize(
    "component", ["grounding", "identity", "coarse", "context", "state"]
)
def test_each_objective_trains_its_intended_modules(objective_batch, component):
    model, batch = objective_batch
    total, parts = training.compute_loss(model, batch, (2, 2), **_only_loss(component))
    assert torch.isfinite(total)
    total.backward()
    assert _has_gradient(model.identity_head) == (component in ("identity", "coarse"))
    if component not in ("identity", "coarse"):
        assert all(p.grad is None for p in model.identity_head.parameters())
    assert _has_gradient(model.reasoner) == (component == "context")
    assert _has_gradient(model.composition) == (component == "context")
    assert _has_gradient(model.state_text_proj) == (component == "state")
    assert _has_gradient(model.state_image_proj) == (component == "state")
    assert model.binding_log_scale.grad is None
    assert not model.binding_log_scale.requires_grad
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if component == "context":
        assert parts["context_active_fraction"].item() == 1
        assert batch["target_persons"].grad[0, 2].count_nonzero() == 0
        assert batch["target_persons"].grad[0, :2].abs().sum() > 0
    if component == "state":
        assert batch["target_global"].grad[0, 2].count_nonzero() == 0
        assert batch["target_global"].grad[0, :2].abs().sum() > 0


def test_context_cannot_reduce_loss_by_changing_frozen_fafa_scale(objective_batch):
    model, batch = objective_batch
    settings = _only_loss("context")
    first, _ = training.compute_loss(model, batch, (2, 2), **settings)
    first.backward()
    gradient = model.reasoner.score[-1].weight.grad.clone()
    model.zero_grad(set_to_none=True)
    model.binding_log_scale.data.fill_(20)
    second, _ = training.compute_loss(model, batch, (2, 2), **settings)
    second.backward()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        gradient, model.reasoner.score[-1].weight.grad, rtol=0, atol=0
    )
    assert model.binding_log_scale.grad is None


def test_no_matched_context_pair_skips_context_without_wrong_identity_fallback(
    objective_batch,
):
    model, batch = objective_batch
    batch["target_mask"][0, 1] = False
    total, parts = training.compute_loss(model, batch, (2, 2), **_only_loss("context"))
    assert parts["context_active_fraction"].item() == parts["context"].item() == 0
    total.backward()
    assert not _has_gradient(model.reasoner)
    # Global state still has an annotated same-identity pair, regardless of detections.
    model.zero_grad(set_to_none=True)
    state, _ = training.compute_loss(model, batch, (2, 2), **_only_loss("state"))
    state.backward()
    assert _has_gradient(model.state_text_proj)


def test_context_requires_same_identity_negatives(objective_batch):
    model, batch = objective_batch
    batch["identity_positive_mask"][0, 1] = False
    total, parts = training.compute_loss(model, batch, (2, 2), **_only_loss("context"))
    assert total.item() == 0
    assert parts["context_active_fraction"].item() == 0
    total.backward()
    assert not _has_gradient(model.reasoner)


def test_coarse_and_context_use_the_correct_positive_sets(objective_batch, monkeypatch):
    model, batch = objective_batch
    coarse_scores = torch.tensor([0.0, -0.1, 0.1], requires_grad=True)
    monkeypatch.setattr(
        "rcr.proposed.losses.identity_scores", lambda *a, **k: coarse_scores
    )
    total, _ = training.compute_loss(model, batch, (2, 2), **_only_loss("coarse"))
    total.backward()
    # A same-ID wrong-state image is an identity positive for coarse retrieval.
    assert coarse_scores.grad[1] < 0
    assert coarse_scores.grad[2] > 0
    context_scores = torch.tensor([0.0, -0.1, 0.1], requires_grad=True)
    monkeypatch.setattr(
        model,
        "score_context",
        lambda *a, **k: (context_scores, SimpleNamespace(coverage=torch.ones(3))),
    )
    total, _ = training.compute_loss(model, batch, (2, 2), **_only_loss("context"))
    total.backward()
    # The same image is a conditional negative; wrong IDs have unknown context.
    assert context_scores.grad[1] > 0
    assert context_scores.grad[2] == 0


def test_disabled_state_cannot_be_used_by_training_validation(tmp_path, monkeypatch):
    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["loss"]["state_weight"] = 0
    monkeypatch.setattr(
        training, "cache_paths", lambda *a: pytest.fail("fail before assets")
    )
    with pytest.raises(
        ValueError, match="state_weight=0 requires retrieval.coarse_beta=0"
    ):
        training.train(cfg)
    assert not Path(cfg["output"]["dir"]).exists()


def test_context_only_training_requires_zero_warmup(tmp_path, monkeypatch):
    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["loss"] = _only_loss("context")
    cfg["retrieval"]["coarse_beta"] = 0
    monkeypatch.setattr(
        training, "cache_paths", lambda *a: pytest.fail("fail before assets")
    )
    with pytest.raises(ValueError, match="warmup requires a non-context loss"):
        training.train(cfg)
    assert not Path(cfg["output"]["dir"]).exists()


def test_warmup_skips_fine_and_preserves_its_parameters_until_joint_training(
    objective_batch, monkeypatch
):
    model, batch = objective_batch
    settings = training.loss_settings({})
    settings["context_weight"] = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    fine = [
        parameter
        for module in (model.composition, model.target_builder, model.reasoner)
        for parameter in module.parameters()
    ] + [model.null_person]
    before = [p.detach().clone() for p in fine]

    def forbidden(*args, **kwargs):
        pytest.fail("warmup must skip composition and fine context scoring")

    with monkeypatch.context() as stage:
        stage.setattr(model.composition, "forward", forbidden)
        stage.setattr(model, "score_context", forbidden)
        value, parts = training.compute_loss(model, batch, (2, 2), **settings)
        value.backward()
        assert parts["context"] == parts["context_active_fraction"] == 0
        assert all(p.grad is None for p in fine)
        assert _has_gradient(model.grounding)
        assert _has_gradient(model.identity_head)
        assert _has_gradient(model.state_text_proj)
        optimizer.step()
        for parameter, original in zip(fine, before, strict=True):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)

    optimizer.zero_grad(set_to_none=True)
    settings["context_weight"] = 1
    value, parts = training.compute_loss(model, batch, (2, 2), **settings)
    value.backward()
    assert parts["context_active_fraction"] == 1
    assert _has_gradient(model.reasoner)
    assert _has_gradient(model.composition)
    assert _has_gradient(model.identity_head)
    optimizer.step()
    assert any(
        not torch.equal(p, original) for p, original in zip(fine, before, strict=True)
    )
    assert model.binding_scale.item() == 1


@pytest.mark.parametrize("query_has_persons", [True, False])
@pytest.mark.parametrize("subjects", [1, 2])
def test_warmup_coarse_query_path_matches_full_encoding(
    objective_batch, monkeypatch, query_has_persons, subjects
):
    model, batch = objective_batch
    for key in (
        "selections",
        "selection_mask",
        "subject_pos",
        "subject_token_mask",
        "grounding_targets",
    ):
        value = batch[key]
        batch[key] = value.repeat(1, subjects, *([1] * (value.ndim - 2)))
    batch["subject_ids"] = torch.arange(1, subjects + 1)[None]
    batch["subject_mask"] = torch.ones(1, subjects, dtype=torch.bool)
    if subjects == 2:
        batch["subject_mask"][0, 1] = False
    batch["query_person_mask"].fill_(query_has_persons)
    if not query_has_persons:
        batch["query_persons"].fill_(float("nan"))
        batch["query_boxes"].fill_(float("nan"))
    settings = training.loss_settings({"context_weight": 0})
    actual, parts = training.compute_loss(model, batch, (2, 2), **settings)
    actual.backward()
    gradients = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)

    def full_encoding(scene, persons, boxes, selections, patch_hw, **kwargs):
        query = model.encode_query(
            scene,
            persons,
            boxes,
            selections,
            batch["change"],
            batch["subject_pos"],
            patch_hw,
            query_person_mask=batch["query_person_mask"],
            subject_mask=batch.get("subject_mask"),
            change_mask=batch.get("change_mask"),
            subject_token_mask=batch.get("subject_token_mask"),
            subject_ids=batch.get("subject_ids"),
            **kwargs,
        )
        return query.logits, query.identity

    monkeypatch.setattr(model, "encode_grounded_identity", full_encoding)
    expected, expected_parts = training.compute_loss(model, batch, (2, 2), **settings)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in parts:
        torch.testing.assert_close(parts[key], expected_parts[key], rtol=0, atol=0)
    expected.backward()
    for name, parameter in model.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)


def test_joint_objective_learns_on_a_fixed_supervised_batch(objective_batch):
    model, batch = objective_batch
    settings = training.loss_settings({})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        value, parts = training.compute_loss(model, batch, (2, 2), **settings)
        assert torch.isfinite(value)
        assert parts["context_active_fraction"].item() == 1
        losses.append(value.item())
        value.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.binding_scale.item() == 1.0
