"""One CPU epoch through the real train and validation loops."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

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

    def resize_token_embeddings(self, _size):
        pass

    def get_input_embeddings(self):
        return self.embedding

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


def test_one_epoch_writes_validation_selected_checkpoint(tmp_path, monkeypatch):
    samples = [_sample("train", "q", "a"), _sample("val", "vq", "va")]
    cache = TinyCache()
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
        "fafa_image_cache": "unused-fafa-images",
        "fafa_query_cache": "unused-fafa-queries",
    }
    cfg["cache"]["supervision_cache"] = str(tmp_path / "supervision.json")
    cfg["runtime"]["device"] = "cpu"
    cfg["model"].update(dim=6, identity_dim=6, state_dim=4, num_heads=2, geo_dim=4)
    cfg["train"].update(epochs=1, batch_size=1, candidates=2, amp=False)
    cfg["evaluation"]["every_epochs"] = 1
    cfg["retrieval"].update(
        top_m=2, fine_batch_size=1, identity_batch_size=2, coarse_batch_size=2
    )
    cfg["candidate_ks"] = [1, 2]
    cfg["output"]["dir"] = str(tmp_path / "run")

    checkpoint = training.train(cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint.name == "best.pt"
    assert saved["best_epoch"] == 1
    assert saved["config"]["checkpoint"] == str(checkpoint)
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
    assert wandb_run.logs[-1]["epoch"] == 1
    assert "val/checkpoint_score" in wandb_run.logs[-1]
    assert wandb_run.summary["best_epoch"] == 1
    assert wandb_run.finished


def test_disabled_wandb_needs_no_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    cfg = {"wandb": {"enabled": False}}
    assert training._wandb_run(cfg, tmp_path, None, 1, 1) is None


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
