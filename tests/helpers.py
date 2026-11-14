"""Small cached inputs and frozen text doubles for CPU workflow tests."""

import re
from types import SimpleNamespace

import torch
from torch import nn

from rcr.proposed import ranking
from rcr.proposed import train as training


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


def _rank(tiny_ranker, **overrides):
    cache, model, text, samples, settings = tiny_ranker
    return ranking.retrieve_rankings(
        samples,
        cache,
        Tokenizer(),
        text,
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        **{**settings, **overrides},
    )
