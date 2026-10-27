"""GT alignment for training or diagnostics, separate from visual caches."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.common.detection import DetectionCache
from rcr.common.io import atomic_write_text


def match_heads_to_persons(heads: Tensor, persons: Tensor) -> Tensor:
    """Greedily match heads to containing persons, one-to-one, by geometry."""
    num_heads = heads.shape[0]
    num_persons = persons.shape[0]
    match = torch.full((num_persons,), -1, device=persons.device, dtype=torch.long)
    if num_heads == 0 or num_persons == 0:
        return match

    hx = (heads[:, 0] + heads[:, 2]) / 2
    hy = (heads[:, 1] + heads[:, 3]) / 2
    x1, y1, x2, y2 = persons.unbind(-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) / 2
    inside = (
        (hx[:, None] >= x1)
        & (hx[:, None] <= x2)
        & (hy[:, None] >= y1)
        & (hy[:, None] <= y2)
    )
    dx = (hx[:, None] - cx) / w
    dy = (hy[:, None] - y1) / h
    cost = (dx.square() + dy.square()).masked_fill(~inside, torch.inf)
    used_heads = torch.zeros(num_heads, device=heads.device, dtype=torch.bool)
    for flat in cost.flatten().argsort().tolist():
        head = flat // num_persons
        person = flat % num_persons
        if not torch.isfinite(cost[head, person]):
            break
        if match[person] < 0 and not used_heads[head]:
            match[person] = head
            used_heads[head] = True
    return match


def align_identities(item, heads):
    """Label existing detections; never create, remove or move a person box."""
    persons = item["boxes_pixel"]
    labels = [None] * len(persons)
    if not len(persons) or not heads:
        return labels
    boxes = persons.new_tensor(
        [[r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]] for r in heads]
    )
    for person, head in enumerate(match_heads_to_persons(boxes, persons).tolist()):
        if head >= 0:
            labels[person] = str(heads[head]["identity_id"])
    return labels


class TrainSupervision:
    """Train-only alignment, optionally reused across runs with identical inputs."""

    def __init__(self, data, cache, *, cache_file=None):
        self.cache = cache
        allowed = split_image_ids(data, "train")
        self.heads = {i: data.gt_head_boxes_by_image.get(i, []) for i in allowed}
        self._ids = {}
        self._saved_count = 0
        self._prepared = False
        self.cache_file = Path(cache_file) if cache_file is not None else None
        if self.cache_file is None:
            return
        detection_id = (
            cache.cache_id
            if isinstance(cache, DetectionCache)
            else cache.detection_cache_id
        )
        if not detection_id:
            raise ValueError("persistent supervision requires a detection cache ID")
        signature = {
            "version": "train-head-alignment-v2",
            "detection_cache_id": detection_id,
            "images": [
                [i, data.images_by_id[i]["path"], self.heads[i]] for i in allowed
            ],
        }
        self.key = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.cache_file.is_file():
            stored = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if stored.get("key") == self.key:
                self._ids = stored["identities"]
                if not self._ids.keys() <= self.heads.keys():
                    raise ValueError("supervision cache contains non-train images")
                self._saved_count = len(self._ids)

    def save(self):
        """Publish completed train alignments atomically."""
        if self.cache_file is None or len(self._ids) == self._saved_count:
            return
        atomic_write_text(
            self.cache_file,
            json.dumps(
                {"key": self.key, "identities": self._ids}, separators=(",", ":")
            )
            + "\n",
        )
        self._saved_count = len(self._ids)

    def _load_image_ids(self, image_id):
        if image_id not in self.heads:
            raise ValueError(
                f"train supervision cannot read non-train image {image_id}"
            )
        if image_id not in self._ids:
            if self._prepared:
                raise RuntimeError(f"train labels were not prepared for {image_id}")
            item = self.cache.load_geometry(self.cache.by_id[image_id])
            self._ids[image_id] = align_identities(
                item,
                self.heads[image_id],
            )
        return self._ids[image_id]

    def prepare(self, samples, *, all_train=False):
        """Align required images once; optionally index the full train gallery."""
        required = set()
        for sample in samples:
            required.add(sample["query_image_id"])
            required.update(sample["positive_image_ids"])
        if not required <= self.heads.keys():
            raise ValueError("train supervision cannot prepare non-train images")
        if all_train:
            required = set(self.heads)
        missing = [i for i in self.heads if i in required and i not in self._ids]
        self._prepared = False
        print(
            f"Train supervision: {len(required) - len(missing)}/{len(required)} "
            f"images cached; preparing {len(missing)} before epoch 1",
            flush=True,
        )
        try:
            for image_id in tqdm(missing, desc="prepare train supervision"):
                self._load_image_ids(image_id)
        finally:
            self.save()
        self._prepared = True

    def identity_crops(self):
        if not self._prepared or not self.heads.keys() <= self._ids.keys():
            raise RuntimeError(
                "prepare the full train gallery before identity sampling"
            )
        crops = defaultdict(list)
        for image_id in self.heads:
            for person, identity in enumerate(self._ids[image_id]):
                if identity is not None:
                    crops[str(identity)].append((image_id, person))
        return dict(crops)

    def load_ids(self, image_ids, max_people):
        result = []
        for image_id in image_ids:
            ids = self._load_image_ids(image_id)
            if len(ids) > max_people:
                raise ValueError(
                    "supervision crop order/count differs from visual batch"
                )
            result.append([*ids, *([None] * (max_people - len(ids)))])
        return result


def prepare_train_supervision(cfg):
    """Prepare reusable train labels without loading encoders or query features."""
    from rcr.proposed.cache.build import cache_paths
    from rcr.proposed.identity import identity_sampling_settings

    cache_paths(cfg)
    data_cfg = cfg["data"]
    cache_file = cfg.get("cache", {}).get("supervision_cache")
    if cache_file is None:
        raise ValueError("prepare-supervision requires cache.supervision_cache")
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    cache = DetectionCache(data_cfg["detection_cache"])
    cache.check_config(cfg["detector"])
    cache.validate_gallery(data.gallery_ids)
    supervision = TrainSupervision(
        data,
        cache,
        cache_file=cache_file,
    )
    settings = identity_sampling_settings(cfg["train"].get("identity_sampling", {}))
    supervision.prepare(
        split_samples(data, "train"),
        all_train=settings["max_crops"] > 0
        and cfg["loss"].get("identity_weight", 0.1) != 0,
    )
    return supervision.cache_file
