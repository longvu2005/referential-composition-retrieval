"""GT alignment for training or diagnostics, separate from visual caches."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.common.io import atomic_write_text
from rcr.common.vision import match_heads_to_persons


def align_identities(item, heads, *, image_path=None, scene_size=None):
    """Label existing detections; never create, remove or move a person box."""
    count = len(item["boxes_scene"])
    labels = [None] * count
    if not count or not heads:
        return labels
    persons = item.get("boxes_pixel")
    if persons is None:
        # Old caches may store only letterboxed coordinates. New scene caches
        # store pixel boxes and need no image access for supervision.
        if image_path is None or scene_size is None:
            raise ValueError(
                "legacy box alignment needs raw images and scene_size; "
                "cached identity_ids are never used"
            )
        from rcr.proposed.cache.fafa import boxes_to_pixels

        with Image.open(image_path) as image:
            h, w = scene_size
            persons = boxes_to_pixels(item["boxes_scene"], image.size, (w, h))
    boxes = persons.new_tensor(
        [[r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]] for r in heads]
    )
    for person, head in enumerate(match_heads_to_persons(boxes, persons).tolist()):
        if head >= 0:
            labels[person] = str(heads[head]["identity_id"])
    return labels


class TrainSupervision:
    """Train-only alignment, optionally reused across runs with identical inputs."""

    def __init__(self, data, cache, *, scene_size=None, cache_file=None):
        self.cache = cache
        allowed = split_image_ids(data, "train")
        self.heads = {i: data.gt_head_boxes_by_image.get(i, []) for i in allowed}
        self.image_paths = {i: Path(data.image_path(i)) for i in allowed}
        self.scene_size = scene_size
        self._ids = {}
        self._saved_count = 0
        self._prepared = False
        self.cache_file = Path(cache_file) if cache_file is not None else None
        if self.cache_file is None:
            return
        if not cache.cache_id:
            raise ValueError("persistent supervision requires a feature cache ID")
        # Feature cache IDs bind immutable crops. Never import legacy shard labels.
        signature = {
            "version": "train-head-alignment-v1",
            "cache_id": cache.cache_id,
            "source_cache_id": getattr(
                getattr(cache, "_scene_cache", None), "cache_id", None
            ),
            "scene_size": scene_size,
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
                image_path=self.image_paths[image_id],
                scene_size=self.scene_size,
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
    from rcr.proposed.cache.build import cache_paths, check_cache_config
    from rcr.proposed.cache.store import GalleryCache
    from rcr.proposed.identity import identity_sampling_settings

    cache_paths(cfg)
    data_cfg = cfg["data"]
    cache_file = cfg.get("cache", {}).get("supervision_cache")
    if cache_file is None:
        raise ValueError("prepare-supervision requires cache.supervision_cache")
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    cache = GalleryCache(
        data_cfg["fafa_image_cache"], scene_root=data_cfg.get("scene_cache")
    )
    check_cache_config(cache, cfg)
    cache.validate_gallery(data.gallery_ids)
    supervision = TrainSupervision(
        data,
        cache,
        scene_size=cfg.get("image_encoder", {}).get("scene_size"),
        cache_file=cache_file,
    )
    settings = identity_sampling_settings(cfg["train"].get("identity_sampling", {}))
    supervision.prepare(
        split_samples(data, "train"),
        all_train=settings["max_crops"] > 0
        and cfg["loss"].get("identity_weight", 0.1) != 0,
    )
    return supervision.cache_file
