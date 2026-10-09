"""Add frozen CLIP features to compatible DINO/FAFA caches without rebuilding them."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.common.io import sha256_file
from rcr.common.runtime import resolve_device
from rcr.common.storage import atomic_torch_save as atomic_save
from rcr.common.storage import (
    format_bytes,
    remove_cache_temporaries,
    require_free_space,
)
from rcr.common.vision import match_heads_to_persons
from rcr.proposed.cache.dino import boxes_to_scene
from rcr.proposed.nn.encoders import CLIPFeatures, QueryTextCache, parse_subjects

CACHE_VERSION = "clip-person-text-v4"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dataset_signature(data):
    # Case labels and GT do not affect inference features.
    return fingerprint([(i, data.images_by_id[i]["path"]) for i in data.gallery_ids])


def completeness_signature(data):
    return fingerprint(
        [
            (i, bool(data.images_by_id[i].get("identities_complete", False)))
            for i in data.gallery_ids
        ]
    )


def text_strings(samples):
    texts = set()
    for sample in samples:
        _, selections, _ = parse_subjects(sample["final_desc"], sample["final_change"])
        texts.update(selections)
        texts.add(sample["final_change"])
    return sorted(texts)


def source_signature(source):
    # Hash indexes once per opened immutable source. Per-image boxes are also
    # checked when loading semantic shards; trainable gallery outputs stay live.
    if not hasattr(source, "_v2_signature"):
        scene = source._scene_cache or source
        source._v2_signature = {
            "cache_id": source.cache_id,
            "scene_cache_id": scene.cache_id,
            "index_sha256": sha256_file(source.root / "index.pt"),
            "scene_index_sha256": sha256_file(scene.root / "index.pt"),
            "encoders": source.encoder_metadata,
            "patch_hw": list(source.patch_hw),
        }
    return source._v2_signature


def check_clip_disk_budget(root, source, config, tokenizer, texts, dtype, cache_id):
    """Estimate remaining raw features before starting gallery inference."""
    vision = config.vision_config
    token_count = (vision.image_size // vision.patch_size) ** 2 + 1
    element_size = torch.empty((), dtype=dtype).element_size()
    person_bytes = (
        token_count * vision.hidden_size + config.projection_dim
    ) * element_size + 4 * 4
    vision_bytes = 0
    for i, count in enumerate(source.mask.sum(1).tolist()):
        path = root / "features" / f"{i}.pt"
        if path.is_file():
            # Read only metadata through mmap; do not scan the tensor contents.
            saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
            if (
                saved.get("cache_id") == cache_id
                and saved.get("image_id") == source.image_ids[i]
                and saved["clip_tokens"].shape
                == (count, token_count, vision.hidden_size)
                and saved["clip_pooled"].shape
                == (count, config.projection_dim)
                and saved["clip_tokens"].dtype == dtype
                and saved["clip_pooled"].dtype == dtype
            ):
                continue
        vision_bytes += count * person_bytes + 8 * 1024
    text_bytes = 64 * 1024
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        text_bytes += len(ids) * (
            config.text_config.hidden_size * element_size + 192
        ) + len(text.encode("utf-8")) + 2048
    # Periodic text saves keep the previous sidecar while writing its replacement.
    sidecar_bytes = len(source.image_ids) * 2048 + 1024**2
    additional_bytes = vision_bytes + 2 * text_bytes + sidecar_bytes
    print(
        f"CLIP disk estimate: remaining vision={format_bytes(vision_bytes)}, "
        f"text write peak={format_bytes(2 * text_bytes)}, "
        f"other sidecars={format_bytes(sidecar_bytes)}",
        flush=True,
    )
    require_free_space(root, additional_bytes, context="remaining CLIP cache")


def align_identities(boxes_scene, heads, image_size, scene_size):
    """Conservative detector/GT alignment for supervision only.

    Detections containing multiple annotated head centers remain unknown. GT
    boxes never change detector proposals, masks, crops, geometry or features.
    """
    if not heads:
        return [None] * len(boxes_scene)
    boxes = torch.tensor(
        [[x["x"], x["y"], x["x"] + x["width"], x["y"] + x["height"]] for x in heads],
        dtype=torch.float32,
    )
    boxes = boxes_to_scene(boxes, image_size, scene_size)
    matches = match_heads_to_persons(boxes, boxes_scene)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (
        (centers[:, None] >= boxes_scene[None, :, :2])
        & (centers[:, None] <= boxes_scene[None, :, 2:])
    ).all(-1)
    return [
        str(heads[j]["identity_id"])
        if j >= 0 and int(inside[:, i].sum()) == 1
        else None
        for i, j in enumerate(matches.tolist())
    ]


def build_clip_cache(cfg, data, source):
    """Resume frozen vision shards; text and annotation sidecars publish atomically."""
    from transformers import (
        CLIPConfig,
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizerFast,
    )

    from rcr.proposed.cache.dino import boxes_to_pixels

    clip_cfg = cfg["clip_encoder"]
    root = Path(cfg["data"]["clip_cache"])
    if root.resolve() in (
        source.root.resolve(),
        Path(cfg["data"]["dino_cache"]).resolve(),
    ):
        raise ValueError("CLIP cache must have its own directory")
    texts = text_strings(data.samples)
    signature = {
        "version": CACHE_VERSION,
        "clip_encoder": clip_cfg,
        "source": source_signature(source),
        "dataset": dataset_signature(data),
        "storage_dtype": cfg["cache"]["storage_dtype"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(exist_ok=True)
    manifest = root / ".build.pt"
    index_path = root / "index.pt"
    old = None
    if index_path.exists() and not (root / ".building").exists():
        old = torch.load(index_path, weights_only=True, map_location="cpu")
        if old["signature"] != signature:
            raise ValueError(
                "CLIP cache source/checkpoint/preprocessing/dataset differs; "
                "choose a new clip_cache"
            )
        for name in ("text", "supervision"):
            path = root / f"{name}.pt"
            if not path.is_file() or sha256_file(path) != old[f"{name}_sha256"]:
                raise ValueError(
                    "CLIP cache checksum mismatch; choose a new clip_cache"
                )
        if (
            old["texts_sha256"] == fingerprint(texts)
            and old["heads_sha256"] == sha256_file(data.final_dir / "head_boxes.jsonl")
            and old["completeness_sha256"] == completeness_signature(data)
        ):
            print(f"Using existing CLIP cache: {root}")
            return
    if manifest.exists():
        saved = torch.load(manifest, weights_only=True, map_location="cpu")
        if saved["signature"] != signature:
            raise ValueError("unfinished CLIP cache settings differ")
        cache_id = saved["cache_id"]
    else:
        cache_id = old["cache_id"] if old else uuid4().hex
        atomic_save({"signature": signature, "cache_id": cache_id}, manifest)
    (root / ".building").write_text(cache_id)
    remove_cache_temporaries(root)
    device = resolve_device(cfg)
    kwargs = {"revision": clip_cfg["revision"]}
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_cfg["model"], **kwargs)
    processor = CLIPImageProcessor.from_pretrained(clip_cfg["model"], **kwargs)
    model_config = CLIPConfig.from_pretrained(clip_cfg["model"], **kwargs)
    dtype = getattr(torch, cfg["cache"]["storage_dtype"])
    check_clip_disk_budget(
        root, source, model_config, tokenizer, texts, dtype, cache_id
    )
    # Transformers 4.57 can download a second, converted checkpoint in a
    # background thread even when the original pinned .bin has already loaded.
    # Disable only this optional task; keep normal format selection/safety checks.
    flag = "DISABLE_SAFETENSORS_CONVERSION"
    previous = os.environ.get(flag)
    os.environ[flag] = "true"
    try:
        backbone = CLIPModel.from_pretrained(
            clip_cfg["model"], config=model_config, **kwargs
        ).to(device)
    finally:
        if previous is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous
    encoder = CLIPFeatures(backbone, tokenizer)
    resolved_revision = getattr(backbone.config, "_commit_hash", None)
    batch_size = clip_cfg["batch_size"]
    height, width = cfg["image_encoder"]["scene_size"]
    labels = {}
    for i, image_id in enumerate(tqdm(source.image_ids, desc="cache CLIP persons")):
        item = source._get_item(i)
        boxes = item["boxes_scene"]
        path = root / "features" / f"{i}.pt"
        with Image.open(data.image_path(image_id)) as source_image:
            image = source_image.convert("RGB")
        labels[image_id] = {
            "identity_ids": align_identities(
                boxes,
                data.gt_head_boxes_by_image.get(image_id, []),
                image.size,
                (width, height),
            ),
            "gt_ids": sorted(
                {
                    str(x["identity_id"])
                    for x in data.gt_head_boxes_by_image.get(image_id, [])
                }
            ),
            "complete": bool(
                data.images_by_id[image_id].get("identities_complete", False)
            ),
        }
        if path.exists():
            cached = torch.load(path, weights_only=True, map_location="cpu")
            if cached.get("cache_id") == cache_id and torch.equal(
                cached["boxes_scene"], boxes
            ):
                continue
        pixels = item.get("boxes_pixel")
        if pixels is None:
            pixels = boxes_to_pixels(boxes, image.size, (width, height))
        pooled, tokens = [], []
        for start in range(0, len(boxes), batch_size):
            crops = [
                image.crop(tuple(box.tolist()))
                for box in pixels[start : start + batch_size]
            ]
            inputs = processor(images=crops, return_tensors="pt")["pixel_values"].to(
                device
            )
            v, v_tokens = encoder.image(inputs)
            pooled.append(v.to(device="cpu", dtype=dtype).clone())
            tokens.append(v_tokens.to(device="cpu", dtype=dtype).clone())
        vision = backbone.config.vision_config
        token_count = (vision.image_size // vision.patch_size) ** 2 + 1
        atomic_save(
            {
                "cache_id": cache_id,
                "image_id": image_id,
                "boxes_scene": boxes,
                "clip_pooled": torch.cat(pooled)
                if pooled
                else torch.empty(0, backbone.config.projection_dim, dtype=dtype),
                "clip_tokens": torch.cat(tokens)
                if tokens
                else torch.empty(0, token_count, vision.hidden_size, dtype=dtype),
            },
            path,
        )
    text_path = root / "text.pt"
    frozen_text = (
        torch.load(text_path, weights_only=True, map_location="cpu")
        if text_path.exists()
        else {}
    )
    for i, text in enumerate(tqdm(texts, desc="cache CLIP text windows")):
        if text not in frozen_text:
            item = encoder.text(text)
            item["tokens"] = item["tokens"].to(dtype)
            frozen_text[text] = item
        if (i + 1) % 256 == 0:
            atomic_save(frozen_text, text_path)
    atomic_save(frozen_text, text_path)
    atomic_save(labels, root / "supervision.pt")
    tokenizer.save_pretrained(root / "tokenizer")
    atomic_save(
        {
            "signature": signature,
            "cache_id": cache_id,
            "image_ids": source.image_ids,
            "dimensions": {
                "clip_dim": backbone.config.projection_dim,
                "token_dim": backbone.config.vision_config.hidden_size,
                "text_dim": backbone.config.text_config.hidden_size,
                "person_dim": source.person_dim,
                "scene_dim": source.scene_dim,
            },
            "resolved_revision": resolved_revision,
            "processor": processor.to_dict(),
            "texts_sha256": fingerprint(texts),
            "heads_sha256": sha256_file(data.final_dir / "head_boxes.jsonl"),
            "completeness_sha256": completeness_signature(data),
            "supervision_sha256": sha256_file(root / "supervision.pt"),
            "text_sha256": sha256_file(text_path),
        },
        index_path,
    )
    (root / ".building").unlink()
    manifest.unlink()


class FeatureCache:
    """New model view: label-free tensors and a separate training supervision store."""

    def __init__(self, cfg, data):
        from rcr.proposed.cache.build import check_cache_config
        from rcr.proposed.cache.store import GalleryCache

        if cfg["person_encoder"]["backend"] != "fafa":
            raise ValueError("the proposed model requires person_encoder.backend=fafa")
        self.source = GalleryCache(
            cfg["data"]["cache"],
            scene_root=cfg["data"]["dino_cache"],
            lru_mib=cfg["cache"].get("lru_mib", 0),
        )
        check_cache_config(self.source, cfg)
        self.source.validate_gallery(data.gallery_ids)
        self.root = Path(cfg["data"]["clip_cache"])
        if (self.root / ".building").exists():
            raise ValueError("CLIP cache incomplete; resume build-cache")
        self.index = torch.load(
            self.root / "index.pt", weights_only=True, map_location="cpu"
        )
        signature = self.index["signature"]
        if (
            signature["version"] != CACHE_VERSION
            or signature["source"] != source_signature(self.source)
            or signature["dataset"] != dataset_signature(data)
            or signature["clip_encoder"] != cfg["clip_encoder"]
            or self.index["image_ids"] != data.gallery_ids
        ):
            raise ValueError("incompatible CLIP cache provenance")
        if self.index["texts_sha256"] != fingerprint(text_strings(data.samples)):
            raise ValueError("text changed; rerun build-cache --cache-stage clip")
        if sha256_file(self.root / "text.pt") != self.index["text_sha256"]:
            raise ValueError("CLIP text cache checksum mismatch")
        self.text = QueryTextCache(
            torch.load(self.root / "text.pt", weights_only=True, map_location="cpu")
        )
        self.data = data
        self.image_ids, self.by_id = self.source.image_ids, self.source.by_id
        self.persons, self.mask = self.source.persons, self.source.mask
        self.patch_hw = self.source.patch_hw
        self.dimensions = self.index["dimensions"]
        self.cache_id = self.index["cache_id"]
        self.encoder_metadata = {
            **self.source.encoder_metadata,
            "clip": signature["clip_encoder"],
            "clip_revision": self.index["resolved_revision"],
        }

    def supervision(self):
        if (
            self.index["heads_sha256"]
            != sha256_file(self.data.final_dir / "head_boxes.jsonl")
            or self.index["supervision_sha256"]
            != sha256_file(self.root / "supervision.pt")
            or self.index["completeness_sha256"] != completeness_signature(self.data)
        ):
            raise ValueError(
                "supervision changed; rerun build-cache --cache-stage clip"
            )
        return torch.load(
            self.root / "supervision.pt", weights_only=True, map_location="cpu"
        )

    def load(self, indices):
        items = []
        for i in indices.tolist():
            raw = self.source._get_item(i)
            semantic = torch.load(
                self.root / "features" / f"{i}.pt",
                weights_only=True,
                map_location="cpu",
            )
            if (
                semantic["cache_id"] != self.cache_id
                or semantic["image_id"] != self.image_ids[i]
                or not torch.equal(semantic["boxes_scene"], raw["boxes_scene"])
            ):
                raise ValueError("CLIP/source person boxes or order differ")
            items.append({**raw, **semantic})
        k = max(len(x["persons"]) for x in items)
        n = max(x["clip_tokens"].shape[1] for x in items)
        out = {"scene": torch.stack([x["scene"] for x in items]).float()}
        for key in ("persons", "clip_pooled", "boxes_scene"):
            value = torch.zeros(len(items), k, items[0][key].shape[-1])
            for j, item in enumerate(items):
                value[j, : len(item[key])] = item[key]
            out["boxes" if key == "boxes_scene" else key] = value
        out["clip_tokens"] = torch.zeros(len(items), k, n, self.dimensions["token_dim"])
        out["person_mask"] = torch.zeros(len(items), k, dtype=torch.bool)
        out["clip_token_mask"] = torch.zeros(len(items), k, n, dtype=torch.bool)
        for j, item in enumerate(items):
            count, tokens = item["clip_tokens"].shape[:2]
            out["clip_tokens"][j, :count, :tokens] = item["clip_tokens"]
            out["person_mask"][j, :count] = True
            out["clip_token_mask"][j, :count, :tokens] = True
        return out
