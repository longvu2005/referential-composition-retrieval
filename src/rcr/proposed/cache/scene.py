"""CLIP patch/global extraction, independent of person detection and FAFA."""

import gc
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from rcr.common.runtime import resolve_device
from rcr.common.torch_io import save_torch
from rcr.proposed.nn.encoders import ImageEncoder


class SceneCache:
    """Read native CLIP globals and scene patches without touching person boxes."""

    def __init__(self, root, *, mmap_shards=False):
        self.root = Path(root)
        self.mmap_shards = mmap_shards
        if (self.root / ".building").exists():
            raise RuntimeError("scene cache incomplete; resume --cache-stage scene")
        if not (self.root / "index.pt").is_file():
            raise FileNotFoundError(
                f"missing scene cache at {self.root}; run --cache-stage scene first"
            )
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True, mmap=True
        )
        # Completed CLIP caches from before detection separation remain usable.
        legacy_clip = (
            index.get("format_version") == 2
            and (index.get("person_encoder") or {}).get("backend") == "clip"
        )
        if index.get("layout") != "clip-scene-v1" and not legacy_clip:
            raise ValueError(
                "build a CLIP scene cache; legacy DINO scene features are incompatible"
            )
        self.image_ids = index["image_ids"]
        self.cache_id = index["cache_id"]
        self.scene_dim = index["scene_dim"]
        self.global_dim = index["global_dim"]
        self.patch_hw = tuple(index["patch_hw"])
        self.encoder_metadata = {"image_encoder": index["image_encoder"]}
        self._global_features = index.get("global_features")
        if self._global_features is None or self._global_features.shape != (
            len(self.image_ids),
            self.global_dim,
        ):
            raise ValueError(
                "CLIP scene cache is missing valid native global embeddings"
            )

    def validate_gallery(self, image_ids):
        if self.image_ids != image_ids:
            raise ValueError("scene gallery IDs/order differ from finalized data")

    def load_global_features(self, indices):
        return self._global_features[indices]

    def _load_item(self, index):
        item = torch.load(
            self.root / "features" / f"{index}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=self.mmap_shards,
        )
        if (
            item.get("cache_id") != self.cache_id
            or item.get("image_id") != self.image_ids[index]
        ):
            raise ValueError("scene shard identity differs from index.pt")
        scene = item["scene"]
        if scene.shape != (self.patch_hw[0] * self.patch_hw[1], self.scene_dim):
            raise ValueError("cache scene shape differs from patch_hw/feature dim")
        return {"scene": scene}


@torch.inference_mode()
def build_cache(
    image_ids,
    image_paths,
    root,
    image_encoder,
    scene_processor,
    device,
    storage_dtype="float32",
    encoder_metadata=None,
):
    """Resume CLIP shards and publish native globals; no detector is loaded."""
    if storage_dtype not in ("float32", "float16"):
        raise ValueError("storage_dtype must be float32 or float16")
    dtype = getattr(torch, storage_dtype)
    image_config = (encoder_metadata or {})["image_encoder"]
    root = Path(root)
    if (root / "index.pt").is_file() and not (root / ".building").exists():
        cache = SceneCache(root)
        cache.validate_gallery(image_ids)
        if cache.encoder_metadata["image_encoder"] != image_config:
            raise ValueError(
                "existing CLIP scene image encoder differs; use a new directory"
            )
        print(f"Using existing CLIP scene cache: {root}", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    features = root / "features"
    features.mkdir(exist_ok=True)
    signature = {
        "image_ids": image_ids,
        "image_paths": [str(Path(path).resolve()) for path in image_paths],
        "image_encoder": image_config,
        "storage_dtype": storage_dtype,
        "layout": "clip-scene-v1",
    }
    manifest_path = root / ".build.pt"
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature:
            raise ValueError(
                "unfinished CLIP scene settings differ; use a new directory"
            )
    else:
        manifest = {"signature": signature, "cache_id": uuid4().hex}
        save_torch(manifest, manifest_path)
    cache_id = manifest["cache_id"]
    (root / ".building").write_text(cache_id, encoding="utf-8")
    image_encoder.eval()
    globals_, patch_hw, scene_dim = [], None, None
    for i, (image_id, path) in enumerate(
        tqdm(
            zip(image_ids, image_paths, strict=True),
            total=len(image_ids),
            desc="CLIP scenes",
        )
    ):
        shard = features / f"{i}.pt"
        saved = (
            torch.load(shard, map_location="cpu", weights_only=True)
            if shard.is_file()
            else None
        )
        if (
            saved is None
            or saved.get("cache_id") != cache_id
            or saved.get("image_id") != image_id
        ):
            with Image.open(path) as source:
                pixels = scene_processor(
                    images=source.convert("RGB"), return_tensors="pt"
                )["pixel_values"]
            patches, global_embedding, current_hw = image_encoder(pixels.to(device))
            saved = {
                "cache_id": cache_id,
                "image_id": image_id,
                "scene": patches[0]
                .detach()
                .to(
                    device="cpu",
                    dtype=dtype,
                    copy=True,
                    memory_format=torch.contiguous_format,
                ),
                "global": global_embedding[0]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                    copy=True,
                    memory_format=torch.contiguous_format,
                ),
                "patch_hw": current_hw,
            }
            save_torch(saved, shard)
        if patch_hw is not None and tuple(saved["patch_hw"]) != tuple(patch_hw):
            raise ValueError("resumed CLIP scene patch grid differs")
        patch_hw = saved["patch_hw"]
        scene_dim = saved["scene"].shape[-1]
        globals_.append(saved["global"].float())
    if not globals_:
        raise ValueError("scene gallery is empty")
    save_torch(
        {
            "layout": "clip-scene-v1",
            "format_version": 3,
            "cache_id": cache_id,
            "image_ids": image_ids,
            "scene_dim": scene_dim,
            "global_dim": globals_[0].shape[-1],
            "patch_hw": patch_hw,
            "image_encoder": image_config,
            "storage_dtype": storage_dtype,
            "global_features": torch.stack(globals_),
        },
        root / "index.pt",
    )
    (root / ".building").unlink()
    manifest_path.unlink()


class _LetterboxProcessor:
    """Apply one fixed letterbox size, then pretrained CLIP normalization."""

    def __init__(self, processor, size):
        self.processor = processor
        self.size = (int(size[1]), int(size[0]))

    def __call__(self, images, return_tensors):
        single = not isinstance(images, list)
        rows = [images] if single else images
        rows = [
            ImageOps.pad(
                image, self.size, method=Image.Resampling.BICUBIC, color=(0, 0, 0)
            )
            for image in rows
        ]
        return self.processor(
            images=rows[0] if single else rows,
            do_resize=False,
            do_center_crop=False,
            return_tensors=return_tensors,
        )


def _prepare_scene(cfg, data, root):
    image_cfg = cfg["image_encoder"]
    if (root / "index.pt").is_file() and not (root / ".building").exists():
        cache = SceneCache(root)
        cache.validate_gallery(data.gallery_ids)
        if cache.encoder_metadata["image_encoder"] != image_cfg:
            raise ValueError("cache image encoder/preprocessing differs from config")
        print(f"Using existing CLIP scene cache: {root}", flush=True)
        return
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

    device = resolve_device(cfg)
    processor = AutoImageProcessor.from_pretrained(
        image_cfg["model"], revision=image_cfg["revision"], use_fast=False
    )
    backbone = CLIPVisionModelWithProjection.from_pretrained(
        image_cfg["model"], revision=image_cfg["revision"]
    ).to(device)
    if image_cfg["scene_size"] != [224, 224] or backbone.config.patch_size != 32:
        raise ValueError("CLIP ViT-B/32 requires a 224x224 scene and 32-pixel patches")
    image_encoder = ImageEncoder(backbone).to(device)
    build_cache(
        data.gallery_ids,
        [data.image_path(i) for i in data.gallery_ids],
        root,
        image_encoder,
        _LetterboxProcessor(processor, image_cfg["scene_size"]),
        device,
        cfg.get("cache", {}).get("storage_dtype", "float32"),
        {"image_encoder": image_cfg},
    )
    del image_encoder, backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
