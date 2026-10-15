"""Build resumable native FAFA caches inside the isolated FAFA environment."""

from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.baselines.fafa import load_fafa
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
from rcr.common.runtime import resolve_device
from rcr.proposed.cache.build import check_cache_config, fafa_spec, load_fafa_config
from rcr.proposed.cache.fafa import boxes_to_pixels
from rcr.proposed.cache.native import (
    NATIVE_VERSION,
    NativeFAFACache,
    query_key,
    subject_conditions,
)
from rcr.proposed.cache.store import GalleryCache


def _save(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _ready(path, cache_id, key, shape, dtype):
    if not path.is_file():
        return False
    item = torch.load(path, map_location="cpu", weights_only=True)
    features = item.get("features")
    return (
        item.get("cache_id") == cache_id
        and item.get("key") == key
        and isinstance(features, torch.Tensor)
        and features.shape == shape
        and features.dtype == dtype
        and bool(torch.isfinite(features).all())
    )


@torch.inference_mode()
def finish_native_cache(cfg):
    gallery = GalleryCache(cfg["data"]["cache"], scene_root=cfg["data"]["dino_cache"])
    check_cache_config(gallery, cfg)
    if gallery.cache_id is None:
        raise ValueError("native FAFA cache requires a versioned person cache")
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    gallery.validate_gallery(data.gallery_ids)
    samples = data.samples
    root = Path(cfg["data"]["native_cache"])
    if root.resolve() in (
        gallery.root.resolve(),
        Path(cfg["data"]["dino_cache"]).resolve(),
    ):
        raise ValueError("native_cache must have a separate directory")
    if (root / "index.pt").is_file() and not (root / ".building").exists():
        NativeFAFACache(root, gallery).validate_samples(samples)
        print(f"Using existing native FAFA cache: {root}", flush=True)
        return

    native = load_fafa_config(cfg)
    device = resolve_device(cfg)
    storage = cfg.get("cache", {}).get("storage_dtype", "float16")
    if storage not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float16 or float32")
    dtype = getattr(torch, storage)
    batch_size = cfg["person_encoder"]["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("person_encoder.batch_size must be positive")
    spec = {
        **fafa_spec(native),
        "feature": NATIVE_VERSION,
        "checkpoint_sha256": sha256_file(native["checkpoint"]["path"]),
        "storage_dtype": storage,
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
    }
    queries = {
        row["sample_id"]: {
            "index": i,
            "key": query_key(row),
            "conditions": subject_conditions(row),
        }
        for i, row in enumerate(samples)
    }
    signature = {
        "version": NATIVE_VERSION,
        "person_cache_id": gallery.cache_id,
        "encoder_metadata": gallery.encoder_metadata,
        "spec": spec,
        "queries": queries,
    }
    for group in ("targets", "queries"):
        (root / group).mkdir(parents=True, exist_ok=True)
    manifest_path = root / ".build.pt"
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature:
            raise ValueError(
                "unfinished native cache settings differ; use a new directory"
            )
    else:
        manifest = {"cache_id": uuid4().hex, "signature": signature}
        _save(manifest_path, manifest)
    cache_id = manifest["cache_id"]
    marker = root / ".building"
    marker.write_text(cache_id)
    model, processors, preprocess, load_info = load_fafa(native, device)
    model.requires_grad_(False).eval()
    print(
        f"Native FAFA checkpoint: {len(load_info.get('missing_keys', []))} missing / "
        f"{len(load_info.get('unexpected_keys', []))} unexpected keys",
        flush=True,
    )
    dim = model.vision_proj.out_features
    tokens = model.query_tokens.shape[1]

    def crop_batches(image_id):
        index = gallery.by_id[image_id]
        # Read only boxes from DINO, not the padded person index.
        item = (gallery._scene_cache or gallery)._load_item(index)
        count = int(gallery.mask[index].sum())
        with Image.open(data.image_path(image_id)) as source:
            image = source.convert("RGB")
            boxes = item.get("boxes_pixel")
            if boxes is None:
                h, w = cfg["image_encoder"]["scene_size"]
                boxes = boxes_to_pixels(item["boxes_scene"], image.size, (w, h))
            if (
                boxes.shape != (count, 4)
                or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError("invalid native FAFA person boxes")
            for start in range(0, count, batch_size):
                pixels = torch.stack(
                    [
                        preprocess(image.crop(tuple(box.tolist())))
                        for box in boxes[start : start + batch_size]
                    ]
                ).to(device)
                yield start, pixels.half() if device.type == "cuda" else pixels.float()

    for i, image_id in enumerate(
        tqdm(gallery.image_ids, desc="cache FAFA target tokens")
    ):
        count = int(gallery.mask[i].sum())
        shape = (count, tokens, dim)
        path = root / "targets" / f"{i}.pt"
        if _ready(path, cache_id, image_id, shape, dtype):
            continue
        features = torch.empty(shape, dtype=dtype)
        if count:
            for start, pixels in crop_batches(image_id):
                value, _ = model.extract_target_features(pixels, mode="mean")
                if value.shape != (len(pixels), tokens, dim):
                    raise ValueError("FAFA target tokens must be [N,L,D_fafa]")
                features[start : start + len(pixels)] = value.float().cpu().to(dtype)
        if not torch.isfinite(features).all():
            raise ValueError("non-finite native FAFA targets")
        _save(path, {"cache_id": cache_id, "key": image_id, "features": features})

    for sample in tqdm(samples, desc="cache FAFA multimodal queries"):
        entry = queries[sample["sample_id"]]
        image_id = sample["query_image_id"]
        count = int(gallery.mask[gallery.by_id[image_id]].sum())
        conditions = entry["conditions"]
        shape = (len(conditions), count, dim)
        path = root / "queries" / f"{entry['index']}.pt"
        # Query embeddings stay FP32, as in the native baseline.
        if _ready(path, cache_id, entry["key"], shape, torch.float32):
            continue
        features = torch.empty(shape)
        if count:
            for start, pixels in crop_batches(image_id):
                for s, condition in enumerate(conditions):
                    value = model.extract_features(
                        {
                            "image": pixels,
                            "text_input": [processors["eval"](condition)] * len(pixels),
                        },
                        mode="multimodal",
                    ).multimodal_embeds.float()
                    if value.ndim == 3 and value.shape[1] == 1:
                        value = value[:, 0]
                    if value.shape != (len(pixels), dim):
                        raise ValueError("FAFA multimodal query must be [N,D_fafa]")
                    features[s, start : start + len(pixels)] = value.cpu()
        if not torch.isfinite(features).all():
            raise ValueError("non-finite native FAFA queries")
        _save(path, {"cache_id": cache_id, "key": entry["key"], "features": features})

    _save(
        root / "index.pt",
        {
            **signature,
            "cache_id": cache_id,
            "image_ids": gallery.image_ids,
            "mask": gallery.mask,
            "dim": dim,
            "tokens": tokens,
        },
    )
    marker.unlink()
    manifest_path.unlink(missing_ok=True)
    print(f"Finished native FAFA cache: {root}", flush=True)
