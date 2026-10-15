"""One frozen FAFA build: pooled persons, target tokens and multimodal queries."""

from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.baselines.fafa import load_fafa
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
from rcr.common.runtime import resolve_device
from rcr.common.torch_io import require_disk_space, save_torch
from rcr.proposed.cache.build import check_cache_config, fafa_spec, load_fafa_config
from rcr.proposed.cache.native import NATIVE_VERSION, query_key, subject_conditions
from rcr.proposed.cache.store import GalleryCache


def boxes_to_pixels(
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    scene_size: tuple[int, int],
) -> torch.Tensor:
    """Invert the legacy ImageOps.pad transform without rerunning detection."""
    width, height = image_size
    scene_width, scene_height = scene_size
    scale = min(scene_width / width, scene_height / height)
    resized_width = min(scene_width, round(width * scale))
    resized_height = min(scene_height, round(height * scale))
    left = round((scene_width - resized_width) * 0.5)
    top = round((scene_height - resized_height) * 0.5)
    size = boxes.new_tensor([scene_width, scene_height] * 2)
    offset = boxes.new_tensor([left, top, left, top])
    factors = boxes.new_tensor([resized_width / width, resized_height / height] * 2)
    return (
        ((boxes * size - offset) / factors)
        .clamp(min=0)
        .minimum(boxes.new_tensor([width, height] * 2))
    )


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
def finish_fafa_cache(cfg: dict) -> None:
    root = Path(cfg["data"]["cache"])
    source_root = Path(cfg["data"]["dino_cache"])
    if root.resolve() == source_root.resolve():
        raise ValueError("FAFA output must differ from the DINO source directory")
    source = GalleryCache(source_root)
    check_cache_config(source, cfg, person_backend="dino")
    if source.cache_id is None:
        raise ValueError("DINO source requires a cache_id to bind FAFA features safely")
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    source.validate_gallery(data.gallery_ids)
    marker = root / ".building"
    if (root / "index.pt").is_file() and not marker.exists():
        ready = GalleryCache(root, scene_root=source_root)
        check_cache_config(ready, cfg)
        ready.validate_samples(data.samples)
        print(f"Using existing unified FAFA cache: {root}", flush=True)
        return

    native_cfg = load_fafa_config(cfg)
    batch_size = cfg["person_encoder"]["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("person_encoder.batch_size must be positive")
    storage = cfg.get("cache", {}).get("storage_dtype", "float16")
    if storage not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float32 or float16")
    dtype = getattr(torch, storage)
    device = resolve_device(cfg)
    queries = {
        row["sample_id"]: {
            "index": i,
            "key": query_key(row),
            "conditions": subject_conditions(row),
        }
        for i, row in enumerate(data.samples)
    }
    signature = {
        "source_cache_id": source.cache_id,
        "spec": fafa_spec(native_cfg),
        "checkpoint_sha256": sha256_file(native_cfg["checkpoint"]["path"]),
        "image_encoder": cfg["image_encoder"],
        "detector": cfg["detector"],
        "storage_dtype": storage,
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
        "queries": queries,
    }
    for group in ("targets", "queries", ".persons"):
        (root / group).mkdir(parents=True, exist_ok=True)
    manifest_path = root / ".build.pt"
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature:
            raise ValueError(
                "unfinished FAFA cache settings differ; use a new directory"
            )
    else:
        manifest = {"cache_id": uuid4().hex, "signature": signature}
        save_torch(manifest, manifest_path)
    cache_id = manifest["cache_id"]
    marker.write_text(cache_id, encoding="utf-8")
    model, processors, preprocess, _ = load_fafa(native_cfg, device)
    model.requires_grad_(False).eval()
    hidden_dim = model.Qformer.config.hidden_size
    dim = model.vision_proj.out_features
    tokens = model.query_tokens.shape[1]
    count_people = int(source.mask.sum())
    query_people = sum(
        len(row["subjects"])
        * int(source.mask[source.by_id[row["query_image_id"]]].sum())
        for row in data.samples
    )
    # Account for final tokens/queries, temporary pooled shards, the padded index,
    # scene means and small per-file headers before starting expensive extraction.
    estimated = (
        count_people * (tokens * dim + hidden_dim) * dtype.itemsize
        + source.mask.numel() * hidden_dim * dtype.itemsize
        + query_people * dim * 4
        + len(source.image_ids) * source.scene_dim * 4
        + (2 * len(source.image_ids) + len(data.samples)) * 4096
    )
    saved_bytes = sum(
        path.stat().st_size
        for group in ("targets", "queries", ".persons")
        for path in (root / group).glob("*.pt")
    )
    require_disk_space(root, max(0, estimated - saved_bytes))
    print(f"Unified FAFA cache estimate: {estimated / 1024**3:.2f} GiB", flush=True)

    def crop_batches(image_id, item):
        count = int(source.mask[source.by_id[image_id]].sum())
        if count == 0:
            return
        with Image.open(data.image_path(image_id)) as image_file:
            image = image_file.convert("RGB")
            boxes = item.get("boxes_pixel")
            if boxes is None:
                h, w = cfg["image_encoder"]["scene_size"]
                boxes = boxes_to_pixels(item["boxes_scene"], image.size, (w, h))
            if (
                boxes.shape != (count, 4)
                or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError("invalid DINO person boxes")
            for start in range(0, count, batch_size):
                pixels = torch.stack(
                    [
                        preprocess(image.crop(tuple(box.tolist())))
                        for box in boxes[start : start + batch_size]
                    ]
                ).to(device)
                yield start, pixels.half() if device.type == "cuda" else pixels.float()

    def write_features(path, key, features):
        if not torch.isfinite(features).all():
            raise ValueError(f"non-finite FAFA features: {path}")
        save_torch({"cache_id": cache_id, "key": key, "features": features}, path)

    global_features = torch.empty(len(source.image_ids), source.scene_dim)
    for i, image_id in enumerate(tqdm(source.image_ids, desc="cache FAFA images")):
        count = int(source.mask[i].sum())
        item = source._load_item(i)
        global_features[i] = item["scene"].float().mean(0)
        person_path, target_path = (
            root / group / f"{i}.pt" for group in (".persons", "targets")
        )
        if _ready(
            person_path, cache_id, image_id, (count, hidden_dim), dtype
        ) and _ready(target_path, cache_id, image_id, (count, tokens, dim), dtype):
            continue
        persons = torch.empty(count, hidden_dim, dtype=dtype)
        targets = torch.empty(count, tokens, dim, dtype=dtype)
        for start, pixels in crop_batches(image_id, item):
            # Both outputs come from the same image-only Q-Former forward.
            value = model.extract_features({"image": pixels}, mode="image")
            hidden, projected = value.image_embeds, value.image_embeds_proj
            if hidden.shape != (len(pixels), tokens, hidden_dim):
                raise ValueError("FAFA image_embeds must be [N,L,hidden_dim]")
            if projected.shape != (len(pixels), tokens, dim):
                raise ValueError("FAFA projected tokens must be [N,L,D_fafa]")
            stop = start + len(pixels)
            persons[start:stop] = hidden.float().mean(1).cpu().to(dtype)
            targets[start:stop] = projected.float().cpu().to(dtype)
        write_features(person_path, image_id, persons)
        write_features(target_path, image_id, targets)

    for row in tqdm(data.samples, desc="cache FAFA queries"):
        entry = queries[row["sample_id"]]
        image_id = row["query_image_id"]
        i = source.by_id[image_id]
        count = int(source.mask[i].sum())
        conditions = entry["conditions"]
        shape = (len(conditions), count, dim)
        path = root / "queries" / f"{entry['index']}.pt"
        if _ready(path, cache_id, entry["key"], shape, torch.float32):
            continue
        features = torch.empty(shape)  # Same FP32 query storage as native baseline.
        for start, pixels in crop_batches(image_id, source._load_item(i)):
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
        write_features(path, entry["key"], features)

    del model  # Release native FAFA before assembling the CPU person index.
    persons_index = torch.zeros(*source.mask.shape, hidden_dim, dtype=dtype)
    for i in range(len(source.image_ids)):
        item = torch.load(
            root / ".persons" / f"{i}.pt", map_location="cpu", weights_only=True
        )
        persons_index[i, : int(source.mask[i].sum())] = item["features"]
    save_torch(
        {
            "format_version": 4,
            "layout": "fafa",
            "cache_id": cache_id,
            "source_cache_id": source.cache_id,
            "image_ids": source.image_ids,
            "persons": persons_index,
            "mask": source.mask,
            "patch_hw": source.patch_hw,
            "scene_dim": source.scene_dim,
            "global_features": global_features,
            "storage_dtype": storage,
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
            "person_encoder": {
                "backend": "fafa",
                "spec": signature["spec"],
                "checkpoint_sha256": signature["checkpoint_sha256"],
                "hidden_dim": hidden_dim,
                "encoder_precision": signature["encoder_precision"],
            },
            "native": {
                "version": NATIVE_VERSION,
                "dim": dim,
                "tokens": tokens,
                "queries": queries,
            },
        },
        root / "index.pt",
    )
    for i in range(len(source.image_ids)):
        (root / ".persons" / f"{i}.pt").unlink()
    (root / ".persons").rmdir()
    marker.unlink()
    manifest_path.unlink(missing_ok=True)
    print(f"Finished unified FAFA cache: {root}", flush=True)
