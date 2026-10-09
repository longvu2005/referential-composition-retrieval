"""Frozen FAFA person features layered over an immutable DINO scene cache."""

import gc
import math
import shutil
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
from rcr.proposed.cache.dino import boxes_to_pixels
from rcr.proposed.cache.store import GalleryCache


@torch.inference_mode()
def extract_person_features(model, pixels: torch.Tensor) -> torch.Tensor:
    """Mean hidden Q-Former image tokens, before the native vision projection."""
    features = model.extract_features({"image": pixels}, mode="image").image_embeds
    if features.ndim != 3:
        raise ValueError("FAFA image_embeds must be [B,query_tokens,hidden_dim]")
    return features.float().mean(dim=1)


def _format_bytes(value: int) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


@torch.inference_mode()
def finish_fafa_cache(cfg: dict) -> None:
    root = Path(cfg["data"]["cache"])
    source_root = Path(cfg["data"]["dino_cache"])
    if root.resolve() == source_root.resolve():
        raise ValueError("FAFA output must differ from the DINO source directory")

    source = GalleryCache(source_root)
    check_cache_config(source, {**cfg, "person_encoder": {"backend": "dino"}})
    if source.cache_id is None:
        raise ValueError("DINO source requires a cache_id to bind FAFA features safely")

    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    source.validate_gallery(data.gallery_ids)

    marker = root / ".building"
    if (root / "index.pt").is_file() and not marker.exists():
        ready = GalleryCache(root, scene_root=source_root)
        check_cache_config(ready, cfg)
        print(f"Using existing FAFA cache: {root}", flush=True)
        return

    native_cfg = load_fafa_config(cfg)
    batch_size = cfg["person_encoder"]["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("person_encoder.batch_size must be positive")

    storage_dtype = cfg.get("cache", {}).get("storage_dtype", "float16")
    if storage_dtype not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float32 or float16")
    dtype = getattr(torch, storage_dtype)

    signature = {
        "source_cache_id": source.cache_id,
        "spec": fafa_spec(native_cfg),
        "checkpoint_sha256": sha256_file(native_cfg["checkpoint"]["path"]),
        "image_encoder": cfg["image_encoder"],
        "detector": cfg["detector"],
        "storage_dtype": storage_dtype,
    }

    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Safe to remove after an interrupted legacy/finalize attempt:
    # the per-image feature shards remain the resumable source of truth.
    for stale in (root / ".persons.bin", root / "index.pt.tmp"):
        if stale.is_file():
            print(f"Removing stale finalize file: {stale}", flush=True)
            stale.unlink()

    manifest_path = root / ".build.pt"
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature:
            raise ValueError(
                "unfinished FAFA cache settings differ; use a new directory"
            )
        cache_id = manifest["cache_id"]
    else:
        cache_id = uuid4().hex
        temporary_manifest = manifest_path.with_suffix(".pt.tmp")
        torch.save(
            {"cache_id": cache_id, "signature": signature},
            temporary_manifest,
        )
        temporary_manifest.replace(manifest_path)

    marker.write_text(cache_id, encoding="utf-8")

    device = resolve_device(cfg)
    model, _, preprocess, _ = load_fafa(native_cfg, device)
    model.requires_grad_(False).eval()
    dim = model.Qformer.config.hidden_size

    height, width = cfg["image_encoder"]["scene_size"]
    global_features = torch.empty(len(source.image_ids), source.scene_dim)

    # Stage 1: extract resumable, per-image FAFA person features.
    for i, image_id in enumerate(tqdm(source.image_ids, desc="cache FAFA persons")):
        feature_path = feature_dir / f"{i}.pt"
        count = int(source.mask[i].sum())

        item = source._load_item(i)
        if item["persons"].shape[0] != count or item["boxes_scene"].shape != (count, 4):
            raise ValueError("DINO scene/person manifest mismatch")

        global_features[i] = item["scene"].float().mean(0)

        if feature_path.is_file():
            saved = torch.load(feature_path, map_location="cpu", weights_only=True)
            if (
                saved.get("cache_id") == cache_id
                and saved.get("source_cache_id") == source.cache_id
                and saved.get("image_id") == image_id
                and saved["persons"].shape == (count, dim)
                and saved["persons"].dtype == dtype
                and torch.isfinite(saved["persons"]).all()
            ):
                continue

        features = torch.empty(count, dim, dtype=dtype)

        if count:
            # Resolve current paths, never old absolute paths from another machine.
            with Image.open(data.image_path(image_id)) as image_file:
                image = image_file.convert("RGB")
                boxes = item.get("boxes_pixel")
                if boxes is None:
                    boxes = boxes_to_pixels(
                        item["boxes_scene"],
                        image.size,
                        (width, height),
                    )

                if (
                    boxes.shape != (count, 4)
                    or not torch.isfinite(boxes).all()
                    or (boxes[:, 2:] <= boxes[:, :2]).any()
                ):
                    raise ValueError("invalid DINO person boxes")

                for start in range(0, count, batch_size):
                    selected = boxes[start : start + batch_size]
                    pixels = torch.stack(
                        [
                            preprocess(image.crop(tuple(box.tolist())))
                            for box in selected
                        ]
                    ).to(device)
                    pixels = pixels.half() if device.type == "cuda" else pixels.float()

                    pooled = extract_person_features(model, pixels)
                    stored = pooled.to(device="cpu", dtype=dtype, copy=True)

                    if (
                        stored.shape != (len(selected), dim)
                        or not torch.isfinite(stored).all()
                    ):
                        raise ValueError("invalid FAFA person features")

                    features[start : start + len(selected)] = stored

        temporary_feature = feature_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "cache_id": cache_id,
                "source_cache_id": source.cache_id,
                "image_id": image_id,
                "persons": features,
            },
            temporary_feature,
        )
        temporary_feature.replace(feature_path)

    # FAFA is no longer needed. Release it before allocating the padded CPU index.
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 2: build the legacy padded person tensor in CPU RAM.
    #
    # The previous implementation used a disk-backed .persons.bin and then
    # torch.save() copied that entire tensor again into index.pt. At finalize
    # time this required two full padded copies on disk, in addition to shards.
    shape = (*source.mask.shape, dim)
    persons_index = torch.zeros(shape, dtype=dtype)

    feature_paths: list[Path] = []
    for i, image_id in enumerate(tqdm(source.image_ids, desc="index FAFA persons")):
        feature_path = feature_dir / f"{i}.pt"
        item = torch.load(feature_path, map_location="cpu", weights_only=True)

        count = int(source.mask[i].sum())
        persons = item["persons"]
        if (
            item.get("cache_id") != cache_id
            or item.get("source_cache_id") != source.cache_id
            or item.get("image_id") != image_id
            or persons.shape != (count, dim)
            or persons.dtype != dtype
            or not torch.isfinite(persons).all()
        ):
            raise ValueError(f"invalid FAFA feature shard: {feature_path}")

        persons_index[i, :count] = persons
        feature_paths.append(feature_path)

    index = {
        "format_version": 3,
        "layout": "persons",
        "cache_id": cache_id,
        "source_cache_id": source.cache_id,
        "image_ids": source.image_ids,
        "persons": persons_index,
        "mask": source.mask,
        "patch_hw": source.patch_hw,
        "scene_dim": source.scene_dim,
        "global_features": global_features,
        "storage_dtype": storage_dtype,
        "image_encoder": cfg["image_encoder"],
        "detector": cfg["detector"],
        "person_encoder": {
            "backend": "fafa",
            "spec": signature["spec"],
            "checkpoint_sha256": signature["checkpoint_sha256"],
            "hidden_dim": dim,
        },
    }

    # Estimate the final write size and keep a safety margin. If disk is tight,
    # delete only as many already-consumed shards as needed. Their data is now
    # resident in persons_index, so this avoids keeping a second full copy on disk.
    persons_bytes = math.prod(shape) * persons_index.element_size()
    auxiliary_bytes = (
        global_features.numel() * global_features.element_size()
        + source.mask.numel() * source.mask.element_size()
    )
    estimated_index_bytes = persons_bytes + auxiliary_bytes
    safety_margin = max(1024**3, estimated_index_bytes // 10)
    required_free = estimated_index_bytes + safety_margin

    disk_free = shutil.disk_usage(root).free
    shard_sizes = [(path, path.stat().st_size) for path in feature_paths]
    reclaimable = sum(size for _, size in shard_sizes)

    print(
        "FAFA finalize disk: "
        f"free={_format_bytes(disk_free)}, "
        f"estimated_index={_format_bytes(estimated_index_bytes)}, "
        f"reclaimable_shards={_format_bytes(reclaimable)}",
        flush=True,
    )

    if disk_free + reclaimable < required_free:
        raise RuntimeError(
            "Not enough disk space to finalize FAFA cache safely. "
            f"Need about {_format_bytes(required_free)}, "
            f"but free + FAFA shards is only "
            f"{_format_bytes(disk_free + reclaimable)}. "
            "No feature shards were deleted."
        )

    if disk_free < required_free:
        # Delete the largest shards first so an interrupted save requires
        # re-extracting as few images as possible on the next run.
        freed = 0
        for feature_path, size in sorted(
            shard_sizes,
            key=lambda pair: pair[1],
            reverse=True,
        ):
            feature_path.unlink()
            freed += size
            if disk_free + freed >= required_free:
                break

        print(
            f"Reclaimed {_format_bytes(freed)} from consumed FAFA shards "
            "before writing index.pt.",
            flush=True,
        )

    temporary = root / "index.pt.tmp"
    try:
        torch.save(index, temporary)
        temporary.replace(root / "index.pt")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    # The published index now owns all FAFA person features. Remove any
    # remaining per-image shards and build metadata.
    for feature_path in feature_paths:
        feature_path.unlink(missing_ok=True)

    try:
        feature_dir.rmdir()
    except OSError:
        # Leave the directory if unrelated files are present.
        pass

    del index, persons_index
    gc.collect()

    marker.unlink()  # Publish only after all person features and the index agree.
    manifest_path.unlink(missing_ok=True)

    print(f"Finished FAFA cache: {root / 'index.pt'}", flush=True)
