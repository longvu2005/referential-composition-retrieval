"""Frozen FAFA person features layered over an immutable DINO scene cache."""

from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.methods.baselines.fafa import load_fafa
from rcr.methods.common.data import load_rcr_data
from rcr.methods.common.experiment import resolve_device
from rcr.methods.common.results import sha256_file
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.person_encoder import fafa_spec, load_fafa_config


@torch.inference_mode()
def extract_person_features(model, pixels: torch.Tensor) -> torch.Tensor:
    """Mean hidden Q-Former image tokens, before the native vision projection."""
    features = model.extract_features({"image": pixels}, mode="image").image_embeds
    if features.ndim != 3:
        raise ValueError("FAFA image_embeds must be [B,query_tokens,hidden_dim]")
    return features.float().mean(dim=1)


def boxes_to_pixels(
    boxes: torch.Tensor, image_size: tuple[int, int], scene_size: tuple[int, int]
) -> torch.Tensor:
    """Invert the legacy ImageOps.pad box transform; no detector is rerun."""
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


@torch.inference_mode()
def finish_fafa_cache(cfg: dict) -> None:
    root = Path(cfg["data"]["cache"])
    source_root = Path(cfg["data"]["dino_cache"])
    if root.resolve() == source_root.resolve():
        raise ValueError("FAFA output must differ from the DINO source directory")
    source = GalleryCache(source_root)
    source.validate_encoders({**cfg, "person_encoder": {"backend": "dino"}})
    source.validate_files()
    if source.cache_id is None:
        raise ValueError("DINO source requires a cache_id to bind FAFA features safely")
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    source.validate_gallery(data.gallery_ids)
    marker = root / ".building"
    if (root / "index.pt").is_file() and not marker.exists():
        ready = GalleryCache(root, scene_root=source_root)
        ready.validate_encoders(cfg)
        ready.validate_files()
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
        torch.save({"cache_id": cache_id, "signature": signature}, temporary_manifest)
        temporary_manifest.replace(manifest_path)
    marker.write_text(cache_id, encoding="utf-8")
    device = resolve_device(cfg)
    model, _, preprocess, _ = load_fafa(native_cfg, device)
    model.requires_grad_(False).eval()
    dim = model.Qformer.config.hidden_size
    height, width = cfg["image_encoder"]["scene_size"]
    global_features = torch.empty(len(source.image_ids), source.scene_dim)
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
                        item["boxes_scene"], image.size, (width, height)
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
        torch.save(
            {
                "cache_id": cache_id,
                "source_cache_id": source.cache_id,
                "image_id": image_id,
                "persons": features,
            },
            feature_path.with_suffix(".pt.tmp"),
        )
        feature_path.with_suffix(".pt.tmp").replace(feature_path)

    # The legacy padded index can be several GB. Assemble on disk so it does
    # not allocate a second huge CPU index while FAFA is resident.
    shape = (*source.mask.shape, dim)
    backing = root / ".persons.bin"
    persons_index = torch.from_file(
        str(backing), shared=True, size=source.mask.numel() * dim, dtype=dtype
    ).reshape(shape)
    persons_index.zero_()
    for i in tqdm(range(len(source.image_ids)), desc="index FAFA persons"):
        item = torch.load(
            feature_dir / f"{i}.pt", map_location="cpu", weights_only=True
        )
        persons_index[i, : len(item["persons"])] = item["persons"]
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
    temporary = root / "index.pt.tmp"
    torch.save(index, temporary)
    temporary.replace(root / "index.pt")
    del index, persons_index
    backing.unlink(missing_ok=True)
    marker.unlink()  # Publish only after all person features and the index agree.
    manifest_path.unlink(missing_ok=True)
