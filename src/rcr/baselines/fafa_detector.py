"""Read exactly the person boxes already detected by Proposed's DINO Tiny.

The shared cache contains inference-time boxes, not identity annotations. No
model is constructed and no person detection is repeated by this baseline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def _pixel_boxes(
    geometry: dict, image_size: tuple[int, int], scene_size: tuple[int, int]
) -> torch.Tensor:
    """Mirror Proposed's legacy boxes_to_pixels only when pixels are absent."""
    if "boxes_pixel" in geometry:
        return geometry["boxes_pixel"]
    boxes = geometry["boxes_scene"]
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


def candidates_from_cache(data, gallery_ids: list[str], cache, cfg: dict):
    """Preserve the same crop order and float pixel coordinates as Proposed."""
    candidates = []
    scene_height, scene_width = cfg["scene_size"]
    scene_size = (scene_width, scene_height)
    for image_id in tqdm(gallery_ids, desc="Shared DINO Tiny boxes", unit="image"):
        index = cache.by_id[image_id]
        geometry = cache.load_geometry(index)
        count = int(cache.mask[index].sum())
        # PIL reads only the image header here; no image is decoded into memory.
        with Image.open(data.image_path(image_id)) as image:
            width, height = image.size
        boxes = _pixel_boxes(geometry, (width, height), scene_size)
        if (
            boxes.shape != (count, 4)
            or not torch.isfinite(boxes).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
            or (boxes < 0).any()
            or (boxes[:, 0::2] > width).any()
            or (boxes[:, 1::2] > height).any()
        ):
            raise ValueError(f"Invalid cached DINO person boxes: {image_id}")
        if count:
            # Proposed records boxes, not detector scores. All boxes have
            # already passed its configured threshold; 1.0 is an eligibility
            # sentinel, NOT an inferred confidence or ranking criterion.
            candidates.append(
                [
                    {"box": box.tolist(), "score": 1.0, "fallback": False}
                    for box in boxes
                ]
            )
        else:
            # Proposed has zero persons; FAFA still requires a gallery crop.
            candidates.append(
                [{"box": [0, 0, width, height], "score": -1.0, "fallback": True}]
            )
    return candidates


@torch.inference_mode()
def detect_gallery(data, gallery_ids: list[str], cfg: dict, device):
    """Load Proposed's complete immutable DINO cache; never run a detector."""
    del device
    if cfg["backend"] != "proposed_dino_tiny_cache":
        raise ValueError("FAFA must read the shared Proposed DINO Tiny cache")
    root = Path(cfg["cache_dir"])
    from rcr.proposed.cache.store import GalleryCache

    cache = GalleryCache(root)
    if cache.cache_id is None:
        raise ValueError("DINO cache has no cache_id; rebuild it with Proposed")
    # The source is built over the full finalized image manifest; baseline
    # ranking still uses its original split-specific gallery_ids argument.
    cache.validate_gallery(data.gallery_ids)
    expected = {key: cfg[key] for key in ("model", "threshold", "text_threshold")}
    observed = cache.encoder_metadata["detector"]
    legacy = observed is None
    if legacy and not cfg.get("allow_legacy_cache", False):
        raise ValueError(
            "DINO cache lacks detector provenance. Rebuild it, or explicitly set "
            "allow_legacy_cache=true only after independently verifying its model "
            "and thresholds"
        )
    if observed is not None and observed != expected:
        raise ValueError(
            "DINO cache detector provenance differs from FAFA configuration; "
            "do not silently mix detector settings"
        )
    image_cfg = cache.encoder_metadata["image_encoder"]
    if image_cfg is not None and tuple(image_cfg["scene_size"]) != tuple(
        cfg["scene_size"]
    ):
        raise ValueError("DINO scene_size differs; cannot invert legacy boxes")
    if any(image_id not in cache.by_id for image_id in gallery_ids):
        raise ValueError("FAFA gallery contains images missing from DINO cache")
    candidates = candidates_from_cache(data, gallery_ids, cache, cfg)
    # The cache id invalidates FAFA gallery/query features if DINO is rebuilt.
    marker = "#legacy-unverified" if legacy else ""
    return candidates, f"{root.resolve()}#cache_id={cache.cache_id}{marker}"
