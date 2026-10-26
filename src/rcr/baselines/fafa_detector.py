"""Read the shared detector's exact pixel boxes; no scene backbone is loaded."""

import torch
from tqdm import tqdm

from rcr.common.detection import DetectionCache


def load_detection_cache(data, cfg):
    if cfg["backend"] != "grounding_dino_cache":
        raise ValueError("FAFA requires the shared grounding_dino_cache backend")
    cache = DetectionCache(cfg["cache_dir"])
    cache.validate_gallery(data.gallery_ids)
    expected = {key: cfg[key] for key in ("model", "threshold", "text_threshold")}
    if "revision" in cfg:
        expected["revision"] = cfg["revision"]
    cache.check_config(expected)
    return cache


def candidates_from_cache(data, gallery_ids, cache, cfg):
    """Preserve pixel coordinates/order and the existing zero-detection fallback."""
    del data, cfg
    candidates = []
    for image_id in tqdm(gallery_ids, desc="Shared person boxes", unit="image"):
        index = cache.by_id[image_id]
        geometry = cache.load_geometry(index)
        boxes = geometry["boxes_pixel"]
        width, height = geometry["image_size"]
        count = int(cache.mask[index].sum())
        if boxes.shape != (count, 4):
            raise ValueError(f"cached detection count differs: {image_id}")
        if count:
            # The cache has thresholded boxes, not confidence scores. 1.0 is
            # an eligibility sentinel; it never ranks or reorders detections.
            candidates.append(
                [
                    {"box": box.tolist(), "score": 1.0, "fallback": False}
                    for box in boxes
                ]
            )
        else:
            candidates.append(
                [{"box": [0, 0, width, height], "score": -1.0, "fallback": True}]
            )
    return candidates


@torch.inference_mode()
def detect_gallery(data, gallery_ids, cfg, device):
    del device
    cache = load_detection_cache(data, cfg)
    if any(name not in cache.by_id for name in gallery_ids):
        raise ValueError("FAFA gallery contains images missing from detection cache")
    return candidates_from_cache(
        data, gallery_ids, cache, cfg
    ), f"detections:{cache.cache_id}"
