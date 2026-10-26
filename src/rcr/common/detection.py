"""Shared, immutable pixel boxes; no scene encoder, text, or training labels."""

import gc
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.common.geometry import boxes_to_pixels
from rcr.common.torch_io import save_torch
from rcr.common.vision import detect

DETECTION_LAYOUT = "person-detections-v1"


def _validate_boxes(boxes):
    if (
        not isinstance(boxes, torch.Tensor)
        or boxes.dtype != torch.float32
        or boxes.ndim != 2
        or boxes.shape[1] != 4
        or not torch.isfinite(boxes).all()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError("invalid detection pixel boxes; crop order must be preserved")


def _cache_id(index):
    """Content identity survives path relocation and identical re-extraction."""
    digest = hashlib.sha256(
        json.dumps(
            {key: index[key] for key in ("layout", "image_ids", "detector", "prompt")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for key in ("image_sizes", "offsets", "boxes_pixel"):
        value = index[key].contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class DetectionCache:
    """Small packed CPU index shared by Proposed, FAFA and diagnostics."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if (self.root / ".building").exists():
            raise RuntimeError(
                "detection cache incomplete; resume --cache-stage detections"
            )
        if not (self.root / "index.pt").is_file():
            raise FileNotFoundError(
                f"missing detection cache at {self.root}; "
                "run --cache-stage detections first"
            )
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True, mmap=True
        )
        if index.get("layout") != DETECTION_LAYOUT or index.get("prompt") != "person":
            raise ValueError(
                "use a separate detection cache; import legacy scene boxes"
            )
        self.image_ids = index["image_ids"]
        self.by_id = {name: i for i, name in enumerate(self.image_ids)}
        self.detector = index["detector"]
        self.encoder_metadata = {"detector": self.detector}
        self.image_sizes = index["image_sizes"]
        self.offsets = index["offsets"]
        self.boxes = index["boxes_pixel"]
        self.cache_id = index["cache_id"]
        count = len(self.image_ids)
        if (
            len(self.by_id) != count
            or self.image_sizes.dtype != torch.int64
            or self.image_sizes.shape != (count, 2)
            or (self.image_sizes <= 0).any()
            or self.offsets.dtype != torch.int64
            or self.offsets.shape != (count + 1,)
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.boxes)
            or (self.offsets.diff() < 0).any()
        ):
            raise ValueError("invalid detection image sizes/offsets")
        if self.cache_id != _cache_id(index):
            raise ValueError("detection cache content differs from its cache_id")
        counts = self.offsets.diff()
        self.mask = (
            torch.arange(int(counts.max()) if count else 0)[None] < counts[:, None]
        )
        # Grounding DINO may predict boxes crossing the image boundary. Keep
        # them exact: PIL crop padding and scene-coordinate clipping are owned
        # by the existing consumers, not an extra detector postprocessing step.
        _validate_boxes(self.boxes)

    def validate_gallery(self, image_ids):
        if self.image_ids != image_ids:
            raise ValueError("detection gallery IDs/order differ from finalized data")

    def check_config(self, detector):
        if self.detector != detector:
            raise ValueError(
                "detection cache detector provenance differs from configuration"
            )

    def load_geometry(self, index):
        start, stop = self.offsets[index : index + 2].tolist()
        return {
            "boxes_pixel": self.boxes[start:stop],
            "image_size": tuple(self.image_sizes[index].tolist()),
        }


def _legacy_source(root, image_ids, detector):
    root = Path(root)
    if (root / ".building").exists():
        raise ValueError("cannot import an unfinished legacy scene cache")
    index = torch.load(
        root / "index.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if index.get("layout") is not None or index.get("detector") != detector:
        raise ValueError("legacy scene cache lacks matching detector provenance")
    if index["image_ids"] != image_ids or not index.get("cache_id"):
        raise ValueError("legacy scene gallery/cache identity differs")
    return index


def build_detection_cache(
    image_ids,
    image_paths,
    root,
    detector_config,
    *,
    model=None,
    processor=None,
    device="cpu",
    import_scene=None,
):
    """Resume per-image shards, then publish only validated boxes and image sizes."""
    root = Path(root)
    if import_scene is not None:
        other, target = Path(import_scene).resolve(), root.resolve()
        if other == target or other in target.parents or target in other.parents:
            raise ValueError(
                "legacy scene and detection cache must be separate directories"
            )
    if (root / "index.pt").is_file() and not (root / ".building").exists():
        cache = DetectionCache(root)
        cache.validate_gallery(image_ids)
        cache.check_config(detector_config)
        print(f"Using existing detection cache: {root}", flush=True)
        return
    legacy = (
        _legacy_source(import_scene, image_ids, detector_config)
        if import_scene
        else None
    )
    signature = {
        "image_ids": image_ids,
        "image_paths": [str(Path(p).resolve()) for p in image_paths],
        "detector": detector_config,
        "prompt": "person",
        "import_cache_id": legacy["cache_id"] if legacy else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / ".build.pt"
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature:
            raise ValueError(
                "unfinished detection settings differ; use a new directory"
            )
    else:
        manifest = {"signature": signature, "build_id": uuid4().hex}
        save_torch(manifest, manifest_path)
    (root / ".building").write_text(manifest["build_id"], encoding="utf-8")
    shards = root / "features"
    shards.mkdir(exist_ok=True)
    if legacy is None and model is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        kwargs = (
            {"revision": detector_config["revision"]}
            if "revision" in detector_config
            else {}
        )
        processor = AutoProcessor.from_pretrained(detector_config["model"], **kwargs)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_config["model"], **kwargs
        ).to(device)
    if model is not None:
        model.eval()
    rows, sizes = [], []
    for i, (image_id, path) in enumerate(
        tqdm(
            zip(image_ids, image_paths, strict=True),
            total=len(image_ids),
            desc="person detections",
        )
    ):
        shard = shards / f"{i}.pt"
        saved = None
        if shard.is_file():
            try:
                saved = torch.load(shard, map_location="cpu", weights_only=True)
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
                pass
        if (
            saved is not None
            and saved.get("build_id") == manifest["build_id"]
            and saved.get("image_id") == image_id
        ):
            boxes, size = saved["boxes_pixel"], saved["image_size"]
        else:
            with Image.open(path) as source:
                size = source.size
                if legacy is None:
                    boxes = (
                        detect(
                            model,
                            processor,
                            source.convert("RGB"),
                            ["person"],
                            device,
                            detector_config["threshold"],
                            detector_config["text_threshold"],
                        )["person"]
                        .detach()
                        .to("cpu", torch.float32)
                        .clone()
                    )
                else:
                    item = torch.load(
                        Path(import_scene) / "features" / f"{i}.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    if (
                        item.get("cache_id") != legacy["cache_id"]
                        or item.get("image_id", image_id) != image_id
                    ):
                        raise ValueError("legacy scene shard identity differs")
                    boxes = item.get("boxes_pixel")
                    if boxes is None:
                        h, w = legacy["image_encoder"]["scene_size"]
                        boxes = boxes_to_pixels(item["boxes_scene"], size, (w, h))
                    boxes = boxes.float().clone()
                    if len(boxes) != int(legacy["mask"][i].sum()):
                        raise ValueError("legacy scene box count differs")
            _validate_boxes(boxes)
            save_torch(
                {
                    "build_id": manifest["build_id"],
                    "image_id": image_id,
                    "boxes_pixel": boxes,
                    "image_size": size,
                },
                shard,
            )
        _validate_boxes(boxes)
        rows.append(boxes)
        sizes.append(size)
    if not rows:
        raise ValueError("detection gallery is empty")
    counts = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    index = {
        "layout": DETECTION_LAYOUT,
        "image_ids": image_ids,
        "detector": detector_config,
        "prompt": "person",
        "image_sizes": torch.tensor(sizes, dtype=torch.int64),
        "offsets": torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))),
        "boxes_pixel": torch.cat(rows),
    }
    index["cache_id"] = _cache_id(index)
    save_torch(index, root / "index.pt")
    (root / ".building").unlink()
    manifest_path.unlink()
    shutil.rmtree(shards)  # The published packed index owns all geometry.
    del model
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
