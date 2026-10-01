"""Build reference-free gallery features for coarse and fine retrieval."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

import torch
import yaml
from PIL import Image, ImageOps
from torch import nn
from tqdm import tqdm

from rcr.methods.common.anchors import match_heads_to_persons
from rcr.methods.common.data import load_rcr_data
from rcr.methods.common.detector import detect
from rcr.methods.proposed.encoders import ImageEncoder


def boxes_to_scene(
    boxes: torch.Tensor, image_size: tuple[int, int], scene_size: tuple[int, int]
) -> torch.Tensor:
    """Map image pixel xyxy boxes to normalized ImageOps.pad letterbox coordinates."""
    width, height = image_size
    scene_width, scene_height = scene_size
    scale = min(scene_width / width, scene_height / height)
    # ImageOps.pad centers a rounded resized image on the canvas.
    resized_width = min(scene_width, round(width * scale))
    resized_height = min(scene_height, round(height * scale))
    left = round((scene_width - resized_width) * 0.5)
    top = round((scene_height - resized_height) * 0.5)
    factors = boxes.new_tensor([resized_width / width, resized_height / height] * 2)
    offset = boxes.new_tensor([left, top, left, top])
    size = boxes.new_tensor([scene_width, scene_height] * 2)
    return ((boxes * factors + offset) / size).clamp(0, 1)


@torch.inference_mode()
def build_cache(
    image_ids: list[str],
    image_paths: list[Path],
    root: str | Path,
    detector: nn.Module,
    detector_processor,
    image_encoder: nn.Module,
    scene_processor,
    person_processor,
    device: torch.device | str,
    gt_heads_by_image: dict[str, list[dict]] | None = None,
    detector_threshold: float = 0.3,
    detector_text_threshold: float = 0.25,
    storage_dtype: str = "float32",
) -> None:
    """Encode gallery images and write a cache compatible with GalleryCache."""

    storage_dtypes = {"float32": torch.float32, "float16": torch.float16}
    if storage_dtype not in storage_dtypes:
        raise ValueError("storage_dtype must be float32 or float16")
    feature_dtype = storage_dtypes[storage_dtype]

    def compact_cpu(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # CLS/patch slices share the backbone output's storage. Even a
        # contiguous one-person CLS slice needs a copy to own compact storage.
        return value.detach().to(
            device="cpu",
            dtype=dtype,
            copy=True,
            memory_format=torch.contiguous_format,
        )

    root = Path(root)
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_id = uuid4().hex
    building = root / ".building"
    building.write_text(cache_id, encoding="utf-8")

    detector.eval()
    image_encoder.eval()

    all_persons = []
    all_global = []
    patch_hw = None

    for index, (image_id, path) in enumerate(
        tqdm(zip(image_ids, image_paths, strict=True), total=len(image_ids))
    ):
        image = Image.open(path).convert("RGB")
        boxes = detect(
            detector,
            detector_processor,
            image,
            ["person"],
            device,
            detector_threshold,
            detector_text_threshold,
        )

        persons = boxes["person"].cpu()

        scene_pixels = scene_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        scene, _, current_hw = image_encoder(scene_pixels.to(device))
        if patch_hw is None:
            patch_hw = current_hw
        elif patch_hw != current_hw:
            raise ValueError("scene processor must produce one fixed patch grid")

        if len(persons):
            person_crops = [image.crop(tuple(box.tolist())) for box in persons]

            pixels = person_processor(images=person_crops, return_tensors="pt")[
                "pixel_values"
            ]
            _, person_features, _ = image_encoder(pixels.to(device))

        else:
            dim = scene.shape[-1]
            person_features = scene.new_empty(0, dim)

        identity_ids: list[str | None] = [None] * len(persons)
        if gt_heads_by_image is not None:
            rows = gt_heads_by_image.get(image_id, [])
            if rows:
                gt_heads = persons.new_tensor(
                    [
                        [
                            row["x"],
                            row["y"],
                            row["x"] + row["width"],
                            row["y"] + row["height"],
                        ]
                        for row in rows
                    ]
                )
                gt_index = match_heads_to_persons(gt_heads, persons)
                for person, gt in enumerate(gt_index.tolist()):
                    if gt >= 0:
                        identity_ids[person] = str(rows[gt]["identity_id"])

        scene_size = (int(scene_pixels.shape[-1]), int(scene_pixels.shape[-2]))
        boxes_scene = boxes_to_scene(persons, image.size, scene_size)
        cached_persons = compact_cpu(person_features, feature_dtype)
        cached_scene = compact_cpu(scene[0], feature_dtype)
        # Match training/legacy pooling exactly, including storage rounding.
        all_global.append(cached_scene.float().mean(dim=0))

        torch.save(
            {
                "cache_id": cache_id,
                "image_id": image_id,
                "scene": cached_scene,
                "persons": cached_persons,
                "identity_ids": identity_ids,
                "boxes_scene": compact_cpu(boxes_scene, torch.float32),
            },
            feature_dir / f"{index}.pt",
        )
        all_persons.append(cached_persons)

    if patch_hw is None:
        raise ValueError("gallery is empty")

    k = max((x.shape[0] for x in all_persons), default=0)
    dim = all_persons[0].shape[-1] if all_persons else 0
    person_index = torch.zeros(len(all_persons), k, dim, dtype=feature_dtype)
    mask = torch.zeros(len(all_persons), k, dtype=torch.bool)
    for i, value in enumerate(all_persons):
        person_index[i, : value.shape[0]] = value
        mask[i, : value.shape[0]] = True

    index_path = root / "index.pt"
    temporary_index = root / "index.pt.tmp"
    torch.save(
        {
            "cache_id": cache_id,
            "image_ids": image_ids,
            "persons": person_index,
            "mask": mask,
            "patch_hw": patch_hw,
            "storage_dtype": storage_dtype,
            "global_features": torch.stack(all_global),
        },
        temporary_index,
    )
    os.replace(temporary_index, index_path)
    building.unlink()


class _LetterboxProcessor:
    """Apply one fixed letterbox size, then reuse the pretrained normalization."""

    def __init__(self, processor, size: list[int]) -> None:
        self.processor = processor
        self.size = (int(size[1]), int(size[0]))

    def __call__(self, images, return_tensors: str):
        single = not isinstance(images, list)
        rows = [images] if single else images
        rows = [
            ImageOps.pad(
                image,
                self.size,
                method=Image.Resampling.BICUBIC,
                color=(0, 0, 0),
            )
            for image in rows
        ]
        return self.processor(
            images=rows[0] if single else rows,
            do_resize=False,
            do_center_crop=False,
            return_tensors=return_tensors,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/build_cache.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])

    detector_cfg = cfg["detector"]
    detector_processor = AutoProcessor.from_pretrained(detector_cfg["model"])
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_cfg["model"]
    ).to(device)

    image_cfg = cfg["image_encoder"]
    processor = AutoImageProcessor.from_pretrained(image_cfg["model"])
    backbone = AutoModel.from_pretrained(image_cfg["model"]).to(device)
    image_encoder = ImageEncoder(backbone, backbone.config.hidden_size).to(device)

    build_cache(
        image_ids=data.gallery_ids,
        image_paths=[data.image_path(image_id) for image_id in data.gallery_ids],
        root=data_cfg["cache"],
        detector=detector,
        detector_processor=detector_processor,
        image_encoder=image_encoder,
        scene_processor=_LetterboxProcessor(processor, image_cfg["scene_size"]),
        person_processor=_LetterboxProcessor(processor, image_cfg["person_size"]),
        device=device,
        gt_heads_by_image=data.gt_head_boxes_by_image,
        detector_threshold=detector_cfg["threshold"],
        detector_text_threshold=detector_cfg["text_threshold"],
        storage_dtype=cfg.get("cache", {}).get("storage_dtype", "float32"),
    )


if __name__ == "__main__":
    main()
