"""CLIP reranking for Full Positive candidate ordering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from rcr.utils.images import image_relative_path as _image_relative_path

IMAGE_ROOT = Path("dataset/data/raw/images")
CACHE_ROOT = Path("cache/features/clip_vit_b32_openai")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 64


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipChangeRanker:
    """Rank candidate images by CLIP similarity to ``final_change``."""

    def __init__(self) -> None:
        self.device = _device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def __call__(
        self,
        change: str,
        candidates: list[tuple[str, str]],
    ) -> list[str]:
        if not candidates:
            return []

        image_features = self._image_features(candidates)
        text = self.tokenizer([change]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature = text_features[0].float().cpu().numpy()

        scores = image_features @ text_feature
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [candidate[0] for candidate, _ in ranked]

    def _image_features(self, candidates: list[tuple[str, str]]) -> np.ndarray:
        features: dict[str, np.ndarray] = {}
        missing: list[tuple[str, str, Path]] = []

        for image_id, url in candidates:
            relative = _image_relative_path(url)
            cache_path = (CACHE_ROOT / relative).with_suffix(".npy")
            if cache_path.exists():
                features[image_id] = np.load(cache_path)
            else:
                missing.append((image_id, url, cache_path))

        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            tensors = []
            for _, url, _ in batch:
                image_path = IMAGE_ROOT / _image_relative_path(url)
                with Image.open(image_path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))

            images = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                encoded = self.model.encode_image(images)
                encoded /= encoded.norm(dim=-1, keepdim=True)
            encoded = encoded.float().cpu().numpy()

            for (image_id, _, cache_path), feature in zip(
                batch,
                encoded,
                strict=True,
            ):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, feature)
                features[image_id] = feature

        return np.stack([features[image_id] for image_id, _ in candidates])
