"""Order Full Positive candidates by independent image and text similarities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from rcr.common.io import image_relative_path as _image_relative_path

IMAGE_ROOT = Path("dataset/data/raw/images")
CACHE_ROOT = Path("cache/features/clip_vit_b32_openai_quickgelu")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
RERANK_POLICY = "max_image_text_cosine"
BATCH_SIZE = 64


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipCandidateRanker:
    """Rank by max(cosine(seed target, candidate), cosine(change, candidate))."""

    def __init__(self) -> None:
        self.device = _device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            force_quick_gelu=True,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self._cached_features: dict[Path, np.ndarray] = {}

    def __call__(
        self,
        reference_image: tuple[str, str],
        change: str,
        candidates: list[tuple[str, str]],
    ) -> list[str]:
        if not candidates:
            return []

        image_scores, text_scores = self._similarities(
            reference_image, change, candidates
        )
        if not np.isfinite(image_scores).all() or not np.isfinite(text_scores).all():
            raise ValueError("CLIP reranking produced non-finite similarities")
        scores = np.maximum(image_scores, text_scores)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [candidate[0] for candidate, _ in ranked]

    def _similarities(
        self,
        reference_image: tuple[str, str],
        change: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self._image_features([reference_image, *candidates])
        reference_feature, image_features = features[0], features[1:]
        text = self.tokenizer([change]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature = text_features[0].float().cpu().numpy()

        return image_features @ reference_feature, image_features @ text_feature

    def _image_features(self, candidates: list[tuple[str, str]]) -> np.ndarray:
        features: dict[str, np.ndarray] = {}
        missing: list[tuple[str, str, Path]] = []

        for image_id, url in candidates:
            relative = _image_relative_path(url)
            cache_path = (CACHE_ROOT / relative).with_suffix(".npy")
            if cache_path in self._cached_features:
                features[image_id] = self._cached_features[cache_path]
            elif cache_path.exists():
                try:
                    feature = np.load(cache_path, allow_pickle=False)
                except (EOFError, ValueError):
                    # An interrupted earlier write can leave an incomplete .npy.
                    missing.append((image_id, url, cache_path))
                else:
                    self._cached_features[cache_path] = feature
                    features[image_id] = feature
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
                temporary = cache_path.with_suffix(".npy.tmp")
                try:
                    with temporary.open("wb") as handle:
                        np.save(handle, feature)
                    temporary.replace(cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                self._cached_features[cache_path] = feature
                features[image_id] = feature

        return np.stack([features[image_id] for image_id, _ in candidates])
