"""OpenAI CLIP image/text/fusion baselines over whole RCR scene images.

Ported scoring from cpr_baseline_bench (ca774cc1). Data and evaluation are RCR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rcr.methods.common.results import cache_directory, image_signature, sha256_file

MODES = ("image", "text", "early_fusion", "late_fusion")


class ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def load_clip(checkpoint: str | Path, device: torch.device):
    """Use prepared local weights; passing a model name would download implicitly."""
    import clip

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}; run prepare_baseline.py first")
    model, preprocess = clip.load(str(checkpoint.resolve()), device=device, jit=False)
    if device.type != "cuda":
        model.float()
    return clip, model.eval(), preprocess


@torch.inference_mode()
def encode_images(model, preprocess, paths, runtime: dict, device) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths, preprocess),
        batch_size=int(runtime["image_batch_size"]),
        num_workers=int(runtime.get("num_workers", 0)),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    features = []
    for images in tqdm(loader, desc="CLIP gallery", unit="batch"):
        encoded = model.encode_image(images.to(device)).float()
        features.append(F.normalize(encoded, dim=-1).cpu().numpy())
    return np.concatenate(features).astype(np.float32, copy=False)


@torch.inference_mode()
def encode_texts(clip_module, model, texts, batch_size: int, device):
    truncated = 0
    for text in texts:
        try:
            clip_module.tokenize([text], truncate=False)
        except RuntimeError:
            truncated += 1
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text", unit="batch"):
        tokens = clip_module.tokenize(texts[start : start + batch_size], truncate=True)
        encoded = model.encode_text(tokens.to(device)).float()
        features.append(F.normalize(encoded, dim=-1).cpu().numpy())
    return np.concatenate(features).astype(np.float32, copy=False), truncated


def score_features(
    gallery: torch.Tensor,
    query_images: torch.Tensor | None,
    query_text: torch.Tensor | None,
    mode: str,
    fusion: dict,
) -> torch.Tensor:
    """Higher is better; late fusion z-scores each branch over the full gallery."""
    if mode not in MODES:
        raise ValueError(f"unsupported CLIP mode {mode!r}")
    if mode == "image":
        return query_images @ gallery.T
    if mode == "text":
        return query_text @ gallery.T
    iw = float(fusion["image_weight"])
    tw = float(fusion["text_weight"])
    if not np.isfinite([iw, tw]).all() or min(iw, tw) < 0 or iw + tw <= 0:
        raise ValueError(
            "fusion weights must be finite, non-negative, with positive sum"
        )
    iw, tw = iw / (iw + tw), tw / (iw + tw)
    if mode == "early_fusion":
        query = F.normalize(iw * query_images + tw * query_text, dim=-1)
        return query @ gallery.T
    eps = float(fusion.get("epsilon", 1e-6))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("fusion epsilon must be finite and positive")
    if fusion.get("score_normalization", "zscore") != "zscore":
        raise ValueError("late fusion requires score_normalization: zscore")

    def zscore(scores):
        return (scores - scores.mean(-1, keepdim=True)) / scores.std(
            -1, keepdim=True, unbiased=False
        ).clamp_min(eps)

    return iw * zscore(query_images @ gallery.T) + tw * zscore(query_text @ gallery.T)


@torch.inference_mode()
def retrieve_clip(data, samples: list[dict], gallery_ids: list[str], cfg: dict, device):
    mode = cfg["mode"]
    if mode not in MODES:
        raise ValueError(f"unsupported CLIP mode {mode!r}")
    runtime = cfg["runtime"]
    checkpoint = Path(cfg["model"]["checkpoint"])
    signature = {
        "version": "clip-rcr-v1",
        "model": cfg["model"]["name"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess": "pinned-openai-clip-default",
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
        "images": image_signature(data, gallery_ids),
    }
    directory = cache_directory(cfg["cache"]["dir"], signature)
    feature_path = directory / "gallery.npy"
    clip_module, model, preprocess = load_clip(checkpoint, device)
    expected_dim = int(model.text_projection.shape[1])
    features = None
    if feature_path.is_file():
        cached = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if (
            cached.shape == (len(gallery_ids), expected_dim)
            and cached.dtype == np.float32
        ):
            features = cached
            print(f"CLIP gallery cache: {feature_path}", flush=True)
    if features is None:
        features = encode_images(
            model,
            preprocess,
            [data.image_path(x) for x in gallery_ids],
            runtime,
            device,
        )
        if features.shape != (len(gallery_ids), expected_dim):
            raise ValueError("CLIP gallery feature shape does not match the model")
        with feature_path.with_suffix(".tmp").open("wb") as handle:
            np.save(handle, features)
        feature_path.with_suffix(".tmp").replace(feature_path)
    text_features = None
    truncated = 0
    if mode != "image":
        field = cfg.get("text_field", "final_instruction")
        if field not in ("final_instruction", "final_change"):
            raise ValueError("text_field must be final_instruction or final_change")
        text_features, truncated = encode_texts(
            clip_module,
            model,
            [sample[field] for sample in samples],
            int(runtime["text_batch_size"]),
            device,
        )
    # Release the encoder before allocating query-gallery score batches.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gallery = torch.from_numpy(np.array(features, copy=True)).to(device)
    by_id = {image_id: index for index, image_id in enumerate(gallery_ids)}
    query_indices = [by_id[sample["query_image_id"]] for sample in samples]
    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "scores.npy.tmp"
    scores = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(samples), len(gallery_ids))
    )
    batch_size = int(runtime["score_batch_size"])
    for start in tqdm(
        range(0, len(samples), batch_size), desc="CLIP scoring", unit="batch"
    ):
        stop = min(start + batch_size, len(samples))
        image_query = gallery[query_indices[start:stop]] if mode != "text" else None
        text_query = (
            torch.from_numpy(text_features[start:stop]).to(device)
            if text_features is not None
            else None
        )
        scores[start:stop] = (
            score_features(
                gallery, image_query, text_query, mode, cfg.get("fusion", {})
            )
            .cpu()
            .numpy()
        )
    scores.flush()
    del scores
    temporary.replace(output / "scores.npy")
    return np.load(output / "scores.npy", mmap_mode="r", allow_pickle=False), {
        "checkpoint_sha256": signature["checkpoint_sha256"],
        "gallery_cache": str(directory),
        "truncated_text_queries": truncated,
        "rcr_training": False,
    }
