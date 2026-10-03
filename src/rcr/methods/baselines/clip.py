"""OpenAI CLIP image/text/fusion baselines over whole RCR scene images.

Ported scoring from cpr_baseline_bench (ca774cc1). Data and evaluation are RCR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rcr.methods.common.results import cache_directory, image_signature, sha256_file

MODES = ("clip_image", "clip_text", "early_fusion", "late_fusion")
MODE_ALIASES = {"image": "clip_image", "text": "clip_text"}


def canonical_mode(mode: str) -> str:
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"unsupported CLIP mode {mode!r}")
    return mode


def fusion_weights(fusion: dict) -> tuple[float, float]:
    """Only two non-negative weights are needed; their ratio sets the mixture."""
    iw, tw = float(fusion["image_weight"]), float(fusion["text_weight"])
    if not np.isfinite([iw, tw]).all() or min(iw, tw) < 0 or iw + tw <= 0:
        raise ValueError(
            "fusion weights must be finite, non-negative, with positive sum"
        )
    return iw / (iw + tw), tw / (iw + tw)


def late_normalize(scores: torch.Tensor, fusion: dict) -> torch.Tensor:
    eps = float(fusion.get("epsilon", 1e-6))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("fusion epsilon must be finite and positive")
    if fusion.get("score_normalization", "zscore") != "zscore":
        raise ValueError("late fusion requires score_normalization: zscore")
    return (scores - scores.mean(-1, keepdim=True)) / scores.std(
        -1, keepdim=True, unbiased=False
    ).clamp_min(eps)


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
    # Load both official JIT archives and raw state dicts. The pinned upstream
    # loader does not rewind after a failed JIT load, breaking raw state dicts.
    from clip.clip import _transform
    from clip.model import build_model

    try:
        state = torch.jit.load(str(checkpoint), map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(state).to(device)
    preprocess = _transform(model.visual.input_resolution)
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
    mode = canonical_mode(mode)
    if mode == "clip_image":
        return query_images @ gallery.T
    if mode == "clip_text":
        return query_text @ gallery.T
    iw, tw = fusion_weights(fusion)
    if mode == "early_fusion":
        query = F.normalize(iw * query_images + tw * query_text, dim=-1)
        return query @ gallery.T
    return iw * late_normalize(query_images @ gallery.T, fusion) + tw * late_normalize(
        query_text @ gallery.T, fusion
    )


def _read_array(path: Path, rows: int, columns: int | None = None):
    if not path.is_file():
        return None
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError):
        return None
    if (
        value.ndim != 2
        or value.shape[0] != rows
        or value.shape[1] < 1
        or value.dtype != np.float32
        or (columns is not None and value.shape[1] != columns)
    ):
        return None
    return value


def _save_array(path: Path, value: np.ndarray):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


@torch.inference_mode()
def _branch_scores(path, queries, gallery, batch_size, device):
    cached = _read_array(path, len(queries), len(gallery))
    if cached is not None:
        return cached
    temporary = path.with_suffix(".tmp")
    scores = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(queries), len(gallery))
    )
    targets = torch.from_numpy(np.array(gallery, copy=True)).to(device)
    for start in range(0, len(queries), batch_size):
        query = torch.from_numpy(
            np.array(queries[start : start + batch_size], copy=True)
        )
        scores[start : start + batch_size] = (
            (query.to(device) @ targets.T).cpu().numpy()
        )
    scores.flush()
    del scores, targets
    temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def prepare_clip_inputs(data, samples, gallery_ids, cfg, device, modes=None):
    """Cache image/text features and branch scores once; never use gold labels.

    Fusion subsequently needs only these arrays and image/text weights. A warm
    cache does not load the encoder or require a second model checkpoint.
    """
    modes = [canonical_mode(m) for m in (modes or [cfg["mode"]])]
    need_image = any(m != "clip_text" for m in modes)
    need_text = any(m != "clip_image" for m in modes)
    runtime = cfg["runtime"]
    checkpoint = Path(cfg["model"]["checkpoint"])
    signature = {
        "version": "clip-rcr-v2",
        "model": cfg["model"]["name"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess": "pinned-openai-clip-default",
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
        "images": image_signature(data, gallery_ids),
    }
    directory = cache_directory(cfg["cache"]["dir"], signature)
    feature_path = directory / "gallery.npy"
    gallery = _read_array(feature_path, len(gallery_ids))
    texts, text_features, text_dir = None, None, None
    truncated = 0
    if need_text:
        field = cfg.get("text_field", "final_instruction")
        if field not in ("final_instruction", "final_change"):
            raise ValueError("text_field must be final_instruction or final_change")
        texts = [sample[field] for sample in samples]
        text_dir = cache_directory(directory / "text", {"field": field, "texts": texts})
        text_features = _read_array(
            text_dir / "features.npy",
            len(samples),
            None if gallery is None else gallery.shape[1],
        )
        metadata = text_dir / "text.json"
        if text_features is not None and metadata.is_file():
            truncated = json.loads(metadata.read_text())["truncated_text_queries"]
        else:
            text_features = None
    if gallery is None or (need_text and text_features is None):
        clip_module, model, preprocess = load_clip(checkpoint, device)
        expected_dim = int(model.text_projection.shape[1])
        if gallery is None:
            gallery = encode_images(
                model,
                preprocess,
                [data.image_path(x) for x in gallery_ids],
                runtime,
                device,
            )
            if gallery.shape != (len(gallery_ids), expected_dim):
                raise ValueError("CLIP gallery feature shape does not match the model")
            _save_array(feature_path, gallery)
        if need_text and text_features is None:
            text_features, truncated = encode_texts(
                clip_module, model, texts, int(runtime["text_batch_size"]), device
            )
            if text_features.shape != (len(samples), expected_dim):
                raise ValueError("CLIP text feature shape does not match the model")
            _save_array(text_dir / "features.npy", text_features)
            metadata.write_text(
                json.dumps({"truncated_text_queries": truncated}) + "\n"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    inputs = {}
    if need_image:
        by_id = {image_id: i for i, image_id in enumerate(gallery_ids)}
        query_ids = [sample["query_image_id"] for sample in samples]
        queries = np.asarray(gallery[[by_id[x] for x in query_ids]])
        query_dir = cache_directory(directory / "image", {"query_image_ids": query_ids})
        inputs["image_features"] = queries
        inputs["image"] = _branch_scores(
            query_dir / "scores.npy",
            queries,
            gallery,
            int(runtime["score_batch_size"]),
            device,
        )
    if need_text:
        inputs["text_features"] = text_features
        inputs["text"] = _branch_scores(
            text_dir / "scores.npy",
            text_features,
            gallery,
            int(runtime["score_batch_size"]),
            device,
        )
    details = {
        "checkpoint_sha256": signature["checkpoint_sha256"],
        "gallery_cache": str(directory),
        "text_cache": str(text_dir) if text_dir is not None else None,
        "truncated_text_queries": truncated,
        "rcr_training": False,
    }
    return inputs, details


def iter_clip_scores(inputs, mode, fusion, batch_size):
    """Mix cached branch scores on CPU, without repeating matrix multiplication."""
    mode = canonical_mode(mode)
    base = inputs["text"] if mode == "clip_text" else inputs["image"]
    if mode in ("early_fusion", "late_fusion"):
        iw, tw = fusion_weights(fusion)
    for start in range(0, len(base), batch_size):
        stop = min(start + batch_size, len(base))
        if mode in ("clip_image", "clip_text"):
            yield start, np.asarray(base[start:stop])
            continue
        image = torch.from_numpy(np.array(inputs["image"][start:stop], copy=True))
        text = torch.from_numpy(np.array(inputs["text"][start:stop], copy=True))
        if mode == "early_fusion":
            qi = torch.from_numpy(
                np.array(inputs["image_features"][start:stop], copy=True)
            )
            qt = torch.from_numpy(
                np.array(inputs["text_features"][start:stop], copy=True)
            )
            # (normalize(iw*qi + tw*qt)) @ gallery.T, by linearity.
            norm = (iw * qi + tw * qt).norm(dim=-1, keepdim=True).clamp_min(1e-12)
            mixed = (iw * image + tw * text) / norm
        else:
            mixed = iw * late_normalize(image, fusion) + tw * late_normalize(
                text, fusion
            )
        yield start, mixed.numpy()


def write_clip_scores(inputs, cfg):
    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    mode = canonical_mode(cfg["mode"])
    base = inputs["text"] if mode == "clip_text" else inputs["image"]
    temporary = output / "scores.npy.tmp"
    scores = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=base.shape
    )
    for start, batch in iter_clip_scores(
        inputs, mode, cfg.get("fusion", {}), int(cfg["runtime"]["score_batch_size"])
    ):
        scores[start : start + len(batch)] = batch
    scores.flush()
    del scores
    temporary.replace(output / "scores.npy")
    return np.load(output / "scores.npy", mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def retrieve_clip(data, samples: list[dict], gallery_ids: list[str], cfg: dict, device):
    inputs, details = prepare_clip_inputs(data, samples, gallery_ids, cfg, device)
    return write_clip_scores(inputs, cfg), details
