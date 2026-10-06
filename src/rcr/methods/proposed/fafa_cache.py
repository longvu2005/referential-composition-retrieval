"""Frozen, image-only FAFA person extraction; imported only in the FAFA venv."""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from rcr.methods.baselines.fafa import load_fafa
from rcr.methods.common.experiment import resolve_device
from rcr.methods.common.results import sha256_file
from rcr.methods.proposed.person_encoder import fafa_spec, load_fafa_config


@torch.inference_mode()
def extract_person_features(model, pixels: torch.Tensor) -> torch.Tensor:
    """Mean hidden Q-Former image tokens, before the native vision projection."""
    features = model.extract_features({"image": pixels}, mode="image").image_embeds
    if features.ndim != 3:
        raise ValueError("FAFA image_embeds must be [B,query_tokens,hidden_dim]")
    return features.float().mean(dim=1)


@torch.inference_mode()
def finish_fafa_cache(cfg: dict) -> None:
    root = Path(cfg["data"]["cache"])
    marker = root / ".building"
    if not marker.is_file():
        raise ValueError("FAFA extraction requires an unfinished scene cache")
    index = torch.load(root / "index.pt", map_location="cpu", weights_only=True)
    if marker.read_text(encoding="utf-8") != index["cache_id"]:
        raise ValueError("unfinished cache ID differs from index.pt")
    native_cfg = load_fafa_config(cfg)
    device = resolve_device(cfg)
    batch_size = cfg["person_encoder"]["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("person_encoder.batch_size must be positive")
    model, _, preprocess, _ = load_fafa(native_cfg, device)
    model.requires_grad_(False).eval()
    dim = model.Qformer.config.hidden_size
    dtype = index["persons"].dtype
    mask = index["mask"].bool()
    persons_index = torch.zeros(*mask.shape, dim, dtype=dtype)
    # The pending index has zero-width persons. Free it before collecting features.
    del index["persons"]
    for i, path in enumerate(tqdm(index["image_paths"], desc="cache FAFA persons")):
        feature_path = root / "features" / f"{i}.pt"
        item = torch.load(feature_path, map_location="cpu", weights_only=True)
        count = len(item["boxes_pixel"])
        if item["cache_id"] != index["cache_id"] or count != int(mask[i].sum()):
            raise ValueError("FAFA scene/person manifest mismatch")
        features = torch.empty(count, dim, dtype=dtype)
        if count:
            with Image.open(path) as source:
                image = source.convert("RGB")
                for start in range(0, count, batch_size):
                    boxes = item["boxes_pixel"][start : start + batch_size]
                    pixels = torch.stack(
                        [preprocess(image.crop(tuple(box.tolist()))) for box in boxes]
                    ).to(device)
                    # The official EVA visual trunk uses fp16 on CUDA, fp32 on CPU.
                    pixels = pixels.half() if device.type == "cuda" else pixels.float()
                    pooled = extract_person_features(model, pixels)
                    if (
                        pooled.shape != (len(boxes), dim)
                        or not torch.isfinite(pooled).all()
                    ):
                        raise ValueError("invalid FAFA person features")
                    features[start : start + len(boxes)] = pooled.cpu().to(dtype)
        item["persons"] = features
        persons_index[i, :count] = features
        temporary = feature_path.with_suffix(".pt.tmp")
        torch.save(item, temporary)
        temporary.replace(feature_path)
    index["persons"] = persons_index
    index["person_encoder"] = {
        "backend": "fafa",
        "spec": fafa_spec(native_cfg),
        "checkpoint_sha256": sha256_file(native_cfg["checkpoint"]["path"]),
        "hidden_dim": dim,
    }
    temporary = root / "index.pt.tmp"
    torch.save(index, temporary)
    temporary.replace(root / "index.pt")
    marker.unlink()  # Publish only after every person feature and the index agree.
