"""Pinned official FAFA model, artifact preparation and native FDA scoring.

FAFA network code is imported unchanged from the authors' checkout. Scene/Subject
adaptation lives in fafa_adapter.py, set matching in fafa_matching.py and metrics
in evaluation/.
"""

from __future__ import annotations

import gc
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rcr.baselines.fafa_adapter import (
    ADAPTER_VERSION,
    VISUAL_CACHE_VERSION,
    build_query_components,
    choose_boxes,
)
from rcr.baselines.fafa_detector import detect_gallery
from rcr.baselines.fafa_matching import aggregate_people
from rcr.common.io import (
    cache_directory,
    image_signature,
    output_directory,
    sha256_file,
    write_json,
)


def official_source(cfg: dict, *, prepare: bool = False) -> Path:
    source = cfg["source"]
    checkout = Path(source["local_checkout"]).resolve()
    if not checkout.exists():
        if not prepare:
            raise FileNotFoundError(
                "FAFA source missing; run the prepare command first"
            )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", source["repository"], str(checkout)], check=True
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    real_changes = [
        line for line in dirty.splitlines() if not line.endswith((".pyc", ".pyo"))
    ]
    if real_changes:
        raise RuntimeError(
            "Pinned FAFA source has tracked edits:\n" + "\n".join(real_changes)
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source["commit"] and prepare:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "origin", source["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", source["commit"]],
            check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != source["commit"]:
        raise RuntimeError("FAFA source commit mismatch; run the prepare command")
    directory = checkout / source.get("subdir", "FAFA_SynCPR")
    if not (directory / "src").is_dir():
        raise FileNotFoundError(directory / "src")
    return directory


def runtime_cache(cfg: dict, *, offline: bool) -> Path:
    root = Path(cfg["checkpoint"]["cache_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root / "torch")
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(root / "xdg")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)
    return root


def official_api(directory: Path):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    src = str(directory / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from data_utils import squarepad_transform_test
    from lavis.models import load_model_and_preprocess

    return squarepad_transform_test, load_model_and_preprocess


def marker_identity(cfg: dict) -> dict:
    return {
        "source_commit": cfg["source"]["commit"],
        "model_name": cfg["checkpoint"]["model_name"],
        "model_type": cfg["checkpoint"]["model_type"],
    }


def assets_ready(cfg: dict) -> bool:
    marker = Path(cfg["checkpoint"]["runtime_assets_marker"])
    root = Path(cfg["checkpoint"]["cache_root"])
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in marker_identity(cfg).items()):
            return False
        files = saved["files"]
        return bool(files) and all(
            (root / item["path"]).is_file()
            and (root / item["path"]).stat().st_size == item["size"]
            for item in files
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def prepare_runtime_assets(cfg: dict, *, force: bool = False) -> None:
    directory = official_source(cfg, prepare=True)
    if assets_ready(cfg) and not force:
        print("FAFA runtime assets already prepared", flush=True)
        return
    root = runtime_cache(cfg, offline=False)
    _, load_model = official_api(directory)
    print("Preparing official FAFA runtime assets on CPU", flush=True)
    model, _, _ = load_model(
        name=cfg["checkpoint"]["model_name"],
        model_type=cfg["checkpoint"]["model_type"],
        is_eval=True,
        device="cpu",
    )
    del model
    gc.collect()
    files = [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise RuntimeError("Official model loaded but the runtime cache is empty")
    marker = Path(cfg["checkpoint"]["runtime_assets_marker"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({**marker_identity(cfg), "files": files}, indent=2), encoding="utf-8"
    )


@contextmanager
def offline_model_load():
    """Fail on missing runtime assets instead of downloading during inference."""
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def blocked(self, address):
        raise RuntimeError("Network blocked in FAFA inference; run the prepare command")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def text_policy(cfg: dict) -> dict[str, int]:
    """Resolve query-only text limits from the pinned FAFA defaults.

    The loader defaults to 32 tokens and its caption processor to 50 words. A
    word cap at least as large as the token cap leaves truncation to the tokenizer.
    """
    length = cfg.get("checkpoint", {}).get("max_txt_len", 32)
    if type(length) is not int or not 2 <= length <= 512:
        raise ValueError("checkpoint.max_txt_len must be an integer in [2, 512]")
    return {"max_txt_len": length, "max_words": max(50, length)}


def load_fafa(cfg: dict, device):
    policy = text_policy(cfg)
    directory = official_source(cfg)
    if not assets_ready(cfg):
        raise RuntimeError("FAFA runtime assets missing/stale; run the prepare command")
    runtime_cache(cfg, offline=True)
    squarepad, load_model = official_api(directory)
    with offline_model_load():
        model, _, processors = load_model(
            name=cfg["checkpoint"]["model_name"],
            model_type=cfg["checkpoint"]["model_type"],
            is_eval=True,
            device=str(device),
        )
    # The authors' released checkpoint is a trusted prepared model artifact.
    checkpoint = torch.load(
        cfg["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not set(state) & set(model.state_dict()):
        raise ValueError("FAFA checkpoint has no matching model parameters")
    # The released checkpoint uses the upstream non-strict loading policy.
    incompatible = model.load_state_dict(state, strict=False)
    model.fda_k = int(cfg["checkpoint"]["fda_k"])
    model.fda_alpha = float(cfg["checkpoint"]["fda_alpha"])
    model.use_soft = bool(cfg["checkpoint"]["use_soft"])
    model.max_txt_len = policy["max_txt_len"]
    processors["eval"].max_words = policy["max_words"]
    print(
        f"FAFA text: max_txt_len={model.max_txt_len} tokens, "
        f"processor.max_words={processors['eval'].max_words}",
        flush=True,
    )
    preprocess = squarepad(
        int(cfg["checkpoint"]["image_size"]),
        need_size=tuple(cfg["checkpoint"]["test_resize_hw"]),
    )
    return (
        model.to(device).eval(),
        processors,
        preprocess,
        {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "text_policy": policy,
        },
    )


class CropDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, box, caption = self.items[index]
        with Image.open(path) as image:
            crop = image.convert("RGB").crop(box)
            return self.preprocess(crop), caption


def _cached_features(path, count, ndim, dtype):
    """Reuse complete NPY arrays only when rank, count and storage precision agree."""
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if (
        value.ndim == ndim
        and len(value) == count
        and all(size > 0 for size in value.shape[1:])
        and value.dtype == np.dtype(dtype)
    ):
        return value
    return None


@torch.inference_mode()
def encode_crops(model, processors, preprocess, items, runtime, device, *, query, path):
    """Reuse compatible arrays; atomically replace missing or invalid features."""
    dtype = np.float32 if query else np.dtype(runtime["gallery_feature_dtype"])
    cached = _cached_features(path, len(items), 2 if query else 3, dtype)
    if cached is not None:
        print(f"FAFA feature cache: {path}", flush=True)
        return cached
    loader = DataLoader(
        CropDataset(items, preprocess),
        batch_size=int(runtime["query_batch_size" if query else "image_batch_size"]),
        num_workers=int(runtime.get("num_workers", 0)),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    features = None
    cursor = 0
    temporary = path.with_suffix(".tmp")
    for images, captions in tqdm(
        loader, desc="FAFA query" if query else "FAFA gallery", unit="batch"
    ):
        images = images.to(device)
        images = images.half() if device.type == "cuda" else images.float()
        if query:
            features_batch = model.extract_features(
                {
                    "image": images,
                    "text_input": [processors["eval"](caption) for caption in captions],
                }
            ).multimodal_embeds.float()
            if features_batch.ndim == 3 and features_batch.shape[1] == 1:
                features_batch = features_batch[:, 0]
        else:
            features_batch, _ = model.extract_target_features(images, mode="mean")
            features_batch = features_batch.float()
            if features_batch.ndim == 2:
                features_batch = features_batch[:, None]
        if features_batch.ndim != (2 if query else 3):
            raise ValueError("Unexpected official FAFA feature shape")
        if features is None:
            features = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=dtype,
                shape=(len(items), *features_batch.shape[1:]),
            )
        features[cursor : cursor + len(images)] = features_batch.cpu().numpy()
        cursor += len(images)
    if features is None or cursor != len(items):
        raise ValueError("FAFA crop dataset is empty or incomplete")
    features.flush()
    del features
    temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def fda_scores(query: torch.Tensor, targets: torch.Tensor, k: int, use_soft=True):
    """Official inference: normalized query dot target tokens, top-k mean."""
    if k < 1:
        raise ValueError("fda_k must be positive")
    similarity = torch.einsum("qd,pkd->qpk", query, targets)
    if use_soft:
        return similarity.topk(min(k, targets.shape[1]), dim=-1).values.mean(-1)
    return similarity.max(-1).values


@torch.inference_mode()
def score_components(
    query_features, gallery_features, path, runtime, device, checkpoint
):
    cached = _cached_features(path, len(query_features), 2, np.float32)
    if cached is not None and cached.shape[1] == len(gallery_features):
        return cached
    temporary = path.with_suffix(".tmp")
    scores = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_features), len(gallery_features)),
    )
    query = torch.from_numpy(np.array(query_features, copy=True)).to(device)
    pb, qb = (
        int(runtime["score_person_batch_size"]),
        int(runtime["score_query_batch_size"]),
    )
    for start in tqdm(
        range(0, len(gallery_features), pb), desc="FAFA FDA", unit="batch"
    ):
        stop = min(start + pb, len(gallery_features))
        target = torch.from_numpy(
            np.array(gallery_features[start:stop], dtype=np.float32)
        ).to(device)
        for qs in range(0, len(query), qb):
            scores[qs : qs + qb, start:stop] = (
                fda_scores(
                    query[qs : qs + qb],
                    target,
                    int(checkpoint["fda_k"]),
                    checkpoint["use_soft"],
                )
                .cpu()
                .numpy()
            )
    scores.flush()
    del scores
    temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def retrieve_fafa(data, samples, gallery_ids, cfg, device):
    official_source(cfg)
    if not assets_ready(cfg):
        raise RuntimeError(
            "FAFA assets not ready; run the prepare command before retrieval"
        )
    checkpoint_sha = sha256_file(cfg["checkpoint"]["path"])
    selector_sha = sha256_file(cfg["localization"]["query_selector"]["checkpoint"])
    detector = cfg["localization"]["detector"]
    candidates, detection_dir = detect_gallery(data, gallery_ids, detector, device)
    gallery_items = []
    gallery_offsets = [0]
    for image_id, boxes in zip(gallery_ids, candidates, strict=True):
        gallery_items.extend(
            (str(data.image_path(image_id)), box["box"], "")
            for box in choose_boxes(boxes, float(detector["score_threshold"]), 1)
        )
        gallery_offsets.append(len(gallery_items))
    signature = {
        "version": VISUAL_CACHE_VERSION,
        "images": image_signature(data, gallery_ids),
        "checkpoint_sha256": checkpoint_sha,
        "selector_sha256": selector_sha,
        # Text changes only invalidate multimodal query features, not images.
        "checkpoint_settings": {
            key: value
            for key, value in cfg["checkpoint"].items()
            if key != "max_txt_len"
        },
        "runtime_marker_sha256": sha256_file(
            cfg["checkpoint"]["runtime_assets_marker"]
        ),
        "detection_cache": str(detection_dir),
        "gallery_items": gallery_items,
        "gallery_feature_dtype": cfg["runtime"]["gallery_feature_dtype"],
        "encoder_precision": "float16" if device.type == "cuda" else "float32",
    }
    gallery_dir = cache_directory(cfg["cache"]["dir"], signature)
    selection_dir = cache_directory(
        gallery_dir / "selections",
        {
            "version": ADAPTER_VERSION,
            "selector_sha256": selector_sha,
            "selector": cfg["localization"]["query_selector"],
            "detector": cfg["localization"]["detector"],
            "detection_cache": str(detection_dir),
            "query_images": image_signature(
                data,
                list(dict.fromkeys(sample["query_image_id"] for sample in samples)),
            ),
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "query_image_id": sample["query_image_id"],
                    "subject_ids": [
                        int(subject["subject_id"]) for subject in sample["subjects"]
                    ],
                    "final_desc": sample["final_desc"],
                    "final_change": sample["final_change"],
                }
                for sample in samples
            ],
        },
    )
    selection_path = selection_dir / "components.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        grouped = selection["grouped"]
        selector_stats = selection["stats"]
        print("FAFA reference selection cache ready; skipping CLIP", flush=True)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        grouped, selector_stats = build_query_components(
            data, samples, gallery_ids, candidates, cfg["localization"], device
        )
        write_json(
            selection_path,
            {"grouped": grouped, "stats": selector_stats},
        )

    query_items = []
    query_offsets = [0]
    for components in grouped:
        query_items.extend((path, box, caption) for path, box, caption, _ in components)
        query_offsets.append(len(query_items))
    query_signature = {
        "version": ADAPTER_VERSION,
        "query_items": query_items,
        "query_offsets": query_offsets,
        "sample_ids": [s["sample_id"] for s in samples],
    }
    policy = text_policy(cfg)
    if policy != text_policy({}):
        query_signature["text_policy"] = policy
    query_dir = cache_directory(gallery_dir / "queries", query_signature)
    runtime = cfg["runtime"]
    gallery_path = gallery_dir / "gallery.npy"
    query_path = query_dir / "query.npy"

    # Both the warm-cache check and extractor use the same validity policy.
    gallery_features = _cached_features(
        gallery_path, len(gallery_items), 3, runtime["gallery_feature_dtype"]
    )
    query_features = _cached_features(query_path, len(query_items), 2, np.float32)
    if gallery_features is not None and query_features is not None:
        load_info = {
            "missing_keys": None,
            "unexpected_keys": None,
            "text_policy": text_policy(cfg),
            "model_loaded_this_run": False,
        }
        print("FAFA feature caches ready; skipping model construction", flush=True)
    else:
        model, processors, preprocess, load_info = load_fafa(cfg, device)
        print(
            f"FAFA checkpoint: {len(load_info['missing_keys'])} missing / "
            f"{len(load_info['unexpected_keys'])} unexpected keys",
            flush=True,
        )
        gallery_features = encode_crops(
            model,
            processors,
            preprocess,
            gallery_items,
            runtime,
            device,
            query=False,
            path=gallery_path,
        )
        query_features = encode_crops(
            model,
            processors,
            preprocess,
            query_items,
            runtime,
            device,
            query=True,
            path=query_path,
        )
        load_info["model_loaded_this_run"] = True
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    component_scores = score_components(
        query_features,
        gallery_features,
        query_dir / "component_scores.npy",
        runtime,
        device,
        cfg["checkpoint"],
    )
    output = output_directory(cfg)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "scores.npy.tmp"
    scores = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(samples), len(gallery_ids))
    )
    offsets = np.asarray(gallery_offsets, dtype=np.int64)
    for qi in tqdm(range(len(samples)), desc="FAFA Partial SetMatch", unit="query"):
        matrix = component_scores[query_offsets[qi] : query_offsets[qi + 1]]
        scores[qi] = aggregate_people(
            matrix, offsets, float(cfg["setmatch"]["unmatched_score"])
        )
    scores.flush()
    del scores
    temporary.replace(output / "scores.npy")
    return np.load(output / "scores.npy", mmap_mode="r", allow_pickle=False), {
        "adapter_version": ADAPTER_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "selector_checkpoint_sha256": selector_sha,
        "detector_source": "grounding_dino_cache",
        "detector_cache_id": str(detection_dir),
        "source_commit": cfg["source"]["commit"],
        "checkpoint_status": cfg["checkpoint"]["status"],
        "checkpoint_load": load_info,
        "gallery_cache": str(gallery_dir),
        "query_cache": str(query_dir),
        "selector": selector_stats,
        "rcr_training": False,
        "limitations": [
            "Group membership is predicted by CLIP threshold/margin; "
            "no GT cardinality.",
            "FAFA runs independently per predicted member; group and relation "
            "conditions are approximated, not jointly reasoned.",
        ],
    }
