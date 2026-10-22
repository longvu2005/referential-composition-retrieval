"""Pinned Grounding DINO person detections used by the CPR FAFA adapter.

Follow cpr_baseline_bench's fixed PyTorch attention, crop, NMS and batching
conventions. Never use annotation boxes or identities during inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rcr.common.io import cache_directory, image_signature, sha256_file

DETECTOR_CACHE_VERSION = "cpr-groundingdino-swin-t-v1"


def _source(cfg: dict, *, prepare: bool = False) -> Path:
    source = cfg["source"]
    path = Path(source["local_checkout"]).resolve()
    if not path.is_dir():
        if not prepare:
            raise FileNotFoundError(
                f"Grounding DINO source missing: {path}; run prepare"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", source["repository"], str(path)], check=True)
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"Grounding DINO source has tracked modifications: {status}")
    current = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != source["commit"] and prepare:
        subprocess.run(
            ["git", "-C", str(path), "fetch", "origin", source["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", source["commit"]],
            check=True,
        )
        current = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    if current != source["commit"]:
        raise RuntimeError("Grounding DINO source commit mismatch; run prepare")
    if sha256_file(path / cfg["config"]) != cfg["config_sha256"]:
        raise RuntimeError("Grounding DINO official config hash mismatch")
    return path


def _hf_cache(cfg: dict, *, offline: bool) -> Path:
    root = Path(cfg["runtime_cache"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(root / "xdg")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)
    return root


def _assets_ready(cfg: dict) -> bool:
    root = Path(cfg["runtime_cache"])
    marker = Path(cfg["runtime_assets_marker"])
    if not marker.is_file():
        return False
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        return (
            meta.get("source_commit") == cfg["source"]["commit"]
            and meta.get("text_encoder") == "bert-base-uncased"
            and bool(meta.get("files"))
            and all(
                (root / row["path"]).is_file()
                and (root / row["path"]).stat().st_size == row["size"]
                for row in meta["files"]
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def prepare_detector(cfg: dict, *, force: bool = False) -> None:
    """Download pinned weights and prewarm BERT once, before offline inference."""
    from rcr.baselines.prepare import download

    _source(cfg, prepare=True)
    download(cfg["checkpoint_url"], Path(cfg["checkpoint"]), force=force)
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    if (not isinstance(state, dict) or not isinstance(state.get("model"), dict)
            or len(state["model"]) < 100):
        raise RuntimeError("Grounding DINO released checkpoint is incomplete")
    del state
    if _assets_ready(cfg) and not force:
        return
    root = _hf_cache(cfg, offline=False)
    from transformers import AutoTokenizer, BertModel

    AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    del model
    files = [
        {"path": str(p.relative_to(root)), "size": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.stat().st_size > 0 and ".locks" not in p.parts
    ]
    if not files:
        raise RuntimeError("Grounding DINO BERT cache is empty")
    marker = Path(cfg["runtime_assets_marker"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"source_commit": cfg["source"]["commit"],
                    "text_encoder": "bert-base-uncased", "files": files}, indent=2),
        encoding="utf-8",
    )


def _import_detector(source: Path) -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    source_path = str(source)
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    import groundingdino
    from groundingdino.models.GroundingDINO import ms_deform_attn as msda

    if Path(groundingdino.__file__).resolve().parent != source / "groundingdino":
        raise RuntimeError("Grounding DINO imported from an unpinned installation")

    class PyTorchMSDeformAttnFunction:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, locations, weights, step):
            return msda.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, locations, weights
            )

    msda.MultiScaleDeformAttnFunction = PyTorchMSDeformAttnFunction


def filter_person_boxes(boxes, confidences, width: int, height: int, cfg: dict):
    """Same deterministic NMS and floor/ceil crop policy as CPR Benchmark."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores) or width < 1 or height < 1:
        raise ValueError("Invalid Grounding DINO outputs")
    boxes *= np.array([width, height, width, height], dtype=np.float32)
    xyxy = np.column_stack((
        boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2,
    ))
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, width)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, height)
    valid = (
        np.isfinite(xyxy).all(axis=1) & np.isfinite(scores)
        & (scores > float(cfg["box_threshold"]))
        & ((xyxy[:, 2] - xyxy[:, 0]) >= float(cfg["min_box_size_px"]))
        & ((xyxy[:, 3] - xyxy[:, 1]) >= float(cfg["min_box_size_px"]))
    )
    indices = sorted(
        np.flatnonzero(valid).tolist(),
        key=lambda i: (-scores[i], *xyxy[i].tolist(), i),
    )
    keep = []
    for i in indices:
        if keep:
            other = xyxy[keep]
            overlap = np.maximum(0, np.minimum(other[:, 2:], xyxy[i, 2:])
                                 - np.maximum(other[:, :2], xyxy[i, :2])).prod(axis=1)
            area = (xyxy[i, 2:] - xyxy[i, :2]).prod()
            union = area + (other[:, 2:] - other[:, :2]).prod(axis=1) - overlap
            if np.any(overlap / union > float(cfg["nms_iou_threshold"])):
                continue
        keep.append(i)
        if len(keep) == int(cfg["max_persons_per_image"]):
            break
    cropped = xyxy[keep]
    cropped[:, :2] = np.floor(cropped[:, :2])
    cropped[:, 2:] = np.ceil(cropped[:, 2:])
    return cropped, scores[keep]


class DetectionImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from groundingdino.util.inference import load_image
        image, tensor = load_image(str(self.paths[i]))
        h, w = image.shape[:2]
        return tensor, w, h


def _collate(batch):
    return tuple(map(list, zip(*batch)))


def _split_batch(tensors, widths, heights, ratio):
    """Match CPR's padding-aware deterministic forward grouping."""
    start = 0
    while start < len(tensors):
        stop = start + 1
        while stop < len(tensors):
            shapes = [x.shape[-2:] for x in tensors[start:stop + 1]]
            padded = max(h for h, w in shapes) * max(w for h, w in shapes) * len(shapes)
            if padded / sum(h * w for h, w in shapes) > ratio:
                break
            stop += 1
        yield tensors[start:stop], widths[start:stop], heights[start:stop]
        start = stop


def _row_digest(row: dict) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _recover(journal: Path, ids: list[str]):
    """Discard an incomplete/corrupt journal suffix before resuming detection."""
    rows = []
    if not journal.is_file():
        return rows
    with journal.open("rb+") as handle:
        valid_end = 0
        while line := handle.readline():
            try:
                row = json.loads(line)
                recorded = row.pop("sha256")
                digest = _row_digest(row)
                if (not line.endswith(b"\n") or recorded != digest
                        or len(rows) >= len(ids) or row["image_id"] != ids[len(rows)]
                        or not row["boxes"] or len(row["boxes"]) != len(row["scores"])):
                    break
                rows.append(row)
                valid_end = handle.tell()
            except (ValueError, TypeError, KeyError):
                break
        handle.truncate(valid_end)
    return rows


@torch.inference_mode()
def detect_gallery(data, gallery_ids: list[str], cfg: dict, device):
    if cfg["backend"] != "groundingdino_swint_ogc":
        raise ValueError("FAFA detector must use pinned Grounding DINO Swin-T")
    if cfg["attention_backend"] != "official_pytorch_fallback":
        raise ValueError("Only the reference PyTorch attention backend is supported")
    signature = {
        "version": DETECTOR_CACHE_VERSION, "config": cfg,
        "checkpoint_sha256": sha256_file(cfg["checkpoint"]),
        "images": image_signature(data, gallery_ids),
    }
    directory = cache_directory(cfg["cache_dir"], signature)
    path = directory / "candidates.json"
    if path.is_file():
        candidates = json.loads(path.read_text(encoding="utf-8"))
        if len(candidates) == len(gallery_ids) and all(candidates):
            print(f"Detector cache: {path}", flush=True)
            return candidates, directory

    source = _source(cfg)
    if not _assets_ready(cfg):
        raise RuntimeError("Grounding DINO BERT assets missing; run prepare")
    _hf_cache(cfg, offline=True)
    _import_detector(source)
    from groundingdino.models import build_model
    from groundingdino.util.inference import preprocess_caption
    from groundingdino.util.misc import clean_state_dict, nested_tensor_from_tensor_list
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(source / cfg["config"]))
    args.device = str(device)
    args.use_checkpoint = args.use_transformer_ckpt = False
    model = build_model(args)
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    del state
    model.to(device).eval()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    journal = directory / "candidates.partial.jsonl"
    rows = _recover(journal, gallery_ids)
    batch_size = int(cfg["batch_size"])
    if batch_size < 1:
        raise ValueError("Detector batch size must be positive")
    # Recover at a full forward batch boundary, preserving padding/numerics.
    if len(rows) % batch_size:
        rows = rows[:len(rows) - len(rows) % batch_size]
        with journal.open("w", encoding="utf-8") as f:
            for row in rows:
                digest = _row_digest(row)
                f.write(json.dumps({**row, "sha256": digest}) + "\n")
    paths = [data.image_path(image_id) for image_id in gallery_ids[len(rows):]]
    loader = DataLoader(DetectionImages(paths), batch_size=batch_size, shuffle=False,
                        num_workers=int(cfg["num_workers"]), collate_fn=_collate)
    caption = preprocess_caption(cfg["text_prompt"])
    with journal.open("a", encoding="utf-8") as handle:
        for batch in tqdm(loader, desc="FAFA Grounding DINO", unit="batch"):
            tensors, widths, heights = batch
            for ts, ws, hs in _split_batch(
                tensors, widths, heights, float(cfg["max_padding_ratio"])
            ):
                output = model(
                    nested_tensor_from_tensor_list([t.to(device) for t in ts]),
                    captions=[caption] * len(ts),
                )
                conf = output["pred_logits"].sigmoid().max(dim=-1).values.cpu().numpy()
                boxes = output["pred_boxes"].cpu().numpy()
                for predicted, confidence, width, height in zip(
                    boxes, conf, ws, hs, strict=True
                ):
                    b, s = filter_person_boxes(
                        predicted, confidence, width, height, cfg
                    )
                    fallback = not len(b)
                    if fallback:
                        b = np.array([[0, 0, width, height]], np.float32)
                        s = np.array([-1.0], np.float32)
                    row = {
                        "image_id": gallery_ids[len(rows)],
                        "boxes": b.tolist(), "scores": s.tolist(),
                        "fallback": [fallback] * len(b),
                    }
                    rows.append(row)
                    digest = _row_digest(row)
                    handle.write(json.dumps({**row, "sha256": digest}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(rows) != len(gallery_ids):
        raise RuntimeError("Grounding DINO detections incomplete")
    candidates = [[{"box": box, "score": score, "fallback": fallback}
                   for box, score, fallback in zip(
                       row["boxes"], row["scores"], row["fallback"], strict=True
                   )]
                  for row in rows]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(candidates), encoding="utf-8")
    temporary.replace(path)
    journal.unlink(missing_ok=True)
    return candidates, directory
