"""RCR scene/Subject adaptation for native single-person FAFA.

Only query images, public Subject IDs/case structure and final text are used.
Predicted set membership is a heuristic, not GT identity supervision. Relations
remain text; independent FAFA components do not provide joint relation reasoning.
"""

from __future__ import annotations

import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from rcr.baselines.clip import load_clip
from rcr.common.data import sample_selection_texts
from rcr.common.io import cache_directory, image_signature, sha256_file

ADAPTER_VERSION = "rcr-fafa-predicted-set-v1"


def subject_changes(sample: dict) -> list[str]:
    """Split DUAL clauses at Subject boundaries; preserve relational clauses."""
    subject_ids = [subject["subject_id"] for subject in sample["subjects"]]
    text = sample["final_change"].removeprefix("then retrieve target images where ")
    parts = re.split(r"\s+and\s+(?=Subject\s+\d+\b)", text)
    by_id = {}
    if sample["case_type"] == "DUAL":
        for part in parts:
            match = re.match(r"Subject\s+(\d+)\b", part)
            if match:
                by_id[int(match[1])] = part
    # If the text cannot be decomposed safely, each Subject receives the intact
    # condition. Never guess a missing subject-specific modification.
    independent = set(by_id) == set(subject_ids)
    changes = []
    for subject_id in subject_ids:
        condition = by_id[subject_id] if independent else text

        def replace(match, current_id=subject_id):
            if int(match[1]) == current_id:
                return "the reference person"
            return f"the other referenced person (Subject {match[1]})"

        changes.append(re.sub(r"\bSubject\s+(\d+)\b", replace, condition))
    return changes


def select_subject_members(
    similarity: np.ndarray,
    *,
    threshold: float,
    margin: float,
    max_members: int,
    allow_multiple: bool,
    valid_extra: np.ndarray | None = None,
) -> list[list[int]]:
    """Hungarian anchors, then disjoint high-similarity predicted extra members.

    Anchor assignment is maximum-sum one-to-one matching. An unused crop is an
    extra member when its cosine score passes the absolute threshold and lies
    within ``margin`` of that Subject's best score. Competing Subjects assign it
    to the highest-scoring eligible Subject. Cardinality is never read from GT.
    """
    if similarity.ndim != 2 or not np.isfinite(similarity).all():
        raise ValueError("selection similarities must be a finite matrix")
    ns, npersons = similarity.shape
    if ns < 1 or npersons < ns:
        raise ValueError("pad missing query candidates before Subject selection")
    if not np.isfinite([threshold, margin]).all() or margin < 0 or max_members < 1:
        raise ValueError("invalid predicted membership settings")
    rows, cols = linear_sum_assignment(-similarity)
    selected = [[] for _ in range(ns)]
    for row, col in zip(rows, cols, strict=True):
        selected[int(row)].append(int(col))
    if not allow_multiple:
        return selected
    used = set(cols.tolist())
    row_best = similarity.max(axis=1)
    valid_extra = np.ones(npersons, dtype=bool) if valid_extra is None else valid_extra
    # Stable high-confidence order makes the per-Subject cap deterministic.
    order = np.argsort(-similarity.max(axis=0), kind="stable")
    for col in order:
        if col in used or not valid_extra[col]:
            continue
        eligible = [
            row
            for row in range(ns)
            if len(selected[row]) < max_members
            and similarity[row, col] >= threshold
            and similarity[row, col] >= row_best[row] - margin
        ]
        if eligible:
            row = max(eligible, key=lambda x, c=col: (similarity[x, c], -x))
            selected[row].append(int(col))
    return selected


def setmatch_score(matrix: np.ndarray, unmatched_score: float = -1.0) -> float:
    """Native adapter rule: max-sum Hungarian, then minimum matched score."""
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("SetMatch needs a finite, non-empty component score matrix")
    if not np.isfinite(unmatched_score):
        raise ValueError("unmatched_score must be finite")
    ncomponents, npersons = matrix.shape
    if npersons < ncomponents:
        matrix = np.concatenate(
            [matrix, np.full((ncomponents, ncomponents - npersons), unmatched_score)],
            axis=1,
        )
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].min())


def choose_boxes(candidates: list[dict], threshold: float, required: int) -> list[dict]:
    selected = [box for box in candidates if box["score"] >= threshold]
    for box in candidates:
        if len(selected) >= required:
            break
        if box not in selected:
            selected.append(box)
    return selected


@torch.inference_mode()
def detect_gallery(data, gallery_ids: list[str], cfg: dict, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import pil_to_tensor

    if cfg["backend"] != "torchvision_fasterrcnn_resnet50_fpn_v2":
        raise ValueError("unsupported FAFA detector backend")
    if int(cfg["max_persons_per_image"]) < 1:
        raise ValueError("max_persons_per_image must be positive")

    signature = {
        "version": ADAPTER_VERSION,
        "detector": cfg,
        "device_type": device.type,
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
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(
        torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    candidates = []
    for image_id in tqdm(gallery_ids, desc="FAFA person detection", unit="image"):
        with Image.open(data.image_path(image_id)) as source:
            image = source.convert("RGB")
            prediction = model([pil_to_tensor(image).float().div(255).to(device)])[0]
            boxes = []
            order = prediction["scores"].argsort(descending=True)
            for index in order.tolist():
                if int(prediction["labels"][index]) != 1:
                    continue
                x1, y1, x2, y2 = prediction["boxes"][index].tolist()
                x1, y1 = (
                    max(0, min(x1, image.width - 1)),
                    max(0, min(y1, image.height - 1)),
                )
                x2 = max(x1 + 1, min(x2, image.width))
                y2 = max(y1 + 1, min(y2, image.height))
                boxes.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "score": float(prediction["scores"][index]),
                        "fallback": False,
                    }
                )
                if len(boxes) == int(cfg["max_persons_per_image"]):
                    break
            if not boxes:
                boxes = [
                    {
                        "box": [0, 0, image.width, image.height],
                        "score": -1.0,
                        "fallback": True,
                    }
                ]
            candidates.append(boxes)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(candidates), encoding="utf-8")
    temporary.replace(path)
    return candidates, directory


@torch.inference_mode()
def build_query_components(data, samples, gallery_ids, candidates, cfg, device):
    """Return (path, predicted box, relative text, Subject ID) for each query."""
    clip_module, model, preprocess = load_clip(
        cfg["query_selector"]["checkpoint"], device
    )
    settings = cfg["query_selector"]
    by_id = {image_id: index for index, image_id in enumerate(gallery_ids)}
    grouped = []
    membership_counts = []
    fallback_slots = 0
    truncated_texts = 0
    for sample in tqdm(samples, desc="FAFA reference selection", unit="query"):
        texts = sample_selection_texts(sample)
        nsubjects = len(texts)
        index = by_id[sample["query_image_id"]]
        boxes = choose_boxes(
            candidates[index], float(cfg["detector"]["score_threshold"]), nsubjects
        )
        path = data.image_path(sample["query_image_id"])
        with Image.open(path) as source:
            image = source.convert("RGB")
            while len(boxes) < nsubjects:
                boxes.append(
                    {
                        "box": [0, 0, image.width, image.height],
                        "score": -1.0,
                        "fallback": True,
                    }
                )
            crops = torch.stack([preprocess(image.crop(box["box"])) for box in boxes])
        for text in texts:
            try:
                clip_module.tokenize([text], truncate=False)
            except RuntimeError:
                truncated_texts += 1
        image_features = F.normalize(
            model.encode_image(crops.to(device)).float(), dim=-1
        )
        tokens = clip_module.tokenize(texts, truncate=True).to(device)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        similarity = (text_features @ image_features.T).cpu().numpy()
        memberships = select_subject_members(
            similarity,
            threshold=float(settings["membership_threshold"]),
            margin=float(settings["membership_margin"]),
            max_members=int(settings["max_members_per_subject"]),
            allow_multiple=sample["case_type"] != "INDIVIDUAL",
            valid_extra=np.array([not box["fallback"] for box in boxes]),
        )
        changes = subject_changes(sample)
        components = []
        for subject, change, members in zip(
            sample["subjects"], changes, memberships, strict=True
        ):
            membership_counts.append(len(members))
            for member in members:
                fallback_slots += int(boxes[member]["fallback"])
                components.append(
                    (str(path), boxes[member]["box"], change, subject["subject_id"])
                )
        grouped.append(components)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return grouped, {
        "predicted_members_per_subject": membership_counts,
        "full_scene_fallback_components": fallback_slots,
        "truncated_selection_texts": truncated_texts,
        "policy": "Hungarian anchors + disjoint threshold/margin set membership",
        "uses_gt_boxes_or_identities": False,
    }
