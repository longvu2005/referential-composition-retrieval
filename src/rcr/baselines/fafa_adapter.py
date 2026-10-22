"""RCR scene/Subject adaptation for native single-person FAFA.

Only query images, public Subject slots and final text are used.
Predicted set membership is a heuristic, not GT identity supervision. Relations
remain text; independent FAFA components do not provide joint relation reasoning.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from rcr.baselines.clip import load_clip
from rcr.common.data import sample_selection_texts

ADAPTER_VERSION = "rcr-fafa-cpr-partial-setmatch-v3"
VISUAL_CACHE_VERSION = "rcr-fafa-groundingdino-v2"


def subject_changes(sample: dict) -> list[str]:
    """Replicate CPR Benchmark's case-free native modification text routing.

    Independent clauses lose their ``Subject N`` prefix. Relational/ambiguous
    clauses remain intact for each Subject; no identity or case labels are used.
    """
    ids = [int(subject["subject_id"]) for subject in sample["subjects"]]
    text = sample["final_change"].removeprefix(
        "then retrieve target images where "
    ).strip()
    mentions = list(re.finditer(r"\bSubject\s+(\d+)\b", text, re.IGNORECASE))
    independent = (
        len(mentions) == len(ids)
        and {int(m[1]) for m in mentions} == set(ids)
        and bool(mentions)
        and mentions[0].start() == 0
    )
    result = {}
    if independent:
        for i, mention in enumerate(mentions):
            end = mentions[i + 1].start() if i + 1 < len(mentions) else len(text)
            fragment = text[mention.end() : end].strip()
            if i + 1 < len(mentions):
                connector = re.search(r"(?:[,;]?\s+and|[,;]?\s+while|[,;])\s*$",
                                      fragment, re.IGNORECASE)
                if connector is None:
                    independent = False
                    break
                fragment = fragment[: connector.start()].strip()
            if not fragment:
                independent = False
                break
            result[int(mention[1])] = fragment
    return [result[subject_id] for subject_id in ids] if independent else [text] * len(ids)


def select_subject_members(
    similarity: np.ndarray,
    *,
    threshold: float,
    margin: float,
    max_members: int,
    valid_extra: np.ndarray | None = None,
) -> list[list[int]]:
    """Hungarian anchors, then disjoint high-similarity predicted extra members.

    Anchor assignment is maximum-sum one-to-one matching. An unused crop is an
    extra member when its cosine score passes the absolute threshold and lies
    within ``margin`` of that Subject's assigned anchor. Competing Subjects assign it
    to the highest-scoring eligible Subject. The same rule applies to every
    query; no case label or GT cardinality is read.
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
    used = set(cols.tolist())
    anchors = {int(row): int(col) for row, col in zip(rows, cols, strict=True)}
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
            and similarity[row, col] >= similarity[row, anchors[row]] - margin
        ]
        if eligible:
            row = max(eligible, key=lambda x, c=col: (similarity[x, c], -x))
            selected[row].append(int(col))
    return selected



def choose_boxes(candidates: list[dict], threshold: float, required: int) -> list[dict]:
    selected = [box for box in candidates if box["score"] >= threshold]
    for box in candidates:
        if len(selected) >= required:
            break
        if box not in selected:
            selected.append(box)
    return selected



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
        "uses_case_type": False,
    }
