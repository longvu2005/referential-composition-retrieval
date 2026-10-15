"""Read frozen FAFA multimodal queries and projected target tokens.

Crop order is inherited from the existing person cache. Query features are
indexed by public instruction/Subject slots, never by selected members or GT.
"""

import hashlib
import json
import re

import torch
from torch import Tensor

NATIVE_VERSION = "fafa-multimodal-fda-v1"


def subject_conditions(sample: dict) -> list[str]:
    """Split only independent, explicit Subject clauses; keep relations intact.

    No case label, GT identity, membership or member count is consulted. The
    complete original change still goes to the RCR Context Reasoner.
    """
    ids = [subject["subject_id"] for subject in sample["subjects"]]
    text = re.sub(
        r"^then retrieve target images where\s+",
        "",
        sample["final_change"],
        flags=re.IGNORECASE,
    ).strip()
    parts = re.split(
        r"\s+and\s+(?=(?:the\s+)?(?:members\s+of\s+)?Subject\s+\d+\b)", text
    )
    clauses = {}
    for part in parts:
        start = re.match(r"(?:the\s+)?(?:members\s+of\s+)?Subject\s+(\d+)\b", part)
        mentions = set(map(int, re.findall(r"\bSubject\s+(\d+)\b", part)))
        if (
            not start
            or not part[start.end() :].strip(" .,;")
            or mentions != {int(start[1])}
            or int(start[1]) in clauses
        ):
            clauses = {}
            break
        clauses[int(start[1])] = part
    independent = set(clauses) == set(ids)
    result = []
    for sid in ids:

        def replace(match, current=sid):
            return (
                "the reference person"
                if int(match[1]) == current
                else f"the other referenced person (Subject {match[1]})"
            )

        condition = re.sub(
            r"(?:the\s+members\s+of\s+|members\s+of\s+)?\bSubject\s+(\d+)\b",
            replace,
            clauses[sid] if independent else text,
        )
        # Per-crop FAFA describes one person even when a public Subject is a group.
        # Keep the condition itself; only fix adjacent plural auxiliary agreement.
        condition = re.sub(
            r"(the reference person|the other referenced person \(Subject \d+\)) "
            r"(are|were|have)\b",
            lambda match: (
                match[1] + " " + {"are": "is", "were": "was", "have": "has"}[match[2]]
            ),
            condition,
        )
        result.append(condition)
    return result


def query_key(sample: dict) -> str:
    """Invalidate stale text or reordered Subject slots without reading labels."""
    value = {
        "version": NATIVE_VERSION,
        "query_image_id": sample["query_image_id"],
        "subject_ids": [row["subject_id"] for row in sample["subjects"]],
        "final_change": sample["final_change"],
        "conditions": subject_conditions(sample),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class NativeFAFACache:
    """Small manifest in RAM, native features read only for requested pairs."""

    def __init__(self, index: dict, gallery) -> None:
        self.root = gallery.root
        if index.get("version") != NATIVE_VERSION:
            raise ValueError("native FAFA cache version differs; use a new directory")
        self.cache_id = gallery.cache_id
        self.dim = index["dim"]
        self.tokens = index["tokens"]
        self.queries = index["queries"]
        self.gallery = gallery
        self.metadata = {
            "version": index["version"],
            "cache_id": self.cache_id,
            "spec": gallery.encoder_metadata["person_encoder"],
            "dim": self.dim,
            "tokens": self.tokens,
        }

    def validate_samples(self, samples) -> None:
        for sample in samples:
            entry = self.queries.get(sample["sample_id"])
            if entry is None or entry["key"] != query_key(sample):
                raise ValueError(
                    f"{sample['sample_id']}: native FAFA query missing/stale; "
                    "build a new native cache for changed instructions"
                )

    def _read(self, group, index, key, shape):
        path = self.root / group / f"{index}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True)
        features = item["features"]
        if (
            item.get("cache_id") != self.cache_id
            or item.get("key") != key
            or features.shape != shape
            or not torch.isfinite(features).all()
        ):
            raise ValueError(f"invalid native FAFA feature shard: {path}")
        return features

    def load_queries(self, samples, max_people: int) -> Tensor:
        self.validate_samples(samples)
        subjects = max(len(row["subjects"]) for row in samples)
        out = torch.zeros(len(samples), subjects, max_people, self.dim)
        for n, sample in enumerate(samples):
            entry = self.queries[sample["sample_id"]]
            count = int(
                self.gallery.mask[self.gallery.by_id[sample["query_image_id"]]].sum()
            )
            if count > max_people:
                raise ValueError("native/query person padding differs")
            shape = (len(sample["subjects"]), count, self.dim)
            out[n, : shape[0], :count] = self._read(
                "queries", entry["index"], entry["key"], shape
            )
        return out

    def load_targets(self, indices: Tensor, max_people: int) -> Tensor:
        out = torch.zeros(len(indices), max_people, self.tokens, self.dim)
        # Repeated positives/candidates are read once per batch.
        rows = {}
        for n, i in enumerate(indices.tolist()):
            count = int(self.gallery.mask[i].sum())
            if count > max_people:
                raise ValueError("native/target person padding differs")
            if i not in rows:
                rows[i] = self._read(
                    "targets",
                    i,
                    self.gallery.image_ids[i],
                    (count, self.tokens, self.dim),
                )
            out[n, :count] = rows[i]
        return out
