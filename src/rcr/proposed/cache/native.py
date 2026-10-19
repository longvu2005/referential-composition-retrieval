"""Read frozen FAFA multimodal queries and projected target tokens.

Crop order is inherited from the DINO person cache. Query features are
indexed by public instruction/Subject slots, never by selected members or GT.
"""

import hashlib
import json
import re
from pathlib import Path

import torch
from torch import Tensor

from rcr.baselines.fafa import text_policy

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


def condition_key(
    signature: dict,
    image_id: str,
    condition: str,
    policy: dict | None = None,
) -> str:
    """One embedding family, reference crop order and public Subject condition."""
    value = [NATIVE_VERSION, signature, image_id, condition]
    if policy is not None and policy != text_policy({}):
        value.append(policy)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def query_entries(samples, signature: dict, policy: dict | None = None) -> dict:
    return {
        row["sample_id"]: {
            "key": query_key(row),
            "features": [
                condition_key(signature, row["query_image_id"], text, policy)
                for text in subject_conditions(row)
            ],
        }
        for row in samples
    }


def cached_text_policy(index: dict) -> dict[str, int]:
    """Use pinned 32-token/50-word defaults when a manifest has no policy."""
    policy = index.get("text_policy", text_policy({}))
    if (
        not isinstance(policy, dict)
        or set(policy) != {"max_txt_len", "max_words"}
        or any(type(value) is not int for value in policy.values())
        or policy != text_policy({"checkpoint": policy})
    ):
        raise ValueError("invalid FAFA query text policy; rebuild fafa-query")
    return dict(policy)


def checkpoint_native_metadata(metadata: dict | None) -> dict | None:
    """Attach the pinned default policy to checkpoints that omit it."""
    if metadata is None or metadata.get("version") != NATIVE_VERSION:
        return metadata
    return {**metadata, "text_policy": cached_text_policy(metadata)}


class NativeQueryCache:
    """Independent, incrementally extendable FP32 query features."""

    def __init__(self, root: str | Path, signature: dict, dim: int) -> None:
        self.root = Path(root)
        if (self.root / ".building").exists():
            raise RuntimeError(
                f"query cache build is incomplete at {self.root}; resume fafa-query"
            )
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True
        )
        if (
            index.get("version") != NATIVE_VERSION
            or index.get("layout") != "fafa-queries"
        ):
            raise ValueError("unsupported FAFA query cache; use a new directory")
        if index.get("signature") != signature or index.get("dim") != dim:
            raise ValueError("FAFA query encoder/crops differ from image cache")
        self.signature = signature
        self.text_policy = cached_text_policy(index)
        self.dim = dim
        self.cache_id = index["cache_id"]
        self.queries = index["queries"]

    def contains_samples(self, samples) -> bool:
        expected = query_entries(samples, self.signature, self.text_policy)
        return all(self.queries.get(key) == value for key, value in expected.items())

    def validate_samples(self, samples) -> None:
        if not self.contains_samples(samples):
            raise ValueError(
                "native FAFA query missing/stale; "
                "run build-cache --cache-stage fafa-query"
            )

    def load(self, sample, count: int) -> Tensor:
        entry = self.queries[sample["sample_id"]]
        features = []
        for key in entry["features"]:
            path = self.root / "features" / f"{key}.pt"
            item = torch.load(path, map_location="cpu", weights_only=True)
            value = item.get("features")
            if (
                item.get("cache_id") != self.cache_id
                or item.get("key") != key
                or not isinstance(value, Tensor)
                or value.shape != (count, self.dim)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"invalid native FAFA feature shard: {path}")
            features.append(value)
        return torch.stack(features)


class NativeFAFACache:
    """Image-owned target tokens plus an optional independent query cache.

    Completed format-4 image caches remain readable without being rewritten.
    Their embedded queries are used only when no separate query root is supplied.
    """

    def __init__(self, index: dict, gallery, query_root=None) -> None:
        self.root = gallery.root
        if index.get("version") != NATIVE_VERSION:
            raise ValueError("native FAFA cache version differs; use a new directory")
        self.cache_id = gallery.cache_id
        self.dim = index["dim"]
        self.tokens = index["tokens"]
        self.query_cache = (
            NativeQueryCache(query_root, gallery.feature_signature, self.dim)
            if query_root is not None
            else None
        )
        self.queries = (
            self.query_cache.queries if self.query_cache else index.get("queries")
        )
        self.gallery = gallery
        self.text_policy = (
            self.query_cache.text_policy if self.query_cache else text_policy({})
        )
        # The checkpoint binds the image features and encoder provenance. Query
        # index UUIDs and sample order are deliberately not part of this contract.
        self.metadata = {
            "version": index["version"],
            "cache_id": self.cache_id,
            "spec": gallery.encoder_metadata["person_encoder"],
            "dim": self.dim,
            "tokens": self.tokens,
            "text_policy": self.text_policy,
        }

    def validate_samples(self, samples) -> None:
        if self.query_cache is not None:
            self.query_cache.validate_samples(samples)
            return
        if self.queries is None:
            raise ValueError("FAFA query cache is required; set data.fafa_query_cache")
        for sample in samples:
            entry = self.queries.get(sample["sample_id"])
            if entry is None or entry["key"] != query_key(sample):
                raise ValueError(
                    "native FAFA query missing/stale; build a separate FAFA query cache"
                )

    def _read(self, group, index, key, shape):
        path = self.root / group / f"{index}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        features = item.get("features")
        if (
            item.get("cache_id") != self.cache_id
            or item.get("key") != key
            or not isinstance(features, Tensor)
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
            count = int(
                self.gallery.mask[self.gallery.by_id[sample["query_image_id"]]].sum()
            )
            if count > max_people:
                raise ValueError("native/query person padding differs")
            shape = (len(sample["subjects"]), count, self.dim)
            if self.query_cache is not None:
                features = self.query_cache.load(sample, count)
            else:
                entry = self.queries[sample["sample_id"]]
                features = self._read("queries", entry["index"], entry["key"], shape)
            out[n, : shape[0], :count] = features
        return out

    def load_targets(self, indices: Tensor, max_people: int) -> Tensor:
        out = torch.zeros(len(indices), max_people, self.tokens, self.dim)
        rows = {}
        for n, i in enumerate(indices.tolist()):
            rows.setdefault(i, []).append(n)
        # Retain one shard at a time alongside the padded output, even when
        # candidate images repeat across queries in a training batch.
        for i, positions in rows.items():
            count = int(self.gallery.mask[i].sum())
            if count > max_people:
                raise ValueError("native/target person padding differs")
            features = self._read(
                "targets",
                i,
                self.gallery.image_ids[i],
                (count, self.tokens, self.dim),
            )
            for n in positions:
                out[n, :count] = features
            del features
        return out
