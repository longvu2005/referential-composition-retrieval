"""Read frozen FAFA multimodal queries and projected target tokens.

Crop order is inherited from the shared detection cache. Query features are
indexed by public instruction/Subject slots, never by selected members or GT.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rcr.proposed.cache.person_encoder import text_policy

NATIVE_VERSION = "fafa-multimodal-fda-v1"


@dataclass
class MatchedTargetTokens:
    """Packed assigned token rows; null/query padding never crosses PCIe."""

    values: Tensor
    positions: Tensor
    shape: tuple[int, int]

    def to_aligned(self, device):
        values = self.values.to(device, non_blocking=True)
        positions = self.positions.to(device, non_blocking=True)
        out = values.new_zeros((self.shape[0] * self.shape[1], *values.shape[1:]))
        out.index_copy_(0, positions, values)
        return out.unflatten(0, self.shape)


class NativeTargetBatch:
    """Read only the current training chunk, before padding and GPU transfer."""

    def __init__(self, cache, image_indices: Tensor, max_people: int | None = None):
        self.cache = cache
        self.image_indices = image_indices
        self.max_people = max_people
        self.prepared = {}

    def select(self, rows: Tensor, max_people: int):
        selected = NativeTargetBatch(self.cache, self.image_indices[rows], max_people)
        selected.prepared = self.prepared
        return selected

    def prepare(self, rows: Tensor, max_bytes: int):
        if max_bytes and hasattr(self.cache, "prepare_native_targets"):
            self.prepared = self.cache.prepare_native_targets(
                self.image_indices[rows], max_bytes
            )

    def load(self, rows: Tensor, max_people: int) -> Tensor:
        return self.cache.native_targets(self.image_indices[rows], max_people)

    def load_matched(self, indices: Tensor) -> Tensor | MatchedTargetTokens:
        """Gather assigned detections on CPU, before allocating/transferring padding."""
        if hasattr(self.cache, "native_matched_targets"):
            return self.cache.native_matched_targets(
                self.image_indices, indices, prepared=self.prepared
            )
        # Compatibility for synthetic/third-party readers implementing the old API.
        if self.max_people is None:
            raise ValueError("deferred native targets need their target person width")
        values = self.cache.native_targets(self.image_indices, self.max_people)
        if not values.shape[1]:
            return values.new_zeros((len(indices), indices.shape[1], *values.shape[2:]))
        shape = (*indices.shape, *values.shape[2:])
        selected = values.gather(1, indices.clamp_min(0)[..., None, None].expand(shape))
        return selected.masked_fill((indices < 0)[..., None, None], 0)


def subject_conditions(sample: dict) -> list[str]:
    """Anchor the complete condition to each public Subject, without splitting.

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
            text,
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

    def _read(self, key: str, count: int) -> Tensor:
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
        return value

    def load(self, sample, count: int, *, memory_cache=None) -> Tensor:
        entry = self.queries[sample["sample_id"]]
        features = []
        for key in entry["features"]:
            if memory_cache is None:
                value = self._read(key, count)
            else:
                value = memory_cache(
                    ("fafa-query", self.cache_id, key, count),
                    lambda key=key: self._read(key, count),
                )
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
        item = torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
            mmap=self.gallery.mmap_shards,
        )
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
                features = self.query_cache.load(
                    sample, count, memory_cache=self.gallery._cached
                )
            else:
                entry = self.queries[sample["sample_id"]]
                features = self.gallery._cached(
                    ("fafa-query", entry["index"], entry["key"], shape),
                    lambda entry=entry, shape=shape: self._read(
                        "queries", entry["index"], entry["key"], shape
                    ),
                )
            out[n, : shape[0], :count] = features
        return out

    def load_targets(self, indices: Tensor, max_people: int) -> Tensor:
        # Transfer storage precision; FDA converts assigned rows to FP32.
        out = None
        rows = {}
        for n, i in enumerate(indices.tolist()):
            rows.setdefault(i, []).append(n)
        # Repeated image IDs share one shard read alongside the padded output.
        for i, positions in rows.items():
            count = int(self.gallery.mask[i].sum())
            if count > max_people:
                raise ValueError("native/target person padding differs")
            shape = (count, self.tokens, self.dim)
            features = self.gallery._cached(
                ("fafa-target", i, shape),
                lambda i=i, shape=shape: self._read(
                    "targets", i, self.gallery.image_ids[i], shape
                ),
            )
            if out is None:
                out = features.new_zeros(
                    len(indices), max_people, self.tokens, self.dim
                )
            for n in positions:
                out[n, :count] = features
            del features
        return (
            out
            if out is not None
            else self.gallery.persons.new_zeros(0, max_people, self.tokens, self.dim)
        )

    def load_matched_targets(
        self, image_indices: Tensor, assignments: Tensor, *, prepared=None
    ) -> MatchedTargetTokens:
        """Read each used image once, retaining only its assigned person token rows."""
        if (
            assignments.device.type != "cpu"
            or assignments.dtype != torch.long
            or assignments.ndim != 2
            or len(assignments) != len(image_indices)
            or (assignments < -1).any()
        ):
            raise ValueError(
                "native assignments must be CPU int64 [images,query_people]"
            )
        rows = {}
        for n, i in enumerate(image_indices.tolist()):
            if (assignments[n] >= 0).any():
                rows.setdefault(i, []).append(n)
        values, locations = [], []
        for i, positions in rows.items():
            count = int(self.gallery.mask[i].sum())
            features = (prepared or {}).get(i)
            if features is None:
                features = self.target_row(i)
            positions = torch.tensor(positions)
            chosen = assignments[positions]
            valid = chosen >= 0
            if (chosen[valid] >= count).any():
                raise ValueError("native assignment exceeds the image person count")
            values.append(features[chosen[valid]])
            flat = positions[:, None] * assignments.shape[1] + torch.arange(
                assignments.shape[1]
            )
            locations.append(flat[valid])
        return MatchedTargetTokens(
            torch.cat(values)
            if values
            else self.gallery.persons.new_empty((0, self.tokens, self.dim)),
            torch.cat(locations) if locations else torch.empty(0, dtype=torch.long),
            tuple(assignments.shape),
        )

    def target_row(self, index):
        count = int(self.gallery.mask[index].sum())
        shape = (count, self.tokens, self.dim)
        return self.gallery._cached(
            ("fafa-target", index, shape),
            lambda: self._read("targets", index, self.gallery.image_ids[index], shape),
        )

    def prepare_targets(self, indices, max_bytes):
        prepared, used = {}, 0
        for index in dict.fromkeys(indices.tolist()):
            count = int(self.gallery.mask[index].sum())
            estimate = (
                count * self.tokens * self.dim * self.gallery.persons.element_size()
            )
            if used + estimate > max_bytes:
                break
            value = self.target_row(index)
            size = value.numel() * value.element_size()
            if used + size > max_bytes:
                break
            prepared[index] = value
            used += size
        return prepared
