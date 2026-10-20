"""Reference-free gallery feature cache."""

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import torch
from torch import Tensor

from rcr.common.data import inference_samples


class GalleryCache:
    """CPU person index plus a bounded LRU for immutable on-disk image features."""

    def __init__(
        self,
        root: str | Path,
        *,
        scene_root: str | Path | None = None,
        query_root: str | Path | None = None,
        lru_mib: int = 0,
        native_lru_mib: int = 0,
        mmap_shards: bool = False,
    ) -> None:
        for name, value in (("lru_mib", lru_mib), ("native_lru_mib", native_lru_mib)):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"cache.{name} must be a nonnegative integer")
        self._limit = lru_mib * 1024**2
        self._native_limit = native_lru_mib * 1024**2
        self._items: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self._native_items: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self._bytes = 0
        self._native_bytes = 0
        self._lock = Lock()
        self.mmap_shards = mmap_shards
        self.root = Path(root)
        if (self.root / ".building").exists():
            raise RuntimeError(
                f"cache build is incomplete at {self.root}; "
                "resume build-cache before use"
            )
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True, mmap=True
        )

        self._scene_cache = None
        if index.get("layout") == "persons":
            raise ValueError("old FAFA person cache; rebuild data.fafa_image_cache")
        if index.get("layout") == "fafa":
            if scene_root is None:
                raise ValueError("FAFA person cache requires data.dino_cache")
            self._scene_cache = GalleryCache(scene_root, mmap_shards=mmap_shards)
            source = self._scene_cache
            source.validate_gallery(index["image_ids"])
            if index["source_cache_id"] != source.cache_id:
                raise ValueError("FAFA cache differs from its source DINO cache")
            if not torch.equal(index["mask"].bool(), source.mask):
                raise ValueError("FAFA/DINO person masks differ")
            if (
                index["scene_dim"] != source.scene_dim
                or tuple(index["patch_hw"]) != source.patch_hw
            ):
                raise ValueError("FAFA/DINO scene specifications differ")

        self.image_ids = index["image_ids"]
        self.by_id = {image_id: i for i, image_id in enumerate(self.image_ids)}
        self.cache_id = index.get("cache_id")
        self.format_version = index.get("format_version", 1)
        self.encoder_metadata = {
            key: index.get(key)
            for key in ("image_encoder", "person_encoder", "detector")
        }
        self.scene_dim = index.get("scene_dim", index["persons"].shape[-1])
        self.person_dim = index["persons"].shape[-1]
        self.persons = index["persons"]
        self.mask = index["mask"].bool()
        self.person_offsets = index.get("person_offsets")
        self.patch_hw = tuple(index["patch_hw"])
        self._global_features = index.get("global_features")
        self._global_rows: dict[int, Tensor] = {}
        if self._global_features is not None and self._global_features.shape != (
            len(self.image_ids),
            self.scene_dim,
        ):
            raise ValueError("cache global features/gallery shape mismatch")
        if self.mask.ndim != 2 or self.mask.shape[0] != len(self.image_ids):
            raise ValueError("cache index persons/mask/gallery shape mismatch")
        if self.person_offsets is None:
            if (
                self.format_version == 6
                or self.persons.ndim != 3
                or self.persons.shape[:2] != self.mask.shape
            ):
                raise ValueError("cache index persons/mask/gallery shape mismatch")
        elif (
            self.format_version != 6
            or index.get("layout") != "fafa"
            or self.persons.ndim != 2
            or not isinstance(self.person_offsets, Tensor)
            or self.person_offsets.dtype != torch.int64
            or self.person_offsets.shape != (len(self.image_ids) + 1,)
            or self.person_offsets[0] != 0
            or self.person_offsets[-1] != len(self.persons)
            or not torch.equal(self.person_offsets.diff(), self.mask.sum(dim=1))
        ):
            raise ValueError("invalid packed person index offsets/mask")
        if index.get("layout") == "fafa":
            # Reconstruct one canonical signature for every supported FAFA index.
            person = self.encoder_metadata["person_encoder"]
            self.feature_signature = {
                "source_cache_id": index["source_cache_id"],
                "spec": person["spec"],
                "checkpoint_sha256": person["checkpoint_sha256"],
                "image_encoder": index["image_encoder"],
                "detector": index["detector"],
                "encoder_precision": person["encoder_precision"],
            }
            if index.get("signature", self.feature_signature) != self.feature_signature:
                raise ValueError("FAFA image signature differs from encoder metadata")
        from rcr.proposed.cache.native import NativeFAFACache

        self.native = (
            NativeFAFACache(index["native"], self, query_root)
            if index.get("layout") == "fafa"
            else None
        )

    def native_queries(self, samples, max_people: int) -> Tensor:
        if self.native is None:
            raise ValueError("FAFA image/query caches are required; run build-cache")
        return self.native.load_queries(inference_samples(samples), max_people)

    def validate_samples(self, samples) -> None:
        if self.native is None:
            raise ValueError("FAFA image/query caches are required; run build-cache")
        self.native.validate_samples(inference_samples(samples))

    def native_targets(self, indices: Tensor, max_people: int) -> Tensor:
        if self.native is None:
            raise ValueError("FAFA image/query caches are required; run build-cache")
        return self.native.load_targets(indices, max_people)

    def validate_gallery(self, image_ids: list[str]) -> None:
        """Require the cached image order to match the finalized benchmark."""
        if self.image_ids != image_ids:
            raise ValueError("cache gallery IDs/order differ from finalized data")

    def person_row(self, index: int) -> Tensor:
        """Actual persons in crop order, without gallery-wide padding."""
        if self.person_offsets is None:
            return self.persons[index, self.mask[index]]
        start, stop = self.person_offsets[index : index + 2].tolist()
        return self.persons[start:stop]

    def load_persons(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Load the coarse index only, padding useful columns for this batch."""
        mask = self.mask[indices]
        columns = mask.any(dim=0)
        mask = mask[:, columns]
        if self.person_offsets is None:
            return self.persons[indices[:, None], columns.nonzero().flatten()], mask
        persons = self.persons.new_zeros(len(indices), mask.shape[1], self.person_dim)
        for n, index in enumerate(indices.tolist()):
            persons[n, mask[n]] = self.person_row(index)
        return persons, mask

    def _read_feature(self, index: int) -> dict:
        item = torch.load(
            self.root / "features" / f"{index}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=self.mmap_shards,
        )
        if item.get("cache_id") != self.cache_id or (
            "image_id" in item and item["image_id"] != self.image_ids[index]
        ):
            raise ValueError(f"cache feature {index} does not match index.pt")
        # Legacy shards can contain identity_ids. They never enter the feature
        # API or its LRU; training computes fresh alignment from train GT only.
        return {
            key: item[key]
            for key in ("scene", "persons", "boxes_scene", "boxes_pixel", "patch_hw")
            if key in item
        }

    def _load_item(self, index: int) -> dict:
        if self._scene_cache is not None:
            # The published person index is authoritative. FAFA shards are only
            # extraction checkpoints and may be removed during finalization.
            # Clone the row so a small LRU item does not retain/count the whole
            # memory-mapped person index as its tensor storage.
            persons = self.person_row(index).clone()
            item = {**self._scene_cache._load_item(index), "persons": persons}
        else:
            item = self._read_feature(index)
        if item["scene"].shape != (
            self.patch_hw[0] * self.patch_hw[1],
            self.scene_dim,
        ):
            raise ValueError("cache scene shape differs from patch_hw/feature dim")
        if self.format_version >= 2:
            count = int(self.mask[index].sum())
            if item["persons"].shape != (count, self.person_dim):
                raise ValueError("cache person shape differs from index.pt")
            if item["boxes_scene"].shape != (count, 4):
                raise ValueError("cache person box count differs from index.pt")
        return item

    def _get_item(self, index: int) -> dict:
        return self._cached(index, lambda: self._load_item(index))

    def _cached(self, key, load):
        """Keep scene and native-shard budgets independent."""
        # Scene keys are integer image indices; native query/target keys are tuples.
        native = isinstance(key, tuple)
        with self._lock:
            items = self._native_items if native else self._items
            limit = self._native_limit if native else self._limit
            used = self._native_bytes if native else self._bytes
            if key in items:
                item, size = items.pop(key)
                items[key] = (item, size)
                return item
            item = load()
            if not limit:
                return item
            values = (item,) if isinstance(item, Tensor) else item.values()
            storages = {
                value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for value in values
                if isinstance(value, Tensor)
            }
            size = sum(storages.values())
            if size <= limit:
                while items and used + size > limit:
                    _, (_, removed) = items.popitem(last=False)
                    used -= removed
                items[key] = (item, size)
                used += size
                if native:
                    self._native_bytes = used
                else:
                    self._bytes = used
            return item

    @property
    def global_features(self) -> Tensor:
        """Explicit full-gallery view; ranking uses scoped loads instead."""
        if self._global_features is None:
            self._global_features = self.load_global_features(
                torch.arange(len(self.image_ids))
            )
            self._global_rows.clear()
        return self._global_features

    def load_global_features(self, indices: Tensor) -> Tensor:
        """Read only requested images; retain their fixed FP32 means on CPU.

        A packed FAFA index can rely on a legacy DINO index with no means.
        Mining the train gallery must not scan val/test/leftover scene shards.
        Reuse any full scene already present in this reader's LRU.
        """
        if self._global_features is not None:
            return self._global_features[indices]
        source = self._scene_cache
        if source is not None and source._global_features is not None:
            return source.load_global_features(indices)
        rows = indices.tolist()
        if not rows:
            return torch.empty(0, self.scene_dim)
        for i in dict.fromkeys(rows):
            if i not in self._global_rows:
                self._global_rows[i] = self._get_item(i)["scene"].float().mean(0)
        return torch.stack([self._global_rows[i] for i in rows])

    def load(
        self, indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, list[list[str | None]], Tensor]:
        """Load and pad full features for selected gallery images."""

        return self.load_groups(indices)[0]

    def load_groups(self, *groups: Tensor) -> tuple:
        """Read unique images once across groups, padding each group separately."""
        rows = [indices.tolist() for indices in groups]
        unique = dict.fromkeys(i for row in rows for i in row)
        items = {i: self._get_item(int(i)) for i in unique}
        return tuple(self._collate([items[i] for i in row]) for row in rows)

    @staticmethod
    def _collate(items: list[dict]) -> tuple:
        if not items:
            raise ValueError("cannot collate an empty cache group")
        # Storage precision is independent of the FP32 train/inference model.
        scene = torch.stack([x["scene"] for x in items]).float()
        k = max(x["persons"].shape[0] for x in items)

        def pad(name: str, dim: int) -> Tensor:
            out = scene.new_zeros(len(items), k, dim)
            for n, item in enumerate(items):
                value = item[name]
                out[n, : value.shape[0]] = value
            return out

        persons = pad("persons", items[0]["persons"].shape[-1])
        boxes = pad("boxes_scene", 4)

        mask = torch.zeros(len(items), k, dtype=torch.bool)
        for n, item in enumerate(items):
            count = item["persons"].shape[0]
            mask[n, :count] = True
        return scene, persons, boxes, mask
