"""Reference-free gallery feature cache."""

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import torch
from torch import Tensor
from tqdm import tqdm


class GalleryCache:
    """CPU person index plus a bounded LRU for immutable on-disk image features."""

    def __init__(
        self,
        root: str | Path,
        *,
        scene_root: str | Path | None = None,
        lru_mib: int = 0,
    ) -> None:
        if not isinstance(lru_mib, int) or lru_mib < 0:
            raise ValueError("cache.lru_mib must be a nonnegative integer")
        self._limit = lru_mib * 1024**2
        self._items: OrderedDict[int, tuple[dict, int]] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
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
            if scene_root is None:
                raise ValueError("FAFA person cache requires data.dino_cache")
            self._scene_cache = GalleryCache(scene_root)
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
        self.patch_hw = tuple(index["patch_hw"])
        self._global_features = index.get("global_features")
        if self._global_features is not None and self._global_features.shape != (
            len(self.image_ids),
            self.scene_dim,
        ):
            raise ValueError("cache global features/gallery shape mismatch")
        if self.persons.shape[:2] != self.mask.shape or self.persons.shape[0] != len(
            self.image_ids
        ):
            raise ValueError("cache index persons/mask/gallery shape mismatch")

    def validate_gallery(self, image_ids: list[str]) -> None:
        """Require the cached image order to match the finalized benchmark."""
        if self.image_ids != image_ids:
            raise ValueError("cache gallery IDs/order differ from finalized data")

    def _read_feature(self, index: int) -> dict:
        item = torch.load(
            self.root / "features" / f"{index}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if item.get("cache_id") != self.cache_id or (
            "image_id" in item and item["image_id"] != self.image_ids[index]
        ):
            raise ValueError(f"cache feature {index} does not match index.pt")
        return item

    def _load_item(self, index: int) -> dict:
        item = self._read_feature(index)
        if self._scene_cache is not None:
            if item.get("source_cache_id") != self._scene_cache.cache_id:
                raise ValueError("FAFA person feature has a different DINO source")
            item = {**self._scene_cache._load_item(index), "persons": item["persons"]}
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
        # A single reader thread shares this cache with the main process.
        # Cached tensors are immutable; collation creates independent batches.
        with self._lock:
            if index in self._items:
                item, size = self._items.pop(index)
                self._items[index] = (item, size)
                return item
            item = self._load_item(index)
            if not self._limit:
                return item
            storages = {
                value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for value in item.values()
                if isinstance(value, torch.Tensor)
            }
            size = sum(storages.values())
            if size <= self._limit:
                while self._items and self._bytes + size > self._limit:
                    _, (_, removed) = self._items.popitem(last=False)
                    self._bytes -= removed
                self._items[index] = (item, size)
                self._bytes += size
            return item

    @property
    def global_features(self) -> Tensor:
        """Mean scene patches [G,D], kept on CPU and reused across evaluations.

        Legacy caches are read once, one feature file at a time. This supports
        read-only Kaggle inputs without rebuilding the detector/image cache.
        New caches store the same FP32 means directly in index.pt.
        """
        if self._global_features is None:
            if self._scene_cache is not None:
                self._global_features = self._scene_cache.global_features
                return self._global_features
            features = torch.empty(len(self.image_ids), self.scene_dim)
            for i in tqdm(
                range(len(self.image_ids)), desc="pool cached global features"
            ):
                item = self._load_item(i)
                features[i] = item["scene"].float().mean(dim=0)
            self._global_features = features
        return self._global_features

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

        identity_ids = []
        mask = torch.zeros(len(items), k, dtype=torch.bool)
        for n, item in enumerate(items):
            count = item["persons"].shape[0]
            mask[n, :count] = True
            ids = item.get("identity_ids", [None] * count)
            identity_ids.append(ids + [None] * (k - count))

        return scene, persons, boxes, identity_ids, mask
