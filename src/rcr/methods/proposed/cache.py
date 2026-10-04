"""Reference-free gallery feature cache."""

from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm


class GalleryCache:
    """Keep reference-free person features in memory and load full Top-M features."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if (self.root / ".building").exists():
            raise RuntimeError(
                f"cache build is incomplete at {self.root}; rebuild before use"
            )
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True
        )

        self.image_ids = index["image_ids"]
        self.cache_id = index.get("cache_id")
        self.persons = index["persons"]
        self.mask = index["mask"].bool()
        self.patch_hw = tuple(index["patch_hw"])
        self._global_features = index.get("global_features")
        if self._global_features is not None and self._global_features.shape != (
            len(self.image_ids),
            self.persons.shape[-1],
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

    def _load_item(self, index: int) -> dict:
        item = torch.load(
            self.root / "features" / f"{index}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if item.get("cache_id") != self.cache_id or (
            "image_id" in item and item["image_id"] != self.image_ids[index]
        ):
            raise ValueError(f"cache feature {index} does not match index.pt")
        if item["scene"].shape != (
            self.patch_hw[0] * self.patch_hw[1],
            self.persons.shape[-1],
        ):
            raise ValueError("cache scene shape differs from patch_hw/feature dim")
        return item

    @property
    def global_features(self) -> Tensor:
        """Mean scene patches [G,D], kept on CPU and reused across evaluations.

        Legacy caches are read once, one feature file at a time. This supports
        read-only Kaggle inputs without rebuilding the detector/image cache.
        New caches store the same FP32 means directly in index.pt.
        """
        if self._global_features is None:
            features = torch.empty(len(self.image_ids), self.persons.shape[-1])
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

        items = []
        for i in indices.tolist():
            items.append(self._load_item(int(i)))

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
