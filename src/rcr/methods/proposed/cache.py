"""Reference-free gallery feature cache."""

from pathlib import Path

import torch
from torch import Tensor


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
        if self.persons.shape[:2] != self.mask.shape or self.persons.shape[0] != len(
            self.image_ids
        ):
            raise ValueError("cache index persons/mask/gallery shape mismatch")

    def validate_gallery(self, image_ids: list[str]) -> None:
        """Require the cached image order to match the finalized benchmark."""
        if self.image_ids != image_ids:
            raise ValueError("cache gallery IDs/order differ from finalized data")

    def load(
        self, indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, list[list[str | None]], Tensor]:
        """Load and pad full features for selected gallery images."""

        items = []
        for i in indices.tolist():
            item = torch.load(
                self.root / "features" / f"{int(i)}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if item.get("cache_id") != self.cache_id or (
                "image_id" in item and item["image_id"] != self.image_ids[int(i)]
            ):
                raise ValueError(f"cache feature {int(i)} does not match index.pt")
            items.append(item)

        # Storage precision is independent of the FP32 train/inference model.
        scene = torch.stack([x["scene"] for x in items]).float()
        if scene.shape[1] != self.patch_hw[0] * self.patch_hw[1]:
            raise ValueError("cache scene patch count differs from patch_hw")
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
