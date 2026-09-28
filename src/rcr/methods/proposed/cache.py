"""Reference-free gallery feature cache."""

from pathlib import Path

import torch
from torch import Tensor


class GalleryCache:
    """Keep reference-free person features in memory and load full Top-M features."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        index = torch.load(
            self.root / "index.pt", map_location="cpu", weights_only=True
        )

        self.image_ids = index["image_ids"]
        self.persons = index["persons"]
        self.mask = index["mask"].bool()
        self.patch_hw = tuple(index["patch_hw"])

    def load(
        self, indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, list[list[str | None]], Tensor]:
        """Load and pad full features for selected gallery images."""

        items = [
            torch.load(
                self.root / "features" / f"{int(i)}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for i in indices.tolist()
        ]

        scene = torch.stack([x["scene"] for x in items])
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
