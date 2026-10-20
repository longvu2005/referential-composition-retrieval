"""Remove legacy GT identity metadata from completed DINO shards, without encoding."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from rcr.common.torch_io import save_torch
from rcr.proposed.cache.store import GalleryCache


def clean_cache(root):
    root = Path(root)
    cache = GalleryCache(root)
    changed = 0
    for i, image_id in enumerate(tqdm(cache.image_ids, desc="strip DINO labels")):
        path = root / "features" / f"{i}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True)
        if (
            item.get("cache_id") != cache.cache_id
            or item.get("image_id", image_id) != image_id
        ):
            raise ValueError(f"cache feature {i} does not match index.pt")
        if "identity_ids" in item:
            del item["identity_ids"]
            save_torch(item, path)
            changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Writable DINO cache directory")
    args = parser.parse_args()
    print(f"Removed identity_ids from {clean_cache(args.cache)} shards")


if __name__ == "__main__":
    main()
