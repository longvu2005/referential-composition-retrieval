"""GT alignment for training or diagnostics, separate from visual caches."""

from pathlib import Path

from PIL import Image

from rcr.common.data import split_image_ids
from rcr.common.vision import match_heads_to_persons


def align_identities(item, heads, *, image_path=None, scene_size=None):
    """Label existing detections; never create, remove or move a person box."""
    count = len(item["boxes_scene"])
    labels = [None] * count
    if not count or not heads:
        return labels
    persons = item.get("boxes_pixel")
    if persons is None:
        # Old caches may store only letterboxed coordinates. New DINO caches
        # store pixel boxes and need no image access for supervision.
        if image_path is None or scene_size is None:
            raise ValueError(
                "legacy box alignment needs raw images and scene_size; "
                "cached identity_ids are never used"
            )
        from rcr.proposed.cache.fafa import boxes_to_pixels

        with Image.open(image_path) as image:
            h, w = scene_size
            persons = boxes_to_pixels(item["boxes_scene"], image.size, (w, h))
    boxes = persons.new_tensor(
        [[r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]] for r in heads]
    )
    for person, head in enumerate(match_heads_to_persons(boxes, persons).tolist()):
        if head >= 0:
            labels[person] = str(heads[head]["identity_id"])
    return labels


class TrainSupervision:
    """Lazy, in-memory alignment restricted to the registered train gallery."""

    def __init__(self, data, cache, *, scene_size=None):
        self.cache = cache
        allowed = split_image_ids(data, "train")
        self.heads = {i: data.gt_head_boxes_by_image.get(i, []) for i in allowed}
        self.image_paths = {i: Path(data.image_path(i)) for i in allowed}
        self.scene_size = scene_size
        self._ids = {}

    def load_ids(self, image_ids, max_people):
        result = []
        for image_id in image_ids:
            if image_id not in self.heads:
                raise ValueError(
                    f"train supervision cannot read non-train image {image_id}"
                )
            if image_id not in self._ids:
                item = self.cache._get_item(self.cache.by_id[image_id])
                self._ids[image_id] = align_identities(
                    item,
                    self.heads[image_id],
                    image_path=self.image_paths[image_id],
                    scene_size=self.scene_size,
                )
            ids = self._ids[image_id]
            if len(ids) > max_people:
                raise ValueError(
                    "supervision crop order/count differs from visual batch"
                )
            result.append([*ids, *([None] * (max_people - len(ids)))])
        return result
