"""Coordinate transforms for full-image letterboxing; independent of encoders."""

import torch


def boxes_to_scene(
    boxes: torch.Tensor, image_size: tuple[int, int], scene_size: tuple[int, int]
) -> torch.Tensor:
    """Map pixel xyxy boxes to normalized ImageOps.pad letterbox coordinates."""
    width, height = image_size
    scene_width, scene_height = scene_size
    scale = min(scene_width / width, scene_height / height)
    resized_width = min(scene_width, round(width * scale))
    resized_height = min(scene_height, round(height * scale))
    left = round((scene_width - resized_width) * 0.5)
    top = round((scene_height - resized_height) * 0.5)
    factors = boxes.new_tensor([resized_width / width, resized_height / height] * 2)
    offset = boxes.new_tensor([left, top, left, top])
    size = boxes.new_tensor([scene_width, scene_height] * 2)
    return ((boxes * factors + offset) / size).clamp(0, 1)
