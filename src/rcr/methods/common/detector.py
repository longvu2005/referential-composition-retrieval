"""Shared GroundingDINO detection helper."""

import torch
from torch import Tensor, nn


@torch.inference_mode()
def detect(
    model: nn.Module,
    processor,
    image,
    labels: list[str],
    device: torch.device | str,
    threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> dict[str, Tensor]:
    """Detect text-labelled boxes and return one tensor per requested label."""

    inputs = processor(images=image, text=[labels], return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    names = result["text_labels"]
    boxes = result["boxes"]
    return {
        label: boxes[
            torch.tensor(
                [name == label for name in names],
                device=boxes.device,
                dtype=torch.bool,
            )
        ]
        for label in labels
    }
