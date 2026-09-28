from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import ImageEncoder
from tools.methods.build_cache import boxes_to_scene, build_cache


class DetectorBatch(dict):
    def to(self, device):
        self.input_ids = torch.tensor([[1]])
        return self


class DetectorProcessor:
    def __call__(self, **kwargs):
        return DetectorBatch()

    def post_process_grounded_object_detection(self, outputs, **kwargs):
        return [
            {
                "boxes": torch.tensor(
                    [
                        [1.0, 1.0, 7.0, 9.0],
                    ]
                ),
                "text_labels": ["person"],
            }
        ]


class Detector(nn.Module):
    def forward(self, **kwargs):
        return object()


class Processor:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, images, return_tensors: str):
        del return_tensors
        count = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.ones(count, 3, self.height, self.width)}


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=6,
            patch_size=2,
            num_register_tokens=0,
        )
        self.patch = nn.Conv2d(3, 6, kernel_size=2, stride=2)

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


def test_build_cache_writes_gallery_features(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10)).save(image_path)

    cache_dir = tmp_path / "cache"
    encoder = ImageEncoder(Backbone(), dim=8)
    build_cache(
        image_ids=["image"],
        image_paths=[image_path],
        root=cache_dir,
        detector=Detector(),
        detector_processor=DetectorProcessor(),
        image_encoder=encoder,
        scene_processor=Processor(8, 8),
        person_processor=Processor(8, 4),
        device="cpu",
    )

    cache = GalleryCache(cache_dir)
    assert cache.image_ids == ["image"]
    assert cache.persons.shape == (1, 1, 8)
    assert cache.mask.tolist() == [[True]]
    assert cache.patch_hw == (4, 4)

    scene, persons, boxes, ids, mask = cache.load(torch.tensor([0]))
    assert scene.shape == (1, 16, 8)
    assert persons.shape == (1, 1, 8)
    assert boxes.shape == (1, 1, 4)
    assert ids == [[None]]
    assert mask.tolist() == [[True]]
    torch.testing.assert_close(boxes[0, 0], torch.tensor([0.1, 0.1, 0.7, 0.9]))


def test_letterbox_boxes_match_patch_coordinate_system() -> None:
    box = torch.tensor([[0.0, 0.0, 20.0, 10.0]])
    torch.testing.assert_close(
        boxes_to_scene(box, (20, 10), (20, 20)), torch.tensor([[0.0, 0.25, 1.0, 0.75]])
    )


def test_detector_without_head_keeps_person(tmp_path) -> None:
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (10, 10)).save(image_path)
    build_cache(
        ["portrait"],
        [image_path],
        tmp_path / "cache",
        Detector(),
        DetectorProcessor(),
        ImageEncoder(Backbone(), 8),
        Processor(8, 8),
        Processor(8, 4),
        device="cpu",
    )
    saved = torch.load(tmp_path / "cache/features/0.pt", weights_only=True)
    assert saved["persons"].shape == (1, 8)
    assert "heads" not in saved
