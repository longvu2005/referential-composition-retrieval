import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from rcr.methods.proposed.build_cache import boxes_to_scene, build_cache
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import ImageEncoder


def test_real_dinov3_processor_and_encoder_without_downloading_weights(tmp_path):
    transformers = pytest.importorskip("transformers", minversion="4.56")
    pytest.importorskip("torchvision")
    from rcr.methods.proposed.build_cache import _LetterboxProcessor

    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "DINOv3ViTImageProcessorFast"})
    )
    processor = transformers.AutoImageProcessor.from_pretrained(tmp_path)
    pixels = _LetterboxProcessor(processor, [32, 48])(
        Image.new("RGB", (80, 160)), return_tensors="pt"
    )["pixel_values"]
    backbone = transformers.DINOv3ViTModel(
        transformers.DINOv3ViTConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_register_tokens=4,
        )
    ).eval()
    with torch.inference_mode():
        patches, person, grid = ImageEncoder(backbone, 32)(pixels)
    assert pixels.shape == (1, 3, 32, 48)
    assert grid == (2, 3) and patches.shape == (1, 6, 32)
    assert person.shape == (1, 32)
    assert torch.isfinite(patches).all() and torch.isfinite(person).all()


class DetectorBatch(dict):
    def to(self, device):
        self.input_ids = torch.tensor([[1]])
        return self


class DetectorProcessor:
    def __init__(self, count=1):
        self.count = count

    def __call__(self, **kwargs):
        return DetectorBatch()

    def post_process_grounded_object_detection(self, outputs, **kwargs):
        return [
            {
                "boxes": torch.tensor([[1.0, 1.0, 7.0, 9.0]]).repeat(self.count, 1),
                "text_labels": ["person"] * self.count,
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


@pytest.mark.parametrize("storage_dtype", [None, "float16"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_build_cache_writes_gallery_features(tmp_path, storage_dtype, count) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10)).save(image_path)

    cache_dir = tmp_path / "cache"
    encoder = ImageEncoder(Backbone(), dim=8)
    build_cache(
        image_ids=["image"],
        image_paths=[image_path],
        root=cache_dir,
        detector=Detector(),
        detector_processor=DetectorProcessor(count),
        image_encoder=encoder,
        scene_processor=Processor(8, 8),
        person_processor=Processor(8, 4),
        device="cpu",
        **({"storage_dtype": storage_dtype} if storage_dtype else {}),
    )

    cache = GalleryCache(cache_dir)
    assert cache.image_ids == ["image"]
    assert cache.persons.shape == (1, count, 8)
    assert cache.mask.tolist() == [[True] * count]
    assert cache.patch_hw == (4, 4)

    scene, persons, boxes, ids, mask = cache.load(torch.tensor([0]))
    assert scene.shape == (1, 16, 8)
    assert persons.shape == (1, count, 8)
    assert boxes.shape == (1, count, 4)
    assert ids == [[None] * count]
    assert mask.tolist() == [[True] * count]
    assert scene.dtype == persons.dtype == boxes.dtype == torch.float32
    torch.testing.assert_close(cache.global_features, scene.mean(dim=1))
    assert cache.global_features.dtype == torch.float32
    if count:
        torch.testing.assert_close(boxes[0, 0], torch.tensor([0.1, 0.1, 0.7, 0.9]))

    dtype = torch.float16 if storage_dtype == "float16" else torch.float32
    assert cache.persons.dtype == dtype
    saved = torch.load(cache_dir / "features/0.pt", weights_only=True)
    assert saved["cache_id"] == cache.cache_id
    assert saved["image_id"] == "image"
    assert not (cache_dir / ".building").exists()
    assert saved["scene"].dtype == saved["persons"].dtype == dtype
    assert saved["boxes_scene"].dtype == torch.float32
    for key in ("scene", "persons", "boxes_scene"):
        value = saved[key]
        # A one-person CLS view can be contiguous while retaining all patches.
        assert value.is_contiguous()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    torch.testing.assert_close(cache.persons[0], saved["persons"])
    with torch.inference_mode():
        expected_scene, _, _ = encoder(torch.ones(1, 3, 8, 8))
        _, expected_persons, _ = encoder(torch.ones(count, 3, 8, 4))
    torch.testing.assert_close(scene[0], expected_scene[0].to(dtype).float())
    torch.testing.assert_close(persons[0], expected_persons.to(dtype).float())


def test_invalid_cache_dtype_fails_before_writing(tmp_path) -> None:
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="storage_dtype"):
        build_cache(
            [], [], root, None, None, None, None, None, "cpu", storage_dtype="int8"
        )
    assert not root.exists()


def test_interrupted_cache_build_cannot_be_read(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10)).save(image_path)

    class FailedDetector(Detector):
        def forward(self, **kwargs):
            raise RuntimeError("detector failed")

    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="detector failed"):
        build_cache(
            ["image"],
            [image_path],
            cache_dir,
            FailedDetector(),
            DetectorProcessor(),
            ImageEncoder(Backbone(), 8),
            Processor(8, 8),
            Processor(8, 4),
            "cpu",
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        GalleryCache(cache_dir)

    build_cache(
        ["image"],
        [image_path],
        cache_dir,
        Detector(),
        DetectorProcessor(),
        ImageEncoder(Backbone(), 8),
        Processor(8, 8),
        Processor(8, 4),
        "cpu",
    )
    assert GalleryCache(cache_dir).image_ids == ["image"]


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
