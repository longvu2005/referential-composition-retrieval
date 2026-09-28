import torch
from torch import nn

from rcr.methods.common.detector import detect


class Batch(dict):
    def to(self, device):
        self.input_ids = torch.tensor([[1, 2]])
        return self


class Processor:
    def __call__(self, **kwargs):
        return Batch()

    def post_process_grounded_object_detection(self, outputs, **kwargs):
        return [
            {
                "boxes": torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 20.0],
                        [2.0, 1.0, 5.0, 6.0],
                        [20.0, 0.0, 30.0, 20.0],
                    ]
                ),
                "text_labels": ["person", "head", "person"],
            }
        ]


class Model(nn.Module):
    def forward(self, **kwargs):
        return object()


class Image:
    width = 32
    height = 24


def test_detect_groups_boxes_by_label() -> None:
    boxes = detect(
        Model(),
        Processor(),
        Image(),
        ["person", "head"],
        "cpu",
    )

    torch.testing.assert_close(
        boxes["person"],
        torch.tensor([[0.0, 0.0, 10.0, 20.0], [20.0, 0.0, 30.0, 20.0]]),
    )
    torch.testing.assert_close(
        boxes["head"],
        torch.tensor([[2.0, 1.0, 5.0, 6.0]]),
    )


def test_detect_accepts_no_detections() -> None:
    class EmptyProcessor(Processor):
        def post_process_grounded_object_detection(self, outputs, **kwargs):
            return [{"boxes": torch.empty(0, 4), "text_labels": []}]

    boxes = detect(Model(), EmptyProcessor(), Image(), ["person", "head"], "cpu")
    assert boxes["person"].shape == boxes["head"].shape == (0, 4)
