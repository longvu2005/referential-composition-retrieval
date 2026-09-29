from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import (
    evaluate_retrieval_output,
    retrieve_rankings,
    slice_retrieval_output,
)


class _Tokenizer:
    marker_id = 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.marker_id if token == "[S1]" else 2

    def __call__(self, texts, padding=False, return_tensors=None):
        del return_tensors
        rows = [texts] if isinstance(texts, str) else texts
        encoded = []
        for text in rows:
            tokens = text.split()
            encoded.append(
                [self.marker_id if token == "[S1]" else 2 for token in tokens]
            )
        length = max(map(len, encoded)) if padding else len(encoded[0])
        ids = torch.zeros(len(encoded), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(3, 8)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids), attention_mask.bool()


class _Cache:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(3)
        self.image_ids = ["q", "a", "b"]
        self.persons = torch.randn(3, 1, 8, generator=generator)
        self.scenes = torch.randn(3, 1, 8, generator=generator)
        self.mask = torch.ones(3, 1, dtype=torch.bool)
        self.patch_hw = (1, 1)

    def load(self, indices):
        indices = indices.long()
        count = len(indices)
        boxes = torch.tensor([0.0, 0.0, 1.0, 1.0]).expand(count, 1, 4)
        identities = [[None] for _ in range(count)]
        return (
            self.scenes[indices],
            self.persons[indices],
            boxes,
            identities,
            self.mask[indices],
        )


def _sample(sample_id: str = "s1") -> dict:
    return {
        "sample_id": sample_id,
        "case_type": "INDIVIDUAL",
        "query_image_id": "q",
        "positive_image_ids": ["a"],
        "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
        "final_desc": "Identify Subject 1 as the person",
        "final_change": "then retrieve target images where Subject 1 is standing",
    }


def test_retrieve_rankings_excludes_query_and_restores_training_modes() -> None:
    torch.manual_seed(4)
    cache = _Cache()
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder().train()
    model = RCRModel(8, 6, 2, max_subjects=1).train()

    output = retrieve_rankings(
        [_sample()],
        cache,
        tokenizer,
        text_encoder,
        model,
        torch.device("cpu"),
        top_m=2,
        fine_batch_size=1,
        identity_batch_size=2,
    )

    assert output["sample_ids"] == ["s1"]
    assert set(output["rankings"][0].tolist()) == {1, 2}
    assert output["coarse_topm"].shape == (1, 2)
    assert model.training
    assert text_encoder.training


def test_slice_and_evaluate_retrieval_output_use_official_metrics() -> None:
    sample = _sample()
    output = {
        "sample_ids": ["s1"],
        "gallery_ids": ["q", "a", "b"],
        "rankings": torch.tensor([[1, 2]], dtype=torch.int32),
        "coarse_topm": torch.tensor([[1, 2]], dtype=torch.int32),
    }
    sliced = slice_retrieval_output(output, 0, 1)
    data = SimpleNamespace(
        gallery_ids=["q", "a", "b"],
        gt_head_boxes_by_image={
            "q": [{"identity_id": "p1"}],
            "a": [{"identity_id": "p1"}],
        },
    )

    result = evaluate_retrieval_output(data, [sample], sliced, candidate_ks=[1, 2])

    assert result["overall"]["full_map"] == 1.0
    assert result["overall"]["id_map"] == 1.0
    assert result["overall"]["candidate_recall_1"] == 1.0


def test_evaluate_retrieval_output_rejects_invalid_tensor_index() -> None:
    sample = _sample()
    output = {
        "sample_ids": ["s1"],
        "gallery_ids": ["q", "a", "b"],
        "rankings": torch.tensor([[1, -1]], dtype=torch.int32),
        "coarse_topm": torch.tensor([[1, 2]], dtype=torch.int32),
    }
    data = SimpleNamespace(
        gallery_ids=["q", "a", "b"],
        gt_head_boxes_by_image={},
    )

    with pytest.raises(ValueError, match="invalid gallery index"):
        evaluate_retrieval_output(data, [sample], output, candidate_ks=[1])
