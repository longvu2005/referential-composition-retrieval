from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rcr.methods.proposed import retrieval
from rcr.methods.proposed.coarse import coarse_scores
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import (
    _coarse_scores_chunked,
    _encode_gallery_identity,
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


@pytest.mark.parametrize(
    "identity_batch_size,coarse_batch_size", [(1, 1), (3, 2), (20, 20)]
)
@pytest.mark.parametrize("empty_query", [False, True])
def test_chunked_coarse_matches_dense_with_padding_and_empty_images(
    identity_batch_size, coarse_batch_size, empty_query
) -> None:
    torch.manual_seed(23)
    # The final two images have no detected people. Valid columns need not be
    # contiguous, so accidentally truncating at the valid count would lose data.
    cache = SimpleNamespace(
        image_ids=[str(index) for index in range(7)],
        persons=torch.randn(7, 5, 8).half(),
        mask=torch.tensor(
            [
                [True, False, True, False, False],
                [True, False, False, False, False],
                [True, True, False, False, True],
                [False, True, False, False, False],
                [True, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    )
    model = RCRModel(8, 6, 2)
    query_identity = torch.randn(0 if empty_query else 3, 6)
    logits = torch.randn(2, len(query_identity))
    query_mask = torch.tensor([] if empty_query else [True, False, True]).bool()
    with torch.inference_mode():
        batches = _encode_gallery_identity(
            cache, model, torch.device("cpu"), identity_batch_size
        )
        expected = coarse_scores(
            query_identity,
            logits,
            model.identity_head(cache.persons.float()),
            cache.mask,
            query_mask=query_mask,
        )
        actual = _coarse_scores_chunked(
            query_identity, logits, query_mask, batches, coarse_batch_size
        )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(
        actual.argsort(descending=True), expected.argsort(descending=True)
    )
    assert all(identity.device.type == "cpu" for identity, _ in batches)
    assert all(identity.dtype == torch.float32 for identity, _ in batches)
    assert sum(len(identity) for identity, _ in batches) == len(cache.image_ids)
    assert not actual.isnan().any()


@pytest.mark.parametrize("coarse_batch_size", [1, 2, 10])
def test_chunked_full_rankings_match_dense_reference(monkeypatch, coarse_batch_size):
    torch.manual_seed(17)
    cache = _Cache()
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder().eval()
    model = RCRModel(8, 6, 2, max_subjects=1).eval()
    samples = [_sample("s1"), {**_sample("s2"), "query_image_id": "a"}]
    kwargs = dict(
        top_m=1,
        fine_batch_size=1,
        identity_batch_size=2,
        coarse_batch_size=coarse_batch_size,
    )
    actual = retrieve_rankings(
        samples, cache, tokenizer, text_encoder, model, torch.device("cpu"), **kwargs
    )

    # Recreate the original full-gallery projection and coarse computation;
    # use the same query/fine path to compare both the shortlist and full tail.
    def dense_gallery(cache, model, device, batch_size):
        del batch_size
        return [(model.identity_head(cache.persons.to(device)), cache.mask.to(device))]

    def dense_scores(query_identity, logits, query_mask, batches, batch_size):
        del batch_size
        identity, mask = batches[0]
        return coarse_scores(
            query_identity, logits, identity, mask, query_mask=query_mask
        )

    monkeypatch.setattr(retrieval, "_encode_gallery_identity", dense_gallery)
    monkeypatch.setattr(retrieval, "_coarse_scores_chunked", dense_scores)
    expected = retrieve_rankings(
        samples, cache, tokenizer, text_encoder, model, torch.device("cpu"), **kwargs
    )
    assert actual["sample_ids"] == expected["sample_ids"]
    assert actual["gallery_ids"] == expected["gallery_ids"]
    assert torch.equal(actual["rankings"], expected["rankings"])
    assert torch.equal(actual["coarse_topm"], expected["coarse_topm"])
    assert not model.training
    assert not text_encoder.training


def test_retrieval_restores_mixed_modes_on_failure(monkeypatch):
    cache = _Cache()
    model = RCRModel(8, 6, 2, max_subjects=1).train()
    text_encoder = _TextEncoder().eval()

    def fail(*args, **kwargs):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(retrieval, "_encode_gallery_identity", fail)
    with pytest.raises(RuntimeError, match="projection failed"):
        retrieve_rankings(
            [_sample()],
            cache,
            _Tokenizer(),
            text_encoder,
            model,
            torch.device("cpu"),
            top_m=2,
            fine_batch_size=1,
            identity_batch_size=2,
        )
    assert model.training
    assert not text_encoder.training


def test_retrieval_rejects_nonpositive_coarse_batch_size():
    with pytest.raises(ValueError, match="must be positive"):
        retrieve_rankings(
            [_sample()],
            _Cache(),
            _Tokenizer(),
            _TextEncoder(),
            RCRModel(8, 6, 2),
            torch.device("cpu"),
            top_m=2,
            fine_batch_size=1,
            identity_batch_size=2,
            coarse_batch_size=0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for VRAM check"
)
def test_gallery_identity_gpu_memory_is_bounded_by_chunks():
    # The dense gallery identity alone is 128 MiB. Projection/coarse chunks
    # must stay below half that amount, including allocator/workspace overhead.
    torch.manual_seed(19)
    device = torch.device("cuda")
    cache = SimpleNamespace(
        image_ids=list(map(str, range(8192))),
        persons=torch.randn(8192, 32, 8),
        mask=torch.ones(8192, 32, dtype=torch.bool),
    )
    model = RCRModel(8, 128, 2).to(device).eval()
    query_identity = torch.randn(5, 128, device=device)
    logits = torch.randn(2, 5, device=device)
    query_mask = torch.ones(5, dtype=torch.bool, device=device)
    with torch.inference_mode():
        # Warm up CUDA libraries before measuring the gallery working set.
        model.identity_head(cache.persons[:1].to(device))
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        batches = _encode_gallery_identity(cache, model, device, 128)
        actual = _coarse_scores_chunked(query_identity, logits, query_mask, batches, 64)
        torch.cuda.synchronize(device)
        peak_extra = torch.cuda.max_memory_allocated(device) - baseline
        dense_bytes = len(cache.image_ids) * 32 * 128 * 4
        assert peak_extra < dense_bytes // 2
        assert all(identity.device.type == "cpu" for identity, _ in batches)
        assert actual.shape == (len(cache.image_ids),)


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
