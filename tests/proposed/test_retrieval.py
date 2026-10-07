from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed import ranking as retrieval
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import (
    _coarse_scores_chunked,
    _encode_gallery_identity,
    mine_hard_negatives,
    retrieve_rankings,
    retrieve_variants,
)
from rcr.proposed.scores import coarse_scores


def test_mining_keeps_only_train_negatives_and_bounded_outputs(monkeypatch):
    torch.manual_seed(15)
    cache = _Cache(["q", "val", "a", "b", "disputed", "n1", "n2", "n3"])
    gallery = [x for x in cache.image_ids if x != "val"]
    model, text_encoder = RCRModel(8, 6, 2).train(), _TextEncoder().train()
    settings = dict(
        top_m=2,
        fine_batch_size=1,
        identity_batch_size=2,
        coarse_batch_size=2,
        coarse_beta=0.4,
        rerank=False,
    )
    sample = {**_sample(), "positive_image_ids": ["a", "b"]}
    monkeypatch.setattr(model, "score_target", lambda *a: pytest.fail("fine mining"))
    full = retrieve_variants(
        [sample],
        cache,
        _Tokenizer(),
        text_encoder,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        variants={"run": settings},
    )["run"]
    forbidden = {"q", "a", "b", "disputed"}
    expected = [gallery[i] for i in full["rankings"][0] if gallery[i] not in forbidden][
        :2
    ]
    pools = mine_hard_negatives(
        [sample],
        cache,
        _Tokenizer(),
        text_encoder,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        retrieval=settings,
        pool_size=2,
        excluded={"s1": {"disputed"}},
    )
    assert pools == {"s1": expected}
    assert model.training and text_encoder.training
    limited = retrieve_variants(
        [sample],
        cache,
        _Tokenizer(),
        text_encoder,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        variants={"run": settings},
        ranking_limit=2,
    )["run"]
    assert limited["rankings"].shape == limited["coarse_rankings"].shape == (1, 2)
    torch.testing.assert_close(limited["rankings"], full["rankings"][:, :2])


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
    def __init__(self, image_ids=None) -> None:
        generator = torch.Generator().manual_seed(3)
        self.image_ids = image_ids or ["q", "a", "b"]
        self.persons = torch.randn(len(self.image_ids), 1, 8, generator=generator)
        self.scenes = torch.randn(len(self.image_ids), 1, 8, generator=generator)
        self.mask = torch.ones(len(self.image_ids), 1, dtype=torch.bool)
        self.patch_hw = (1, 1)
        self.global_features = self.scenes.mean(dim=1)

    @property
    def by_id(self):
        return {image_id: i for i, image_id in enumerate(self.image_ids)}

    def load_groups(self, *groups):
        return tuple(self.load(indices) for indices in groups)

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
        "target_image_id": "a",
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
        gallery_ids=cache.image_ids,
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
@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_chunked_coarse_matches_dense_with_padding_and_empty_images(
    identity_batch_size, coarse_batch_size, empty_query, beta
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
    state = dict(
        query_state=torch.nn.functional.normalize(torch.randn(4), dim=-1),
        gallery_state=torch.nn.functional.normalize(torch.randn(7, 4), dim=-1),
        beta=beta,
    )
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
            **state,
        )
        actual = _coarse_scores_chunked(
            query_identity, logits, query_mask, batches, coarse_batch_size, **state
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
        gallery_ids=cache.image_ids,
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
    def dense_gallery(cache, model, device, batch_size, image_indices):
        del batch_size
        return [
            (
                model.identity_head(cache.persons[image_indices].to(device)),
                cache.mask[image_indices].to(device),
            )
        ]

    def dense_scores(query_identity, logits, query_mask, batches, batch_size, **state):
        del batch_size
        identity, mask = batches[0]
        return coarse_scores(
            query_identity, logits, identity, mask, query_mask=query_mask, **state
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
            gallery_ids=cache.image_ids,
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
            gallery_ids=["q", "a", "b"],
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


@pytest.mark.parametrize("beta", [0.0, 0.3])
@pytest.mark.parametrize("top_m", [1, 20])
def test_split_retrieval_uses_local_indices_and_only_loads_selected_images(beta, top_m):
    cache = _Cache()
    loaded = []
    load = cache.load

    def record(indices):
        loaded.extend(indices.tolist())
        return load(indices)

    cache.load = record
    output = retrieve_rankings(
        [{**_sample(), "target_image_id": "b", "positive_image_ids": ["b"]}],
        cache,
        _Tokenizer(),
        _TextEncoder(),
        RCRModel(8, 6, 2, max_subjects=1, coarse_beta=beta),
        torch.device("cpu"),
        gallery_ids=["b", "q"],  # Cache indices 2,0; query is local index 1.
        top_m=top_m,
        fine_batch_size=1,
        identity_batch_size=1,
        coarse_batch_size=1,
    )

    assert output["gallery_ids"] == ["b", "q"]
    assert output["rankings"].tolist() == [[0]]
    assert output["coarse_topm"].tolist() == [[0]]
    assert loaded == [0, 2]


def test_evaluate_retrieval_output_uses_official_split_metrics() -> None:
    sample = _sample()
    output = {
        "sample_ids": ["s1"],
        "gallery_ids": ["q", "a", "b"],
        "rankings": torch.tensor([[1, 2]], dtype=torch.int32),
        "coarse_topm": torch.tensor([[1, 2]], dtype=torch.int32),
    }
    data = SimpleNamespace(
        gallery_ids=["q", "a", "b"],
        images_by_id={x: {"path": f"test/{x}.jpg"} for x in ["q", "a", "b"]},
        splits={"test": ["s1"]},
        gt_head_boxes_by_image={
            "q": [{"identity_id": "p1"}],
            "a": [{"identity_id": "p1"}],
        },
    )

    result = evaluate_retrieval_output(
        data, [sample], output, candidate_ks=[1, 2], split="test"
    )

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
        images_by_id={x: {"path": f"test/{x}.jpg"} for x in ["q", "a", "b"]},
        splits={"test": ["s1"]},
        gt_head_boxes_by_image={},
    )

    with pytest.raises(ValueError, match="invalid gallery index"):
        evaluate_retrieval_output(
            data, [sample], output, candidate_ks=[1], split="test"
        )


def test_evaluation_rejects_rankings_from_the_old_global_gallery():
    ids = ["q", "a", "b", "train_image", "leftover_image"]
    data = SimpleNamespace(
        gallery_ids=ids,
        images_by_id={
            image_id: {"path": f"{split}/{image_id}.jpg"}
            for image_id, split in zip(
                ids, ["test", "test", "test", "train", "leftover"], strict=True
            )
        },
        splits={"test": ["s1"]},
        gt_head_boxes_by_image={},
    )
    output = {
        "sample_ids": ["s1"],
        "gallery_ids": ids,
        "rankings": torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        "coarse_topm": torch.tensor([[1, 2]], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="do not match the test gallery"):
        evaluate_retrieval_output(data, [_sample()], output, [1, 2], split="test")


def test_sweep_shares_scores_and_matches_independent_normalized_runs(monkeypatch):
    torch.manual_seed(42)
    cache = _Cache(["q", "a", "b", "c", "d"])
    cache.mask[-1] = False
    model, encoder = RCRModel(8, 6, 2, max_subjects=1), _TextEncoder()
    args = ([_sample()], cache, _Tokenizer(), encoder, model, torch.device("cpu"))
    base = dict(
        top_m=3,
        fine_batch_size=2,
        identity_batch_size=2,
        coarse_batch_size=2,
        coarse_normalization="zscore",
        rerank=False,
    )
    variants = {
        "id": {**base, "coarse_mode": "identity_only"},
        "state": {**base, "coarse_mode": "state_only"},
        "beta0": {**base, "coarse_beta": 0.0},
        "beta1": {**base, "coarse_beta": 1.0},
        "fine": {**base, "coarse_beta": 1.0, "rerank": True},
        "fine_small": {**base, "coarse_beta": 1.0, "rerank": True, "top_m": 1},
    }
    # Joint 2D sweep, with zero weights and the existing (.4, .4) pair.
    variants.update(
        {
            f"joint_{beta}_{weight}": {
                **base,
                "coarse_beta": beta,
                "fine_coarse_weight": weight,
                "rerank": True,
            }
            for beta in (0.0, 0.4, 1.0)
            for weight in (0.0, 0.4, 1.0)
        }
    )
    calls, loads, combinations = [], [], []
    score_identity, load = retrieval._coarse_scores_chunked, cache.load
    combine = retrieval.combine_scores

    def count_scores(*args, **kwargs):
        calls.append(1)
        return score_identity(*args, **kwargs)

    def count_loads(indices):
        loads.extend(indices.tolist())
        return load(indices)

    def count_combinations(*args, **kwargs):
        combinations.append(1)
        return combine(*args, **kwargs)

    monkeypatch.setattr(retrieval, "_coarse_scores_chunked", count_scores)
    monkeypatch.setattr(cache, "load", count_loads)
    monkeypatch.setattr(retrieval, "combine_scores", count_combinations)
    actual = retrieve_variants(*args, gallery_ids=cache.image_ids, variants=variants)
    assert len(calls) == 1  # One ID score vector for all beta/mode variants.
    assert len(combinations) == 5  # Reuse normalization/sort across final weights.
    assert loads[0] == 0 and len(loads[1:]) == len(set(loads[1:])) == 3
    for name, cfg in variants.items():
        # Change chunk sizes to catch normalization accidentally performed per chunk.
        for batch_size in (1, 20):
            expected = retrieve_rankings(
                *args,
                gallery_ids=cache.image_ids,
                **{
                    **cfg,
                    "identity_batch_size": batch_size,
                    "coarse_batch_size": batch_size,
                },
            )
            for key in ("rankings", "coarse_topm", "coarse_rankings"):
                assert torch.equal(actual[name][key], expected[key])
            if cfg["rerank"]:
                assert torch.equal(
                    actual[name]["fine_rankings"], expected["fine_rankings"]
                )
        assert actual[name]["rankings"].shape == (1, 4)
        assert set(actual[name]["rankings"][0].tolist()) == {1, 2, 3, 4}
    assert torch.equal(actual["id"]["rankings"], actual["beta0"]["rankings"])
    assert torch.equal(actual["beta1"]["rankings"], actual["beta1"]["coarse_rankings"])


def test_state_only_coarse_never_evaluates_identity_or_fine(monkeypatch):
    cache = _Cache(["q", "a", "b", "c"])
    cache.mask[:] = False
    model = RCRModel(8, 6, 2, max_subjects=1, state_dim=2)

    def fail(*args, **kwargs):
        pytest.fail("state-only coarse used the ID/fine branch")

    monkeypatch.setattr(model, "encode_query", fail)
    monkeypatch.setattr(model, "score_target", fail)
    monkeypatch.setattr(retrieval, "_encode_gallery_identity", fail)
    monkeypatch.setattr(retrieval, "_coarse_scores_chunked", fail)
    monkeypatch.setattr(
        model, "encode_text_state", lambda *args: torch.tensor([[1.0, 0]])
    )
    monkeypatch.setattr(
        retrieval,
        "_encode_gallery_state",
        lambda *args: torch.tensor([[0.0, 1], [0.1, 0], [1.0, 0], [0.5, 0]]),
    )
    output = retrieve_rankings(
        [_sample()],
        cache,
        _Tokenizer(),
        _TextEncoder(),
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        top_m=1,
        fine_batch_size=1,
        identity_batch_size=2,
        coarse_mode="state_only",
        coarse_beta=0,
        coarse_normalization="zscore",
        rerank=False,
    )
    assert output["rankings"].tolist() == [[2, 3, 1]]
    assert output["coarse_topm"].tolist() == [[2]]


def test_runtime_beta_changes_ranking_without_mutating_checkpoint_beta(monkeypatch):
    cache = _Cache()
    model = RCRModel(8, 6, 2, max_subjects=1, state_dim=2, coarse_beta=0.0)
    monkeypatch.setattr(
        retrieval,
        "_coarse_scores_chunked",
        lambda *args: torch.tensor([0.0, 1.0, -1.0]),
    )
    monkeypatch.setattr(
        model, "encode_text_state", lambda *args: torch.tensor([[1.0, 0.0]])
    )
    monkeypatch.setattr(
        retrieval,
        "_encode_gallery_state",
        lambda *args: torch.tensor([[0.0, 0], [-1.0, 0], [1.0, 0]]),
    )
    settings = dict(
        top_m=1,
        fine_batch_size=1,
        identity_batch_size=2,
        rerank=False,
        coarse_normalization="zscore",
    )
    results = retrieve_variants(
        [_sample()],
        cache,
        _Tokenizer(),
        _TextEncoder(),
        model,
        torch.device("cpu"),
        gallery_ids=cache.image_ids,
        variants={"inherited": settings, "override": {**settings, "coarse_beta": 2.0}},
    )
    assert results["inherited"]["rankings"].tolist() == [[1, 2]]
    assert results["override"]["rankings"].tolist() == [[2, 1]]
    assert model.coarse_beta == 0.0


@pytest.mark.parametrize("query_batch_size", [1, 2, 8])
@pytest.mark.parametrize(
    "mode,beta",
    [
        ("identity_only", 0.4),
        ("state_only", 0.4),
        ("identity_state", 0.0),
        ("identity_state", 0.4),
    ],
)
@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_batched_mining_matches_single_query_rankings_without_composition(
    tmp_path, monkeypatch, query_batch_size, mode, beta, normalization
):
    from rcr.proposed.cache.store import GalleryCache
    from tests.proposed.test_batch import _TextEncoder as VariableTextEncoder
    from tests.proposed.test_batch import _Tokenizer as TwoSubjectTokenizer
    from tests.proposed.test_pipeline import write_cache

    torch.manual_seed(31)
    ids = write_cache(tmp_path, counts=(0, 3, 1, 2, 4, 0, 2, 3, 2))
    cache = GalleryCache(tmp_path, lru_mib=1)
    # Non-contiguous/reordered train gallery: exclude a simulated validation image.
    gallery = [ids[i] for i in (4, 1, 0, 6, 2, 3, 5, 7)]
    samples = []
    for i in range(4):
        sample = {
            **_sample(f"s{i}"),
            "query_image_id": ids[i],
            "positive_image_ids": [ids[4], ids[6]],
        }
        if i % 2:
            sample["subjects"] = [
                {"subject_id": 1, "identity_ids": ["0"]},
                {"subject_id": 2, "identity_ids": ["1"]},
            ]
            sample["final_desc"] = (
                "Identify Subject 1 as the man and Subject 2 as the woman"
            )
            sample["final_change"] = (
                "Subject 2 stands beside Subject 1 and Subject 2 smiles"
            )
        samples.append(sample)
    tokenizer = TwoSubjectTokenizer()
    encoder = VariableTextEncoder(8).eval()
    model = RCRModel(8, 6, 2, state_dim=4).train()
    settings = dict(
        top_m=2,
        fine_batch_size=1,
        identity_batch_size=3,
        coarse_batch_size=2,
        coarse_mode=mode,
        coarse_beta=beta,
        coarse_normalization=normalization,
        rerank=False,
    )
    excluded = {row["sample_id"]: {ids[7]} for row in samples}
    expected = retrieve_rankings(
        samples,
        cache,
        tokenizer,
        encoder,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        **settings,
    )
    pools = {}
    for row, ranking in zip(samples, expected["rankings"], strict=True):
        forbidden = {
            row["query_image_id"],
            *row["positive_image_ids"],
            *excluded[row["sample_id"]],
        }
        pools[row["sample_id"]] = [
            gallery[i] for i in ranking if gallery[i] not in forbidden
        ][:3]

    def fail(*args, **kwargs):
        pytest.fail("coarse mining must not compose or run fine scoring")

    monkeypatch.setattr(model.composition, "forward", fail)
    monkeypatch.setattr(model, "score_target", fail)
    if mode == "state_only":
        monkeypatch.setattr(model, "encode_grounded_identity", fail)
    calls = []
    handle = encoder.register_forward_hook(lambda *args: calls.append(1))
    actual = mine_hard_negatives(
        samples,
        cache,
        tokenizer,
        encoder,
        model,
        torch.device("cpu"),
        gallery_ids=gallery,
        retrieval=settings,
        pool_size=3,
        excluded=excluded,
        query_batch_size=query_batch_size,
    )
    handle.remove()
    assert actual == pools and list(actual) == [row["sample_id"] for row in samples]
    assert len(calls) == 4 * ((2 + query_batch_size - 1) // query_batch_size)
    assert model.training and not encoder.training


def test_batched_mining_retains_canonical_ties_and_restores_modes_on_failure(
    monkeypatch,
):
    torch.manual_seed(21)
    cache = _Cache(["q", "a", "b", "c", "d"])
    cache.mask[0] = False  # No query identity evidence: all nonempty images tie.
    model, encoder = RCRModel(8, 6, 2).train(), _TextEncoder().eval()
    samples = [_sample("s1"), _sample("s2")]
    settings = dict(
        identity_batch_size=2,
        coarse_mode="identity_only",
        coarse_normalization="zscore",
    )
    kwargs = dict(
        gallery_ids=cache.image_ids, retrieval=settings, pool_size=2, query_batch_size=2
    )
    assert mine_hard_negatives(
        samples, cache, _Tokenizer(), encoder, model, torch.device("cpu"), **kwargs
    ) == {"s1": ["b", "c"], "s2": ["b", "c"]}

    def fail(*args, **kwargs):
        raise RuntimeError("mining failed")

    monkeypatch.setattr(model, "encode_grounded_identity", fail)
    with pytest.raises(RuntimeError, match="mining failed"):
        mine_hard_negatives(
            samples, cache, _Tokenizer(), encoder, model, torch.device("cpu"), **kwargs
        )
    assert model.training and not encoder.training
