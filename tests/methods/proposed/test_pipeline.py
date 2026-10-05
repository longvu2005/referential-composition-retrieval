"""Regression checks for CPU preparation, cache reuse and mixed precision."""

import re
import threading

import pytest
import torch

from rcr.methods.common.data import sample_selection_texts
from rcr.methods.proposed.batch import (
    finish_batch,
    prefetch_batches,
    prepare_visual_batch,
)
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import QueryTextCache, TextEncoder
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.objective import compute_loss
from tests.methods.proposed.test_batch import _samples, _TextEncoder, _Tokenizer


def write_cache(root, counts=(0, 2, 4), dim=8, patches=2):
    """Small on-disk cache, including an image with no detections."""
    root.mkdir(exist_ok=True)
    (root / "features").mkdir()
    generator = torch.Generator().manual_seed(71)
    ids = [f"image{i}" for i in range(len(counts))]
    people = torch.randn(len(counts), max(counts), dim, generator=generator).half()
    masks = torch.arange(max(counts))[None] < torch.tensor(counts)[:, None]
    scenes = torch.randn(len(counts), patches, dim, generator=generator).half()
    torch.save(
        {
            "image_ids": ids,
            "persons": people,
            "mask": masks,
            "patch_hw": (1, patches),
            "global_features": scenes.float().mean(dim=1),
        },
        root / "index.pt",
    )
    for i, count in enumerate(counts):
        torch.save(
            {
                "scene": scenes[i].clone(),
                "persons": people[i, :count].clone(),
                "boxes_scene": torch.tensor([[0.1, 0.2, 0.8, 0.9]] * count).reshape(
                    count, 4
                ),
                "identity_ids": [str(j) for j in range(count)],
            },
            root / "features" / f"{i}.pt",
        )
    return ids


def record_reads(cache, monkeypatch):
    reads, original = [], cache._load_item

    def load(index):
        reads.append(index)
        return original(index)

    monkeypatch.setattr(cache, "_load_item", load)
    return reads


@pytest.mark.parametrize("lru_mib", [0, 1])
def test_group_dedup_preserves_padding_order_and_immutable_storage(
    tmp_path, monkeypatch, lru_mib
):
    write_cache(tmp_path)
    cache = GalleryCache(tmp_path, lru_mib=lru_mib)
    reads = record_reads(cache, monkeypatch)
    queries, targets = cache.load_groups(
        torch.tensor([0, 1, 0]), torch.tensor([1, 2, 1])
    )
    assert reads == [0, 1, 2]
    assert queries[1].shape == (3, 2, 8)
    assert targets[1].shape == (3, 4, 8)
    assert queries[4].tolist() == [[False, False], [True, True], [False, False]]
    assert targets[3] == [
        ["0", "1", None, None],
        ["0", "1", "2", "3"],
        ["0", "1", None, None],
    ]
    torch.testing.assert_close(queries[0][1], targets[0][0])
    torch.testing.assert_close(targets[0][0], targets[0][2])
    expected = targets[0][0].clone()
    targets[0][0].zero_()
    actual = cache.load(torch.tensor([1]))
    torch.testing.assert_close(actual[0][0], expected)
    assert reads == ([0, 1, 2] if lru_mib else [0, 1, 2, 1])
    assert actual[0].dtype == torch.float32
    if lru_mib:
        assert cache._items[1][0]["scene"].dtype == torch.float16


def test_lru_evicts_least_recent_image_with_a_byte_budget(tmp_path, monkeypatch):
    # Two ~400 KiB images fit in 1 MiB; a third requires eviction.
    write_cache(tmp_path, counts=(1, 1, 1), dim=512, patches=400)
    cache = GalleryCache(tmp_path, lru_mib=1)
    reads = record_reads(cache, monkeypatch)
    for i in (0, 1, 0, 2, 0, 1):
        cache.load(torch.tensor([i]))
        assert cache._bytes == sum(size for _, size in cache._items.values())
        assert cache._bytes <= 1024**2
    assert reads == [0, 1, 2, 1]
    assert list(cache._items) == [0, 1]


def test_lru_does_not_retain_an_oversized_item(tmp_path, monkeypatch):
    write_cache(tmp_path, counts=(1,), dim=512, patches=1200)
    cache = GalleryCache(tmp_path, lru_mib=1)
    reads = record_reads(cache, monkeypatch)
    cache.load(torch.tensor([0, 0]))
    cache.load(torch.tensor([0]))
    assert reads == [0, 0]
    assert cache._bytes == 0 and not cache._items


def local_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[S1]",
        "[S2]",
        "then",
        "the",
        "man",
        "woman",
        "in",
        "white",
        "black",
        "blue",
        "red",
        "is",
        "beside",
        "and",
        "behind",
        "stands",
        "looking",
        "at",
    ]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(path))
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S1]", "[S2]"]})
    return tokenizer


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_cached_tokens_match_direct_tokenizer_with_all_subject_mentions(
    tmp_path, monkeypatch, padding_side
):
    tokenizer = local_tokenizer(tmp_path)
    tokenizer.padding_side = padding_side
    samples = _samples()
    samples[0]["final_change"] = (
        "Subject 1 is beside Subject 2 and Subject 1 is looking"
    )
    samples[1]["subjects"] = list(reversed(samples[1]["subjects"]))
    samples[1]["final_desc"] = (
        "Identify Subject 2 as the woman and Subject 1 as the man in blue"
    )
    texts = [text for sample in samples for text in sample_selection_texts(sample)]
    changes = [
        re.sub(r"Subject\s+(\d)", r"[S\1]", row["final_change"]) for row in samples
    ]
    expected_selection = tokenizer(texts, padding=True, return_tensors="pt")
    expected_change = tokenizer(changes, padding=True, return_tensors="pt")
    calls, original = [], type(tokenizer).__call__

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", counted)
    cache = QueryTextCache(tokenizer)
    cache.prepare(samples + samples)
    assert len(cache.rows) == 2 and len(calls) == 2
    for _ in range(2):
        tokens = cache.batch(samples)
        torch.testing.assert_close(
            tokens["selection_ids"], expected_selection["input_ids"]
        )
        torch.testing.assert_close(
            tokens["selection_mask"], expected_selection["attention_mask"]
        )
        torch.testing.assert_close(tokens["change_ids"], expected_change["input_ids"])
        torch.testing.assert_close(
            tokens["change_mask"], expected_change["attention_mask"]
        )
        for n, sample in enumerate(samples):
            for s, subject in enumerate(sample["subjects"]):
                sid = subject["subject_id"]
                marker = tokenizer.convert_tokens_to_ids(f"[S{sid}]")
                positions = expected_change["input_ids"][n] == marker
                assert torch.equal(tokens["subject_token_mask"][n, s], positions)
                assert tokens["subject_pos"][n, s] == positions.nonzero()[0, 0]
                assert tokens["subject_ids"][n, s] == sid
    assert len(calls) == 2  # No tokenizer call during either batch.


def test_prefetch_keeps_rng_in_caller_and_matches_synchronous_order():
    main_thread = threading.get_ident()

    def run(depth):
        torch.manual_seed(19)
        generator = torch.Generator().manual_seed(11)
        workers, result = [], []
        ahead = threading.Event()

        def jobs():
            for i in range(6):
                assert threading.get_ident() == main_thread
                yield i, torch.randint(1000, (7,), generator=generator)

        def prepare(job):
            workers.append(threading.get_ident())
            if job[0] == 1:
                ahead.set()
            return job

        with prefetch_batches(jobs(), prepare, depth) as batches:
            for i, sample in batches:
                if depth and i == 0:
                    assert ahead.wait(5), "next batch was not prefetched"
                result.append((i, sample.tolist(), torch.rand(3).tolist()))
        assert len(set(workers)) == 1
        assert (workers[0] == main_thread) == (depth == 0)
        return result

    assert run(0) == run(1)
    assert not any(t.name.startswith("rcr-cache") for t in threading.enumerate())


@pytest.mark.parametrize("abort", ["failure", "break"])
def test_prefetch_closes_worker_on_error_or_early_exit(abort):
    def prepare(value):
        if abort == "failure" and value == 1:
            raise RuntimeError("feature read failed")
        return value

    if abort == "failure":
        with (
            pytest.raises(RuntimeError, match="feature read failed"),
            prefetch_batches(range(4), prepare, 1) as batches,
        ):
            list(batches)
    else:
        with prefetch_batches(range(4), prepare, 1) as batches:
            assert next(batches) == 0
    assert not any(t.name.startswith("rcr-cache") for t in threading.enumerate())
    with prefetch_batches([], prepare, 1) as batches:
        assert list(batches) == []


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required for FP16 training"
            ),
        ),
    ],
)
@pytest.mark.parametrize("counts", [(0, 0, 0), (2, 0, 4)])
def test_amp_training_finite_with_empty_images_and_padded_boxes(
    tmp_path, device, counts
):
    torch.manual_seed(5)
    ids = write_cache(tmp_path, counts=counts, dim=8, patches=14)
    cache = GalleryCache(tmp_path, lru_mib=1)
    samples = []
    for i, sample in enumerate(_samples()):
        samples.append(
            {
                **sample,
                "query_image_id": ids[i],
                "positive_image_ids": [ids[2]],
                "subjects": [
                    {"subject_id": 1, "identity_ids": ["0"]},
                    {"subject_id": 2, "identity_ids": ["1"]},
                ],
            }
        )
    candidates = [[ids[2], ids[1]], [ids[2], ids[0]]]
    visual = prepare_visual_batch(
        samples, candidates, cache, state_image_ids=[set(ids)] * 2
    )
    tokens = QueryTextCache(_Tokenizer()).batch(samples)
    model = RCRModel(8, 6, 2, state_dim=4, geo_dim=4).to(device).train()
    encoder = _TextEncoder(8).to(device).train()
    optimizer = torch.optim.AdamW([*model.parameters(), *encoder.parameters()], lr=1e-3)
    # A modest initial scale tests a real update on the tiny fixture.
    scaler = torch.amp.GradScaler(device, init_scale=16)
    before = model.reasoner.score[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16):
            batch = finish_batch(visual, tokens, encoder, device)
            loss, parts = compute_loss(model, batch, cache.patch_hw)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in parts.values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [*model.parameters(), *encoder.parameters()]
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert encoder.embedding.weight.grad.abs().sum() > 0
        assert all(p.dtype == torch.float32 for p in parameters)
        scaler.step(optimizer)
        scaler.update()
    assert not torch.equal(before, model.reasoner.score[-1].weight)


def test_text_cache_keeps_real_bert_trainable_across_steps(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizer = local_tokenizer(tmp_path)
    backbone = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    encoder = TextEncoder(backbone, 8).train()
    text_cache = QueryTextCache(tokenizer)
    text_cache.prepare(_samples())
    tokens = text_cache.batch(_samples())
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    before = backbone.embeddings.word_embeddings.weight.detach().clone()
    from rcr.methods.proposed.encoders import encode_tokenized_text

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        encoded = encode_tokenized_text(tokens, encoder, "cpu")
        (
            encoded["change"][..., 0].mean() + encoded["selections"][..., 1].mean()
        ).backward()
        grad = backbone.embeddings.word_embeddings.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(before, backbone.embeddings.word_embeddings.weight)
