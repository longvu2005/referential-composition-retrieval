"""Numerical contracts and work bounds for the GPU execution paths."""

import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rcr.proposed import losses, ranking
from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.cache.native import NativeTargetBatch
from rcr.proposed.matching import partial_identity_matching, unique_box_mask
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.scores import (
    fafa_binding_scores,
    identity_scores,
    identity_scores_paired,
)
from tests.helpers import TextBackbone, TinyCache, Tokenizer, _only_loss, _rank, _sample
from tools import diagnose_gpu

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_warmup_skips_target_scene_reads_with_identical_loss_and_gradients(
    device, monkeypatch
):
    cache = TinyCache()
    model = RCRModel(dim=8, identity_dim=6, num_heads=2, state_dim=4).to(device)
    reference = deepcopy(model)
    text = TextEncoder(TextBackbone(), dim=8).to(device)
    reference_text = deepcopy(text)
    supervision = SimpleNamespace(
        load_ids=lambda ids, width: [["p1"] * width for _ in ids]
    )
    args = ([_sample("train", "q", "a")], [["a", "n"]], cache, Tokenizer())
    options = {"supervision": supervision, "identity_image_ids": [{"a", "n"}]}
    expected_batch = build_batch(*args, reference_text, device, **options)
    reads = []
    load = cache.load

    def track(indices):
        reads.extend(cache.image_ids[i] for i in indices.tolist())
        return load(indices)

    monkeypatch.setattr(cache, "load", track)
    batch = build_batch(*args, text, device, load_target_context=False, **options)
    assert reads == ["q"]
    assert "target_scene" not in batch and "target_boxes" not in batch
    for name, value in batch.items():
        torch.testing.assert_close(value, expected_batch[name])
    settings = losses.loss_settings({"context_weight": 0.0})
    expected, _ = losses.compute_loss(reference, expected_batch, (2, 2), **settings)
    actual, _ = losses.compute_loss(model, batch, (2, 2), **settings)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    parameters = [*model.parameters(), *text.proj.parameters()]
    reference_parameters = [*reference.parameters(), *reference_text.proj.parameters()]
    for parameter, other in zip(parameters, reference_parameters, strict=True):
        torch.testing.assert_close(parameter.grad, other.grad)


def test_context_cannot_omit_required_target_scenes():
    with pytest.raises(ValueError, match="context training requires target scenes"):
        prepare_visual_batch(
            [],
            [],
            None,
            supervision=None,
            identity_image_ids=[],
            fafa_pooling="subject_mean",
            load_target_context=False,
        )


def test_diagnostics_records_cuda_kernel_failure(monkeypatch):
    cfg = {"runtime": {"device": "cuda"}, "train": {"amp": True}}
    monkeypatch.setattr(diagnose_gpu, "_command", lambda args: {})
    monkeypatch.setattr(diagnose_gpu.shutil, "which", lambda name: None)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "RTX 5060")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    def unsupported(*args, **kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "randn", unsupported)
    report = diagnose_gpu.environment_report(cfg, torch.device("cuda"))
    assert report["compute_capability"] == [12, 0]
    assert report["cuda_kernel_smoke"]["status"] == "failed"
    assert "no kernel image" in report["cuda_kernel_smoke"]["error"]


def test_diagnostics_saves_failure_before_attempting_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose_gpu, "load_config", lambda path: {})
    environment = {"cuda_kernel_smoke": {"status": "failed", "error": "unsupported"}}
    monkeypatch.setattr(diagnose_gpu, "environment_report", lambda *args: environment)
    monkeypatch.setattr(
        diagnose_gpu, "cache_report", lambda *args, **kwargs: pytest.fail("cache read")
    )
    output = tmp_path / "diagnostics.json"
    status = diagnose_gpu.main(
        ["--device", "cuda", "--cache-batches", "1", "--output", str(output)]
    )
    assert status == 1
    assert json.loads(output.read_text()) == {"environment": environment}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "priority,mask,threshold,expected",
    [
        (None, [True, True, True], 0.8, [True, False, True]),
        ([1, 3, 2], [True, True, True], 0.8, [False, True, False]),
        (None, [False, True, True], 0.8, [False, True, False]),
        (None, [True, True, True], 1.0, [True, True, True]),
        (None, [False, False, False], 0.8, [False, False, False]),
    ],
)
def test_box_suppression_keeps_stable_greedy_semantics(
    device, priority, mask, threshold, expected
):
    # A overlaps B, B overlaps C, A does not overlap C at this threshold.
    boxes = torch.tensor(
        [[[0, 0, 1, 1], [0.1, 0, 1.1, 1], [0.2, 0, 1.2, 1]]], device=device
    )
    valid = torch.tensor([mask], device=device)
    boxes.masked_fill_(~valid[..., None], torch.nan)
    priority = None if priority is None else torch.tensor([priority], device=device)
    actual = unique_box_mask(boxes, valid, threshold=threshold, priority=priority)
    assert actual.cpu().tolist() == [expected]


def test_box_suppression_has_no_per_box_tensor_scalar_reads():
    class Counts(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = Counter()

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            self.calls[str(function)] += 1
            return function(*args, **(kwargs or {}))

    boxes = torch.rand(64, 8, 4)
    mask = torch.ones(64, 8, dtype=torch.bool)
    with Counts() as counts:
        unique_box_mask(boxes, mask, threshold=0.9)
    assert not counts.calls["aten.nonzero.default"]
    assert not counts.calls["aten._local_scalar_dense.default"]


@pytest.mark.parametrize("device", DEVICES)
def test_matching_preserves_private_nulls_duplicates_and_masked_nan(device):
    query = torch.tensor([[[1.0, 0], [0, 1], [torch.nan, torch.nan]]], device=device)
    target = torch.tensor([[[1.0, 0], [torch.nan, torch.nan], [0, 1]]], device=device)
    membership = torch.tensor([[[0.9, 0, 0], [0, 0.8, 0]]], device=device)
    subjects = torch.ones(1, 2, device=device, dtype=torch.bool)
    boxes = torch.tensor([[[0.0, 0, 1, 1], [0, 0, 1, 1], [2, 0, 3, 1]]], device=device)
    matched = partial_identity_matching(
        query, target, membership, subjects, target_boxes=boxes
    )
    assert matched.indices.cpu().tolist() == [[0, 2, -1]]
    torch.testing.assert_close(matched.coverage, torch.ones(1, device=device))
    nulls = partial_identity_matching(
        query, target, membership, subjects, target_boxes=boxes, identity_threshold=1.0
    )
    assert nulls.indices.cpu().tolist() == [[-1, -1, -1]]
    assert nulls.coverage.cpu().tolist() == [0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_paired_coarse_scores_match_per_query_scores_and_gradients(device):
    torch.manual_seed(17)
    query = torch.randn(3, 4, 6, device=device, requires_grad=True)
    logits = torch.randn(3, 2, 4, device=device, requires_grad=True)
    target = torch.randn(3, 5, 7, 6, device=device, requires_grad=True)
    mask = torch.rand(3, 5, 7, device=device) > 0.3
    mask[1, 2] = False
    query_mask = torch.rand(3, 4, device=device) > 0.2
    subjects = torch.tensor([[True, False], [True, True], [True, True]], device=device)
    expected = torch.stack(
        [
            identity_scores(
                query[i], logits[i], target[i], mask[i], query_mask[i], subjects[i]
            )
            for i in range(3)
        ]
    )
    actual = identity_scores_paired(query, logits, target, mask, query_mask, subjects)
    torch.testing.assert_close(actual, expected)
    inputs = (query, logits, target)
    expected_grad = torch.autograd.grad(
        expected.masked_fill(~expected.isfinite(), 0).sum(), inputs
    )
    actual_grad = torch.autograd.grad(
        actual.masked_fill(~actual.isfinite(), 0).sum(), inputs
    )
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference)


def _query(model, batch):
    return model.encode_query(
        batch["query_scene"],
        batch["query_persons"],
        batch["query_boxes"],
        patch_hw=(2, 2),
        query_person_mask=batch["query_person_mask"],
        fafa_query=batch["fafa_query"],
        **{
            key: batch[key]
            for key in (
                "selections",
                "change",
                "subject_pos",
                "selection_mask",
                "change_mask",
                "subject_token_mask",
                "subject_ids",
            )
        },
    )


def _dense_context(model, query, batch, patch_hw, size, **kwargs):
    queries, candidates = batch["positive_mask"].shape
    repeated = query.repeat_candidates(candidates)
    raw, matching = model.score_context(
        repeated,
        batch["target_scene"].flatten(0, 1),
        batch["target_persons"].flatten(0, 1),
        batch["target_boxes"].flatten(0, 1),
        patch_hw,
        batch["target_mask"].flatten(0, 1),
    )
    binding = fafa_binding_scores(
        repeated.fafa,
        batch["fafa_targets"].flatten(0, 1),
        matching=matching,
        k=model.fda_k,
        missing_penalty=model.fine_matching["missing_penalty"],
        pooling=batch["fafa_pooling"],
    )
    shape = (queries, candidates)
    return (
        raw.view(shape),
        matching.coverage.view(shape),
        binding.view(shape),
        matching.indices.unflatten(0, shape),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True])
def test_sparse_deferred_context_matches_dense_loss_and_gradients(
    objective_batch, monkeypatch, device, amp
):
    model, batch = objective_batch
    model.to(device)
    batch = {
        key: value.detach().to(device).requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
    reference = deepcopy(model)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with monkeypatch.context() as guard:
        guard.setattr(losses, "_context_scores", _dense_context)
        with torch.autocast(device, dtype=dtype, enabled=amp):
            expected, _ = losses.compute_loss(
                reference, deepcopy(batch), (2, 2), **_only_loss("context")
            )
        expected.backward()
    cache, loaded = TinyCache(), []
    load = cache.native_targets

    def native_targets(indices, width):
        loaded.append((indices.tolist(), width))
        return load(indices, width)

    monkeypatch.setattr(cache, "native_targets", native_targets)
    batch["fafa_targets"] = NativeTargetBatch(cache, torch.tensor([1, 2, 4]))
    calls = []
    match = model.match_target

    def matched(query, identity, boxes, mask):
        calls.append(len(identity))
        return match(query, identity, boxes, mask)

    monkeypatch.setattr(model, "match_target", matched)
    with torch.autocast(device, dtype=dtype, enabled=amp):
        actual, _ = losses.compute_loss(
            model, batch, (2, 2), context_batch_size=1, **_only_loss("context")
        )
    actual.backward()
    torch.testing.assert_close(
        actual, expected, rtol=2e-3 if amp else 1e-5, atol=2e-4 if amp else 1e-6
    )
    assert loaded == [([1], 1), ([2], 1)]
    assert calls == [1, 1]  # Backward recomputes the network, not assignment or I/O.
    for (name, parameter), other in zip(
        model.named_parameters(), reference.parameters(), strict=True
    ):
        if other.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                other.grad,
                msg=name,
                rtol=1e-2 if amp else 3e-4,
                atol=5e-4 if amp else 2e-6,
            )


def test_zero_context_mining_skips_reasoner_and_preserves_binding(objective_batch):
    model, batch = objective_batch
    query = _query(model, batch).repeat_candidates(3)
    args = [
        batch[key].flatten(0, 1)
        for key in ("target_scene", "target_persons", "target_boxes")
    ]
    options = dict(
        person_mask=batch["target_mask"].flatten(0, 1),
        fafa_targets=batch["fafa_targets"].flatten(0, 1),
        context_weight=0.0,
        return_details=True,
    )
    with torch.inference_mode():
        expected = model.score_target(query, *args, (2, 2), **options)
        calls = []
        handle = model.reasoner.register_forward_hook(lambda *args: calls.append(1))
        try:
            actual = model.score_target(
                query, *args, (2, 2), compute_context=False, **options
            )
        finally:
            handle.remove()
    assert not calls
    for key in (
        "score",
        "coverage",
        "matched_indices",
        "binding",
        "binding_subject_mean",
    ):
        torch.testing.assert_close(actual[key], expected[key])


def test_retrieval_query_batching_preserves_order_and_evidence(tiny_ranker):
    expected = _rank(tiny_ranker, query_batch_size=1)
    actual = _rank(tiny_ranker, query_batch_size=8)
    assert actual["sample_ids"] == expected["sample_ids"]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)


def test_retrieval_batching_restores_mixed_subject_order(tiny_ranker):
    cache, model, text, samples, settings = tiny_ranker
    dual = deepcopy(samples[0])
    dual["sample_id"] = "dual"
    dual["subjects"].append({"subject_id": 2, "identity_ids": ["p2"]})
    dual["final_desc"] += " and Subject 2 as another person"
    dual["final_change"] = "then retrieve images where Subject 1 is beside Subject 2"
    mixed = [samples[0], dual, samples[1]]
    args = (mixed, cache, Tokenizer(), text, model, torch.device("cpu"))
    expected = ranking.retrieve_rankings(
        *args, gallery_ids=cache.image_ids, query_batch_size=1, **settings
    )
    actual = ranking.retrieve_rankings(
        *args, gallery_ids=cache.image_ids, query_batch_size=8, **settings
    )
    assert actual["sample_ids"] == [row["sample_id"] for row in mixed]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)


def test_context_trims_target_padding_before_native_load(objective_batch, monkeypatch):
    model, batch = objective_batch
    for name in ("target_persons", "target_boxes", "target_mask"):
        value = batch[name]
        shape = (*value.shape[:2], 4, *value.shape[3:])
        padded = value.new_zeros(shape)
        padded[:, :, :1] = value.detach()
        batch[name] = padded
    cache, widths = TinyCache(), []
    native = cache.native_targets

    def load(indices, width):
        widths.append(width)
        return native(indices, width)

    monkeypatch.setattr(cache, "native_targets", load)
    batch["fafa_targets"] = NativeTargetBatch(cache, torch.tensor([1, 2, 4]))
    total, _ = losses.compute_loss(
        model, batch, (2, 2), context_batch_size=1, **_only_loss("context")
    )
    total.backward()
    assert widths == [1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gallery_device_cache_honors_budget():
    batches = [(torch.randn(512, 8, 128), torch.ones(512, 8, dtype=torch.bool))]
    state = torch.randn(512, 128)
    original, original_state = ranking._cache_gallery_on_device(
        batches, state, torch.device("cuda"), 1
    )
    assert original is batches and original_state is state
    resident, resident_state = ranking._cache_gallery_on_device(
        batches, state, torch.device("cuda"), 8
    )
    assert resident[0][0].is_cuda and resident_state.is_cuda
    torch.testing.assert_close(resident[0][0].cpu(), batches[0][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_text_cache_hits_accept_cpu_ids_without_running_backbone():
    from rcr.proposed.nn.encoders import TextEncoder, encode_query_text
    from tests.helpers import TextBackbone, _sample

    encoder = TextEncoder(TextBackbone(), dim=8, cache_mib=1).cuda()
    sample = _sample("q", "q", "a")
    expected = encode_query_text([sample], Tokenizer(), encoder, "cuda")
    handle = encoder.backbone.register_forward_hook(
        lambda *args: pytest.fail("cache hit reran the frozen backbone")
    )
    try:
        actual = encode_query_text([sample], Tokenizer(), encoder, "cuda")
    finally:
        handle.remove()
    for key in expected:
        assert actual[key].is_cuda
        torch.testing.assert_close(actual[key], expected[key])
