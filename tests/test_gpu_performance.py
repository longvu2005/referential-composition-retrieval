"""Numerical contracts and work bounds for the GPU execution paths."""

import json
from collections import Counter
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rcr.common.tensor_checks import TensorChecks, check_tensor
from rcr.proposed import losses, ranking
from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.cache.native import NativeFAFACache, NativeTargetBatch
from rcr.proposed.cache.store import GalleryCache, SceneTargetBatch
from rcr.proposed.identity import IdentityLayout, select_identity_observations
from rcr.proposed.matching import (
    PersonMatching,
    partial_identity_matching,
    unique_box_mask,
)
from rcr.proposed.nn.encoders import TextEncoder
from rcr.proposed.nn.model import QueryEncoding, RCRModel
from rcr.proposed.prefetch import OrderedPrefetch, PackedPersons, VisualStager
from rcr.proposed.scores import (
    fafa_binding_scores,
    identity_scores,
    identity_scores_paired,
)
from tests.helpers import TextBackbone, TinyCache, Tokenizer, _only_loss, _rank, _sample
from tools import diagnose_gpu
from tools.profile_gpu import TransferAudit

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["packed", "legacy"])
@pytest.mark.parametrize("indices", [[], [0, 0], [2, 0, 1, 3, 1, 2]])
def test_direct_packed_index_preserves_crop_order_without_dense_cpu_persons(
    dtype, layout, indices, monkeypatch
):
    cache = GalleryCache.__new__(GalleryCache)
    cache.mask = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    counts = cache.mask.sum(1)
    cache.person_dim = 4
    values = torch.arange(int(counts.sum()) * 4).reshape(-1, 4).to(dtype)
    cache.person_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(0))
    )
    cache.persons = values
    crops = [cache.person_row(i).clone() for i in range(len(counts))]
    if layout == "legacy":
        cache.persons = torch.full((4, 8, 4), torch.nan, dtype=dtype)
        for i, row in enumerate(crops):
            cache.persons[i, cache.mask[i]] = row
        cache.person_offsets = None
    cache.detections = object()
    geometry = [torch.arange(len(row) * 4).float().reshape(-1, 4) for row in crops]
    monkeypatch.setattr(cache, "_scene_boxes", lambda i: geometry[i])
    indices = torch.tensor(indices, dtype=torch.long)
    mask = cache.mask[indices]
    mask = mask[:, mask.any(0)]
    expected = torch.zeros(len(indices), mask.shape[1], 4, dtype=dtype)
    boxes = torch.zeros(len(indices), mask.shape[1], 4)
    for n, i in enumerate(indices.tolist()):
        expected[n, mask[n]] = crops[i]
        boxes[n, mask[n]] = geometry[i]
    dense, dense_mask = cache.load_persons(indices)
    torch.testing.assert_close(
        dense.masked_fill(~mask[..., None], 0), expected, rtol=0, atol=0
    )
    assert torch.equal(dense_mask, mask)
    monkeypatch.setattr(
        cache, "load_persons", lambda *args: pytest.fail("padded CPU persons")
    )
    monkeypatch.setattr(
        cache,
        "load_person_geometry",
        lambda *args: pytest.fail("dense geometry reader"),
    )
    monkeypatch.setattr(
        cache, "person_row", lambda *args: pytest.fail("per-image crop gather")
    )
    for geometry_required in (False, True):
        packed, actual_boxes, actual_mask = PackedPersons.from_cache(
            cache, indices, geometry=geometry_required
        )
        assert packed.values.dtype == dtype
        assert len(packed.values) == int(mask.sum())
        torch.testing.assert_close(
            packed.to_dense("cpu"), expected.float(), rtol=0, atol=0
        )
        assert torch.equal(actual_mask, mask)
        if geometry_required:
            torch.testing.assert_close(actual_boxes, boxes, rtol=0, atol=0)
        else:
            assert actual_boxes is None


@pytest.mark.parametrize("device", DEVICES)
def test_device_checks_and_values_share_one_readback(device):
    values = torch.tensor([[0.8, -0.3]], device=device, requires_grad=True)
    positives = torch.tensor([[True, False]], device=device)
    with TransferAudit() as audit:
        with TensorChecks() as checks:
            loss = losses.retrieval_loss(values, positives)
            check_tensor(torch.isfinite(loss), "invalid loss")
            result = checks.cpu_values({"scores": values, "loss": loss})
    assert result["scores"].device.type == "cpu"
    torch.testing.assert_close(result["scores"], values.detach().cpu())
    assert audit.counts["cuda_scalar_reads"] == 0
    assert audit.counts["cuda_to_cpu"] == (1 if device == "cuda" else 0)
    assert (
        audit.counts["cuda_boolean_index"] == audit.counts["cuda_dynamic_shapes"] == 0
    )
    loss.backward()
    assert values.grad is not None
    with pytest.raises(ValueError, match="first invalid predicate"):
        with TensorChecks() as checks:
            check_tensor(torch.tensor(False, device=device), "first invalid predicate")
            check_tensor(torch.tensor(False), "second invalid predicate")
            checks.cpu_values({"value": values})
    # Failed/nested scopes must not leak a pending check into later API calls.
    losses.retrieval_loss(values.detach(), positives)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "labels", [[-1, -1, -1], [7, -1, -1], [2, -1, 2, 3, 3], [1, 2, -1]]
)
def test_identity_unknown_rows_match_filtered_loss_and_gradients_without_dynamic_shapes(
    device, labels
):
    torch.manual_seed(41)
    labels = torch.tensor(labels, device=device)
    features = (
        torch.randn(len(labels), 7, device=device)
        .masked_fill((labels < 0)[:, None], torch.nan)
        .requires_grad_()
    )
    original = features.detach().clone().requires_grad_()
    keep = labels >= 0
    valid = torch.nn.functional.normalize(original[keep].float(), dim=-1)
    ids = labels[keep]
    if len(valid) < 2:
        expected = valid.sum() * 0
    else:
        logits = valid @ valid.T / 0.2
        diagonal = torch.eye(len(valid), device=device, dtype=torch.bool)
        positive = (ids[:, None] == ids[None]) & ~diagonal
        probability = logits - logits.masked_fill(diagonal, -torch.inf).logsumexp(
            1, keepdim=True
        )
        counts = positive.sum(1)
        per_anchor = -probability.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1)
        expected = per_anchor.masked_fill(counts == 0, 0).sum() / (
            counts > 0
        ).sum().clamp_min(1)
    with TransferAudit() as audit:
        actual = losses.identity_loss(features, labels, 0.2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(features.grad, original.grad, rtol=3e-5, atol=3e-6)
    assert torch.isfinite(features.grad).all()
    assert (
        audit.counts["cuda_scalar_reads"]
        == audit.counts["cuda_dynamic_shapes"]
        == audit.counts["cuda_boolean_index"]
        == 0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_identity_cpu_layout_preserves_gradients_and_invalidates_after_edit(
    objective_batch, device
):
    model, batch = objective_batch
    model = model.to(device)
    reference = deepcopy(model)
    layout = IdentityLayout.from_batch(batch)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["identity_layout"] = layout
    layout.bind(batch)
    baseline = {key: value for key, value in batch.items() if key != "identity_layout"}
    settings = _only_loss("identity")
    expected, _ = losses.compute_loss(reference, baseline, (2, 2), **settings)
    with TransferAudit() as audit:
        actual, _ = losses.compute_loss(model, batch, (2, 2), **settings)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for parameter, other in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        if other.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-5, atol=3e-6)
    assert (
        audit.counts["cuda_scalar_reads"]
        == audit.counts["cuda_dynamic_shapes"]
        == audit.counts["cuda_boolean_index"]
        == 0
    )
    assert layout.current(batch)
    original_labels = batch["query_identity_labels"]
    batch["query_identity_labels"] = original_labels.clone()
    assert not layout.current(batch)
    batch["query_identity_labels"] = original_labels
    assert layout.current(batch)
    original_labels.add_(0)
    assert not layout.current(batch)


@pytest.mark.parametrize("device", DEVICES)
def test_hard_identity_layout_reads_only_hardness_and_preserves_reserved_slots(device):
    source = {
        "query_identity_labels": torch.tensor([[0, 1]]),
        "query_identity_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_identity_labels": torch.tensor([[[0, 2], [1, 0]]]),
        "target_identity_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "identity_anchor_mask": torch.ones(1, 2, dtype=torch.bool),
        "supplemental_target_indices": torch.tensor([1, 2, 3]),
        "supplemental_target_labels": torch.tensor([2, 1, 0]),
        "extra_identity_positive_mask": torch.tensor([True, False, False, False]),
        "extra_identity_target_indices": torch.tensor([2, -1, 1, 3]),
        "extra_identity_labels": torch.tensor([0, 2, 2, 0]),
    }
    query = torch.eye(2, device=device)[None]
    targets = torch.tensor(
        [[[[1.0, 0], [0, 1]], [[0.9, 0.1], [0.1, 0.9]]]], device=device
    )
    settings = {"max_crops": 3, "hard_negative_fraction": 0.5}
    batch = {key: value.to(device) for key, value in source.items()}
    expected = select_identity_observations(query, targets, batch, settings)
    batch["identity_layout"] = IdentityLayout.from_batch(source)
    batch["identity_layout"].bind(batch)
    with TransferAudit() as audit:
        hard, extra = select_identity_observations(query, targets, batch, settings)
    torch.testing.assert_close(hard, expected[0].cpu())
    torch.testing.assert_close(extra, expected[1].cpu())
    assert hard.tolist() == [1] and extra.tolist() == [0, 1]
    assert audit.counts["cuda_to_cpu"] == (1 if device == "cuda" else 0)
    assert (
        audit.counts["cuda_scalar_reads"]
        == audit.counts["cuda_boolean_index"]
        == audit.counts["cuda_dynamic_shapes"]
        == 0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("all_null", [False, True])
def test_packed_native_transfer_matches_dense_fda_and_skips_null_images(
    device, all_null
):
    torch.manual_seed(23)
    counts = torch.tensor([0, 3, 8])
    features = [torch.randn(int(k), 5, 3, dtype=torch.float16) for k in counts]
    reads = []
    native = NativeFAFACache.__new__(NativeFAFACache)
    native.tokens, native.dim = 5, 3
    native.gallery = SimpleNamespace(
        persons=torch.empty(0, 3, dtype=torch.float16),
        mask=torch.arange(8)[None] < counts[:, None],
        image_ids=["empty", "three", "eight"],
        _cached=lambda key, load: load(),
    )

    def read(group, index, key, shape):
        reads.append(index)
        assert features[index].shape == shape
        return features[index]

    native._read = read
    images = torch.tensor([1, 2, 1, 0])
    assignments = torch.tensor(
        [[0, -1, 2, -1], [7, 1, -1, 0], [-1, 1, -1, -1], [-1, -1, -1, -1]]
    )
    if all_null:
        assignments.fill_(-1)
    packed = native.load_matched_targets(images, assignments)
    assert reads == ([] if all_null else [1, 2])
    assert len(packed.values) == int((assignments >= 0).sum())
    assert packed.values.numel() < 4 * 8 * 5 * 3
    member = torch.tensor([[[1.0, 0.7, 0, 0], [0, 0, 0.8, 1]]]).expand(4, -1, -1)
    subject_coverage = (member * (assignments >= 0)[:, None]).sum(-1) / member.sum(-1)
    matching = PersonMatching(
        assignments.to(device),
        member.to(device),
        torch.ones(4, 2, dtype=torch.bool, device=device),
        subject_coverage.to(device),
        subject_coverage.mean(-1).to(device),
        assignments,
    )
    query = torch.randn(4, 2, 4, 3, device=device)
    dense = native.load_targets(images, 8).to(device)
    cache = SimpleNamespace(native_matched_targets=native.load_matched_targets)
    expected = fafa_binding_scores(
        query, dense, matching=matching, return_poolings=True
    )
    actual = fafa_binding_scores(
        query,
        NativeTargetBatch(cache, images, 8),
        matching=matching,
        return_poolings=True,
    )
    torch.testing.assert_close(actual, expected)
    prepared = native.prepare_targets(images, 3 * 5 * 3 * 2)
    assert set(prepared) == {1}
    assert sum(t.numel() * t.element_size() for t in prepared.values()) <= 90


def test_lazy_scenes_load_only_eligible_pairs_once_outside_checkpoint(
    objective_batch, monkeypatch
):
    model, batch = objective_batch
    reference = deepcopy(model)
    expected_batch = deepcopy(batch)
    scene = batch["target_scene"].flatten(0, 1)
    scene_reads, native_reads = [], []
    cache = TinyCache()
    images = torch.tensor([1, 2, 4])

    def load(indices):
        scene_reads.extend(indices.tolist())
        positions = torch.tensor([images.tolist().index(i) for i in indices.tolist()])
        return scene[positions]

    native = cache.native_targets

    def read_native(indices, width):
        native_reads.extend(indices.tolist())
        return native(indices, width)

    monkeypatch.setattr(cache, "load_scenes", load, raising=False)
    monkeypatch.setattr(cache, "native_targets", read_native)
    batch["target_scene"] = SceneTargetBatch(cache, images)
    batch["fafa_targets"] = NativeTargetBatch(cache, images)
    batch["context_layout"] = {
        "identity_rows": torch.tensor([0, 1]),
        "rejection_rows": torch.tensor([0, 1]),
        "widths": torch.ones(3, dtype=torch.long),
    }
    settings = losses.loss_settings({})
    expected, _ = losses.compute_loss(reference, expected_batch, (2, 2), **settings)
    actual, _ = losses.compute_loss(
        model, batch, (2, 2), context_batch_size=1, **settings
    )
    expected.backward()
    actual.backward()
    assert (
        scene_reads == native_reads == [1, 2]
    )  # No cache I/O is repeated in backward.
    torch.testing.assert_close(actual, expected)
    for parameter, other in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("identity_weight", [0.0, 0.1])
@pytest.mark.parametrize("amp", [False, True])
def test_packed_person_projection_preserves_objective_and_gradients(
    objective_batch, device, identity_weight, amp
):
    model, batch = objective_batch
    model.to(device)
    persons = batch["target_persons"].detach()
    padded = persons.new_full((1, 3, 8, persons.shape[-1]), torch.nan)
    padded[:, :, :1] = persons
    mask = torch.zeros(1, 3, 8, dtype=torch.bool)
    mask[:, :, :1] = True
    for name in ("target_identity_labels", "target_identity_mask"):
        value = batch[name]
        expanded = value.new_zeros((1, 3, 8))
        expanded[:, :, :1] = value
        batch[name] = expanded
    batch["target_persons"], batch["target_mask"] = padded, mask
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    reference = deepcopy(model)
    expected_batch = deepcopy(batch)
    packed = PackedPersons.from_padded(padded, mask)
    batch["target_persons"] = packed.to_dense(device)
    batch["target_layout"] = {"positions": packed.positions}
    assert len(packed.values) == 3  # 24 padded positions would otherwise be projected.
    settings = losses.loss_settings(
        {"context_weight": 0.0, "identity_weight": identity_weight}
    )
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=dtype, enabled=amp):
        expected, _ = losses.compute_loss(reference, expected_batch, (2, 2), **settings)
        actual, _ = losses.compute_loss(model, batch, (2, 2), **settings)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(
        actual, expected, rtol=2e-3 if amp else 1e-5, atol=2e-4 if amp else 1e-6
    )
    for parameter, other in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.grad,
            other.grad,
            rtol=1e-2 if amp else 3e-4,
            atol=5e-4 if amp else 2e-6,
        )


def test_prefetch_bounds_future_work_and_joins_on_early_exit():
    produced, ready = [], Event()

    def prepare(index):
        produced.append(index)
        if index == 1:
            ready.set()
        return index

    with OrderedPrefetch(range(5), prepare, enabled=True) as queue:
        iterator = iter(queue)
        assert next(iterator) == 0
        assert ready.wait(5)
        assert produced == [0, 1]
        iterator.close()
    assert queue.pool is None
    assert produced == [0, 1]


def test_prefetch_propagates_worker_errors_and_closes():
    def prepare(index):
        if index == 1:
            raise ValueError("broken cache shard")
        return index

    with OrderedPrefetch(range(3), prepare, enabled=True) as queue:
        iterator = iter(queue)
        assert next(iterator) == 0
        with pytest.raises(ValueError, match="broken cache shard"):
            next(iterator)
    assert queue.pool is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_staging_preserves_storage_precision_padding_and_stream_lifetimes():
    values = torch.randn(2, 3, 4, 8).half()
    mask = torch.rand(2, 3, 4) > 0.5
    values.masked_fill_(~mask[..., None], 0)
    packed = PackedPersons.from_padded(values, mask)
    scenes = SceneTargetBatch(None, torch.arange(2))
    scenes.values = torch.randn(2, 4, 8).half()
    scenes.positions = torch.arange(2)
    expected_scenes = scenes.values.clone()
    with VisualStager("cuda", enabled=True) as stager:
        labels = torch.arange(6).reshape(2, 3)
        with TransferAudit() as audit:
            staged = stager.stage(
                {
                    "target_persons": packed,
                    "target_scene": scenes,
                    "mask": mask,
                    "mask_copy": ~mask,
                    "labels": labels,
                    "labels_copy": labels + 3,
                }
            )
        visual = staged.consume()
        assert audit.counts["cpu_to_cuda"] == 5
        assert (
            visual["mask"].untyped_storage().data_ptr()
            == visual["mask_copy"].untyped_storage().data_ptr()
        )
        assert (
            visual["labels"].untyped_storage().data_ptr()
            == visual["labels_copy"].untyped_storage().data_ptr()
        )
        assert staged.sources and all(t.is_pinned() for t in staged.sources)
        assert visual["target_persons"].dtype == torch.float32
        assert scenes.values.dtype == torch.float16
        torch.testing.assert_close(visual["target_persons"].cpu(), values.float())
        torch.testing.assert_close(
            scenes.load(torch.tensor([1, 0])).cpu(), expected_scenes.flip(0)
        )
        torch.testing.assert_close(visual["mask"].cpu(), mask)
        torch.testing.assert_close(visual["mask_copy"].cpu(), ~mask)
        torch.testing.assert_close(visual["labels"].cpu(), labels)
        torch.testing.assert_close(visual["labels_copy"].cpu(), labels + 3)


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
        if name == "identity_layout":
            torch.testing.assert_close(value.host, expected_batch[name].host)
            torch.testing.assert_close(value.positions, expected_batch[name].positions)
        else:
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefetch", [0, 1])
def test_rejection_mining_reuses_matching_and_loads_only_wrong_id_scenes(
    tiny_ranker, monkeypatch, device, prefetch
):
    cache, model, text, samples, settings = tiny_ranker
    model.to(device)
    text.to(device)
    model.fine_matching.update(selection_threshold=0.0001, identity_threshold=-1.0)
    scenes, natives, matched, reasoned = [], [], [], []
    native_targets, match_target = cache.native_targets, model.match_target

    def load_scenes(indices):
        scenes.extend(indices.tolist())
        return cache.scenes[indices]

    def load_native(indices, width):
        natives.extend(indices.tolist())
        return native_targets(indices, width)

    def match(query, identity, *args, **kwargs):
        matched.append(len(identity))
        return match_target(query, identity, *args, **kwargs)

    monkeypatch.setattr(
        cache,
        "load_person_geometry",
        lambda indices: (
            cache.persons[indices],
            cache.boxes[indices],
            cache.mask[indices],
        ),
        raising=False,
    )
    monkeypatch.setattr(cache, "load_scenes", load_scenes, raising=False)
    monkeypatch.setattr(cache, "native_targets", load_native)
    monkeypatch.setattr(model, "match_target", match)
    handle = model.reasoner.register_forward_pre_hook(
        lambda module, inputs: reasoned.append(len(inputs[0]))
    )
    try:
        with TransferAudit() as audit:
            pools = ranking.mine_hard_negatives(
                samples[:1],
                cache,
                Tokenizer(),
                text,
                model,
                torch.device(device),
                gallery_ids=cache.image_ids,
                retrieval={
                    **settings,
                    "coarse_beta": 0,
                    "fine_batch_size": 1,
                    "prefetch_batches": prefetch,
                },
                pool_size=64,
                identity_image_ids={"train": {"a", "n", "vq"}},
                semantic_hard=True,
                identity_rejection=True,
                rejection_mining_context_weight=0.005,
            )
    finally:
        handle.remove()
    assert Counter(scenes) == Counter(cache.by_id[i] for i in ("va", "vn"))
    assert Counter(natives) == Counter(cache.by_id[i] for i in ("n", "vq", "va", "vn"))
    assert sum(matched) == 4  # Two semantic, two wrong-ID: no union rescore.
    assert sum(reasoned) == 2
    assert set(pools["train"]["semantic"]) == {"n", "vq"}
    assert set(pools["train"]["identity"]) == {"va", "vn"}
    assert (
        audit.counts["cuda_boolean_index"] == audit.counts["cuda_dynamic_shapes"] == 0
    )


def test_retrieval_query_batching_preserves_order_and_evidence(tiny_ranker):
    expected = _rank(tiny_ranker, query_batch_size=1)
    actual = _rank(tiny_ranker, query_batch_size=8)
    assert actual["sample_ids"] == expected["sample_ids"]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_sparse_eval_retrieval_preserves_topm_artifacts_and_cpu_replay(
    tiny_ranker, monkeypatch, prefetch
):
    from rcr.proposed.replay import replay_rankings

    cache, _, _, _, _ = tiny_ranker
    expected = _rank(tiny_ranker, query_batch_size=8)
    monkeypatch.setattr(
        cache,
        "load_person_geometry",
        lambda indices: (
            cache.persons[indices],
            cache.boxes[indices],
            cache.mask[indices],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cache, "load_scenes", lambda indices: cache.scenes[indices], raising=False
    )
    actual = _rank(tiny_ranker, query_batch_size=8, prefetch_batches=prefetch)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
    for field in ("rankings", "fine_rankings", "coarse_rankings", "fafa_rankings"):
        assert torch.equal(actual[field], expected[field])
    assert torch.equal(replay_rankings(actual)["rankings"], actual["rankings"])
    assert set(actual["timings_seconds"]) == {
        "gallery_encoding",
        "query_and_coarse",
        "fine_scoring",
        "ranking",
        "total",
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("case", ["mixed", "missing", "mining"])
def test_eval_packs_people_and_skips_unmatched_evidence_without_changing_scores(
    device, prefetch, case, monkeypatch
):
    torch.manual_seed(1812)
    cache = TinyCache()
    cache.mask = torch.arange(5)[None] < torch.tensor([3, 1, 0, 4, 2, 1])[:, None]
    cache.persons = (
        torch.randn(6, 5, 8).half().masked_fill(~cache.mask[..., None], torch.nan)
    )
    low = torch.arange(5).float() / 10
    cache.boxes = torch.stack((low, low, low + 0.08, low + 0.08), -1)[None].expand(
        6, -1, -1
    )
    cache.scenes = cache.scenes.half()
    model = (
        RCRModel(
            dim=8,
            identity_dim=6,
            num_heads=2,
            fine_matching={"identity_threshold": 1.0 if case == "missing" else -1.0},
        )
        .eval()
        .to(device)
    )
    semantic = model.encode_semantic(cache.persons[:1, :3].to(device).float())
    membership = torch.tensor([[[0.8, 0.6, 0], [0, 0, 0.7]]], device=device)
    subjects = torch.ones(1, 2, device=device, dtype=torch.bool)
    reference, mask, prior, condition = model._encode_condition(
        semantic,
        membership,
        subjects,
        torch.randn(1, 9, 8, device=device),
        torch.tensor([[2, 5]], device=device),
        None,
        None,
        None,
    )
    query = QueryEncoding(
        torch.zeros_like(membership),
        model.identity_head(semantic),
        reference,
        mask,
        prior,
        condition,
        torch.randn(1, 2, 3, 5, device=device),
        membership,
        subjects,
    )
    targets = torch.tensor([4, 1, 2, 3, 5, 1])  # Repeated images and non-count order.
    context = case != "mining"
    with torch.inference_mode():
        scene, persons, boxes, person_mask = cache.load(targets)
        expected = model.score_target(
            query.repeat_candidates(len(targets)),
            scene.to(device).float(),
            persons.to(device).float(),
            boxes.to(device),
            cache.patch_hw,
            person_mask.to(device),
            fafa_targets=cache.native_targets(targets, 5).to(device),
            context_weight=0.1 if context else 0.0,
            compute_context=context,
            return_details=True,
        )
    scene_reads, native_reads, projected, reasoned = [], [], [], []

    def load_geometry(indices):
        return cache.persons[indices], cache.boxes[indices], cache.mask[indices]

    def load_scenes(indices):
        scene_reads.extend(indices.tolist())
        return cache.scenes[indices]

    native = cache.native_targets

    def load_native(indices, width):
        native_reads.extend(indices.tolist())
        return native(indices, width)

    monkeypatch.setattr(cache, "load_person_geometry", load_geometry, raising=False)
    monkeypatch.setattr(cache, "load_scenes", load_scenes, raising=False)
    monkeypatch.setattr(cache, "native_targets", load_native)
    handles = [
        model.person_proj.register_forward_pre_hook(
            lambda m, x: projected.append(len(x[0]))
        ),
        model.reasoner.register_forward_pre_hook(
            lambda m, x: reasoned.append(len(x[0]))
        ),
    ]
    try:
        with torch.inference_mode(), TransferAudit() as audit:
            actual = ranking._score_target_components(
                query,
                cache,
                model,
                torch.device(device),
                targets,
                2,
                "subject_mean",
                context_weight=0.1 if context else 0.0,
                compute_context=context,
                prefetch_batches=prefetch,
                return_cpu=case == "mixed",
            )
    finally:
        for handle in handles:
            handle.remove()
    for key, value in actual.items():
        torch.testing.assert_close(
            value, expected[key].to(value.device), rtol=3e-5, atol=3e-6
        )
    assert (
        audit.counts["cuda_scalar_reads"]
        == audit.counts["cuda_dynamic_shapes"]
        == audit.counts["cuda_boolean_index"]
        == 0
    )
    assert (
        audit.counts["cuda_to_cpu"] <= 4
    )  # Three matching chunks plus one final packet.
    used = targets[expected["coverage"].cpu() > 0].tolist()
    assert Counter(native_reads) == Counter(used)
    assert Counter(scene_reads) == Counter(used if context else [])
    assert sum(projected) == int(cache.mask[targets].sum())
    assert sum(reasoned) == (len(used) if context else 0)
    if case == "mixed":
        assert ((actual["coverage"] > 0) & (actual["coverage"] < 1)).any()


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
    cache = TinyCache()
    cache.persons = torch.randn(512, 8, 8)
    cache.mask = torch.ones(512, 8, dtype=torch.bool)
    cache.scenes = torch.randn(512, 4, 8)
    model = RCRModel(dim=8, identity_dim=128, state_dim=128, num_heads=2).cuda().eval()
    indices = torch.arange(512)
    with TransferAudit() as fallback:
        batches, state = ranking._encode_gallery(
            cache,
            model,
            torch.device("cuda"),
            512,
            indices,
            state=True,
            budget_mib=1,
        )
    assert batches[0][0].device.type == state.device.type == "cpu"
    assert fallback.counts["cuda_to_cpu"] == 1
    with TransferAudit() as cached:
        resident, resident_state = ranking._encode_gallery(
            cache,
            model,
            torch.device("cuda"),
            512,
            indices,
            state=True,
            budget_mib=8,
        )
    assert cached.counts["cuda_to_cpu"] == 0
    assert resident[0][0].is_cuda and resident_state.is_cuda
    torch.testing.assert_close(resident[0][0].cpu(), batches[0][0])
    torch.testing.assert_close(resident_state.cpu(), state)


def test_gallery_residency_is_decided_before_projection(monkeypatch):
    cache = TinyCache()
    cache.mask = torch.ones(512, 8, dtype=torch.bool)
    model = RCRModel(dim=8, identity_dim=128, state_dim=128, num_heads=2)
    cuda = torch.device("cuda")
    indices = torch.arange(512)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a: (16 * 1024**2, 16 * 1024**2)
    )
    assert (
        ranking._gallery_output_device(cache, model, cuda, 512, indices, True, 1).type
        == "cpu"
    )
    assert (
        ranking._gallery_output_device(cache, model, cuda, 512, indices, True, 8)
        == cuda
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1024**2, 16 * 1024**2))
    assert (
        ranking._gallery_output_device(cache, model, cuda, 512, indices, True, 8).type
        == "cpu"
    )
    # Training's unused allocator blocks can hold the gallery without eviction.
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 16 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 2 * 1024**2)
    assert (
        ranking._gallery_output_device(cache, model, cuda, 512, indices, True, 8)
        == cuda
    )


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


@pytest.mark.parametrize("device", DEVICES)
def test_text_cache_reuses_fresh_device_states_and_uploads_unique_hits(device):
    encoder = TextEncoder(TextBackbone(), dim=8, cache_mib=1).to(device)
    ids = torch.tensor([[5, 1, 2, 6, 0, 0], [5, 1, 2, 6, 0, 0], [5, 4, 6, 0, 0, 0]])
    mask = (ids != 0).long()
    with torch.no_grad():
        expected = encoder._backbone_tokens(ids.to(device), mask.to(device))
        expected = expected.masked_fill(~mask.to(device).bool()[..., None], 0)
        with TransferAudit() as cold:
            actual = encoder._cached_backbone(ids, mask)
        with TransferAudit() as warm:
            repeated = encoder._cached_backbone(ids, mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)
    if device == "cuda":
        assert cold.counts["cuda_to_cpu"] == 1
        assert (
            cold.counts["cpu_to_cuda"] == 2
        )  # Token IDs/mask only; no hidden-state upload.
        assert warm.counts["cuda_to_cpu"] == 0
        assert warm.counts["cpu_to_cuda"] == 1
        assert warm.counts["cpu_to_cuda_source_bytes"] == 2 * 6 * 8 * 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deduplicate", [0.9, 1.0])
@pytest.mark.parametrize("people", [1, 2])
def test_context_flags_share_the_exact_matching_readback(device, deduplicate, people):
    identities = torch.eye(2, device=device)[None]
    membership = torch.tensor([[[0.8, 0], [0, 0.7]]], device=device)
    subjects = torch.ones(1, 2, device=device, dtype=torch.bool)
    mask = torch.ones(1, 2, device=device, dtype=torch.bool)
    boxes = torch.tensor([[[0, 0, 0.4, 1], [0.6, 0, 1, 1]]], device=device)
    flags = torch.tensor([[True, False, True, True, False]], device=device)
    args = (
        identities,
        identities[:, :people],
        membership,
        subjects,
        mask[:, :people],
        boxes[:, :people],
    )
    expected = partial_identity_matching(*args, duplicate_iou=deduplicate)
    with TransferAudit() as audit:
        actual = partial_identity_matching(
            *args, duplicate_iou=deduplicate, context_flags=flags
        )
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.coverage, expected.coverage)
    torch.testing.assert_close(actual.context_flags_cpu, flags.cpu())
    assert audit.counts["cuda_to_cpu"] == (1 if device == "cuda" else 0)
    assert audit.counts["cuda_scalar_reads"] == audit.counts["cuda_dynamic_shapes"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_single_person_suppression_has_no_cpu_readback(device):
    mask = torch.tensor([[True], [False]], device=device)
    boxes = torch.tensor([[[0.0, 0, 1, 1]], [[torch.nan, 0, 0, 0]]], device=device)
    with TransferAudit() as audit:
        actual = unique_box_mask(boxes, mask, threshold=0.9)
    torch.testing.assert_close(actual, mask)
    assert audit.counts["cuda_to_cpu"] == audit.counts["cuda_scalar_reads"] == 0
    with pytest.raises(ValueError, match="boxes must have shape"):
        unique_box_mask(boxes[..., :3], mask, threshold=0.9)
