"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.common.data import inference_samples
from rcr.common.runtime import evaluating
from rcr.proposed.matching import PersonMatching
from rcr.proposed.scores import (
    DEFAULT_CONTEXT_WEIGHT,
    DEFAULT_COVERAGE_WEIGHT,
    FINE_SCORE_MODE,
    MINING_CONTEXT_WEIGHT,
    POOLING_MODES,
    combine_scores,
    fafa_binding_scores,
    identity_scores_batched,
    linear_score_components,
    pool_fafa_scores,
)

SCORE_SCHEMA_VERSION = 4
SCORE_EPS = 1e-6
DEFAULT_FINE_COARSE_WEIGHT = 0.15
if TYPE_CHECKING:
    from rcr.proposed.cache.store import GalleryCache
    from rcr.proposed.nn.encoders import QueryTextCache


def _encode_gallery_identity(
    cache: GalleryCache,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_indices: Tensor | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Project once, retaining only CPU batches with useful person columns.

    Packed caches pad only the current batch. Legacy dense caches drop columns
    masked out for an entire batch without removing evidence.
    Never concatenate these batches on the accelerator (or duplicate them in RAM).
    """

    if image_indices is None:
        image_indices = torch.arange(len(cache.image_ids))
    batches = []
    dtype = model.identity_head.proj.weight.dtype
    for start in range(0, len(image_indices), batch_size):
        indices = image_indices[start : start + batch_size]
        persons, mask = cache.load_persons(indices)
        persons = persons.to(device=device, dtype=dtype)
        identity = model.encode_identity(persons)
        batches.append((identity.cpu(), mask.cpu()))
        # Release GPU outputs before allocating the next projection batch.
        del persons, identity
    return batches


def _coarse_scores_chunked(
    query_identity: Tensor,
    logits: Tensor,
    query_mask: Tensor,
    gallery_batches: list[tuple[Tensor, Tensor]],
    batch_size: int,
) -> Tensor:
    """Score the entire gallery with bounded accelerator working memory."""

    single = query_identity.ndim == 2
    if single:
        query_identity, logits, query_mask = (
            value[None] for value in (query_identity, logits, query_mask)
        )
    num_images = sum(len(identity) for identity, _ in gallery_batches)
    scores = query_identity.new_empty(len(query_identity), num_images)
    offset = 0
    for identity_cpu, mask_cpu in gallery_batches:
        for start in range(0, len(identity_cpu), batch_size):
            stop = min(start + batch_size, len(identity_cpu))
            identity = identity_cpu[start:stop].to(
                query_identity.device, non_blocking=True
            )
            mask = mask_cpu[start:stop].to(query_identity.device, non_blocking=True)
            scores[:, offset + start : offset + stop] = identity_scores_batched(
                query_identity,
                logits,
                identity,
                mask,
                query_mask=query_mask,
            )
            del identity, mask
        offset += len(identity_cpu)
    return scores[0] if single else scores


def _encode_gallery_state(
    cache: GalleryCache,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_indices: Tensor,
) -> Tensor:
    """Project global image features once per retrieval call, retaining CPU output."""
    projection = model.state_image_proj.weight
    state = torch.empty(len(image_indices), projection.shape[0], dtype=projection.dtype)
    for start in range(0, len(image_indices), batch_size):
        stop = min(start + batch_size, len(image_indices))
        chunk = cache.load_global_features(image_indices[start:stop]).to(
            device=device, dtype=projection.dtype
        )
        state[start:stop] = model.encode_image_state(chunk).cpu()
    return state


def _cache_gallery_on_device(batches, state, device, budget_mib):
    """Keep small projected galleries resident within an explicit VRAM budget."""
    if device.type != "cuda" or not budget_mib:
        return batches, state
    tensors = [value for batch in batches for value in batch]
    if state is not None:
        tensors.append(state)
    size = sum(value.numel() * value.element_size() for value in tensors)
    free, _ = torch.cuda.mem_get_info(device)
    # Leave room for fine scoring and the CUDA allocator's working buffers.
    if size > min(budget_mib * 1024**2, free // 2):
        return batches, state
    resident = [
        tuple(value.to(device, non_blocking=True) for value in batch)
        for batch in batches
    ]
    state = None if state is None else state.to(device, non_blocking=True)
    return resident, state


def fine_score_components(
    binding: Tensor,
    context: Tensor,
    coverage: Tensor,
    binding_scale: float,
    fine_score_mode: str = FINE_SCORE_MODE,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
) -> dict[str, Tensor]:
    """Apply the shared pairwise scorer to a shortlist in candidate order."""
    if binding.ndim != 1:
        raise ValueError("fine score components must be 1D tensors")
    if fine_score_mode != FINE_SCORE_MODE:
        raise ValueError(f"unknown fine_score_mode: {fine_score_mode!r}")
    return linear_score_components(
        binding, context, coverage, binding_scale, context_weight, coverage_weight
    )


def normalize_fusion_scores(
    scores: Tensor, support: Tensor, eps: float = SCORE_EPS
) -> Tensor:
    """Normalize one branch on the common finite coarse shortlist support."""
    if scores.ndim != 1 or scores.shape != support.shape or support.dtype != torch.bool:
        raise ValueError("normalization requires aligned 1D scores and boolean support")
    result = torch.full_like(scores, -torch.inf, dtype=torch.float32)
    if support.any():
        values = scores[support].float()
        if not torch.isfinite(values).all():
            raise ValueError("fusion support contains NaN/Inf")
        result[support] = (
            torch.zeros_like(values)
            if (values == values[0]).all()
            else (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)
        )
    return result


def normalize_fusion_rows(scores: Tensor, support: Tensor) -> Tensor:
    """Batch CPU calibration statistics over rows with the same support size.

    Pack only supported entries, preserving their order and the population
    normalization used by the single-query replay scorer.
    """
    if scores.ndim != 2 or scores.shape != support.shape or support.dtype != torch.bool:
        raise ValueError("normalization requires aligned score/mask matrices")
    result = torch.full_like(scores, -torch.inf, dtype=torch.float32)
    counts = support.sum(1)
    for count in counts.unique().tolist():
        if not count:
            continue
        rows = (counts == count).nonzero().flatten()
        mask = support.index_select(0, rows)
        values = scores.index_select(0, rows)[mask].float().reshape(-1, count)
        if not torch.isfinite(values).all():
            raise ValueError("fusion support contains NaN/Inf")
        normalized = (values - values.mean(1, keepdim=True)) / values.std(
            1, unbiased=False, keepdim=True
        ).clamp_min(SCORE_EPS)
        normalized = normalized.masked_fill(
            (values == values[:, :1]).all(1)[:, None], 0
        )
        group = result.index_select(0, rows)
        group[mask] = normalized.flatten()
        result.index_copy_(0, rows, group)
    return result


def fuse_fine_coarse_scores(
    fine: Tensor, coarse: Tensor, weight: float, eps: float = SCORE_EPS
) -> Tensor:
    """Fuse population z-scores on the same finite shortlist support.

    Zero weight returns raw fine scores. For positive weights, candidates with
    no finite coarse evidence remain last. Statistics use the complete shortlist,
    independently of inference chunk sizes; constant branches contribute zero.
    """
    if fine.ndim != 1 or fine.shape != coarse.shape:
        raise ValueError("fine and coarse scores must be matching 1D tensors")
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("fine_coarse_weight must be finite and nonnegative")
    fine, coarse = fine.float(), coarse.float()
    if not torch.isfinite(fine).all():
        raise ValueError("fine scores must be finite before fusion")
    if weight == 0:
        return fine
    valid = torch.isfinite(coarse)
    fused = normalize_fusion_scores(
        fine, valid, eps
    ) + weight * normalize_fusion_scores(coarse, valid, eps)
    if not torch.isfinite(fused[valid]).all():
        raise ValueError("fine/coarse fusion overflowed float32")
    return fused


def rerank_topm(order: Tensor, top_indices: Tensor, scores: Tensor) -> Tensor:
    """Stable shortlist sort; ties retain Coarse order, tail is unchanged."""
    ranked = top_indices[torch.argsort(scores, descending=True, stable=True)]
    return torch.cat((ranked, order[len(top_indices) :]))


def score_configuration(settings: dict, binding_scale: float) -> dict:
    """Weights-only-safe convention shared by artifacts and run metadata."""
    return {
        **{
            key: settings[key]
            for key in (
                "fine_score_mode",
                "context_weight",
                "coverage_weight",
                "fine_coarse_weight",
                "coarse_beta",
                "top_m",
            )
        },
        "pooling": settings.get("pooling", "penalized"),
        "binding_scale": binding_scale,
        "normalization": "per_query_coarse_topm_population",
        "normalization_dtype": "float32",
        "normalization_device": "cpu",
        "eps": SCORE_EPS,
        "std_unbiased": False,
        "tie_break": "stable_coarse_order_then_canonical_gallery",
        "nonfinite_coarse": "masked_in_fusion_when_weight_positive",
    }


def retrieval_settings(cfg: dict, default_beta: float) -> dict:
    """Resolve runtime overrides and the model's default coarse weight."""
    unknown = cfg.keys() - {
        "top_m",
        "fine_batch_size",
        "identity_batch_size",
        "coarse_batch_size",
        "coarse_beta",
        "fine_score_mode",
        "context_weight",
        "coverage_weight",
        "fine_coarse_weight",
        "pooling",
        "query_batch_size",
        "gpu_cache_mib",
        "prefetch_batches",
    }
    if unknown:
        raise ValueError(f"unknown retrieval settings: {', '.join(sorted(unknown))}")
    settings = {
        "coarse_batch_size": 512,
        "coarse_beta": default_beta,
        "fine_score_mode": FINE_SCORE_MODE,
        "context_weight": DEFAULT_CONTEXT_WEIGHT,
        "coverage_weight": DEFAULT_COVERAGE_WEIGHT,
        "fine_coarse_weight": DEFAULT_FINE_COARSE_WEIGHT,
        "pooling": "penalized",
        "query_batch_size": 1,
        "gpu_cache_mib": 512,
        "prefetch_batches": 1,
        **cfg,
    }
    if settings["coarse_beta"] is None:
        settings["coarse_beta"] = default_beta
    sizes = (
        "top_m",
        "fine_batch_size",
        "identity_batch_size",
        "coarse_batch_size",
        "query_batch_size",
    )
    if any(
        key in settings and (type(settings[key]) is not int or settings[key] < 1)
        for key in sizes
    ):
        raise ValueError("retrieval batch sizes and top_m must be positive")
    if type(settings["gpu_cache_mib"]) is not int or settings["gpu_cache_mib"] < 0:
        raise ValueError("retrieval.gpu_cache_mib must be a nonnegative integer")
    if type(settings["prefetch_batches"]) is not int or settings[
        "prefetch_batches"
    ] not in (
        0,
        1,
    ):
        raise ValueError("retrieval.prefetch_batches must be 0 or 1")
    if any(
        type(settings[key]) not in (int, float)
        or not math.isfinite(settings[key])
        or settings[key] < 0
        for key in (
            "coarse_beta",
            "context_weight",
            "coverage_weight",
            "fine_coarse_weight",
        )
    ):
        raise ValueError("retrieval weights must be finite and nonnegative")
    if settings["fine_score_mode"] != FINE_SCORE_MODE:
        raise ValueError(f"unknown fine_score_mode: {settings['fine_score_mode']!r}")
    if settings["pooling"] not in POOLING_MODES:
        raise ValueError(f"unknown FAFA pooling: {settings['pooling']!r}")
    return settings


def _gallery_scope(cache, gallery_ids, samples, label):
    gallery_ids = list(gallery_ids)
    if not gallery_ids or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError(f"{label} gallery_ids must be non-empty and unique")
    if not set(gallery_ids) <= set(cache.by_id):
        raise ValueError(f"{label} gallery contains images missing from the cache")
    local = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if any(sample["query_image_id"] not in local for sample in samples):
        raise ValueError(f"{label} query must belong to the retrieval gallery")
    indices = torch.tensor([cache.by_id[image_id] for image_id in gallery_ids])
    return gallery_ids, local, indices


def _score_target_components(
    query,
    cache,
    model,
    device,
    image_indices,
    batch_size,
    pooling,
    context_weight=DEFAULT_CONTEXT_WEIGHT,
    coverage_weight=DEFAULT_COVERAGE_WEIGHT,
    *,
    compute_context=True,
    prefetch_batches=1,
):
    """Collect pairwise evidence without changing its scoring population."""
    from rcr.proposed.cache.native import NativeTargetBatch
    from rcr.proposed.prefetch import OrderedPrefetch, PackedPersons

    components = {
        key: torch.empty(len(image_indices), device=device, dtype=torch.float32)
        for key in (
            "context",
            "coverage",
            *(f"binding_{mode}" for mode in POOLING_MODES),
        )
    }
    if pooling not in POOLING_MODES:
        raise ValueError(f"unknown FAFA pooling: {pooling!r}")
    if not compute_context and context_weight != 0:
        raise ValueError("skipping context requires context_weight=0")
    if not len(image_indices):
        return components
    sparse = hasattr(cache, "load_person_geometry") and hasattr(cache, "load_scenes")
    layout = (
        _reasoning_layout(query)
        if sparse and compute_context and not model.training
        else None
    )
    missing_bindings = {}
    rows = torch.arange(len(image_indices))
    if sparse and hasattr(cache, "mask"):
        # Change execution order only. Fusion still sees the original complete Top-M.
        counts = cache.mask[image_indices].sum(1)
        rows = rows[torch.argsort(counts, stable=True)]

    def prepare(rows):
        targets = image_indices[rows]
        if sparse:
            persons, boxes, mask = cache.load_person_geometry(targets)
            packed = PackedPersons.from_padded(persons, mask)
            if device.type == "cuda":
                packed.values = packed.values.pin_memory()
                packed.positions = packed.positions.pin_memory()
                boxes, mask = boxes.pin_memory(), mask.pin_memory()
            return (
                rows,
                targets,
                (packed, boxes, mask),
            )
        return rows, targets, None

    with OrderedPrefetch(
        rows.split(batch_size), prepare, enabled=sparse and bool(prefetch_batches)
    ) as batches:
        for rows, targets, prepared in batches:
            if sparse:
                details = _score_packed_targets(
                    query,
                    cache,
                    model,
                    device,
                    targets,
                    prepared,
                    compute_context,
                    layout,
                    missing_bindings,
                )
            else:
                # Keep the public reader/scorer API for legacy and third-party caches.
                if not compute_context and hasattr(cache, "load_person_geometry"):
                    persons, boxes, mask = cache.load_person_geometry(targets)
                    scene = None
                else:
                    scene, persons, boxes, mask = cache.load(targets)
                details = model.score_target(
                    query.repeat_candidates(len(targets)),
                    None if scene is None else scene.to(device, non_blocking=True),
                    persons.to(device, non_blocking=True).float(),
                    boxes.to(device, non_blocking=True),
                    cache.patch_hw,
                    mask.to(device, non_blocking=True),
                    fafa_targets=NativeTargetBatch(cache, targets, persons.shape[1]),
                    return_details=True,
                    pooling=pooling,
                    context_weight=context_weight,
                    coverage_weight=coverage_weight,
                    compute_context=compute_context,
                )
            positions = rows.to(device, non_blocking=True)
            for key, output in components.items():
                output.index_copy_(0, positions, details[key].float())
    # Validate once per shortlist, avoiding an extra host synchronization per chunk.
    if not torch.stack(
        [torch.isfinite(value).all() for value in components.values()]
    ).all():
        raise ValueError("fine score components contain NaN/Inf")
    return components


def _reasoning_layout(query):
    """Resolve query-only masks once on CPU, outside the candidate/chunk loop."""
    from rcr.proposed.nn.reasoning import ReasoningLayout

    selected = query.subjects[..., None] & (query.membership > 0)
    offset = query.reference.shape[1] - selected.shape[1] * selected.shape[2]
    valid = query.mask[0].clone()
    valid[offset:] &= selected[0].flatten()
    flags = torch.cat((valid, query.subjects[0])).cpu()
    reference_mask, subjects = flags[: len(valid)], flags[len(valid) :]
    reference = reference_mask.nonzero().flatten()
    first, second = torch.triu_indices(len(subjects), len(subjects), offset=1)
    pairs = (subjects[first] & subjects[second]).nonzero().flatten()
    persons = reference[reference >= offset] - offset
    return ReasoningLayout(
        reference.to(query.reference.device, non_blocking=True),
        torch.cat((reference, len(valid) + pairs)).to(
            query.reference.device, non_blocking=True
        ),
        persons.to(query.reference.device, non_blocking=True),
        int(reference_mask[:offset].sum()),
    )


def _score_packed_targets(
    query,
    cache,
    model,
    device,
    targets,
    prepared,
    compute_context,
    layout,
    missing_bindings,
):
    """Match actual detections, then load/score evidence only where H can be nonzero."""
    from rcr.proposed.cache.native import NativeTargetBatch

    if query.fafa is None:
        raise ValueError("fine retrieval requires native FAFA query features")
    packed, boxes, mask = prepared
    positions = packed.positions.to(device, non_blocking=True)
    values = model.encode_semantic(packed.values.to(device, non_blocking=True).float())

    def unpack(values):
        output = values.new_zeros((mask.numel(), values.shape[-1]))
        return output.index_copy_(0, positions, values).unflatten(0, mask.shape)

    semantic = unpack(values)
    identity = unpack(model.identity_head(values))
    boxes = boxes.to(device, non_blocking=True)
    mask_device = mask.to(device, non_blocking=True)
    query = query.repeat_candidates(len(targets))
    matching = model.match_target(query, identity, boxes, mask_device)
    indices_cpu = matching.indices_cpu
    if indices_cpu is None:
        indices_cpu = matching.indices.cpu()
    rows = (indices_cpu >= 0).any(1).nonzero().flatten()
    if not missing_bindings:
        # Preserve native null pooling arithmetic, including GROUP weight rounding.
        # It depends on the query, so compute it once across all candidate chunks.
        missing = PersonMatching(
            torch.full_like(matching.indices[:1], -1),
            matching.membership[:1],
            matching.subjects[:1],
            matching.subject_coverage[:1],
            matching.coverage[:1],
        )
        missing_bindings.update(
            pool_fafa_scores(
                torch.zeros_like(missing.membership),
                missing,
                model.fine_matching["missing_penalty"],
            )
        )
    components = {
        "context": matching.coverage.new_zeros(len(targets)),
        "coverage": matching.coverage,
        **{
            f"binding_{mode}": value.expand(len(targets)).clone()
            for mode, value in missing_bindings.items()
        },
    }
    if not len(rows):
        # All pooling modes equal the configured missing penalty; H is exactly zero.
        return components
    selected = rows.to(device, non_blocking=True)
    if len(rows) == len(targets):
        chosen, chosen_matching = query, matching
    else:
        chosen = query.index_select(selected)
        chosen_matching = PersonMatching(
            matching.indices.index_select(0, selected),
            matching.membership.index_select(0, selected),
            matching.subjects.index_select(0, selected),
            matching.subject_coverage.index_select(0, selected),
            matching.coverage.index_select(0, selected),
            indices_cpu.index_select(0, rows),
        )
    native = NativeTargetBatch(cache, targets[rows], mask.shape[1])
    poolings = fafa_binding_scores(
        chosen.fafa,
        native,
        matching=chosen_matching,
        k=model.fda_k,
        missing_penalty=model.fine_matching["missing_penalty"],
        return_poolings=True,
    )
    for mode, values in poolings.items():
        components[f"binding_{mode}"].index_copy_(0, selected, values)
    if compute_context:
        scene = cache.load_scenes(targets[rows]).to(device, non_blocking=True).float()
        context = model.score_aligned_context(
            chosen,
            scene,
            semantic.index_select(0, selected),
            boxes.index_select(0, selected),
            chosen_matching.indices,
            cache.patch_hw,
            layout=layout,
        )
        components["context"].index_copy_(0, selected, context)
    return components


@torch.inference_mode()
def retrieve_rankings(
    samples: Sequence[dict],
    cache: GalleryCache,
    tokenizer: Any,
    text_encoder: nn.Module,
    model: nn.Module,
    device: torch.device,
    *,
    gallery_ids: Sequence[str],
    top_m: int,
    fine_batch_size: int,
    identity_batch_size: int,
    coarse_batch_size: int = 512,
    coarse_beta: float | None = None,
    fine_score_mode: str = FINE_SCORE_MODE,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
    fine_coarse_weight: float = DEFAULT_FINE_COARSE_WEIGHT,
    pooling: str = "penalized",
    description: str = "retrieve",
    text_cache: QueryTextCache | None = None,
    state_trained: bool = True,
    query_batch_size: int = 1,
    gpu_cache_mib: int = 512,
    prefetch_batches: int = 1,
) -> dict[str, Any]:
    """Rank the gallery once for every query.

    Coarse scores order the whole gallery. Only the Top-M shortlist is scored by
    the pairwise model and reranked; the coarse tail is unchanged. Training
    validation, retrieval and offline replay share this one scoring path.
    """
    from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text

    started = perf_counter()
    timings = dict.fromkeys(
        ("gallery_encoding", "query_and_coarse", "fine_scoring", "ranking"), 0.0
    )
    if not samples:
        raise ValueError("retrieval requires samples")
    samples = inference_samples(samples)
    settings = retrieval_settings(
        {
            "top_m": top_m,
            "fine_batch_size": fine_batch_size,
            "identity_batch_size": identity_batch_size,
            "coarse_batch_size": coarse_batch_size,
            "coarse_beta": coarse_beta,
            "fine_score_mode": fine_score_mode,
            "context_weight": context_weight,
            "coverage_weight": coverage_weight,
            "fine_coarse_weight": fine_coarse_weight,
            "pooling": pooling,
            "query_batch_size": query_batch_size,
            "gpu_cache_mib": gpu_cache_mib,
            "prefetch_batches": prefetch_batches,
        },
        model.coarse_beta,
    )
    beta = settings["coarse_beta"]
    if not state_trained and beta != 0:
        raise ValueError("checkpoint state branch is untrained; use coarse_beta=0")
    binding_scale = model.binding_scale.item()
    gallery_ids, gallery_by_id, image_indices = _gallery_scope(
        cache, gallery_ids, samples, "retrieval"
    )
    if len(gallery_ids) < 2:
        raise ValueError("retrieval requires at least two gallery images")
    need_state = beta != 0
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))
    text_stats = text_cache.statistics(list(samples))
    rows = [None] * len(samples)
    with evaluating(model, text_encoder):
        stage = perf_counter()
        gallery_batches = _encode_gallery_identity(
            cache, model, device, identity_batch_size, image_indices
        )
        gallery_state = (
            _encode_gallery_state(
                cache, model, device, identity_batch_size, image_indices
            )
            if need_state
            else None
        )
        gallery_batches, gallery_state = _cache_gallery_on_device(
            gallery_batches, gallery_state, device, gpu_cache_mib
        )
        timings["gallery_encoding"] += perf_counter() - stage
        subject_groups = {}
        for index, sample in enumerate(samples):
            subject_groups.setdefault(len(sample["subjects"]), []).append(
                (index, sample)
            )
        batches = [
            group[start : start + query_batch_size]
            for group in subject_groups.values()
            for start in range(0, len(group), query_batch_size)
        ]
        progress = tqdm(total=len(samples), desc=description)
        for batch in batches:
            stage = perf_counter()
            batch_samples = [sample for _, sample in batch]
            text = encode_query_text(
                batch_samples, tokenizer, text_encoder, device, text_cache=text_cache
            )
            query_idx = torch.tensor(
                [cache.by_id[sample["query_image_id"]] for sample in batch_samples]
            )
            q_scene, q_persons, q_boxes, q_mask = cache.load(query_idx)
            q_mask = q_mask.to(device, non_blocking=True)
            queries = model.encode_query(
                q_scene.to(device, non_blocking=True),
                q_persons.to(device, non_blocking=True),
                q_boxes.to(device, non_blocking=True),
                patch_hw=cache.patch_hw,
                query_person_mask=q_mask,
                fafa_query=cache.native_queries(batch_samples, q_persons.shape[1]).to(
                    device, non_blocking=True
                ),
                **{key: value for key, value in text.items() if key != "change_global"},
            )
            identity = (
                _coarse_scores_chunked(
                    queries.identity,
                    queries.logits,
                    q_mask,
                    gallery_batches,
                    coarse_batch_size,
                )
                .float()
                .cpu()
            )
            state = None
            if need_state:
                query_state = model.encode_text_state(text["change_global"])
                state = (
                    torch.cat(
                        [
                            query_state
                            @ gallery_state[start : start + coarse_batch_size]
                            .to(device, non_blocking=True)
                            .T
                            for start in range(0, len(gallery_ids), coarse_batch_size)
                        ],
                        dim=1,
                    )
                    .float()
                    .cpu()
                )
            timings["query_and_coarse"] += perf_counter() - stage
            for n, (sample_index, sample) in enumerate(batch):
                stage = perf_counter()
                query = queries.select(n)
                query_index = gallery_by_id[sample["query_image_id"]]
                coarse = combine_scores(
                    identity[n],
                    None if state is None else state[n],
                    beta=beta,
                    exclude_index=query_index,
                )
                order = torch.argsort(coarse, descending=True, stable=True)
                order = order[order != query_index]
                top = order[:top_m]
                timings["ranking"] += perf_counter() - stage
                stage = perf_counter()
                evidence = {
                    key: value.cpu()
                    for key, value in _score_target_components(
                        query,
                        cache,
                        model,
                        device,
                        image_indices[top],
                        fine_batch_size,
                        pooling,
                        context_weight,
                        coverage_weight,
                        prefetch_batches=prefetch_batches,
                    ).items()
                }
                timings["fine_scoring"] += perf_counter() - stage
                stage = perf_counter()
                raw = {
                    "binding": evidence[f"binding_{pooling}"],
                    "context": evidence["context"],
                    "coverage": evidence["coverage"],
                }
                scored = fine_score_components(
                    **raw,
                    binding_scale=binding_scale,
                    fine_score_mode=fine_score_mode,
                    context_weight=context_weight,
                    coverage_weight=coverage_weight,
                )
                fine = rerank_topm(order, top, scored["fine_scores"]).to(torch.int32)
                fused = fuse_fine_coarse_scores(
                    scored["fine_scores"], coarse[top], fine_coarse_weight
                )
                rows[sample_index] = {
                    **scored,
                    **evidence,
                    "query_indices": torch.tensor(query_index),
                    "coarse_rankings": order.to(torch.int32),
                    "coarse_topm": top.to(torch.int32),
                    "coarse_topm_scores": coarse[top],
                    "fafa_rankings": rerank_topm(order, top, raw["binding"]).to(
                        torch.int32
                    ),
                    "fine_rankings": fine,
                    "fused_topm_scores": fused,
                    "rankings": rerank_topm(order, top, fused).to(torch.int32),
                }
                timings["ranking"] += perf_counter() - stage
            progress.update(len(batch))
        progress.close()
    output = {
        "timings_seconds": timings,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "gallery_ids": gallery_ids,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "schema_version": SCORE_SCHEMA_VERSION,
        "clip_text_stats": text_stats,
        "binding_scale": binding_scale,
        "state_trained": state_trained,
        "fine_matching": dict(model.fine_matching),
        "score_config": score_configuration(settings, binding_scale),
        "provenance": {},
        **{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
    }
    # Alias of the selected pooling's matrix, not a separate score matrix.
    output["fafa_binding"] = output[f"binding_{pooling}"]
    timings["total"] = perf_counter() - started
    tqdm.write(
        f"{description} timing (s): "
        + ", ".join(f"{key}={value:.2f}" for key, value in timings.items())
    )
    return output


@torch.inference_mode()
def mine_hard_negatives(
    samples,
    cache,
    tokenizer,
    text_encoder,
    model,
    device,
    *,
    gallery_ids,
    retrieval,
    pool_size,
    excluded=None,
    query_batch_size=1,
    text_cache=None,
    identity_image_ids=None,
    semantic_hard=False,
    identity_rejection=False,
):
    """Mine train-only coarse or fine-ranked semantic/identity negative pools.

    Coarse filters each requested group to ``pool_size`` images. Fine scores use
    binding, coverage and coarse fusion over their union, with zero context
    weight. Labels classify pools outside the scorer; normalization is shared
    across groups and inference chunks.
    """
    from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text

    settings = retrieval_settings(retrieval, model.coarse_beta)
    settings["context_weight"] = MINING_CONTEXT_WEIGHT
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("mining pool_size must be a positive integer")
    if type(query_batch_size) is not int or query_batch_size < 1:
        raise ValueError("mining query_batch_size must be a positive integer")
    fine_mining = semantic_hard or identity_rejection
    if fine_mining and (
        identity_image_ids is None
        or any(row["sample_id"] not in identity_image_ids for row in samples)
    ):
        raise ValueError("fine mining requires train identity candidate sets")
    gallery_ids, gallery_by_id, image_indices = _gallery_scope(
        cache, gallery_ids, samples, "mining"
    )
    excluded = excluded or {}
    forbidden = {
        row["sample_id"]: set(row["positive_image_ids"])
        | {row["query_image_id"]}
        | set(excluded.get(row["sample_id"], ()))
        for row in samples
    }
    limit = pool_size + max(map(len, forbidden.values()))
    samples = inference_samples(samples)
    if fine_mining:
        cache.validate_samples(samples)
    text_cache = text_cache or QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))

    subject_groups = {}
    for sample in samples:
        subject_groups.setdefault(len(sample["subjects"]), []).append(sample)
    need_state = settings["coarse_beta"] != 0
    pools = {}

    with evaluating(model, text_encoder):
        gallery_identity = _encode_gallery_identity(
            cache,
            model,
            device,
            settings["identity_batch_size"],
            image_indices,
        )
        gallery_state = (
            _encode_gallery_state(
                cache,
                model,
                device,
                settings["identity_batch_size"],
                image_indices,
            )
            if need_state
            else None
        )
        gallery_identity, gallery_state = _cache_gallery_on_device(
            gallery_identity, gallery_state, device, settings["gpu_cache_mib"]
        )
        progress = tqdm(total=len(samples), desc="mine train negatives")
        for sample_group in subject_groups.values():
            for start in range(0, len(sample_group), query_batch_size):
                rows = sample_group[start : start + query_batch_size]
                text = encode_query_text(
                    rows, tokenizer, text_encoder, device, text_cache=text_cache
                )
                indices = torch.tensor(
                    [cache.by_id[row["query_image_id"]] for row in rows]
                )
                scene, people, boxes, mask = cache.load(indices)
                mask = mask.to(device)
                if fine_mining:
                    queries = model.encode_query(
                        scene.to(device),
                        people.to(device),
                        boxes.to(device),
                        patch_hw=cache.patch_hw,
                        query_person_mask=mask,
                        fafa_query=cache.native_queries(rows, people.shape[1]).to(
                            device
                        ),
                        **{
                            key: value
                            for key, value in text.items()
                            if key != "change_global"
                        },
                    )
                    logits, query_identity = queries.logits, queries.identity
                else:
                    logits, query_identity = model.encode_grounded_identity(
                        scene.to(device),
                        people.to(device),
                        boxes.to(device),
                        text["selections"],
                        cache.patch_hw,
                        text["selection_mask"],
                    )

                identity = query_identity.new_empty(len(rows), len(gallery_ids))
                offset = 0
                for people_cpu, mask_cpu in gallery_identity:
                    for j in range(0, len(people_cpu), settings["coarse_batch_size"]):
                        stop = min(j + settings["coarse_batch_size"], len(people_cpu))
                        identity[:, offset + j : offset + stop] = (
                            identity_scores_batched(
                                query_identity,
                                logits,
                                people_cpu[j:stop].to(device),
                                mask_cpu[j:stop].to(device),
                                query_mask=mask,
                            )
                        )
                    offset += len(people_cpu)

                state = None
                if need_state:
                    query_state = model.encode_text_state(text["change_global"])
                    state = query_identity.new_empty(len(rows), len(gallery_ids))
                    for j in range(0, len(gallery_ids), settings["coarse_batch_size"]):
                        stop = j + settings["coarse_batch_size"]
                        state[:, j:stop] = (
                            query_state @ gallery_state[j:stop].to(device).T
                        )

                for n, row in enumerate(rows):
                    query_index = gallery_by_id[row["query_image_id"]]
                    scores = combine_scores(
                        identity[n],
                        None if state is None else state[n],
                        beta=settings["coarse_beta"],
                        exclude_index=query_index,
                    )
                    order = torch.argsort(scores, descending=True, stable=True)
                    sample_id = row["sample_id"]
                    pools[sample_id] = {}
                    if not identity_rejection:
                        pools[sample_id]["coarse"] = [
                            gallery_ids[index]
                            for index in order[:limit].cpu().tolist()
                            if gallery_ids[index] not in forbidden[sample_id]
                        ][:pool_size]
                    if not fine_mining:
                        continue
                    # Build the bounded union in coarse order for stable ties.
                    scores, order = scores.cpu(), order.cpu()
                    eligible = (
                        torch.tensor(
                            [i not in forbidden[sample_id] for i in gallery_ids]
                        )
                        & scores.isfinite()
                    )
                    same_id = torch.tensor(
                        [i in identity_image_ids[sample_id] for i in gallery_ids]
                    )
                    negative_groups = {}
                    if semantic_hard:
                        negative_groups["semantic"] = eligible & same_id
                    if identity_rejection:
                        negative_groups["identity"] = eligible & ~same_id
                    screened = [
                        order[membership[order]][:pool_size]
                        for membership in negative_groups.values()
                    ]
                    selected = torch.cat(screened)
                    shortlist = order[torch.isin(order, selected)]
                    pools[sample_id].update({name: [] for name in negative_groups})
                    if not shortlist.numel():
                        continue
                    components = _score_target_components(
                        queries.select(n),
                        cache,
                        model,
                        device,
                        image_indices[shortlist],
                        settings["fine_batch_size"],
                        settings["pooling"],
                        settings["context_weight"],
                        settings["coverage_weight"],
                        compute_context=False,
                        prefetch_batches=settings["prefetch_batches"],
                    )
                    fine = fine_score_components(
                        components[f"binding_{settings['pooling']}"].cpu(),
                        components["context"].cpu(),
                        components["coverage"].cpu(),
                        model.binding_scale.item(),
                        settings["fine_score_mode"],
                        settings["context_weight"],
                        settings["coverage_weight"],
                    )
                    fused = fuse_fine_coarse_scores(
                        fine["fine_scores"],
                        scores[shortlist],
                        settings["fine_coarse_weight"],
                    )
                    ranked = fused.argsort(descending=True, stable=True)
                    covered = components["coverage"].cpu() > 0
                    for name, membership in negative_groups.items():
                        keep = covered & membership[shortlist]
                        pools[sample_id][name] = [
                            gallery_ids[i]
                            for i in shortlist[ranked[keep[ranked]]].tolist()
                        ]
                progress.update(len(rows))
        progress.close()
    return pools
