"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.common.data import inference_samples
from rcr.common.runtime import evaluating
from rcr.proposed.scores import (
    DEFAULT_CONTEXT_WEIGHT,
    DEFAULT_COVERAGE_WEIGHT,
    FINE_SCORE_MODE,
    MINING_CONTEXT_WEIGHT,
    POOLING_MODES,
    combine_scores,
    identity_scores_batched,
    linear_score_components,
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
):
    """Collect pairwise evidence without changing its scoring population."""
    components = {
        key: torch.empty(len(image_indices), device=device, dtype=torch.float32)
        for key in (
            "context",
            "coverage",
            *(f"binding_{mode}" for mode in POOLING_MODES),
        )
    }
    for start in range(0, len(image_indices), batch_size):
        targets = image_indices[start : start + batch_size]
        scene, persons, boxes, mask = cache.load(targets)
        details = model.score_target(
            query.repeat_candidates(len(targets)),
            scene.to(device, non_blocking=True),
            persons.to(device, non_blocking=True),
            boxes.to(device, non_blocking=True),
            cache.patch_hw,
            mask.to(device, non_blocking=True),
            fafa_targets=cache.native_targets(targets, persons.shape[1]).to(
                device, non_blocking=True
            ),
            return_details=True,
            pooling=pooling,
            context_weight=context_weight,
            coverage_weight=coverage_weight,
            compute_context=compute_context,
        )
        finite = torch.stack([torch.isfinite(details[key]).all() for key in components])
        if not finite.all():
            raise ValueError("fine score components contain NaN/Inf")
        for key, output in components.items():
            values = details[key].float()
            output[start : start + len(targets)] = values
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
) -> dict[str, Any]:
    """Rank the gallery once for every query.

    Coarse scores order the whole gallery. Only the Top-M shortlist is scored by
    the pairwise model and reranked; the coarse tail is unchanged. Training
    validation, retrieval and offline replay share this one scoring path.
    """
    from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text

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
            for n, (sample_index, sample) in enumerate(batch):
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
                    ).items()
                }
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
            progress.update(len(batch))
        progress.close()
    output = {
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
