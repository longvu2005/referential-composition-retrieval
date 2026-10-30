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
    POOLING_MODES,
    combine_scores,
    identity_scores,
    identity_scores_batched,
)

SCORE_SCHEMA_VERSION = 1
SCORE_EPS = 1e-6
FINE_SCORE_MODES = ("legacy_raw", "context_zscore")
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

    num_images = sum(len(identity) for identity, _ in gallery_batches)
    scores = query_identity.new_empty(num_images)
    offset = 0
    for identity_cpu, mask_cpu in gallery_batches:
        for start in range(0, len(identity_cpu), batch_size):
            stop = min(start + batch_size, len(identity_cpu))
            identity = identity_cpu[start:stop].to(query_identity.device)
            mask = mask_cpu[start:stop].to(query_identity.device)
            scores[offset + start : offset + stop] = identity_scores(
                query_identity,
                logits,
                identity,
                mask,
                query_mask=query_mask,
            )
            del identity, mask
        offset += len(identity_cpu)
    return scores


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


def _score_statistics(values: Tensor) -> tuple[Tensor, Tensor]:
    values = values.float()
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("score statistics require a non-empty finite 1D tensor")
    # Repeated FP32 values can have a rounded mean different from that value.
    if (values == values[0]).all():
        return values[0], values.new_zeros(())
    mean, std = values.mean(), values.std(unbiased=False)
    if not torch.isfinite(mean) or not torch.isfinite(std):
        raise ValueError("score statistics overflowed float32")
    return mean, std


def _zscore_1d(values: Tensor, eps: float = SCORE_EPS) -> Tensor:
    """Population z-score for one finite score vector; constants become zero."""
    values = values.float()
    if values.numel() == 0:
        return values
    mean, std = _score_statistics(values)
    return (values - mean) / std.clamp_min(eps)


def fine_score_components(
    binding: Tensor,
    context: Tensor,
    coverage: Tensor,
    binding_scale: float,
    fine_score_mode: str = "context_zscore",
    context_weight: float = 0.03,
    *,
    context_statistics: tuple[Tensor, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Score one complete Coarse Top-M, never an inference mini-batch.

    Raw gamma and both-z-score are offline ablations only. All vectors retain
    candidate order. Non-finite fine evidence is an error, including at C=0.
    Cached context statistics must come from this exact shortlist.
    """
    if (
        binding.ndim != 1
        or binding.shape != context.shape
        or binding.shape != coverage.shape
    ):
        raise ValueError("binding, context and coverage must be matching 1D tensors")
    if fine_score_mode not in (*FINE_SCORE_MODES, "raw_gamma", "both_zscore"):
        raise ValueError(f"unknown fine_score_mode: {fine_score_mode!r}")
    if any(not math.isfinite(x) or x < 0 for x in (binding_scale, context_weight)):
        raise ValueError(
            "binding scale and context weight must be finite and nonnegative"
        )
    binding, context, coverage = binding.float(), context.float(), coverage.float()
    if not all(torch.isfinite(x).all() for x in (binding, context, coverage)):
        raise ValueError("fine score components contain NaN/Inf")
    gated = coverage * context
    mean, std = (
        _score_statistics(gated) if context_statistics is None else context_statistics
    )
    scaled_binding = binding_scale * binding
    raw = scaled_binding + gated
    if fine_score_mode == "legacy_raw":
        fine = raw
    elif fine_score_mode == "raw_gamma":
        fine = scaled_binding + context_weight * gated
    else:
        fine = (
            _zscore_1d(scaled_binding)
            if fine_score_mode == "both_zscore"
            else scaled_binding
        ) + context_weight * ((gated - mean) / std.clamp_min(SCORE_EPS))
    if not torch.isfinite(raw).all() or not torch.isfinite(fine).all():
        raise ValueError("fine scores overflowed float32")
    return {
        "gated_context": gated,
        "fine_raw_scores": raw,
        "fine_scores": fine,
        "context_mean": mean,
        "context_std": std,
    }


def rerank_topm(order: Tensor, top_indices: Tensor, scores: Tensor) -> Tensor:
    """Stable shortlist sort; ties retain Coarse order, tail is unchanged."""
    ranked = top_indices[torch.argsort(scores, descending=True, stable=True)]
    return torch.cat((ranked, order[len(top_indices) :]))


def score_configuration(
    settings: dict, binding_scale: float, *, full_gallery_scores: bool = False
) -> dict:
    """Weights-only-safe convention shared by artifacts and run metadata."""
    return {
        **{
            key: settings[key]
            for key in (
                "fine_score_mode",
                "context_weight",
                "coarse_beta",
                "fine_coarse_weight",
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
        "score_alignment": "gallery_ids" if full_gallery_scores else "coarse_topm",
        "tie_break": "stable_coarse_order_then_canonical_gallery",
        "nonfinite_coarse": "masked_in_fusion_when_weight_positive",
    }


def fuse_fine_coarse_scores(
    fine: Tensor,
    coarse: Tensor,
    weight: float,
    eps: float = SCORE_EPS,
    *,
    normalized_scores: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Fuse fine and coarse evidence inside the same Top-M shortlist.

    ``weight=0`` returns the raw fine scores exactly. For positive weights, both
    branches are population-z-scored over the same candidates with finite coarse
    support, then combined as
    ``z(fine) + weight * z(coarse)``. Non-finite coarse candidates stay last
    instead of being rescued by a fine score despite having no coarse support.
    Cached normalized_scores use this exact shortlist's finite coarse support.
    """
    if fine.ndim != 1 or coarse.ndim != 1 or fine.shape != coarse.shape:
        raise ValueError("fine and coarse scores must be matching 1D tensors")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("fine_coarse_weight must be finite and nonnegative")
    if not torch.isfinite(fine).all():
        raise ValueError("fine scores must be finite before ranking")
    if weight == 0:
        return fine

    valid = torch.isfinite(coarse)

    fused = torch.full_like(fine, -torch.inf, dtype=torch.float32)
    if valid.any():
        fine_z, coarse_z = (
            (_zscore_1d(fine[valid], eps), _zscore_1d(coarse[valid], eps))
            if normalized_scores is None
            else normalized_scores
        )
        fused[valid] = fine_z + weight * coarse_z
        if not torch.isfinite(fused[valid]).all():
            raise ValueError("fine/coarse fusion overflowed float32")
    return fused


def retrieval_settings(cfg: dict, default_beta: float) -> dict:
    """Resolve runtime overrides and the model's default coarse weight."""
    unknown = cfg.keys() - {
        "top_m",
        "fine_batch_size",
        "identity_batch_size",
        "coarse_batch_size",
        "coarse_beta",
        "fine_coarse_weight",
        "fine_score_mode",
        "context_weight",
        "pooling",
    }
    if unknown:
        raise ValueError(f"unknown retrieval settings: {', '.join(sorted(unknown))}")
    settings = {
        "coarse_batch_size": 512,
        "coarse_beta": default_beta,
        "fine_coarse_weight": 0.0,
        "fine_score_mode": "context_zscore",
        "context_weight": 0.03,
        "pooling": "penalized",
        **cfg,
    }
    if settings["coarse_beta"] is None:
        settings["coarse_beta"] = default_beta
    sizes = ("top_m", "fine_batch_size", "identity_batch_size", "coarse_batch_size")
    if any(
        key in settings and (type(settings[key]) is not int or settings[key] < 1)
        for key in sizes
    ):
        raise ValueError("retrieval batch sizes and top_m must be positive")
    if any(
        not math.isfinite(float(settings[key])) or settings[key] < 0
        for key in ("coarse_beta", "fine_coarse_weight", "context_weight")
    ):
        raise ValueError("retrieval weights must be finite and nonnegative")
    if settings["fine_score_mode"] not in FINE_SCORE_MODES:
        raise ValueError(f"unknown fine_score_mode: {settings['fine_score_mode']!r}")
    if settings["pooling"] not in POOLING_MODES:
        raise ValueError(f"unknown FAFA pooling: {settings['pooling']!r}")
    return settings


def uses_state(cfg: dict, default_beta: float) -> bool:
    settings = retrieval_settings(cfg, default_beta)
    return settings["coarse_beta"] != 0


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
    fine_coarse_weight: float = 0.0,
    fine_score_mode: str = "context_zscore",
    context_weight: float = 0.03,
    pooling: str = "penalized",
    description: str = "retrieve",
    text_cache: QueryTextCache | None = None,
    full_gallery_scores: bool = False,
    state_trained: bool = True,
) -> dict[str, Any]:
    """Rank once; official validation also retains every raw gallery score.

    Training validation and test only fine-score the selected Top-M. Full-gallery
    collection scores that shortlist first, then the remaining non-self images,
    so exporting extra evidence cannot change the direct ranking arithmetic.
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
            "fine_coarse_weight": fine_coarse_weight,
            "fine_score_mode": fine_score_mode,
            "context_weight": context_weight,
            "pooling": pooling,
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
    need_state = beta != 0 or full_gallery_scores
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))
    text_stats = text_cache.statistics(list(samples))
    rows = []
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
        for sample in tqdm(samples, desc=description):
            query_index = gallery_by_id[sample["query_image_id"]]
            text = encode_query_text(
                [sample], tokenizer, text_encoder, device, text_cache=text_cache
            )
            query_idx = torch.tensor([cache.by_id[sample["query_image_id"]]])
            q_scene, q_persons, q_boxes, q_mask = cache.load(query_idx)
            q_mask = q_mask.to(device)
            query = model.encode_query(
                q_scene.to(device),
                q_persons.to(device),
                q_boxes.to(device),
                patch_hw=cache.patch_hw,
                query_person_mask=q_mask,
                fafa_query=cache.native_queries([sample], q_persons.shape[1]).to(
                    device
                ),
                **{key: value for key, value in text.items() if key != "change_global"},
            )
            identity = (
                _coarse_scores_chunked(
                    query.identity[0],
                    query.logits[0],
                    q_mask[0],
                    gallery_batches,
                    coarse_batch_size,
                )
                .float()
                .cpu()
            )
            state = None
            if need_state:
                query_state = model.encode_text_state(text["change_global"])[0]
                state = (
                    torch.cat(
                        [
                            gallery_state[start : start + coarse_batch_size].to(device)
                            @ query_state
                            for start in range(0, len(gallery_ids), coarse_batch_size)
                        ]
                    )
                    .float()
                    .cpu()
                )
            coarse = combine_scores(
                identity, state, beta=beta, exclude_index=query_index
            )
            order = torch.argsort(coarse, descending=True, stable=True)
            order = order[order != query_index]
            top = order[:top_m]
            top_device = top.to(device)
            binding_key = f"binding_{pooling}"
            components = {
                key: torch.zeros(len(gallery_ids), device=device, dtype=torch.float32)
                for key in (
                    "context",
                    "coverage",
                    *(f"binding_{mode}" for mode in POOLING_MODES),
                )
            }

            def score_targets(indices, *, query=query, components=components):
                for start in range(0, len(indices), fine_batch_size):
                    local = indices[start : start + fine_batch_size]
                    targets = image_indices[local.cpu()]
                    scene, persons, boxes, mask = cache.load(targets)
                    details = model.score_target(
                        query.repeat_candidates(len(local)),
                        scene.to(device),
                        persons.to(device),
                        boxes.to(device),
                        cache.patch_hw,
                        mask.to(device),
                        fafa_targets=cache.native_targets(targets, persons.shape[1]).to(
                            device
                        ),
                        return_details=True,
                        pooling=pooling,
                    )
                    for key in components:
                        values = details[key].float()
                        if not torch.isfinite(values).all():
                            raise ValueError("fine score components contain NaN/Inf")
                        components[key][local] = values

            score_targets(top_device)
            raw = {
                "binding": components[binding_key][top_device].cpu(),
                **{
                    key: components[key][top_device].cpu()
                    for key in ("context", "coverage")
                },
            }
            scored = fine_score_components(
                **raw,
                binding_scale=binding_scale,
                fine_score_mode=fine_score_mode,
                context_weight=context_weight,
            )
            fused = fuse_fine_coarse_scores(
                scored["fine_scores"], coarse[top], fine_coarse_weight
            )
            row = {
                **scored,
                "query_indices": torch.tensor(query_index),
                "coarse_rankings": order.to(torch.int32),
                "coarse_topm": top.to(torch.int32),
                "coarse_topm_scores": coarse[top],
                "fused_topm_scores": fused,
                "fafa_rankings": rerank_topm(order, top, raw["binding"]).to(
                    torch.int32
                ),
                "fine_rankings": rerank_topm(order, top, scored["fine_scores"]).to(
                    torch.int32
                ),
                "rankings": rerank_topm(order, top, fused).to(torch.int32),
            }
            if full_gallery_scores:
                # The coarse tail excludes self and contains no scored Top-M pair.
                score_targets(order[len(top) :].to(device))
                row.update(identity_scores=identity, state_scores=state)
            for field, key in (
                ("context", "context"),
                ("coverage", "coverage"),
                *((f"binding_{mode}", f"binding_{mode}") for mode in POOLING_MODES),
            ):
                row[field] = (
                    components[key].cpu()
                    if full_gallery_scores
                    else components[key][top_device].cpu()
                )
            rows.append(row)
    output = {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "gallery_ids": gallery_ids,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "schema_version": SCORE_SCHEMA_VERSION,
        "clip_text_stats": text_stats,
        "binding_scale": binding_scale,
        "state_trained": state_trained,
        "fine_matching": dict(model.fine_matching),
        "score_config": score_configuration(
            settings, binding_scale, full_gallery_scores=full_gallery_scores
        ),
        "provenance": {},
        **{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
    }
    # The selected binding is an alias, not another full score matrix.
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
):
    """Return each query's highest-scoring non-positive coarse candidates."""
    from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text

    settings = retrieval_settings(retrieval, model.coarse_beta)
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
    text_cache = text_cache or QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))

    groups = {}
    for sample in samples:
        groups.setdefault(len(sample["subjects"]), []).append(sample)
    need_state = uses_state(settings, model.coarse_beta)
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
        progress = tqdm(total=len(samples), desc="mine train negatives")
        for group in groups.values():
            for start in range(0, len(group), query_batch_size):
                rows = group[start : start + query_batch_size]
                text = encode_query_text(
                    rows, tokenizer, text_encoder, device, text_cache=text_cache
                )
                indices = torch.tensor(
                    [cache.by_id[row["query_image_id"]] for row in rows]
                )
                scene, people, boxes, mask = cache.load(indices)
                mask = mask.to(device)
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
                    pools[row["sample_id"]] = [
                        gallery_ids[index]
                        for index in order[:limit].cpu().tolist()
                        if gallery_ids[index] not in forbidden[row["sample_id"]]
                    ][:pool_size]
                progress.update(len(rows))
        progress.close()
    return pools
