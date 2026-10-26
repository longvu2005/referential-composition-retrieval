"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.common.data import inference_samples
from rcr.common.runtime import evaluating
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text
from rcr.proposed.scores import (
    combine_scores,
    identity_scores,
    identity_scores_batched,
)

SCORE_SCHEMA_VERSION = 1
SCORE_EPS = 1e-6
FINE_SCORE_MODES = ("legacy_raw", "context_zscore")
TUNING_CONVENTIONS = {
    "schema_version": 1,
    "scope": "full_gallery",
    "score_alignment": "gallery_ids",
    "normalization_dtype": "float32",
    "normalization_device": "cpu",
    "eps": SCORE_EPS,
    "std_unbiased": False,
}


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
) -> dict[str, Tensor]:
    """Score one complete Coarse Top-M, never an inference mini-batch.

    Raw gamma and both-z-score are offline ablations only. All vectors retain
    candidate order. Non-finite fine evidence is an error, including at C=0.
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
    mean, std = _score_statistics(gated)
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


def score_configuration(settings: dict, binding_scale: float) -> dict:
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
        "binding_scale": binding_scale,
        "normalization": "per_query_coarse_topm_population",
        "normalization_dtype": "float32",
        "normalization_device": "cpu",
        "eps": SCORE_EPS,
        "std_unbiased": False,
        "score_alignment": "coarse_topm",
        "tie_break": "stable_coarse_order_then_canonical_gallery",
        "nonfinite_coarse": "masked_in_fusion_when_weight_positive",
    }


def fuse_fine_coarse_scores(
    fine: Tensor,
    coarse: Tensor,
    weight: float,
    eps: float = SCORE_EPS,
) -> Tensor:
    """Fuse fine and coarse evidence inside the same Top-M shortlist.

    ``weight=0`` returns the raw fine scores exactly. For positive weights, both
    branches are population-z-scored over the same candidates with finite coarse
    support, then combined as
    ``z(fine) + weight * z(coarse)``. Non-finite coarse candidates stay last
    instead of being rescued by a fine score despite having no coarse support.
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
        fused[valid] = _zscore_1d(fine[valid], eps) + weight * _zscore_1d(
            coarse[valid], eps
        )
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
    }
    if unknown:
        raise ValueError(f"unknown retrieval settings: {', '.join(sorted(unknown))}")
    settings = {
        "coarse_batch_size": 512,
        "coarse_beta": default_beta,
        "fine_coarse_weight": 0.0,
        "fine_score_mode": "context_zscore",
        "context_weight": 0.03,
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
    description: str = "retrieve",
    text_cache: QueryTextCache | None = None,
    save_tuning: bool = False,
    state_trained: bool = True,
) -> dict[str, Any]:
    """Single-run interface shared by training validation and standalone use."""
    settings = {
        "top_m": top_m,
        "fine_batch_size": fine_batch_size,
        "identity_batch_size": identity_batch_size,
        "coarse_batch_size": coarse_batch_size,
        "coarse_beta": coarse_beta,
        "fine_coarse_weight": fine_coarse_weight,
        "fine_score_mode": fine_score_mode,
        "context_weight": context_weight,
    }
    settings = retrieval_settings(settings, model.coarse_beta)
    return rank_weight_grid(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=gallery_ids,
        settings=settings,
        weights=[(settings["coarse_beta"], settings["fine_coarse_weight"])],
        description=description,
        text_cache=text_cache,
        save_tuning=save_tuning,
        state_trained=state_trained,
    )[0]


@torch.inference_mode()
def rank_weight_grid(
    samples: Sequence[dict],
    cache: GalleryCache,
    tokenizer: Any,
    text_encoder: nn.Module,
    model: nn.Module,
    device: torch.device,
    *,
    gallery_ids: Sequence[str],
    settings: dict,
    weights: Sequence[tuple[float, float]],
    description: str = "retrieve",
    ranking_limit: int | None = None,
    text_cache: QueryTextCache | None = None,
    save_tuning: bool = False,
    state_trained: bool = True,
) -> list[dict]:
    """Share model scores while calibrating (coarse beta, fine/coarse weight).

    Only the split gallery is projected. CPU gallery projections and GPU scoring
    chunks bound accelerator memory. Coarse normalization happens after gathering
    one complete score vector, never per chunk. Equal beta values share coarse
    scores and rankings; each query-target pair is fine-scored once. Outputs keep
    full gallery order; top_m only limits fine reranking. Optional ranking_limit
    retains only a prefix to bound CPU output memory. Such truncated outputs
    are never accepted by the official full-rank evaluator.
    """
    if not samples or not weights:
        raise ValueError("retrieval requires samples and at least one weight pair")
    samples = inference_samples(samples)
    if ranking_limit is not None and ranking_limit < 1:
        raise ValueError("ranking_limit must be positive")
    if save_tuning and ranking_limit is not None:
        raise ValueError("tuning export requires complete gallery rankings")
    settings = retrieval_settings(settings, model.coarse_beta)
    batch_keys = ("fine_batch_size", "identity_batch_size", "coarse_batch_size")
    for weight in (value for pair in weights for value in pair):
        if (
            not isinstance(weight, int | float)
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("retrieval weights must be finite and nonnegative")
    if not state_trained and any(beta != 0 for beta, _ in weights):
        raise ValueError("checkpoint state branch is untrained; use coarse_beta=0")
    fine_batch_size, identity_batch_size, coarse_batch_size = (
        settings[key] for key in batch_keys
    )
    binding_scale = model.binding_scale.item()
    gallery_ids, gallery_by_id, image_indices = _gallery_scope(
        cache, gallery_ids, samples, "retrieval"
    )
    by_id = cache.by_id
    need_state = any(beta != 0 for beta, _ in weights) or (
        save_tuning and state_trained
    )
    tuning_rows = {
        key: []
        for key in (
            "query_indices",
            "identity_scores",
            "state_scores",
            "fafa_binding",
            "context",
            "coverage",
        )
    }
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))
    text_stats = text_cache.statistics(list(samples))
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
        shared_fields = (
            "coarse_rankings",
            "coarse_topm",
            "fine_rankings",
            "fafa_rankings",
            "fafa_binding",
            "context",
            "coverage",
            "coarse_topm_scores",
            "gated_context",
            "fine_raw_scores",
            "fine_scores",
            "context_mean",
            "context_std",
        )
        shared_outputs = {
            beta: {key: [] for key in shared_fields} for beta, _ in weights
        }
        outputs = [{"rankings": [], "fused_topm_scores": []} for _ in weights]
        for sample in tqdm(samples, desc=description):
            query_index = gallery_by_id[sample["query_image_id"]]
            text = encode_query_text(
                [sample], tokenizer, text_encoder, device, text_cache=text_cache
            )
            query_idx = torch.tensor([by_id[sample["query_image_id"]]])
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
            components = {
                key: torch.zeros(len(gallery_ids), device=device, dtype=torch.float32)
                for key in ("binding", "context", "coverage")
            }
            fine_done = torch.zeros(len(gallery_ids), dtype=torch.bool, device=device)

            def score_missing(
                indices, *, query=query, components=components, fine_done=fine_done
            ):
                for start in range(0, len(indices), fine_batch_size):
                    local = indices[start : start + fine_batch_size]
                    scene, persons, boxes, mask = cache.load(image_indices[local.cpu()])
                    details = model.score_target(
                        query.repeat_candidates(len(local)),
                        scene.to(device),
                        persons.to(device),
                        boxes.to(device),
                        cache.patch_hw,
                        mask.to(device),
                        fafa_targets=cache.native_targets(
                            image_indices[local.cpu()], persons.shape[1]
                        ).to(device),
                        return_details=True,
                    )
                    for key in components:
                        values = details[key].float()
                        if not torch.isfinite(values).all():
                            raise ValueError("fine score components contain NaN/Inf")
                        components[key][local] = values
                    fine_done[local] = True

            coarse_orders = {}
            fine_shortlists = {}
            recorded_shared = set()
            for output, (beta, fine_coarse_weight) in zip(
                outputs, weights, strict=True
            ):
                # A joint sweep changes the final fusion weight repeatedly for
                # each beta. Normalize/sort coarse only once per distinct setting.
                coarse_key = beta
                if coarse_key not in coarse_orders:
                    coarse = combine_scores(
                        identity,
                        state,
                        beta=beta,
                        exclude_index=query_index,
                    )
                    order = torch.argsort(coarse, descending=True, stable=True)
                    order = order[order != query_index]
                    coarse_orders[coarse_key] = coarse, order
                coarse, order = coarse_orders[coarse_key]
                top_indices = order[: settings["top_m"]]
                device_indices = top_indices.to(device)
                score_missing(device_indices[~fine_done[device_indices]])
                if coarse_key not in fine_shortlists:
                    # CPU FP32 scoring makes saved-score replay use the exact
                    # same reductions/sort, including after accelerator inference.
                    raw = {
                        key: value[device_indices].float().cpu()
                        for key, value in components.items()
                    }
                    scored = fine_score_components(
                        **raw,
                        binding_scale=binding_scale,
                        fine_score_mode=settings["fine_score_mode"],
                        context_weight=settings["context_weight"],
                    )
                    fine_shortlists[coarse_key] = (
                        raw,
                        scored,
                        top_indices.cpu(),
                        order.cpu(),
                        coarse[top_indices].float().cpu(),
                    )
                raw, scored, top_indices, order_cpu, coarse_topm = fine_shortlists[
                    coarse_key
                ]
                rerank_scores = fuse_fine_coarse_scores(
                    scored["fine_scores"],
                    coarse_topm,
                    float(fine_coarse_weight),
                )
                output["fused_topm_scores"].append(rerank_scores)
                final_order = rerank_topm(order_cpu, top_indices, rerank_scores)
                output["rankings"].append(
                    final_order[:ranking_limit].to(torch.int32).cpu()
                )
                if coarse_key in recorded_shared:
                    continue
                recorded_shared.add(coarse_key)
                shared = shared_outputs[coarse_key]
                # Diagnostics use exactly this shortlist and identity assignment.
                shared["fafa_rankings"].append(
                    rerank_topm(order_cpu, top_indices, raw["binding"])[:ranking_limit]
                    .to(torch.int32)
                    .cpu()
                )
                for field, key in (
                    ("fafa_binding", "binding"),
                    ("context", "context"),
                    ("coverage", "coverage"),
                ):
                    shared[field].append(raw[key])
                shared["fine_rankings"].append(
                    rerank_topm(order_cpu, top_indices, scored["fine_scores"])[
                        :ranking_limit
                    ]
                    .to(torch.int32)
                    .cpu()
                )
                for field, value in scored.items():
                    shared[field].append(value)
                shared["coarse_topm_scores"].append(coarse_topm)
                shared["coarse_rankings"].append(
                    order[:ranking_limit].to(torch.int32).cpu()
                )
                shared["coarse_topm"].append(top_indices.to(torch.int32))
            if save_tuning:
                # Keep normal shortlist forwards first. Compute only additional
                # targets, once per query, and never score the query against itself.
                remaining = ~fine_done
                remaining[query_index] = False
                score_missing(remaining.nonzero(as_tuple=True)[0])
                tuning_rows["query_indices"].append(torch.tensor(query_index))
                tuning_rows["identity_scores"].append(identity)
                if state is not None:
                    tuning_rows["state_scores"].append(state)
                for field, key in (
                    ("fafa_binding", "binding"),
                    ("context", "context"),
                    ("coverage", "coverage"),
                ):
                    tuning_rows[field].append(components[key].cpu())
        shared_tensors = {
            beta: {key: torch.stack(rows) for key, rows in output.items()}
            for beta, output in shared_outputs.items()
        }
        tuning = (
            {
                **TUNING_CONVENTIONS,
                "state_trained": state_trained,
                "fixed_fine_matching": dict(model.fine_matching),
                **{
                    key: torch.stack(rows) if rows else None
                    for key, rows in tuning_rows.items()
                },
            }
            if save_tuning
            else None
        )
        return [
            {
                "sample_ids": [sample["sample_id"] for sample in samples],
                "clip_text_stats": text_stats,
                "gallery_ids": gallery_ids,
                "score_schema_version": SCORE_SCHEMA_VERSION,
                "binding_scale": binding_scale,
                "score_config": score_configuration(
                    {
                        **settings,
                        "coarse_beta": beta,
                        "fine_coarse_weight": fine_weight,
                    },
                    binding_scale,
                ),
                **shared_tensors[beta],
                **{key: torch.stack(rows) for key, rows in output.items()},
                **({"tuning_scores": tuning} if save_tuning else {}),
            }
            for output, (beta, fine_weight) in zip(outputs, weights, strict=True)
        ]


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
