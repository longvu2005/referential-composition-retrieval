"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.common.runtime import evaluating
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import QueryTextCache, encode_query_text
from rcr.proposed.scores import (
    combine_scores,
    identity_scores,
    identity_scores_batched,
)


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
    features = cache.global_features
    projection = model.state_image_proj.weight
    state = torch.empty(len(image_indices), projection.shape[0], dtype=projection.dtype)
    for start in range(0, len(image_indices), batch_size):
        stop = min(start + batch_size, len(image_indices))
        chunk = features[image_indices[start:stop]].to(
            device=device, dtype=projection.dtype
        )
        state[start:stop] = model.encode_image_state(chunk).cpu()
    return state


def _zscore_1d(values: Tensor, eps: float = 1e-6) -> Tensor:
    """Population z-score for one finite score vector; constants become zero."""
    values = values.float()
    if values.numel() == 0:
        return values
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)


def fuse_fine_coarse_scores(
    fine: Tensor,
    coarse: Tensor,
    weight: float,
    eps: float = 1e-6,
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
    }
    if unknown:
        raise ValueError(f"unknown retrieval settings: {', '.join(sorted(unknown))}")
    settings = {
        "coarse_batch_size": 512,
        "coarse_beta": default_beta,
        "fine_coarse_weight": 0.0,
        **cfg,
    }
    if settings["coarse_beta"] is None:
        settings["coarse_beta"] = default_beta
    sizes = ("top_m", "fine_batch_size", "identity_batch_size", "coarse_batch_size")
    if any(key in settings and settings[key] < 1 for key in sizes):
        raise ValueError("retrieval batch sizes and top_m must be positive")
    if any(
        not math.isfinite(float(settings[key])) or settings[key] < 0
        for key in ("coarse_beta", "fine_coarse_weight")
    ):
        raise ValueError("retrieval weights must be finite and nonnegative")
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
    description: str = "retrieve",
    text_cache: QueryTextCache | None = None,
) -> dict[str, Any]:
    """Single-run interface shared by training validation and standalone use."""
    settings = dict(
        top_m=top_m,
        fine_batch_size=fine_batch_size,
        identity_batch_size=identity_batch_size,
        coarse_batch_size=coarse_batch_size,
        coarse_beta=coarse_beta,
        fine_coarse_weight=fine_coarse_weight,
    )
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
    if ranking_limit is not None and ranking_limit < 1:
        raise ValueError("ranking_limit must be positive")
    settings = retrieval_settings(settings, model.coarse_beta)
    batch_keys = ("fine_batch_size", "identity_batch_size", "coarse_batch_size")
    for weight in (value for pair in weights for value in pair):
        if (
            not isinstance(weight, int | float)
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("retrieval weights must be finite and nonnegative")
    fine_batch_size, identity_batch_size, coarse_batch_size = (
        settings[key] for key in batch_keys
    )
    gallery_ids, gallery_by_id, image_indices = _gallery_scope(
        cache, gallery_ids, samples, "retrieval"
    )
    by_id = cache.by_id
    need_state = any(beta != 0 for beta, _ in weights)
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))
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
        outputs = [
            {
                key: []
                for key in (
                    "rankings",
                    "coarse_rankings",
                    "coarse_topm",
                    "fine_rankings",
                    "fafa_rankings",
                    "fafa_binding",
                    "context",
                    "coverage",
                )
            }
            for _ in weights
        ]
        for sample in tqdm(samples, desc=description):
            query_index = gallery_by_id[sample["query_image_id"]]
            text = encode_query_text(
                [sample], tokenizer, text_encoder, device, text_cache=text_cache
            )
            query_idx = torch.tensor([by_id[sample["query_image_id"]]])
            q_scene, q_persons, q_boxes, _, q_mask = cache.load(query_idx)
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
                **text,
            )
            identity = _coarse_scores_chunked(
                query.identity[0],
                query.logits[0],
                q_mask[0],
                gallery_batches,
                coarse_batch_size,
            )
            state = None
            if need_state:
                query_state = model.encode_text_state(
                    text["change"], text["change_mask"]
                )[0]
                state = torch.cat(
                    [
                        gallery_state[start : start + coarse_batch_size].to(device)
                        @ query_state
                        for start in range(0, len(gallery_ids), coarse_batch_size)
                    ]
                )
            fine_scores = torch.empty(len(gallery_ids), device=device)
            components = {
                key: torch.empty_like(fine_scores)
                for key in ("binding", "context", "coverage")
            }
            fine_done = torch.zeros(len(gallery_ids), dtype=torch.bool, device=device)
            coarse_orders = {}
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
                missing = top_indices[~fine_done[top_indices]]
                for start in range(0, len(missing), fine_batch_size):
                    local = missing[start : start + fine_batch_size]
                    scene, persons, boxes, _, mask = cache.load(
                        image_indices[local.cpu()]
                    )
                    count = len(local)
                    details = model.score_target(
                        query.repeat_candidates(count),
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
                    fine_scores[local] = details["score"]
                    for key in components:
                        components[key][local] = details[key]
                    fine_done[local] = True
                scores = fine_scores
                # Diagnostics use exactly this shortlist and identity assignment.
                # They do not claim an unconstrained full-gallery FAFA baseline.
                fafa_only = top_indices[
                    torch.argsort(
                        components["binding"][top_indices], descending=True, stable=True
                    )
                ]
                output["fafa_rankings"].append(
                    torch.cat((fafa_only, order[len(top_indices) :]))[:ranking_limit]
                    .to(torch.int32)
                    .cpu()
                )
                for field, key in (
                    ("fafa_binding", "binding"),
                    ("context", "context"),
                    ("coverage", "coverage"),
                ):
                    output[field].append(
                        components[key][top_indices][:ranking_limit].cpu()
                    )
                fine_only = top_indices[
                    torch.argsort(scores[top_indices], descending=True, stable=True)
                ]
                output["fine_rankings"].append(
                    torch.cat((fine_only, order[len(top_indices) :]))[:ranking_limit]
                    .to(torch.int32)
                    .cpu()
                )
                rerank_scores = fuse_fine_coarse_scores(
                    scores[top_indices],
                    coarse[top_indices],
                    float(fine_coarse_weight),
                )
                fine_order = top_indices[
                    torch.argsort(rerank_scores, descending=True, stable=True)
                ]
                final_order = torch.cat((fine_order, order[len(top_indices) :]))
                output["rankings"].append(
                    final_order[:ranking_limit].to(torch.int32).cpu()
                )
                output["coarse_rankings"].append(
                    order[:ranking_limit].to(torch.int32).cpu()
                )
                output["coarse_topm"].append(
                    top_indices[:ranking_limit].to(torch.int32).cpu()
                )
        return [
            {
                "sample_ids": [sample["sample_id"] for sample in samples],
                "gallery_ids": gallery_ids,
                **{key: torch.stack(rows) for key, rows in output.items()},
            }
            for output in outputs
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
    text_cache: QueryTextCache | None = None,
):
    """Batch train queries through coarse scoring; never build fine compositions.

    Gallery tensors are copied once per query batch and gallery chunk. Only the
    current batch's full score vectors are retained, preserving per-query zscores
    and canonical tie ordering without a queries-by-gallery output matrix.
    """
    cfg = retrieval_settings(retrieval, model.coarse_beta)
    if not samples or pool_size < 1 or query_batch_size < 1:
        raise ValueError("mining requires samples and positive pool/query batch sizes")
    gallery_ids, gallery_by_id, image_indices = _gallery_scope(
        cache, gallery_ids, samples, "mining"
    )
    by_id = cache.by_id
    excluded = excluded or {}
    forbidden = {
        row["sample_id"]: set(row["positive_image_ids"])
        | {row["query_image_id"]}
        | set(excluded.get(row["sample_id"], ()))
        for row in samples
    }
    limit = pool_size + max(map(len, forbidden.values()))
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    text_cache.prepare(list(samples))
    groups = {}
    for sample in samples:
        groups.setdefault(len(sample["subjects"]), []).append(sample)
    need_state = uses_state(cfg, model.coarse_beta)
    pools = {}
    with evaluating(model, text_encoder):
        gallery_batches = _encode_gallery_identity(
            cache, model, device, cfg["identity_batch_size"], image_indices
        )
        gallery_state = (
            _encode_gallery_state(
                cache, model, device, cfg["identity_batch_size"], image_indices
            )
            if need_state
            else None
        )
        with tqdm(total=len(samples), desc="mine train negatives") as progress:
            for group in groups.values():
                for start in range(0, len(group), query_batch_size):
                    rows = group[start : start + query_batch_size]
                    text = encode_query_text(
                        rows, tokenizer, text_encoder, device, text_cache=text_cache
                    )
                    indices = torch.tensor(
                        [by_id[row["query_image_id"]] for row in rows]
                    )
                    scene, persons, boxes, _, mask = cache.load(indices)
                    mask = mask.to(device)
                    logits, query_identity = model.encode_grounded_identity(
                        scene.to(device),
                        persons.to(device),
                        boxes.to(device),
                        text["selections"],
                        cache.patch_hw,
                        text["selection_mask"],
                    )
                    identity = query_identity.new_empty(len(rows), len(gallery_ids))
                    offset = 0
                    for people_cpu, mask_cpu in gallery_batches:
                        for j in range(0, len(people_cpu), cfg["coarse_batch_size"]):
                            stop = min(j + cfg["coarse_batch_size"], len(people_cpu))
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
                        query_state = model.encode_text_state(
                            text["change"], text["change_mask"]
                        )
                        state = query_state.new_empty(len(rows), len(gallery_ids))
                        for j in range(0, len(gallery_ids), cfg["coarse_batch_size"]):
                            stop = j + cfg["coarse_batch_size"]
                            state[:, j:stop] = (
                                query_state @ gallery_state[j:stop].to(device).T
                            )
                    for n, row in enumerate(rows):
                        query_index = gallery_by_id[row["query_image_id"]]
                        scores = combine_scores(
                            identity[n],
                            None if state is None else state[n],
                            beta=cfg["coarse_beta"],
                            exclude_index=query_index,
                        )
                        order = torch.argsort(scores, descending=True, stable=True)
                        order = order[order != query_index][:limit].cpu().tolist()
                        pools[row["sample_id"]] = [
                            gallery_ids[i]
                            for i in order
                            if gallery_ids[i] not in forbidden[row["sample_id"]]
                        ][:pool_size]
                    progress.update(len(rows))
        return {row["sample_id"]: pools[row["sample_id"]] for row in samples}
