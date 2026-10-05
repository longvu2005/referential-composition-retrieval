"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.coarse import coarse_scores, combine_scores
from rcr.methods.proposed.encoders import encode_query_text


def _encode_gallery_identity(
    cache: GalleryCache,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_indices: Tensor | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Project once, retaining only CPU batches with useful person columns.

    The cache pads every image to the gallery's maximum person count. Dropping
    columns masked out for an entire batch saves RAM without removing evidence.
    Never concatenate these batches on the accelerator (or duplicate them in RAM).
    """

    if image_indices is None:
        image_indices = torch.arange(len(cache.image_ids))
    batches = []
    dtype = model.identity_head.proj.weight.dtype
    for start in range(0, len(image_indices), batch_size):
        indices = image_indices[start : start + batch_size]
        mask = cache.mask[indices]
        columns = mask.any(dim=0)
        mask = mask[:, columns]
        persons = cache.persons[indices[:, None], columns.nonzero().flatten()].to(
            device=device, dtype=dtype
        )
        identity = model.identity_head(persons)
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
    *,
    query_state: Tensor | None = None,
    gallery_state: Tensor | None = None,
    beta: float = 0.0,
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
            scores[offset + start : offset + stop] = coarse_scores(
                query_identity,
                logits,
                identity,
                mask,
                query_mask=query_mask,
                query_state=query_state,
                gallery_state=(
                    gallery_state[offset + start : offset + stop].to(
                        query_identity.device
                    )
                    if beta != 0 and gallery_state is not None
                    else None
                ),
                beta=beta,
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

    ``weight=0`` returns the raw fine scores exactly, preserving the historical
    reranker. For positive weights, both branches are population-z-scored over
    the same candidates with finite coarse support, then combined as
    ``z(fine) + weight * z(coarse)``. Non-finite coarse candidates stay last
    instead of being rescued by a fine score despite having no coarse support.
    """
    if fine.ndim != 1 or coarse.ndim != 1 or fine.shape != coarse.shape:
        raise ValueError("fine and coarse scores must be matching 1D tensors")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("fine_coarse_weight must be finite and nonnegative")
    if weight == 0:
        return fine

    valid = torch.isfinite(coarse)
    if not torch.isfinite(fine[valid]).all():
        raise ValueError("fine scores must be finite on coarse-supported candidates")

    fused = torch.full_like(fine, -torch.inf, dtype=torch.float32)
    if valid.any():
        fused[valid] = _zscore_1d(fine[valid], eps) + weight * _zscore_1d(
            coarse[valid], eps
        )
    return fused


def retrieval_settings(cfg: dict, default_beta: float) -> dict:
    """Resolve runtime overrides while preserving old checkpoint behavior."""
    settings = {
        "coarse_batch_size": 512,
        "coarse_mode": "identity_state",
        "coarse_beta": default_beta,
        "coarse_normalization": "none",
        "fine_coarse_weight": 0.0,
        "rerank": True,
        **cfg,
    }
    if settings["coarse_beta"] is None:
        settings["coarse_beta"] = default_beta
    return settings


def uses_state(cfg: dict, default_beta: float) -> bool:
    settings = retrieval_settings(cfg, default_beta)
    return settings["coarse_mode"] == "state_only" or (
        settings["coarse_mode"] == "identity_state" and settings["coarse_beta"] != 0
    )


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
    coarse_mode: str = "identity_state",
    coarse_beta: float | None = None,
    coarse_normalization: str = "none",
    fine_coarse_weight: float = 0.0,
    rerank: bool = True,
    description: str = "retrieve",
) -> dict[str, Any]:
    """Single-run interface shared by training validation and standalone use."""
    settings = dict(
        top_m=top_m,
        fine_batch_size=fine_batch_size,
        identity_batch_size=identity_batch_size,
        coarse_batch_size=coarse_batch_size,
        coarse_mode=coarse_mode,
        coarse_beta=coarse_beta,
        coarse_normalization=coarse_normalization,
        fine_coarse_weight=fine_coarse_weight,
        rerank=rerank,
    )
    return retrieve_variants(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=gallery_ids,
        variants={"run": settings},
        description=description,
    )["run"]


@torch.inference_mode()
def retrieve_variants(
    samples: Sequence[dict],
    cache: GalleryCache,
    tokenizer: Any,
    text_encoder: nn.Module,
    model: nn.Module,
    device: torch.device,
    *,
    gallery_ids: Sequence[str],
    variants: Mapping[str, dict],
    description: str = "retrieve",
    ranking_limit: int | None = None,
) -> dict[str, dict]:
    """Reuse encodings and raw ID/state/fine scores across inference ablations.

    Only the split gallery is projected. CPU gallery projections and GPU scoring
    chunks bound accelerator memory. Coarse normalization happens after gathering
    one complete score vector, never per chunk. Fine/coarse final fusion is
    normalized only inside each variant's Top-M shortlist. All outputs retain the
    full gallery order; top_m only limits fine reranking. Training mining may
    retain only ranking_limit entries to bound CPU output memory. Such outputs
    are never accepted by the official full-rank evaluator.
    """
    settings = {
        name: retrieval_settings(cfg, model.coarse_beta)
        for name, cfg in variants.items()
    }
    if not samples or not settings:
        raise ValueError("retrieval requires samples and at least one variant")
    if ranking_limit is not None and ranking_limit < 1:
        raise ValueError("ranking_limit must be positive")
    first = next(iter(settings.values()))
    batch_keys = ("fine_batch_size", "identity_batch_size", "coarse_batch_size")
    for cfg in settings.values():
        if min(cfg[key] for key in (*batch_keys, "top_m")) < 1:
            raise ValueError("retrieval batch sizes and top_m must be positive")
        if any(cfg[key] != first[key] for key in batch_keys):
            raise ValueError("shared retrieval variants require the same batch sizes")
        weight = cfg["fine_coarse_weight"]
        if (
            not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("fine_coarse_weight must be finite and nonnegative")
    fine_batch_size, identity_batch_size, coarse_batch_size = (
        first[key] for key in batch_keys
    )
    gallery_ids = list(gallery_ids)
    if not gallery_ids or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("retrieval gallery_ids must be non-empty and unique")
    by_id = {image_id: index for index, image_id in enumerate(cache.image_ids)}
    if not set(gallery_ids) <= set(by_id):
        raise ValueError("retrieval gallery contains images missing from the cache")
    gallery_by_id = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if any(sample["query_image_id"] not in gallery_by_id for sample in samples):
        raise ValueError("query image must belong to the retrieval gallery")
    image_indices = torch.tensor([by_id[image_id] for image_id in gallery_ids])
    need_identity = any(c["coarse_mode"] != "state_only" for c in settings.values())
    need_state = any(
        c["coarse_mode"] == "state_only"
        or (c["coarse_mode"] == "identity_state" and c["coarse_beta"] != 0)
        for c in settings.values()
    )
    need_fine = any(c["rerank"] for c in settings.values())
    model_was_training, text_was_training = model.training, text_encoder.training
    model.eval()
    text_encoder.eval()
    try:
        gallery_batches = (
            _encode_gallery_identity(
                cache, model, device, identity_batch_size, image_indices
            )
            if need_identity
            else None
        )
        gallery_state = (
            _encode_gallery_state(
                cache, model, device, identity_batch_size, image_indices
            )
            if need_state
            else None
        )
        outputs = {
            name: {"rankings": [], "coarse_rankings": [], "coarse_topm": []}
            for name in settings
        }
        for sample in tqdm(samples, desc=description):
            query_index = gallery_by_id[sample["query_image_id"]]
            text = encode_query_text([sample], tokenizer, text_encoder, device)
            if need_identity or need_fine:
                query_idx = torch.tensor([by_id[sample["query_image_id"]]])
                q_scene, q_persons, q_boxes, _, q_mask = cache.load(query_idx)
                q_mask = q_mask.to(device)
                logits, query_identity, query, query_mask, prior = model.encode_query(
                    q_scene.to(device),
                    q_persons.to(device),
                    q_boxes.to(device),
                    patch_hw=cache.patch_hw,
                    query_person_mask=q_mask,
                    **text,
                )
            identity = (
                _coarse_scores_chunked(
                    query_identity[0],
                    logits[0],
                    q_mask[0],
                    gallery_batches,
                    coarse_batch_size,
                )
                if need_identity
                else None
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
            # Fine scores depend on the query/target pair, not on beta, Top-M,
            # or the final fine/coarse fusion weight, so variants can share them.
            fine_scores = torch.empty(len(gallery_ids), device=device)
            fine_done = torch.zeros(len(gallery_ids), dtype=torch.bool, device=device)
            for name, cfg in settings.items():
                coarse = combine_scores(
                    identity,
                    state,
                    mode=cfg["coarse_mode"],
                    beta=cfg["coarse_beta"],
                    normalization=cfg["coarse_normalization"],
                    exclude_index=query_index,
                )
                order = torch.argsort(coarse, descending=True, stable=True)
                order = order[order != query_index]
                top_indices = order[: cfg["top_m"]]
                final_order = order
                if cfg["rerank"]:
                    missing = top_indices[~fine_done[top_indices]]
                    for start in range(0, len(missing), fine_batch_size):
                        local = missing[start : start + fine_batch_size]
                        scene, persons, boxes, _, mask = cache.load(
                            image_indices[local.cpu()]
                        )
                        count = len(local)
                        fine_scores[local] = model.score_target(
                            query.expand(count, -1, -1),
                            query_mask.expand(count, -1),
                            prior.expand(count, -1),
                            scene.to(device),
                            persons.to(device),
                            boxes.to(device),
                            cache.patch_hw,
                            mask.to(device),
                        )
                        fine_done[local] = True
                    rerank_scores = fuse_fine_coarse_scores(
                        fine_scores[top_indices],
                        coarse[top_indices],
                        float(cfg["fine_coarse_weight"]),
                    )
                    fine_order = top_indices[
                        torch.argsort(rerank_scores, descending=True, stable=True)
                    ]
                    final_order = torch.cat((fine_order, order[len(top_indices) :]))
                outputs[name]["rankings"].append(
                    final_order[:ranking_limit].to(torch.int32).cpu()
                )
                outputs[name]["coarse_rankings"].append(
                    order[:ranking_limit].to(torch.int32).cpu()
                )
                outputs[name]["coarse_topm"].append(
                    top_indices[:ranking_limit].to(torch.int32).cpu()
                )
        return {
            name: {
                "sample_ids": [sample["sample_id"] for sample in samples],
                "gallery_ids": gallery_ids,
                **{key: torch.stack(rows) for key, rows in output.items()},
            }
            for name, output in outputs.items()
        }
    finally:
        model.train(model_was_training)
        text_encoder.train(text_was_training)


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
):
    """Coarse-only train mining, without retaining full-gallery ranking matrices.

    Caller supplies TRAIN queries/gallery exclusively. No labels enter scoring;
    train labels only remove positives and disputed negatives from the pool.
    """
    excluded = excluded or {}
    forbidden = {
        sample["sample_id"]: set(sample["positive_image_ids"])
        | {sample["query_image_id"]}
        | set(excluded.get(sample["sample_id"], ()))
        for sample in samples
    }
    limit = pool_size + max(map(len, forbidden.values()))
    result = retrieve_variants(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        gallery_ids=gallery_ids,
        variants={"mining": {**retrieval, "rerank": False, "top_m": limit}},
        ranking_limit=limit,
        description="mine train negatives",
    )["mining"]
    return {
        sample_id: [
            gallery_ids[i]
            for i in row.tolist()
            if gallery_ids[i] not in forbidden[sample_id]
        ][:pool_size]
        for sample_id, row in zip(result["sample_ids"], result["rankings"], strict=True)
    }
