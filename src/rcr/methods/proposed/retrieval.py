"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.evaluation.evaluate import evaluate_rankings
from rcr.methods.common.data import RCRData
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.coarse import coarse_scores
from rcr.methods.proposed.encoders import encode_query_text


def _encode_gallery_identity(
    cache: GalleryCache,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Project once, retaining only CPU batches with useful person columns.

    The cache pads every image to the gallery's maximum person count. Dropping
    columns masked out for an entire batch saves RAM without removing evidence.
    Never concatenate these batches on the accelerator (or duplicate them in RAM).
    """

    batches = []
    dtype = model.identity_head.proj.weight.dtype
    for start in range(0, len(cache.image_ids), batch_size):
        stop = start + batch_size
        mask = cache.mask[start:stop]
        columns = mask.any(dim=0)
        mask = mask[:, columns]
        persons = cache.persons[start:stop, columns].to(device=device, dtype=dtype)
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
    cache: GalleryCache, model: nn.Module, device: torch.device, batch_size: int
) -> Tensor:
    """Project global image features once per retrieval call, retaining CPU output."""
    features = cache.global_features
    projection = model.state_image_proj.weight
    state = torch.empty(
        len(cache.image_ids), projection.shape[0], dtype=projection.dtype
    )
    for start in range(0, len(cache.image_ids), batch_size):
        stop = min(start + batch_size, len(cache.image_ids))
        chunk = features[start:stop].to(device=device, dtype=projection.dtype)
        state[start:stop] = model.encode_image_state(chunk).cpu()
    return state


@torch.inference_mode()
def retrieve_rankings(
    samples: Sequence[dict],
    cache: GalleryCache,
    tokenizer: Any,
    text_encoder: nn.Module,
    model: nn.Module,
    device: torch.device,
    *,
    top_m: int,
    fine_batch_size: int,
    identity_batch_size: int,
    coarse_batch_size: int = 512,
    description: str = "retrieve",
) -> dict[str, Any]:
    """Return full and coarse gallery rankings for an ordered sample sequence.

    Gallery identities stay in CPU RAM; only projection/coarse-scoring chunks
    and Top-M fine batches use the accelerator. Scores still cover every image.
    Module train/eval modes are restored even if retrieval fails.
    """

    if not samples:
        raise ValueError("retrieval requires at least one sample")
    if min(top_m, fine_batch_size, identity_batch_size, coarse_batch_size) < 1:
        raise ValueError("retrieval batch sizes and top_m must be positive")
    if not cache.image_ids:
        raise ValueError("retrieval requires a non-empty gallery")

    model_was_training = model.training
    text_was_training = text_encoder.training
    model.eval()
    text_encoder.eval()

    try:
        gallery_batches = _encode_gallery_identity(
            cache, model, device, identity_batch_size
        )
        gallery_state = (
            _encode_gallery_state(cache, model, device, identity_batch_size)
            if model.coarse_beta != 0
            else None
        )

        by_id = {image_id: index for index, image_id in enumerate(cache.image_ids)}
        sample_ids = []
        rankings = []
        coarse_topm = []

        for sample in tqdm(samples, desc=description):
            query_index = by_id[sample["query_image_id"]]
            query_idx = torch.tensor([query_index])
            q_scene, q_persons, q_boxes, _, q_mask = cache.load(query_idx)
            q_scene = q_scene.to(device)
            q_persons = q_persons.to(device)
            q_boxes = q_boxes.to(device)
            q_mask = q_mask.to(device)

            text = encode_query_text([sample], tokenizer, text_encoder, device)
            logits, query_identity, query, query_mask, prior = model.encode_query(
                q_scene,
                q_persons,
                q_boxes,
                patch_hw=cache.patch_hw,
                query_person_mask=q_mask,
                **text,
            )

            coarse = _coarse_scores_chunked(
                query_identity[0],
                logits[0],
                q_mask[0],
                gallery_batches,
                coarse_batch_size,
                query_state=(
                    model.encode_text_state(text["change"], text["change_mask"])[0]
                    if model.coarse_beta != 0
                    else None
                ),
                gallery_state=gallery_state,
                beta=model.coarse_beta,
            )
            coarse[query_index] = -torch.inf
            coarse_order = torch.argsort(coarse, descending=True)
            coarse_order = coarse_order[coarse_order != query_index]
            top_indices = coarse_order[: min(top_m, len(coarse_order))]

            fine_scores = []
            for start in range(0, len(top_indices), fine_batch_size):
                indices = top_indices[start : start + fine_batch_size].cpu()
                scene, persons, boxes, _, target_mask = cache.load(indices)
                scene = scene.to(device)
                persons = persons.to(device)
                boxes = boxes.to(device)
                target_mask = target_mask.to(device)

                count = len(indices)
                score = model.score_target(
                    query.expand(count, -1, -1),
                    query_mask.expand(count, -1),
                    prior.expand(count, -1),
                    scene,
                    persons,
                    boxes,
                    cache.patch_hw,
                    target_mask,
                )
                fine_scores.append(score)

            scores = torch.cat(fine_scores) if fine_scores else coarse.new_empty(0)
            fine_order = top_indices[torch.argsort(scores, descending=True)]
            final_order = torch.cat((fine_order, coarse_order[len(top_indices) :]))

            sample_ids.append(sample["sample_id"])
            rankings.append(final_order.to(torch.int32).cpu())
            coarse_topm.append(top_indices.to(torch.int32).cpu())

        return {
            "sample_ids": sample_ids,
            "gallery_ids": cache.image_ids,
            "rankings": torch.stack(rankings),
            "coarse_topm": torch.stack(coarse_topm),
        }
    finally:
        model.train(model_was_training)
        text_encoder.train(text_was_training)


def slice_retrieval_output(
    output: dict[str, Any], start: int, stop: int
) -> dict[str, Any]:
    """Slice query rows while preserving the shared gallery index."""

    return {
        "sample_ids": output["sample_ids"][start:stop],
        "gallery_ids": output["gallery_ids"],
        "rankings": output["rankings"][start:stop],
        "coarse_topm": output["coarse_topm"][start:stop],
    }


def evaluate_retrieval_output(
    data: RCRData,
    samples: Sequence[dict],
    output: dict[str, Any],
    candidate_ks: Sequence[int],
) -> dict:
    """Evaluate index-based retrieval output with the official RCR metrics."""

    gallery_ids = output["gallery_ids"]
    sample_ids = output["sample_ids"]
    expected_sample_ids = [sample["sample_id"] for sample in samples]
    if gallery_ids != data.gallery_ids:
        raise ValueError("saved gallery_ids do not match the current benchmark gallery")
    if sample_ids != expected_sample_ids:
        raise ValueError("saved sample_ids do not match the requested samples")

    def decode(rows: Tensor, label: str) -> dict[str, list[str]]:
        decoded = {}
        for sample_id, row in zip(sample_ids, rows, strict=True):
            indices = row.tolist()
            if any(index < 0 or index >= len(gallery_ids) for index in indices):
                raise ValueError(f"{label} contains an invalid gallery index")
            decoded[sample_id] = [gallery_ids[index] for index in indices]
        return decoded

    rankings = decode(output["rankings"], "rankings")
    coarse = decode(output["coarse_topm"], "coarse_topm")
    identities_by_image = {
        image_id: {
            box["identity_id"] for box in data.gt_head_boxes_by_image.get(image_id, [])
        }
        for image_id in gallery_ids
    }
    return evaluate_rankings(
        samples,
        gallery_ids,
        identities_by_image,
        rankings,
        coarse_rankings=coarse,
        candidate_ks=tuple(candidate_ks),
    )
