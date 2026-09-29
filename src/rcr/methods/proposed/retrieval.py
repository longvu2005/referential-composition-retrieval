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
    description: str = "retrieve",
) -> dict[str, Any]:
    """Return full and coarse gallery rankings for an ordered sample sequence.

    Module train/eval modes are restored before returning, so this function is
    safe to call between training epochs.
    """

    if not samples:
        raise ValueError("retrieval requires at least one sample")
    if top_m < 1 or fine_batch_size < 1 or identity_batch_size < 1:
        raise ValueError("retrieval batch sizes and top_m must be positive")

    model_was_training = model.training
    text_was_training = text_encoder.training
    model.eval()
    text_encoder.eval()

    try:
        gallery_identity = []
        identity_dtype = model.identity_head.proj.weight.dtype
        for start in range(0, len(cache.image_ids), identity_batch_size):
            persons = cache.persons[start : start + identity_batch_size].to(
                device=device,
                dtype=identity_dtype,
            )
            gallery_identity.append(model.identity_head(persons))
        gallery_identity = torch.cat(gallery_identity)
        gallery_mask = cache.mask.to(device)

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

            coarse = coarse_scores(
                query_identity[0],
                logits[0],
                gallery_identity,
                gallery_mask,
                query_mask=q_mask[0],
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
