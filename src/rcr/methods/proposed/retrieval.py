"""Retrieval helpers shared by training-time evaluation and the CLI."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from rcr.evaluation.evaluate import evaluate_rankings
from rcr.methods.common.data import RCRData, sample_selection_texts
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.coarse import coarse_scores
from rcr.methods.proposed.composition import composed_query_mask, reference_key_bias


def _change_text(sample: dict) -> str:
    """Replace canonical Subject references with tokenizer marker tokens."""

    text = sample["final_change"]
    for subject in sample["subjects"]:
        subject_id = int(subject["subject_id"])
        text = re.sub(
            rf"\bSubject\s+{subject_id}\b",
            f"[S{subject_id}]",
            text,
        )
    return text


def _subject_positions(
    sample: dict,
    change_ids: Tensor,
    tokenizer: Any,
) -> tuple[Tensor, Tensor]:
    """Return first marker positions and all marker-token locations."""

    subjects = sample["subjects"]
    positions = torch.empty(
        1,
        len(subjects),
        dtype=torch.long,
        device=change_ids.device,
    )
    token_mask = torch.zeros(
        1,
        len(subjects),
        change_ids.shape[1],
        dtype=torch.bool,
        device=change_ids.device,
    )
    for index, subject in enumerate(subjects):
        marker = f"[S{int(subject['subject_id'])}]"
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        locations = (change_ids[0] == marker_id).nonzero(as_tuple=False).flatten()
        if locations.numel() == 0:
            raise ValueError(f"{marker} must be one tokenizer token and occur")
        positions[0, index] = locations[0]
        token_mask[0, index, locations] = True
    return positions, token_mask


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

            selection_inputs = tokenizer(
                sample_selection_texts(sample),
                padding=True,
                return_tensors="pt",
            )
            selection, selection_mask = text_encoder(
                selection_inputs["input_ids"].to(device),
                selection_inputs["attention_mask"].to(device),
            )
            selection = selection.unsqueeze(0)
            selection_mask = selection_mask.unsqueeze(0)

            change_inputs = tokenizer(_change_text(sample), return_tensors="pt")
            change_ids = change_inputs["input_ids"].to(device)
            change, change_mask = text_encoder(
                change_ids,
                change_inputs["attention_mask"].to(device),
            )
            subject_pos, subject_token_mask = _subject_positions(
                sample,
                change_ids,
                tokenizer,
            )

            logits = model.grounding(
                q_scene,
                q_persons,
                q_boxes,
                selection,
                cache.patch_hw,
                selection_mask,
            )
            query_identity = model.identity_head(q_persons)
            composition_logits = logits.masked_fill(~q_mask[:, None], -torch.inf)

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

            subject_ids = torch.tensor(
                [[int(subject["subject_id"]) for subject in sample["subjects"]]],
                device=device,
            )
            query = model.composition(
                change,
                query_identity,
                composition_logits,
                subject_pos,
                change_mask,
                subject_token_mask=subject_token_mask,
                subject_ids=subject_ids,
            )
            query_mask = composed_query_mask(change, composition_logits, change_mask)
            prior = reference_key_bias(change, composition_logits, change_mask)

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
