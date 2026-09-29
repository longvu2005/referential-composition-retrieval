"""Build training batches from RCR samples and cached visual features."""

import torch
from torch import Tensor, nn

from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import encode_query_text


def _identity_labels(
    identity_ids: list[list[str | None]], vocab: dict[str, int]
) -> Tensor:
    labels = torch.full((len(identity_ids), len(identity_ids[0])), -1, dtype=torch.long)
    for n, row in enumerate(identity_ids):
        for i, identity_id in enumerate(row):
            if identity_id is not None:
                identity_id = str(identity_id)
                if identity_id not in vocab:
                    vocab[identity_id] = len(vocab)
                labels[n, i] = vocab[identity_id]
    return labels


def build_supervision(
    identity_ids: list[list[str | None]],
    subjects: list[list[dict]],
    identity_vocab: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return grounding targets [B,S,K] and local identity labels [B,K]."""

    b = len(identity_ids)
    s = len(subjects[0])
    k = len(identity_ids[0])

    targets = torch.zeros(b, s, k)
    labels = _identity_labels(
        identity_ids, {} if identity_vocab is None else identity_vocab
    )

    for n in range(b):
        for i, identity_id in enumerate(identity_ids[n]):
            if identity_id is None:
                continue

            for subject, row in enumerate(subjects[n]):
                if str(identity_id) in map(str, row["identity_ids"]):
                    targets[n, subject, i] = 1

    return targets, labels


def build_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """Build one training batch; candidate selection is handled outside."""

    b = len(samples)
    c = len(candidate_image_ids[0])
    by_id = {image_id: i for i, image_id in enumerate(cache.image_ids)}

    query_idx = torch.tensor([by_id[x["query_image_id"]] for x in samples])
    target_idx = torch.tensor(
        [by_id[image_id] for rows in candidate_image_ids for image_id in rows]
    )

    q_scene, q_persons, q_boxes, q_ids, q_mask = cache.load(query_idx)
    t_scene, t_persons, t_boxes, t_ids, t_mask = cache.load(target_idx)

    subjects = [sample["subjects"] for sample in samples]
    identity_vocab: dict[str, int] = {}
    grounding_targets, identity_labels = build_supervision(
        q_ids, subjects, identity_vocab
    )
    target_identity_labels = _identity_labels(t_ids, identity_vocab)

    text = encode_query_text(samples, tokenizer, text_encoder, device)

    positives = [
        set(sample["positive_image_ids"])
        if "positive_image_ids" in sample
        else {sample["target_image_id"]}
        for sample in samples
    ]
    positive_mask = torch.tensor(
        [
            [image_id in positive for image_id in rows]
            for positive, rows in zip(positives, candidate_image_ids, strict=True)
        ],
        dtype=torch.bool,
        device=device,
    )

    # A cached crop is one observation even when its image occurs in multiple
    # queries/candidate lists. Deduplicate observations, never identity classes.
    seen: set[tuple[str, int]] = set()

    def unique_people(image_id: str, valid: Tensor) -> Tensor:
        keep = torch.zeros_like(valid, dtype=torch.bool)
        for person in valid.nonzero(as_tuple=False).flatten().tolist():
            key = (image_id, person)
            if key not in seen:
                keep[person] = True
                seen.add(key)
        return keep

    query_identity_mask = torch.stack(
        [
            unique_people(
                sample["query_image_id"], q_mask[n] & (identity_labels[n] >= 0)
            )
            for n, sample in enumerate(samples)
        ]
    )
    target_identity_mask = torch.zeros_like(t_mask, dtype=torch.bool)
    for n, rows in enumerate(candidate_image_ids):
        for j, image_id in enumerate(rows):
            if image_id in positives[n]:
                index = n * c + j
                target_identity_mask[index] = unique_people(
                    image_id, t_mask[index] & (target_identity_labels[index] >= 0)
                )

    k_t = t_persons.shape[1]
    return {
        "query_scene": q_scene.to(device),
        "query_persons": q_persons.to(device),
        "query_boxes": q_boxes.to(device),
        "query_person_mask": q_mask.to(device),
        "query_identity_labels": identity_labels.to(device),
        "query_identity_mask": query_identity_mask.to(device),
        "grounding_targets": grounding_targets.to(device),
        **text,
        "target_scene": t_scene.to(device).reshape(b, c, *t_scene.shape[1:]),
        "target_persons": t_persons.to(device).reshape(b, c, k_t, t_persons.shape[-1]),
        "target_boxes": t_boxes.to(device).reshape(b, c, k_t, 4),
        "target_mask": t_mask.to(device).reshape(b, c, k_t),
        "target_identity_labels": target_identity_labels.to(device).reshape(b, c, k_t),
        "target_identity_mask": target_identity_mask.to(device).reshape(b, c, k_t),
        "positive_mask": positive_mask,
    }
