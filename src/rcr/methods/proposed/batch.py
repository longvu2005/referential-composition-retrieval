"""Build training batches from RCR samples and cached visual features."""

import re

import torch
from torch import Tensor, nn

from rcr.methods.common.data import sample_selection_texts
from rcr.methods.proposed.cache import GalleryCache

SUBJECT_MARKERS = {1: "[S1]", 2: "[S2]"}


def build_supervision(
    identity_ids: list[list[str | None]],
    subjects: list[list[dict]],
) -> tuple[Tensor, Tensor]:
    """Return grounding targets [B,S,K] and local identity labels [B,K]."""

    b = len(identity_ids)
    s = len(subjects[0])
    k = len(identity_ids[0])

    targets = torch.zeros(b, s, k)
    labels = torch.full((b, k), -1, dtype=torch.long)

    vocab: dict[str, int] = {}
    for n in range(b):
        for i, identity_id in enumerate(identity_ids[n]):
            if identity_id is None:
                continue

            if identity_id not in vocab:
                vocab[identity_id] = len(vocab)
            labels[n, i] = vocab[identity_id]

            for subject, row in enumerate(subjects[n]):
                if identity_id in map(str, row["identity_ids"]):
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
    s = len(samples[0]["subjects"])
    c = len(candidate_image_ids[0])
    by_id = {image_id: i for i, image_id in enumerate(cache.image_ids)}

    query_idx = torch.tensor([by_id[x["query_image_id"]] for x in samples])
    target_idx = torch.tensor(
        [by_id[image_id] for rows in candidate_image_ids for image_id in rows]
    )

    q_scene, q_persons, q_boxes, q_ids, q_mask = cache.load(query_idx)
    t_scene, t_persons, t_boxes, _, t_mask = cache.load(target_idx)

    subjects = [sample["subjects"] for sample in samples]
    grounding_targets, identity_labels = build_supervision(q_ids, subjects)

    selection_texts = [
        text for sample in samples for text in sample_selection_texts(sample)
    ]
    selection = tokenizer(selection_texts, padding=True, return_tensors="pt")
    selection_tokens, selection_mask = text_encoder(
        selection["input_ids"].to(device),
        selection["attention_mask"].to(device),
    )
    selection_tokens = selection_tokens.reshape(b, s, *selection_tokens.shape[1:])
    selection_mask = selection_mask.reshape(b, s, -1)

    changes = []
    for sample in samples:
        text = sample["final_change"]
        for subject in sample["subjects"]:
            subject_id = int(subject["subject_id"])
            marker = SUBJECT_MARKERS[subject_id]
            text = re.sub(rf"\bSubject\s+{subject_id}\b", marker, text)
        changes.append(text)

    encoded_change = tokenizer(changes, padding=True, return_tensors="pt")
    change_ids = encoded_change["input_ids"].to(device)
    change, change_mask = text_encoder(
        change_ids,
        encoded_change["attention_mask"].to(device),
    )

    subject_pos = torch.empty(b, s, dtype=torch.long, device=device)
    subject_token_mask = torch.zeros(
        b, s, change_ids.shape[1], dtype=torch.bool, device=device
    )
    for n, sample in enumerate(samples):
        for j, subject in enumerate(sample["subjects"]):
            marker = SUBJECT_MARKERS[int(subject["subject_id"])]
            marker_id = tokenizer.convert_tokens_to_ids(marker)
            pos = (change_ids[n] == marker_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                raise ValueError(f"{marker} must be one tokenizer token and occur")
            subject_pos[n, j] = pos[0]
            subject_token_mask[n, j, pos] = True

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

    def move(x: Tensor) -> Tensor:
        return x.to(device)

    k_t = t_persons.shape[1]
    return {
        "query_scene": move(q_scene),
        "query_persons": move(q_persons),
        "query_boxes": move(q_boxes),
        "query_person_mask": move(q_mask),
        "query_identity_labels": move(identity_labels),
        "selections": selection_tokens,
        "selection_mask": selection_mask,
        "grounding_targets": move(grounding_targets),
        "change": change,
        "change_mask": change_mask,
        "subject_pos": subject_pos,
        "subject_ids": torch.tensor(
            [[int(x["subject_id"]) for x in row] for row in subjects], device=device
        ),
        "subject_token_mask": subject_token_mask,
        "target_scene": move(t_scene).reshape(b, c, *t_scene.shape[1:]),
        "target_persons": move(t_persons).reshape(b, c, k_t, t_persons.shape[-1]),
        "target_boxes": move(t_boxes).reshape(b, c, k_t, 4),
        "target_mask": move(t_mask).reshape(b, c, k_t),
        "positive_mask": positive_mask,
    }
