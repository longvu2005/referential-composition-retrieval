"""Create one training batch from samples and frozen cache features."""

import torch
from torch import Tensor, nn

from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import QueryTextCache, encode_tokenized_text


def _identity_labels(rows, vocabulary: dict[str, int]) -> Tensor:
    labels = torch.full((len(rows), len(rows[0])), -1, dtype=torch.long)
    for n, identities in enumerate(rows):
        for person, identity in enumerate(identities):
            if identity is None:
                continue
            identity = str(identity)
            labels[n, person] = vocabulary.setdefault(identity, len(vocabulary))
    return labels


def build_supervision(identity_rows, subjects, vocabulary=None):
    """Build Subject membership targets and local identity-class labels."""
    vocabulary = {} if vocabulary is None else vocabulary
    labels = _identity_labels(identity_rows, vocabulary)
    targets = torch.zeros(len(identity_rows), len(subjects[0]), len(identity_rows[0]))
    for n, (identities, subject_rows) in enumerate(
        zip(identity_rows, subjects, strict=True)
    ):
        required = [set(map(str, subject["identity_ids"])) for subject in subject_rows]
        for person, identity in enumerate(identities):
            if identity is None:
                continue
            for subject, identity_set in enumerate(required):
                targets[n, subject, person] = str(identity) in identity_set
    return targets, labels


def _deduplicated_masks(
    samples,
    candidates,
    positives,
    query_labels,
    target_labels,
    query_mask,
    target_mask,
):
    """Use each annotated crop once in the supervised contrastive loss."""
    seen = set()

    def unseen(image_id, valid):
        keep = torch.zeros_like(valid)
        for person in valid.nonzero(as_tuple=False).flatten().tolist():
            key = (image_id, person)
            if key not in seen:
                seen.add(key)
                keep[person] = True
        return keep

    query_keep = torch.stack(
        [
            unseen(row["query_image_id"], query_mask[n] & (query_labels[n] >= 0))
            for n, row in enumerate(samples)
        ]
    )
    target_keep = torch.zeros_like(target_mask)
    width = len(candidates[0])
    for n, image_ids in enumerate(candidates):
        for j, image_id in enumerate(image_ids):
            if image_id in positives[n]:
                i = n * width + j
                target_keep[i] = unseen(
                    image_id, target_mask[i] & (target_labels[i] >= 0)
                )
    return query_keep, target_keep, seen


def prepare_visual_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    *,
    supervision,
    state_image_ids: list[set[str]] | None = None,
    identity_crops=(),
) -> dict[str, Tensor]:
    """Load CPU visual features and labels. No model is executed here."""
    batch_size = len(samples)
    candidates_per_query = len(candidate_image_ids[0])
    query_indices = torch.tensor(
        [cache.by_id[row["query_image_id"]] for row in samples]
    )
    target_indices = torch.tensor(
        [cache.by_id[image_id] for row in candidate_image_ids for image_id in row]
    )
    query, target = cache.load_groups(query_indices, target_indices)
    q_scene, q_people, q_boxes, q_mask = query
    t_scene, t_people, t_boxes, t_mask = target

    positives = [set(row["positive_image_ids"]) for row in samples]
    q_ids = supervision.load_ids(
        [row["query_image_id"] for row in samples], q_people.shape[1]
    )
    t_ids = [[None] * t_people.shape[1] for _ in range(len(target_indices))]
    positive_slots = [
        (n * candidates_per_query + j, image_id)
        for n, image_ids in enumerate(candidate_image_ids)
        for j, image_id in enumerate(image_ids)
        if image_id in positives[n]
    ]
    if positive_slots:
        rows = supervision.load_ids(
            [image_id for _, image_id in positive_slots], t_people.shape[1]
        )
        for (index, _), identities in zip(positive_slots, rows, strict=True):
            t_ids[index] = identities

    vocabulary: dict[str, int] = {}
    grounding_targets, query_labels = build_supervision(
        q_ids, [row["subjects"] for row in samples], vocabulary
    )
    target_labels = _identity_labels(t_ids, vocabulary)
    positive_mask = torch.tensor(
        [
            [image_id in positive for image_id in image_ids]
            for positive, image_ids in zip(positives, candidate_image_ids, strict=True)
        ],
        dtype=torch.bool,
    )
    query_identity_mask, target_identity_mask, seen = _deduplicated_masks(
        samples,
        candidate_image_ids,
        positives,
        query_labels,
        target_labels,
        q_mask,
        t_mask,
    )

    extra = []
    for image_id, person, identity in identity_crops:
        if (image_id, person) not in seen:
            seen.add((image_id, person))
            extra.append((image_id, person, identity))
    extra_batch = {}
    if extra:
        people_by_image = {
            image_id: cache.person_row(cache.by_id[image_id])
            for image_id in dict.fromkeys(image_id for image_id, _, _ in extra)
        }
        extra_batch = {
            "extra_identity_persons": torch.stack(
                [people_by_image[image_id][person] for image_id, person, _ in extra]
            ).float(),
            "extra_identity_labels": _identity_labels(
                [[identity for _, _, identity in extra]], vocabulary
            )[0],
        }

    k_target = t_people.shape[1]
    batch = {
        **extra_batch,
        "query_scene": q_scene,
        "query_persons": q_people,
        "query_boxes": q_boxes,
        "query_person_mask": q_mask,
        "fafa_query": cache.native_queries(samples, q_people.shape[1]),
        "query_identity_labels": query_labels,
        "query_identity_mask": query_identity_mask,
        "grounding_targets": grounding_targets,
        "target_scene": t_scene.unflatten(0, (batch_size, candidates_per_query)),
        "target_persons": t_people.unflatten(0, (batch_size, candidates_per_query)),
        "target_boxes": t_boxes.unflatten(0, (batch_size, candidates_per_query)),
        "target_mask": t_mask.unflatten(0, (batch_size, candidates_per_query)),
        "fafa_targets": cache.native_targets(target_indices, k_target).unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "target_identity_labels": target_labels.unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "target_identity_mask": target_identity_mask.unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "positive_mask": positive_mask,
    }
    if state_image_ids is not None:
        batch["state_mask"] = torch.tensor(
            [
                [image_id in eligible for image_id in image_ids]
                for image_ids, eligible in zip(
                    candidate_image_ids, state_image_ids, strict=True
                )
            ],
            dtype=torch.bool,
        )
        if (positive_mask & ~batch["state_mask"]).any():
            raise ValueError("every Full Positive must contain all required IDs")
    return batch


def build_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
    *,
    supervision,
    state_image_ids: list[set[str]] | None = None,
    identity_crops=(),
    text_cache: QueryTextCache | None = None,
) -> dict[str, Tensor]:
    text_cache = text_cache or QueryTextCache(tokenizer)
    visual = prepare_visual_batch(
        samples,
        candidate_image_ids,
        cache,
        supervision=supervision,
        state_image_ids=state_image_ids,
        identity_crops=identity_crops,
    )
    visual = {key: value.to(device) for key, value in visual.items()}
    text = encode_tokenized_text(text_cache.batch(samples), text_encoder, device)
    return {**visual, **text}
