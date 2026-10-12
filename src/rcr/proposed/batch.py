"""Build training batches from RCR samples and cached visual features."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import QueryTextCache, encode_tokenized_text


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


def prepare_visual_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    *,
    state_image_ids: list[set[str]] | None = None,
) -> dict[str, Tensor]:
    """Read and collate CPU features/supervision without running a model."""

    b = len(samples)
    c = len(candidate_image_ids[0])
    by_id = cache.by_id

    query_idx = torch.tensor([by_id[x["query_image_id"]] for x in samples])
    target_idx = torch.tensor(
        [by_id[image_id] for rows in candidate_image_ids for image_id in rows]
    )

    query, target = cache.load_groups(query_idx, target_idx)
    q_scene, q_persons, q_boxes, q_ids, q_mask = query
    t_scene, t_persons, t_boxes, t_ids, t_mask = target

    subjects = [sample["subjects"] for sample in samples]
    identity_vocab: dict[str, int] = {}
    grounding_targets, identity_labels = build_supervision(
        q_ids, subjects, identity_vocab
    )
    target_identity_labels = _identity_labels(t_ids, identity_vocab)

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
    batch = {
        "query_scene": q_scene,
        "query_persons": q_persons,
        "query_boxes": q_boxes,
        "query_person_mask": q_mask,
        "query_identity_labels": identity_labels,
        "query_identity_mask": query_identity_mask,
        "grounding_targets": grounding_targets,
        "target_scene": t_scene.reshape(b, c, *t_scene.shape[1:]),
        "target_persons": t_persons.reshape(b, c, k_t, t_persons.shape[-1]),
        "target_boxes": t_boxes.reshape(b, c, k_t, 4),
        "target_mask": t_mask.reshape(b, c, k_t),
        "target_identity_labels": target_identity_labels.reshape(b, c, k_t),
        "target_identity_mask": target_identity_mask.reshape(b, c, k_t),
        "positive_mask": positive_mask,
        "grounding_complete": torch.tensor(
            [
                [
                    set(map(str, subject["identity_ids"]))
                    <= {str(identity) for identity in ids if identity is not None}
                    for subject in sample["subjects"]
                ]
                for sample, ids in zip(samples, q_ids, strict=True)
            ],
            dtype=torch.bool,
        ),
    }
    if state_image_ids is not None:
        # Full GT image identities, not detector-aligned labels: a missed person
        # must not change which target conditions supervise the global state.
        batch["state_mask"] = torch.tensor(
            [
                [image_id in eligible for image_id in rows]
                for rows, eligible in zip(
                    candidate_image_ids, state_image_ids, strict=True
                )
            ],
            dtype=torch.bool,
        )
        if (positive_mask & ~batch["state_mask"]).any():
            raise ValueError("every Full Positive must contain all required IDs")
    return batch


def finish_batch(visual, tokens, text_encoder, device):
    """Transfer a prepared batch and encode text in the training process."""
    return {
        **{key: value.to(device, non_blocking=True) for key, value in visual.items()},
        **encode_tokenized_text(tokens, text_encoder, device),
    }


def build_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
    *,
    state_image_ids: list[set[str]] | None = None,
    text_cache: QueryTextCache | None = None,
) -> dict[str, Tensor]:
    """Synchronous interface, also used to check the prefetched path."""
    text_cache = text_cache if text_cache is not None else QueryTextCache(tokenizer)
    visual = prepare_visual_batch(
        samples, candidate_image_ids, cache, state_image_ids=state_image_ids
    )
    return finish_batch(visual, text_cache.batch(samples), text_encoder, device)


@contextmanager
def prefetch_batches(jobs, prepare, depth: int = 0):
    """One CPU batch ahead; job generation (and its RNG) stays in the caller."""
    if depth not in (0, 1):
        raise ValueError("cache.prefetch_batches must be 0 or 1")
    if depth == 0:
        yield map(prepare, jobs)
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rcr-cache") as executor:

        def batches():
            iterator = iter(jobs)
            pending = None
            try:
                first = next(iterator, None)
                if first is None:
                    return
                pending = executor.submit(prepare, first)
                for job in iterator:
                    batch = pending.result()
                    pending = executor.submit(prepare, job)
                    yield batch
                yield pending.result()
            finally:
                if pending is not None:
                    pending.cancel()

        iterator = batches()
        try:
            yield iterator
        finally:
            iterator.close()
