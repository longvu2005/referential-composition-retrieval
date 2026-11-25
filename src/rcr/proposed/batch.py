"""Create one training batch from samples and frozen cache features."""

import torch
from torch import Tensor, nn

from rcr.common.data import inference_sample, inference_samples, required_identity_ids
from rcr.proposed.cache.native import NativeTargetBatch
from rcr.proposed.cache.store import GalleryCache, SceneTargetBatch
from rcr.proposed.identity import IdentityLayout
from rcr.proposed.nn.encoders import QueryTextCache, encode_tokenized_text
from rcr.proposed.prefetch import PackedPersons, StagedVisual
from rcr.proposed.scores import POOLING_MODES


def _identity_labels(rows, vocabulary: dict[str, int]) -> Tensor:
    return torch.tensor(
        [
            [
                -1
                if identity is None
                else vocabulary.setdefault(str(identity), len(vocabulary))
                for identity in identities
            ]
            for identities in rows
        ],
        dtype=torch.long,
    ).reshape(len(rows), len(rows[0]))


def build_supervision(identity_rows, subjects, vocabulary=None):
    """Build Subject membership targets and local identity-class labels."""
    vocabulary = {} if vocabulary is None else vocabulary
    labels = _identity_labels(identity_rows, vocabulary)
    memberships = []
    for identities, subject_rows in zip(identity_rows, subjects, strict=True):
        required = [set(map(str, subject["identity_ids"])) for subject in subject_rows]
        memberships.append(
            [
                [
                    identity is not None and str(identity) in ids
                    for identity in identities
                ]
                for ids in required
            ]
        )
    targets = torch.tensor(memberships, dtype=torch.get_default_dtype()).reshape(
        len(identity_rows), len(subjects[0]), len(identity_rows[0])
    )
    return targets, labels


def _multi_head_crops(image_ids, canonical_rows, conservative_ids):
    """Flag labeled crops excluded by the existing single-head check."""
    return torch.tensor(
        [
            [
                identity is not None and conservative_ids[image_id][person] is None
                for person, identity in enumerate(row)
            ]
            for image_id, row in zip(image_ids, canonical_rows, strict=True)
        ],
        dtype=torch.bool,
    )


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


def load_visual_features(
    cache,
    query_indices,
    target_indices,
    *,
    target_context=True,
    defer_target_scenes=False,
    preserve_storage=False,
    pack_target_persons=False,
):
    """Skip target scene shards and geometry when only identity/state are trained."""
    if target_context and not defer_target_scenes:
        return cache.load_groups(query_indices, target_indices)
    query = cache.load(query_indices)
    if pack_target_persons:
        persons, boxes, mask = PackedPersons.from_cache(
            cache, target_indices, geometry=target_context
        )
        if not preserve_storage:
            persons.values = persons.values.float()
        scene = SceneTargetBatch(cache, target_indices) if target_context else None
    elif target_context:
        if hasattr(cache, "load_person_geometry"):
            persons, boxes, mask = cache.load_person_geometry(target_indices)
        else:
            _, persons, boxes, mask = cache.load(target_indices)
        scene = SceneTargetBatch(cache, target_indices)
    else:
        persons, mask = cache.load_persons(target_indices)
        scene = boxes = None
    # The trainable projection expects FP32 inputs outside CUDA autocast.
    if not preserve_storage and isinstance(persons, Tensor):
        persons = persons.float()
    return query, (scene, persons, boxes, mask)


def prepare_visual_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    *,
    supervision,
    identity_image_ids: list[set[str]],
    identity_crops=(),
    identity_sampling=None,
    hard_identity_image_ids=None,
    fafa_pooling: str | None = None,
    defer_native_targets: bool = False,
    load_target_context: bool = True,
    audit_context_members: bool = False,
    defer_target_scenes: bool = False,
    preserve_storage: bool = False,
    pack_target_persons: bool = False,
) -> dict[str, Tensor | dict | str | NativeTargetBatch]:
    """Load CPU visual features and labels. No model is executed here.

    ``fafa_pooling`` is set only for the context ranking objective. It adds the
    frozen native FAFA query/target tokens that objective needs and records the
    pooling used for them. Without it no native token is read.
    Deferred targets retain only image indices and load useful chunks in the loss.
    Target scene/box tensors can be omitted when the context objective is disabled.
    """
    if fafa_pooling is not None and fafa_pooling not in POOLING_MODES:
        raise ValueError(f"unknown FAFA pooling: {fafa_pooling!r}")
    if (fafa_pooling is not None or audit_context_members) and not load_target_context:
        raise ValueError("context training requires target scenes and boxes")
    if not samples or len(samples) != len(candidate_image_ids):
        raise ValueError("training requires matching non-empty query/candidate rows")
    batch_size = len(samples)
    candidates_per_query = len(candidate_image_ids[0])
    if candidates_per_query < 2 or any(
        len(row) != candidates_per_query
        or len(set(row)) != len(row)
        or sample["query_image_id"] in row
        for sample, row in zip(samples, candidate_image_ids, strict=True)
    ):
        raise ValueError(
            "training candidates must be unique, exclude self and have width >= 2"
        )
    query_image_ids = [row["query_image_id"] for row in samples]
    query_indices = torch.tensor(
        [cache.by_id[image_id] for image_id in query_image_ids]
    )
    target_image_ids = [image_id for row in candidate_image_ids for image_id in row]
    target_indices = torch.tensor(
        [cache.by_id[image_id] for image_id in target_image_ids]
    )
    query, target = load_visual_features(
        cache,
        query_indices,
        target_indices,
        target_context=load_target_context,
        defer_target_scenes=defer_target_scenes,
        preserve_storage=preserve_storage,
        pack_target_persons=pack_target_persons,
    )
    q_scene, q_people, q_boxes, q_mask = query
    t_scene, t_people, t_boxes, t_mask = target

    positives = [set(row["positive_image_ids"]) for row in samples]
    q_ids = supervision.load_ids(query_image_ids, q_people.shape[1])
    t_ids = [[None] * t_people.shape[1] for _ in range(len(target_indices))]
    context_ids = t_ids.copy() if audit_context_members else None
    label_scope = identity_image_ids if audit_context_members else positives
    labeled_slots = [
        (n * candidates_per_query + j, image_id)
        for n, image_ids in enumerate(candidate_image_ids)
        for j, image_id in enumerate(image_ids)
        if image_id in label_scope[n]
    ]
    if labeled_slots:
        rows = supervision.load_ids(
            [image_id for _, image_id in labeled_slots], t_people.shape[1]
        )
        for (index, image_id), identities in zip(labeled_slots, rows, strict=True):
            if image_id in positives[index // candidates_per_query]:
                t_ids[index] = identities
            if context_ids is not None:
                context_ids[index] = identities

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

    # This branch is absent at fraction=0, during warmup, or with no ID budget.
    mining = (
        identity_sampling is not None
        and identity_sampling["max_crops"] > 0
        and identity_sampling["hard_negative_fraction"] > 0
    )
    safe_ids = {}
    if mining or audit_context_members:
        other_ids = (
            target_image_ids + [image_id for image_id, _, _ in identity_crops]
            if mining
            else [image_id for _, image_id in labeled_slots]
        )
        image_ids = list(dict.fromkeys(query_image_ids + other_ids))
        safe_ids = dict(
            zip(
                image_ids,
                supervision.load_ids(image_ids, supplemental=True),
                strict=True,
            )
        )

    extra = []
    for image_id, person, identity in identity_crops:
        if mining and (
            identity is None
            or person >= len(safe_ids[image_id])
            or safe_ids[image_id][person] != str(identity)
        ):
            continue
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

    if mining:
        # Deduplicate physical detections on CPU, once, before model execution.
        # Extra crops may reuse target embeddings instead of a second ID forward.
        base_seen = {
            (row["query_image_id"], person)
            for n, row in enumerate(samples)
            for person in q_mask[n].nonzero(as_tuple=True)[0].tolist()
        }
        base_seen.update(
            (image_id, person)
            for n, row in enumerate(candidate_image_ids)
            for j, image_id in enumerate(row)
            if image_id in positives[n]
            for person in t_mask[n * candidates_per_query + j]
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        target_positions, supplemental = {}, []
        for n, row in enumerate(candidate_image_ids):
            for j, image_id in enumerate(row):
                if image_id in positives[n]:
                    continue
                slot = n * candidates_per_query + j
                for person, identity in enumerate(safe_ids[image_id]):
                    key = (image_id, person)
                    if identity is None or not t_mask[slot, person] or key in base_seen:
                        continue
                    base_seen.add(key)
                    index = slot * t_people.shape[1] + person
                    target_positions[key] = index
                    supplemental.append((index, identity))
        required = {str(i) for row in samples for i in required_identity_ids(row)}
        anchor_mask = torch.zeros_like(q_mask)
        anchor_seen = set()
        for n, (row, ids) in enumerate(zip(samples, q_ids, strict=True)):
            requested = set(map(str, required_identity_ids(row)))
            safe = safe_ids[row["query_image_id"]]
            for person, identity in enumerate(ids):
                key = (row["query_image_id"], person)
                if (
                    q_mask[n, person]
                    and identity is not None
                    and person < len(safe)
                    and safe[person] == str(identity)
                    and str(identity) in requested
                    and key not in anchor_seen
                ):
                    anchor_seen.add(key)
                    anchor_mask[n, person] = True
        # Deduplicate requested anchors separately: an earlier query of the same
        # image may contain this person without requesting their identity.
        extra_batch.update(
            identity_sampling=identity_sampling,
            identity_anchor_mask=anchor_mask,
            supplemental_target_indices=torch.tensor(
                [index for index, _ in supplemental], dtype=torch.long
            ),
            supplemental_target_labels=_identity_labels(
                [[identity for _, identity in supplemental]], vocabulary
            )[0],
            extra_identity_positive_mask=torch.tensor(
                [str(identity) in required for _, _, identity in extra],
                dtype=torch.bool,
            ),
            extra_identity_target_indices=torch.tensor(
                [
                    target_positions.get((image_id, person), -1)
                    for image_id, person, _ in extra
                ],
                dtype=torch.long,
            ),
        )

    batch = {
        **extra_batch,
        "query_scene": q_scene,
        "query_persons": q_people,
        "query_boxes": q_boxes,
        "query_person_mask": q_mask,
        "query_identity_labels": query_labels,
        "query_identity_mask": query_identity_mask,
        "grounding_targets": grounding_targets,
        "target_global": cache.load_global_features(target_indices).unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "target_persons": (
            PackedPersons(
                t_people.values,
                t_people.positions,
                (batch_size, candidates_per_query, *t_people.shape[1:]),
            )
            if isinstance(t_people, PackedPersons)
            else t_people.unflatten(0, (batch_size, candidates_per_query))
        ),
        "target_mask": t_mask.unflatten(0, (batch_size, candidates_per_query)),
        "target_identity_labels": target_labels.unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "target_identity_mask": target_identity_mask.unflatten(
            0, (batch_size, candidates_per_query)
        ),
        "positive_mask": positive_mask,
    }
    if load_target_context:
        batch["target_scene"] = (
            t_scene
            if isinstance(t_scene, SceneTargetBatch)
            else t_scene.unflatten(0, (batch_size, candidates_per_query))
        )
        batch["target_boxes"] = t_boxes.unflatten(0, (batch_size, candidates_per_query))
    if context_ids is not None:
        # Keep extra Context labels out of the existing identity supervision.
        shape = (batch_size, candidates_per_query)
        batch.update(
            context_target_identity_labels=_identity_labels(
                context_ids, vocabulary
            ).unflatten(0, shape),
            context_required_member_counts=torch.tensor(
                [
                    [
                        len(set(map(str, subject["identity_ids"])))
                        for subject in row["subjects"]
                    ]
                    for row in samples
                ],
                dtype=torch.long,
            ),
            context_query_multi_head_crops=_multi_head_crops(
                query_image_ids, q_ids, safe_ids
            ),
            context_target_multi_head_crops=_multi_head_crops(
                target_image_ids, context_ids, safe_ids
            ).unflatten(0, shape),
        )
    batch["identity_positive_mask"] = torch.tensor(
        [
            [image_id in eligible for image_id in image_ids]
            for image_ids, eligible in zip(
                candidate_image_ids, identity_image_ids, strict=True
            )
        ],
        dtype=torch.bool,
    )
    if (positive_mask & ~batch["identity_positive_mask"]).any():
        raise ValueError("every Full Positive must contain all required IDs")
    hard_ids = (
        [set() for _ in samples]
        if hard_identity_image_ids is None
        else hard_identity_image_ids
    )
    if len(hard_ids) != batch_size or any(
        not set(hard) <= set(candidates)
        for hard, candidates in zip(hard_ids, candidate_image_ids, strict=True)
    ):
        raise ValueError("hard identity provenance must match sampled candidates")
    hard_mask = torch.tensor(
        [
            [image_id in hard for image_id in candidates]
            for hard, candidates in zip(hard_ids, candidate_image_ids, strict=True)
        ],
        dtype=torch.bool,
    )
    if (hard_mask & batch["identity_positive_mask"]).any():
        raise ValueError("hard identity negatives cannot contain all required IDs")
    batch["hard_identity_negative_mask"] = hard_mask
    if defer_target_scenes and load_target_context:
        width = (
            torch.where(t_mask, torch.arange(t_mask.shape[1]), -1).amax(1) + 1
            if t_mask.shape[1]
            else torch.zeros(len(t_mask), dtype=torch.long)
        )
        batch["context_layout"] = {
            "identity_rows": batch["identity_positive_mask"]
            .flatten()
            .nonzero()
            .flatten(),
            "rejection_rows": (batch["identity_positive_mask"] | hard_mask)
            .flatten()
            .nonzero()
            .flatten(),
            "widths": width,
        }
    if fafa_pooling is not None:
        # Public query inputs only: native query features never see annotations.
        batch["fafa_query"] = cache.native_queries(
            inference_samples(samples), q_people.shape[1]
        )
        batch["fafa_targets"] = (
            NativeTargetBatch(cache, target_indices)
            if defer_native_targets
            else cache.native_targets(target_indices, t_people.shape[1]).unflatten(
                0, (batch_size, candidates_per_query)
            )
        )
        batch["fafa_pooling"] = fafa_pooling
    if pack_target_persons:
        packed = batch["target_persons"]
        if not isinstance(packed, PackedPersons):
            packed = PackedPersons.from_padded(packed, batch["target_mask"])
        batch["target_persons"] = packed
        batch["target_layout"] = {"positions": packed.positions}
    batch["identity_layout"] = IdentityLayout.from_batch(batch)
    return batch


def _swap_batch(
    samples, candidates, supervision, visual, text_cache, text_encoder, device
):
    """Pick at most one eligible Full Positive per query and encode its edited text."""
    positions, variants, query_roles, target_roles = [], [], [], []
    query_width = visual["query_persons"].shape[1]
    target_width = visual["target_persons"].shape[2]
    for row, (sample, images) in enumerate(zip(samples, candidates, strict=True)):
        edit = supervision.direction_swaps.get(sample["sample_id"])
        if edit is None:
            continue
        change, verified = edit
        possible = [
            (col, image) for col, image in enumerate(images) if image in verified
        ]
        if not possible:
            continue
        subjects = sample["subjects"]
        query_ids = supervision.load_ids(
            [sample["query_image_id"]], query_width, supplemental=True
        )
        query_mask = build_supervision(query_ids, [subjects])[0][0].bool()
        if not query_mask.any(1).all():
            continue
        ids = supervision.load_ids(
            [image for _, image in possible], target_width, supplemental=True
        )
        masks = build_supervision(ids, [subjects] * len(ids))[0].bool()
        for (col, _), mask in zip(possible, masks, strict=True):
            if mask.any(1).all():
                positions.append((row, col))
                variant = inference_sample(sample)
                variant["final_change"] = change
                variant.pop("final_instruction", None)
                variants.append(variant)
                query_roles.append(query_mask)
                target_roles.append(mask)
                break
    if not positions:
        return None

    tokens = text_cache.batch(variants)
    change, mask, _ = text_encoder(tokens["change_ids"], tokens["change_mask"])
    return {
        # Annotation routing stays on CPU. The same detached matching workspace
        # decides eligibility before any alternative neural forward is launched.
        "positions": torch.tensor(positions),
        "query_roles": torch.stack(query_roles),
        "target_roles": torch.stack(target_roles),
        "text": {
            "change": change,
            "change_mask": mask,
            **{
                key: tokens[key].to(device)
                for key in ("subject_pos", "subject_token_mask", "subject_ids")
            },
        },
    }


def build_batch(
    samples: list[dict],
    candidate_image_ids: list[list[str]],
    cache: GalleryCache,
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device | str,
    *,
    supervision,
    identity_image_ids: list[set[str]],
    identity_crops=(),
    identity_sampling=None,
    hard_identity_image_ids=None,
    text_cache: QueryTextCache | None = None,
    fafa_pooling: str | None = None,
    include_direction_swaps: bool = False,
    defer_native_targets: bool = False,
    load_target_context: bool = True,
    audit_context_members: bool = False,
    defer_target_scenes: bool = False,
    preserve_storage: bool = False,
    pack_target_persons: bool = False,
    prepared_visual: dict | StagedVisual | None = None,
) -> dict[str, Tensor | dict | str | NativeTargetBatch]:
    text_cache = text_cache or QueryTextCache(tokenizer)
    if isinstance(prepared_visual, StagedVisual):
        prepared_visual = prepared_visual.consume()
    visual = (
        prepared_visual
        if prepared_visual is not None
        else prepare_visual_batch(
            samples,
            candidate_image_ids,
            cache,
            supervision=supervision,
            identity_image_ids=identity_image_ids,
            identity_crops=identity_crops,
            identity_sampling=identity_sampling,
            hard_identity_image_ids=hard_identity_image_ids,
            fafa_pooling=fafa_pooling,
            defer_native_targets=defer_native_targets,
            load_target_context=load_target_context,
            audit_context_members=audit_context_members,
            defer_target_scenes=defer_target_scenes,
            preserve_storage=preserve_storage,
            pack_target_persons=pack_target_persons,
        )
    )

    def transfer(value):
        if isinstance(value, PackedPersons):
            return value.to_dense(device)
        if isinstance(value, Tensor):
            value = value.to(device, non_blocking=True)
            return value.float() if value.is_floating_point() else value
        return value

    visual = {key: transfer(value) for key, value in visual.items()}
    visual["identity_layout"].bind(visual)
    text = encode_tokenized_text(text_cache.batch(samples), text_encoder, device)
    if include_direction_swaps:
        swap = _swap_batch(
            samples,
            candidate_image_ids,
            supervision,
            visual,
            text_cache,
            text_encoder,
            device,
        )
        if swap is not None:
            visual["direction_swap"] = swap
    return {**visual, **text}
