"""Case-balanced query batches and reproducible candidate sampling."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch

from rcr.common.data import required_identity_ids


def training_batches(samples, batch_size, generator, *, case_balanced=True):
    """Draw one epoch and group rows by Subject count for tensor collation."""
    if case_balanced:
        counts = Counter(row["case_type"] for row in samples)
        weights = torch.tensor(
            [1 / math.sqrt(counts[row["case_type"]]) for row in samples],
            dtype=torch.double,
        )
        order = torch.multinomial(
            weights, len(samples), replacement=True, generator=generator
        )
    else:
        order = torch.randperm(len(samples), generator=generator)

    groups = defaultdict(list)
    for index in order.tolist():
        row = samples[index]
        groups[len(row["subjects"])].append(row)
    batches = [
        rows[start : start + batch_size]
        for rows in groups.values()
        for start in range(0, len(rows), batch_size)
    ]
    order = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[index] for index in order]


@dataclass(frozen=True, slots=True)
class PartialIdentity:
    """Distinct missing IDs and fully absent Subject slots from train GT."""

    missing_ids: frozenset[str]
    missing_subjects: frozenset[int]
    overlap: int


def annotation_candidate_pools(data, samples, gallery_ids, *, include_partial=True):
    """Index train GT once for identity supersets and overlapping wrong-ID images."""
    by_identity = defaultdict(set)
    by_image = {}
    for image_id in gallery_ids:
        identities = {
            str(box["identity_id"])
            for box in data.gt_head_boxes_by_image.get(image_id, [])
        }
        by_image[image_id] = identities
        for identity in identities:
            by_identity[identity].add(image_id)

    gallery_order = {image_id: index for index, image_id in enumerate(gallery_ids)}
    pools, partial = {}, {}
    for sample in samples:
        identities = set(map(str, required_identity_ids(sample)))
        images = set.intersection(*(by_identity[identity] for identity in identities))
        images.discard(sample["query_image_id"])
        sample_id = sample["sample_id"]
        pools[sample_id] = sorted(images, key=gallery_order.__getitem__)
        if not include_partial or len(identities) < 2:
            continue
        subjects = [set(map(str, s["identity_ids"])) for s in sample["subjects"]]
        overlapping = set.union(*(by_identity[identity] for identity in identities))
        partial[sample_id] = {}
        patterns = {}
        for image_id in sorted(overlapping - images, key=gallery_order.__getitem__):
            missing = frozenset(identities - by_image[image_id])
            if 0 < len(missing) < len(identities):
                if missing not in patterns:
                    patterns[missing] = PartialIdentity(
                        missing,
                        frozenset(
                            i for i, ids in enumerate(subjects) if ids <= missing
                        ),
                        len(identities) - len(missing),
                    )
                partial[sample_id][image_id] = patterns[missing]
    return pools, partial


def identity_candidate_pools(data, samples, gallery_ids):
    """Images containing every required identity, using train annotations only."""
    return annotation_candidate_pools(
        data, samples, gallery_ids, include_partial=False
    )[0]


def sampling_settings(config):
    settings = {
        "positives_per_query": 2,
        "identity_fraction": 0.5,
        "hard_fraction": 0.0,
        "semantic_hard_fraction": 0.0,
        "semantic_context_quota": 0,
        "warmup_epochs": 1,
        "refresh_every_epochs": 2,
        "pool_size": 100,
        "partial_id_quota": 0,
        "rejection_mining_context_weight": 0.0,
    }
    if config.keys() - settings.keys():
        raise ValueError("unknown train.sampling setting")
    settings.update(config)
    if settings["positives_per_query"] < 1:
        raise ValueError("positives_per_query must be positive")
    fractions = settings["identity_fraction"], settings["hard_fraction"]
    if (
        any(
            type(value) not in (int, float) or not 0 <= value <= 1
            for value in (*fractions, settings["semantic_hard_fraction"])
        )
        or sum(fractions) > 1
    ):
        raise ValueError("negative fractions must be in [0, 1] and sum to <= 1")
    if settings["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    if settings["refresh_every_epochs"] < 1 or settings["pool_size"] < 1:
        raise ValueError("hard-negative intervals and pool size must be positive")
    for key in ("partial_id_quota", "semantic_context_quota"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    weight = settings["rejection_mining_context_weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError(
            "rejection_mining_context_weight must be finite and nonnegative"
        )
    return settings


class CandidateIndex:
    """Precomputed positives, exclusions, same-ID and partial-ID candidates."""

    def __init__(
        self, samples, gallery_ids, identity_pools, excluded=None, *, partial_pools=None
    ):
        self.gallery = list(gallery_ids)
        self.gallery_set = gallery = set(gallery_ids)
        self.rows = {}
        self.partial = {}
        for sample in samples:
            sample_id = sample["sample_id"]
            positives = list(
                dict.fromkeys(
                    image_id
                    for image_id in sample["positive_image_ids"]
                    if image_id in gallery and image_id != sample["query_image_id"]
                )
            )
            forbidden = (
                set(positives)
                | {sample["query_image_id"]}
                | set((excluded or {}).get(sample_id, ()))
            )
            identity = [
                image_id
                for image_id in identity_pools.get(sample_id, ())
                if image_id in gallery and image_id not in forbidden
            ]
            self.rows[sample_id] = positives, forbidden, identity
            self.partial[sample_id] = {
                image_id: info
                for image_id, info in (partial_pools or {}).get(sample_id, {}).items()
                if image_id in gallery and image_id not in forbidden
            }


def _draw(pool, count, generator):
    if not pool or count == 0:
        return []
    order = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return [pool[index] for index in order]


def _random_negatives(gallery, forbidden, count, generator):
    """Uniform sampling without replacement; avoid scanning a large gallery."""
    selected = []
    while len(selected) < count:
        draws = torch.randint(len(gallery), (max(16, 2 * count),), generator=generator)
        for index in draws.tolist():
            image_id = gallery[index]
            if image_id not in forbidden:
                selected.append(image_id)
                forbidden.add(image_id)
                if len(selected) == count:
                    break
        if len(gallery) - len(forbidden) < count - len(selected):
            raise ValueError("not enough negative gallery images")
    return selected


def _partial_negatives(sample, pool, row, count, quota, generator):
    """Top up partial IDs, reserving error types before weighted diversity draws."""
    if not pool:
        return []
    present = [pool[image_id] for image_id in row if image_id in pool]
    count = min(count, max(0, quota - len(present)))
    if not count:
        return []
    row_ids = set(row)
    available = {i: info for i, info in pool.items() if i not in row_ids}
    missing_counts = Counter(i for info in present for i in info.missing_ids)
    selected = []

    def choose(images):
        single = [i for i in images if len(available[i].missing_ids) == 1]
        if single:
            frequency = {
                i: missing_counts[next(iter(available[i].missing_ids))] for i in single
            }
            least = min(frequency.values())
            images = [i for i in single if frequency[i] == least]
        image_id = _draw(images, 1, generator)[0]
        info = available.pop(image_id)
        selected.append(image_id)
        present.append(info)
        missing_counts.update(info.missing_ids)

    requirements = [
        lambda info: bool(info.missing_subjects),
        lambda info: not info.missing_subjects,
    ]
    subjects = [set(map(str, s["identity_ids"])) for s in sample["subjects"]]
    if len(subjects) == 2 and all(len(ids) == 1 for ids in subjects):
        requirements = [
            lambda info, slot=slot: slot in info.missing_subjects for slot in range(2)
        ]
    uncovered = [
        requirement
        for requirement in requirements
        if not any(requirement(info) for info in present)
        and any(requirement(info) for info in available.values())
    ]
    # Reserve representatives only if they fit together. With one remaining
    # slot, the single-ID priority takes precedence over an absent multi-ID Subject.
    if len(uncovered) <= count:
        for requirement in uncovered:
            choose([i for i, info in available.items() if requirement(info)])

    single_target = min(4, quota)
    while len(selected) < count:
        single = [i for i, info in available.items() if len(info.missing_ids) == 1]
        if (
            sum(len(info.missing_ids) == 1 for info in present) >= single_target
            or not single
        ):
            break
        choose(single)

    while len(selected) < count and available:
        groups = defaultdict(list)
        for image_id, info in available.items():
            groups[info.overlap].append(image_id)
        overlaps = sorted(groups)
        # Rescale the powers by their maximum to keep large ID sets finite.
        weights = torch.tensor([2.0 ** (r - max(overlaps)) for r in overlaps])
        group = torch.multinomial(weights, 1, generator=generator).item()
        image_id = _draw(groups[overlaps[group]], 1, generator)[0]
        selected.append(image_id)
        available.pop(image_id)
    return selected


@dataclass
class SampledCandidates:
    """Candidate IDs and the train-only provenance of actual hard selections."""

    images: list[list[str]]
    hard_identity_ids: list[set[str]]
    semantic_hard_count: int
    identity_hard_count: int
    partial_id_count: int = 0
    rejection_context_count: int = 0
    semantic_context_ids: list[set[str]] = field(default_factory=list)


def sample_candidates(
    samples,
    num_candidates,
    generator,
    *,
    index: CandidateIndex,
    positives_per_query=2,
    identity_fraction=0.5,
    hard_pools=None,
    hard_fraction=0.0,
    semantic_hard_fraction=0.0,
    semantic_context_quota=0,
    identity_rejection=False,
    partial_id_quota=0,
    rejection_mining_context_weight=0.0,
):
    """Draw positives, same-ID, mined wrong-ID, partial-ID, then random images.

    Small same-ID pools are retained in full. Fallback images are never tagged
    as mined hard identity negatives, even if they happen to have a wrong ID.
    """
    rows, hard_identity_ids, semantic_context_ids = [], [], []
    semantic_hard_count = identity_hard_count = 0
    partial_id_count = rejection_context_count = 0
    for sample in samples:
        positives, base_forbidden, identity_pool = index.rows[sample["sample_id"]]
        positive_count = min(positives_per_query, len(positives), num_candidates - 1)
        row = _draw(positives, positive_count, generator)
        forbidden = set(base_forbidden)

        negative_count = num_candidates - positive_count
        pools = (hard_pools or {}).get(sample["sample_id"], {})
        identity_set = set(identity_pool)

        def draw(pool, count, *, hard=False, row=row, forbidden=forbidden):
            pool = list(
                dict.fromkeys(
                    i for i in pool if i in index.gallery_set and i not in forbidden
                )
            )
            # Draw from a small hardest window instead of repeating a fixed top-K.
            if hard:
                pool = pool[: 2 * count]
            chosen = _draw(pool, count, generator)
            row.extend(chosen)
            forbidden.update(chosen)
            return chosen

        identity_quota = int(negative_count * identity_fraction)
        semantic_quota = math.ceil(identity_quota * semantic_hard_fraction)
        semantic_hard = []
        if len(identity_pool) > identity_quota:
            semantic_hard = draw(
                [i for i in pools.get("semantic", ()) if i in identity_set],
                semantic_quota,
                hard=True,
            )
        semantic_hard_count += len(semantic_hard)
        draw(identity_pool, identity_quota - len(semantic_hard))

        hard_quota = int(negative_count * hard_fraction)
        if identity_rejection:
            context_hard = (
                draw(
                    [
                        i
                        for i in pools.get("rejection_context", ())[:2]
                        if i not in identity_set
                    ],
                    1,
                )
                if hard_quota and rejection_mining_context_weight > 0
                else []
            )
            rejection_context_count += len(context_hard)
            identity_hard = context_hard + draw(
                [i for i in pools.get("identity", ()) if i not in identity_set],
                hard_quota - len(context_hard),
                hard=True,
            )
        else:
            # The zero-feature ablation retains the original coarse hard pool.
            draw(pools.get("coarse", ()), hard_quota)
            identity_hard = []
        hard_identity_ids.append(set(identity_hard))
        identity_hard_count += len(identity_hard)
        if partial_id_quota:
            partial = _partial_negatives(
                sample,
                index.partial[sample["sample_id"]],
                row,
                num_candidates - len(row),
                partial_id_quota,
                generator,
            )
            row.extend(partial)
            forbidden.update(partial)
        row.extend(
            _random_negatives(
                index.gallery,
                forbidden,
                num_candidates - len(row),
                generator,
            )
        )
        partial_id_count += sum(i in index.partial[sample["sample_id"]] for i in row)
        # Start from the completed base draw. Replace only mined same-ID slots
        # with new images; preserve positives, random diversity and wrong-ID tags.
        count = min(semantic_context_quota, len(semantic_hard))
        window = pools.get("semantic_context", ())[: 2 * count]
        new_ids = [
            image
            for image in dict.fromkeys(window)
            if image in identity_set and image not in row
        ]
        replacements = _draw(new_ids, count, generator)
        for old, new in zip(reversed(semantic_hard), replacements, strict=False):
            row[row.index(old)] = new
        semantic_context_ids.append(set(replacements))
        permutation = torch.randperm(num_candidates, generator=generator).tolist()
        rows.append([row[position] for position in permutation])
    return SampledCandidates(
        rows,
        hard_identity_ids,
        semantic_hard_count,
        identity_hard_count,
        partial_id_count,
        rejection_context_count,
        semantic_context_ids,
    )
