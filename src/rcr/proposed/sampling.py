"""Case-balanced query batches and reproducible candidate sampling."""

import math
from collections import Counter, defaultdict

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


def identity_candidate_pools(data, samples, gallery_ids):
    """Images containing every required identity, using train annotations only."""
    by_identity = defaultdict(set)
    for image_id in gallery_ids:
        for box in data.gt_head_boxes_by_image.get(image_id, []):
            by_identity[str(box["identity_id"])].add(image_id)

    gallery_order = {image_id: index for index, image_id in enumerate(gallery_ids)}
    pools = {}
    for sample in samples:
        identities = set(map(str, required_identity_ids(sample)))
        images = set.intersection(*(by_identity[identity] for identity in identities))
        images.discard(sample["query_image_id"])
        pools[sample["sample_id"]] = sorted(images, key=gallery_order.__getitem__)
    return pools


def sampling_settings(config):
    settings = {
        "positives_per_query": 2,
        "identity_fraction": 0.5,
        "hard_fraction": 0.0,
        "warmup_epochs": 1,
        "refresh_every_epochs": 2,
        "pool_size": 100,
    }
    if config.keys() - settings.keys():
        raise ValueError("unknown train.sampling setting")
    settings.update(config)
    if settings["positives_per_query"] < 1:
        raise ValueError("positives_per_query must be positive")
    fractions = settings["identity_fraction"], settings["hard_fraction"]
    if any(not 0 <= value <= 1 for value in fractions) or sum(fractions) > 1:
        raise ValueError("negative fractions must be in [0, 1] and sum to <= 1")
    if settings["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    if settings["refresh_every_epochs"] < 1 or settings["pool_size"] < 1:
        raise ValueError("hard-negative intervals and pool size must be positive")
    return settings


class CandidateIndex:
    """Precomputed positives, exclusions and same-identity negatives."""

    def __init__(self, samples, gallery_ids, identity_pools, excluded=None):
        self.gallery = list(gallery_ids)
        gallery = set(gallery_ids)
        self.rows = {}
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
):
    """Sample positives, structured negatives, then uniform negatives."""
    rows = []
    for sample in samples:
        positives, base_forbidden, identity_pool = index.rows[sample["sample_id"]]
        positive_count = min(positives_per_query, len(positives), num_candidates - 1)
        row = _draw(positives, positive_count, generator)
        forbidden = set(base_forbidden)

        negative_count = num_candidates - positive_count
        hard_pool = list(
            dict.fromkeys(
                image_id
                for image_id in (hard_pools or {}).get(sample["sample_id"], ())
                if image_id not in forbidden
            )
        )
        for pool, fraction in (
            (identity_pool, identity_fraction),
            (hard_pool, hard_fraction),
        ):
            pool = [image_id for image_id in pool if image_id not in forbidden]
            chosen = _draw(
                pool, min(int(negative_count * fraction), len(pool)), generator
            )
            row.extend(chosen)
            forbidden.update(chosen)
        row.extend(
            _random_negatives(
                index.gallery,
                forbidden,
                num_candidates - len(row),
                generator,
            )
        )
        permutation = torch.randperm(num_candidates, generator=generator).tolist()
        rows.append([row[index] for index in permutation])
    return rows
