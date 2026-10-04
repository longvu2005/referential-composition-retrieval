"""Train-only identity pools and mixed negative sampling."""

import math
from collections import defaultdict

import torch


def identity_candidate_pools(data, samples, gallery_image_ids):
    """All images with every required identity, using GT TRAIN labels only.

    Pool order follows the gallery, so sampling is reproducible across processes.
    Detector misses must not turn a state-positive image into a state negative.
    """
    by_identity = defaultdict(set)
    order = {image_id: i for i, image_id in enumerate(gallery_image_ids)}
    for image_id in gallery_image_ids:
        for box in data.gt_head_boxes_by_image.get(image_id, []):
            by_identity[str(box["identity_id"])].add(image_id)
    pools = {}
    for sample in samples:
        required = {
            str(identity)
            for subject in sample["subjects"]
            for identity in subject["identity_ids"]
        }
        if not required:
            raise ValueError(f"{sample['sample_id']}: missing required identities")
        images = set.intersection(*(by_identity[identity] for identity in required))
        images.discard(sample["query_image_id"])
        if not set(sample["positive_image_ids"]) <= images:
            raise ValueError(f"{sample['sample_id']}: Full Positive lacks required IDs")
        pools[sample["sample_id"]] = sorted(images, key=order.__getitem__)
    return pools


def sampling_settings(cfg: dict) -> dict:
    settings = {
        "identity_fraction": 0.5,
        "hard_fraction": 0.0,
        "warmup_epochs": 1,
        "refresh_every_epochs": 2,
        "pool_size": 100,
        **cfg,
    }
    fractions = [settings[k] for k in ("identity_fraction", "hard_fraction")]
    if (
        not all(math.isfinite(x) and 0 <= x <= 1 for x in fractions)
        or sum(fractions) > 1
    ):
        raise ValueError("sampling fractions must be in [0,1] and sum to at most 1")
    for key in ("warmup_epochs", "refresh_every_epochs", "pool_size"):
        value = settings[key]
        if not isinstance(value, int) or value < (0 if key == "warmup_epochs" else 1):
            raise ValueError(f"invalid sampling.{key}")
    return settings


def sample_candidates(
    samples: list[dict],
    gallery_image_ids: list[str],
    num_candidates: int,
    generator: torch.Generator | None = None,
    *,
    identity_pools: dict[str, list[str]] | None = None,
    hard_pools: dict[str, list[str]] | None = None,
    excluded: dict[str, set[str]] | None = None,
    identity_fraction: float = 0.0,
    hard_fraction: float = 0.0,
    stats: dict | None = None,
) -> list[list[str]]:
    """One reviewed positive, identity negatives, mined negatives, then random.

    Quotas are floor(fraction * (C-1)); unused slots fall back to random. Every
    stage excludes self, ALL known positives, disputed negatives and prior picks.
    The hard pool is sampled uniformly, not always taking its highest ranks.
    """

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")
    sampling_settings(
        {"identity_fraction": identity_fraction, "hard_fraction": hard_fraction}
    )
    if identity_fraction and identity_pools is None:
        raise ValueError("identity sampling requires identity_pools")

    gallery = list(dict.fromkeys(gallery_image_ids))
    gallery_set = set(gallery)
    rows: list[list[str]] = []

    for sample in samples:
        positives = sample.get("positive_image_ids")
        if positives is None:
            positives = [sample["target_image_id"]]

        positives = list(
            dict.fromkeys(
                x
                for x in positives
                if x in gallery_set and x != sample["query_image_id"]
            )
        )
        if not positives:
            raise ValueError(f"no positive target in gallery for {sample['sample_id']}")

        positive = positives[
            torch.randint(len(positives), (), generator=generator).item()
        ]

        forbidden = set(positives)
        forbidden.add(sample["query_image_id"])
        forbidden.update((excluded or {}).get(sample["sample_id"], ()))
        negatives = [x for x in gallery if x not in forbidden]

        need = num_candidates - 1
        if len(negatives) < need:
            raise ValueError("not enough negative gallery images")

        row = [positive]

        sample_id = sample["sample_id"]
        pools = (
            (
                (identity_pools or {}).get(sample_id, ()),
                int(need * identity_fraction),
                "sampled_identity",
            ),
            (
                (hard_pools or {}).get(sample_id, ()),
                int(need * hard_fraction),
                "sampled_hard",
            ),
            (negatives, None, "sampled_random"),
        )
        for pool, count, kind in pools:
            available = list(
                dict.fromkeys(
                    image_id
                    for image_id in pool
                    if image_id in gallery_set and image_id not in forbidden
                )
            )
            count = min(
                num_candidates - len(row) if count is None else count, len(available)
            )
            indices = torch.randperm(len(available), generator=generator)[:count]
            selected = [available[i] for i in indices.tolist()]
            row.extend(selected)
            forbidden.update(selected)
            if stats is not None:
                stats[kind] = stats.get(kind, 0) + count

        order = torch.randperm(num_candidates, generator=generator)
        rows.append([row[i] for i in order.tolist()])

    return rows
