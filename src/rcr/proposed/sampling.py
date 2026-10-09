"""Uniform batches and train-only positive/negative sampling."""

import math
from collections import defaultdict

import torch


def training_batches(samples, batch_size, generator):
    """Uniform permutation, no case or annotation-cardinality routing."""
    if batch_size < 1 or not samples:
        raise ValueError("training requires samples and a positive batch_size")
    order = torch.randperm(len(samples), generator=generator).tolist()
    return [
        [samples[i] for i in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def negative_exclusions(samples):
    """Ignore disputed negatives across equivalent identity/condition queries.

    Case is intentionally absent from equivalence and training decisions.
    """
    groups = defaultdict(list)
    for sample in samples:
        key = (
            tuple(
                (s["subject_id"], tuple(sorted(map(str, s["identity_ids"]))))
                for s in sample["subjects"]
            ),
            sample["final_change"],
        )
        groups[key].append(sample)
    result = {}
    for rows in groups.values():
        positives = set().union(*(set(r["positive_image_ids"]) for r in rows))
        for row in rows:
            result[row["sample_id"]] = (
                positives - set(row["positive_image_ids"]) - {row["query_image_id"]}
            )
    return result


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
        "positives_per_query": 2,
        "identity_fraction": 0.5,
        "hard_fraction": 0.0,
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
    for key in ("refresh_every_epochs", "pool_size"):
        value = settings[key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid sampling.{key}")
    positives = settings["positives_per_query"]
    if isinstance(positives, bool) or not isinstance(positives, int) or positives < 1:
        raise ValueError("sampling.positives_per_query must be a positive integer")
    return settings


class CandidateIndex:
    """Static gallery, positives and exclusions reused by every training batch."""

    def __init__(self, samples, gallery_image_ids, identity_pools=None, excluded=None):
        self.gallery = list(dict.fromkeys(gallery_image_ids))
        self.gallery_set = set(self.gallery)
        self.rows = {}
        for sample in samples:
            sample_id = sample["sample_id"]
            positives = sample.get("positive_image_ids")
            if positives is None:
                positives = [sample["target_image_id"]]
            positives = list(
                dict.fromkeys(
                    x
                    for x in positives
                    if x in self.gallery_set and x != sample["query_image_id"]
                )
            )
            if not positives:
                raise ValueError(f"no positive target in gallery for {sample_id}")
            forbidden = (
                set(positives)
                | {sample["query_image_id"]}
                | set((excluded or {}).get(sample_id, ()))
            ) & self.gallery_set
            identity = list(
                dict.fromkeys(
                    x
                    for x in (identity_pools or {}).get(sample_id, ())
                    if x in self.gallery_set and x not in forbidden
                )
            )
            self.rows[sample_id] = (positives, forbidden, identity)


def _random_negatives(gallery, forbidden, count, generator):
    """Uniform draws without replacement, with a bounded dense fallback."""
    selected = []
    # Rejection is cheap when C is small compared with the eligible gallery.
    # Accepted draws remain uniform; dense fallback samples the remaining set.
    for _ in range(8):
        remaining = count - len(selected)
        if remaining == 0:
            return selected
        if len(gallery) - len(forbidden) < max(2 * remaining, len(gallery) // 4):
            break
        draws = torch.randint(
            len(gallery), (max(16, 2 * remaining),), generator=generator
        )
        for index in draws.tolist():
            image_id = gallery[index]
            if image_id not in forbidden:
                selected.append(image_id)
                forbidden.add(image_id)
                if len(selected) == count:
                    return selected
    available = [x for x in gallery if x not in forbidden]
    indices = torch.randperm(len(available), generator=generator)[
        : count - len(selected)
    ]
    tail = [available[i] for i in indices.tolist()]
    forbidden.update(tail)
    return selected + tail


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
    index: CandidateIndex | None = None,
    positives_per_query: int = 2,
) -> list[list[str]]:
    """Up to P distinct positives, mixed negatives, and at least one negative.

    Negative quotas use the actual C-P slots. Unselected positives stay excluded;
    short identity/mined pools fall back to random negatives.
    """
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")
    sampling_settings(
        {
            "identity_fraction": identity_fraction,
            "hard_fraction": hard_fraction,
            "positives_per_query": positives_per_query,
        }
    )
    if identity_fraction and identity_pools is None:
        raise ValueError("identity sampling requires identity_pools")
    if index is None:
        index = CandidateIndex(samples, gallery_image_ids, identity_pools, excluded)
    rows = []
    for sample in samples:
        sample_id = sample["sample_id"]
        positives, base_forbidden, identity = index.rows[sample_id]
        positive_count = min(positives_per_query, len(positives), num_candidates - 1)
        need = num_candidates - positive_count
        if len(index.gallery) - len(base_forbidden) < need:
            raise ValueError("not enough negative gallery images")
        chosen = torch.randperm(len(positives), generator=generator)[:positive_count]
        forbidden = set(base_forbidden)
        row = [positives[i] for i in chosen.tolist()]
        if stats is not None:
            stats["sampled_positive"] = (
                stats.get("sampled_positive", 0) + positive_count
            )
        wrong_identity = (
            set(index.gallery)
            - set((identity_pools or {}).get(sample_id, ()))
            - forbidden
        )
        if wrong_identity:
            available = [x for x in index.gallery if x in wrong_identity]
            chosen_wrong = available[
                int(torch.randint(len(available), (1,), generator=generator))
            ]
            row.append(chosen_wrong)
            forbidden.add(chosen_wrong)
            if stats is not None:
                stats["sampled_wrong_identity"] = (
                    stats.get("sampled_wrong_identity", 0) + 1
                )
        for pool, fraction, kind in (
            (identity, identity_fraction, "sampled_identity"),
            ((hard_pools or {}).get(sample_id, ()), hard_fraction, "sampled_hard"),
        ):
            quota = min(int(need * fraction), num_candidates - len(row))
            if not quota:
                available = []
            elif kind == "sampled_identity":
                available = identity  # Already filtered by the static index.
            else:
                available = list(
                    dict.fromkeys(
                        x for x in pool if x in index.gallery_set and x not in forbidden
                    )
                )
            count = min(quota, len(available))
            indices = (
                torch.randperm(len(available), generator=generator)[:count].tolist()
                if count
                else []
            )
            chosen = [available[i] for i in indices]
            row.extend(chosen)
            forbidden.update(chosen)
            if stats is not None:
                stats[kind] = stats.get(kind, 0) + count
        count = num_candidates - len(row)
        row.extend(_random_negatives(index.gallery, forbidden, count, generator))
        if stats is not None:
            stats["sampled_random"] = stats.get("sampled_random", 0) + count
        order = torch.randperm(num_candidates, generator=generator).tolist()
        rows.append([row[i] for i in order])
    return rows
