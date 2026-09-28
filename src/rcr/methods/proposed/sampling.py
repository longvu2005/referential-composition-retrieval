"""Candidate sampling for RCR training."""

import torch


def sample_candidates(
    samples: list[dict],
    gallery_image_ids: list[str],
    num_candidates: int,
    generator: torch.Generator | None = None,
) -> list[list[str]]:
    """Sample one positive and random negatives for each query."""

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")

    gallery = list(dict.fromkeys(gallery_image_ids))
    gallery_set = set(gallery)
    rows: list[list[str]] = []

    for sample in samples:
        positives = sample.get("positive_image_ids")
        if positives is None:
            positives = [sample["target_image_id"]]

        positives = [
            x for x in positives if x in gallery_set and x != sample["query_image_id"]
        ]
        if not positives:
            raise ValueError(f"no positive target in gallery for {sample['sample_id']}")

        positive = positives[
            torch.randint(len(positives), (), generator=generator).item()
        ]

        forbidden = set(positives)
        forbidden.add(sample["query_image_id"])
        negatives = [x for x in gallery if x not in forbidden]

        need = num_candidates - 1
        if len(negatives) < need:
            raise ValueError("not enough negative gallery images")

        order = torch.randperm(len(negatives), generator=generator)[:need]
        row = [positive] + [negatives[i] for i in order.tolist()]

        order = torch.randperm(num_candidates, generator=generator)
        rows.append([row[i] for i in order.tolist()])

    return rows
