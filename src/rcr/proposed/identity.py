"""Small train-only crop batches for identity learning, independent of retrieval."""

from collections import defaultdict

import torch

from rcr.common.data import required_identity_ids


def identity_sampling_settings(cfg: dict) -> dict:
    settings = {
        "max_crops": 0,  # Older configs retain query/Full Positive supervision.
        "crops_per_identity": 2,
        "random_identities": 8,
        **cfg,
    }
    for name, minimum in (
        ("max_crops", 0),
        ("crops_per_identity", 1),
        ("random_identities", 0),
    ):
        value = settings[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid train.identity_sampling.{name}")
    return settings


class IdentityCropSampler:
    """Sample distinct-image observations from prepared detector-aligned IDs."""

    def __init__(self, supervision, settings):
        self.settings = settings
        self.pools = {}
        for identity, crops in supervision.identity_crops().items():
            images = defaultdict(list)
            for image_id, person in crops:
                images[image_id].append(person)
            self.pools[identity] = dict(images)
        # Single-image IDs can support negatives in the ordinary batch, but do
        # not supply a cross-image positive pair for an auxiliary random ID.
        self.random_ids = [i for i, images in self.pools.items() if len(images) >= 2]

    def sample(self, samples, candidates, generator):
        """Return (image_id, crop_index, identity_id), never retrieval candidates."""
        limit = self.settings["max_crops"]
        if limit == 0:
            return []
        used_images = {s["query_image_id"] for s in samples}
        for sample, row in zip(samples, candidates, strict=True):
            positives = (
                set(sample["positive_image_ids"])
                if "positive_image_ids" in sample
                else {sample["target_image_id"]}
            )
            used_images.update(i for i in row if i in positives)
        required = list(
            dict.fromkeys(
                str(i) for sample in samples for i in required_identity_ids(sample)
            )
        )
        order = torch.randperm(len(required), generator=generator).tolist()
        required = [required[i] for i in order]
        required_set = set(required)
        random_ids = [i for i in self.random_ids if i not in required_set]
        order = torch.randperm(len(random_ids), generator=generator).tolist()
        random_ids = [
            random_ids[i] for i in order[: self.settings["random_identities"]]
        ]
        per_id = self.settings["crops_per_identity"]
        # Reserve at most half the budget for other train IDs, including IDs
        # that never occur as query Subjects. Short pools simply use fewer crops.
        query_limit = limit - min(len(random_ids) * per_id, limit // 2)
        result = []
        for identities, budget in ((required, query_limit), (random_ids, limit)):
            for identity in identities:
                count = min(per_id, budget - len(result))
                if count <= 0:
                    break
                images = [
                    (i, people)
                    for i, people in self.pools.get(identity, {}).items()
                    if i not in used_images
                ]
                order = torch.randperm(len(images), generator=generator)[:count]
                for index in order.tolist():
                    image_id, people = images[index]
                    person = people[
                        int(torch.randint(len(people), (), generator=generator))
                    ]
                    result.append((image_id, person, identity))
        return result
