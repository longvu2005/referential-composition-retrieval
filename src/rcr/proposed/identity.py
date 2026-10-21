"""Train-only identity crops sampled independently of retrieval candidates."""

from collections import defaultdict

import torch

from rcr.common.data import required_identity_ids


def identity_sampling_settings(config):
    settings = {
        "max_crops": 0,
        "crops_per_identity": 2,
        "random_identities": 8,
        **config,
    }
    if set(settings) != {"max_crops", "crops_per_identity", "random_identities"}:
        raise ValueError("unknown train.identity_sampling setting")
    if settings["max_crops"] < 0 or settings["crops_per_identity"] < 1:
        raise ValueError("invalid identity crop limits")
    if settings["random_identities"] < 0:
        raise ValueError("random_identities must be nonnegative")
    return settings


class IdentityCropSampler:
    """Draw different-image observations from detector-aligned train IDs."""

    def __init__(self, supervision, settings):
        self.settings = settings
        self.pools = {}
        for identity, crops in supervision.identity_crops().items():
            images = defaultdict(list)
            for image_id, person in crops:
                images[image_id].append(person)
            self.pools[identity] = dict(images)
        self.random_ids = [
            identity for identity, images in self.pools.items() if len(images) >= 2
        ]

    def sample(self, samples, candidates, generator):
        limit = self.settings["max_crops"]
        if not limit:
            return []

        used_images = {sample["query_image_id"] for sample in samples}
        for sample, row in zip(samples, candidates, strict=True):
            positives = set(sample["positive_image_ids"])
            used_images.update(image_id for image_id in row if image_id in positives)

        required = list(
            dict.fromkeys(
                str(identity)
                for sample in samples
                for identity in required_identity_ids(sample)
            )
        )
        order = torch.randperm(len(required), generator=generator).tolist()
        required = [required[index] for index in order]
        other = [identity for identity in self.random_ids if identity not in required]
        if other:
            order = torch.randperm(len(other), generator=generator).tolist()
            other = [
                other[index]
                for index in order[: self.settings["random_identities"]]
            ]

        per_identity = self.settings["crops_per_identity"]
        query_budget = limit - min(len(other) * per_identity, limit // 2)
        result = []
        for identities, budget in ((required, query_budget), (other, limit)):
            for identity in identities:
                count = min(per_identity, budget - len(result))
                if count <= 0:
                    break
                images = [
                    (image_id, people)
                    for image_id, people in self.pools.get(identity, {}).items()
                    if image_id not in used_images
                ]
                order = torch.randperm(len(images), generator=generator)[:count]
                for index in order.tolist():
                    image_id, people = images[index]
                    person = people[
                        int(torch.randint(len(people), (), generator=generator))
                    ]
                    result.append((image_id, person, identity))
        return result
