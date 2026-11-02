"""Train-only extra observations and batch-local hard identity crop selection."""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from rcr.common.data import required_identity_ids


def identity_sampling_settings(config):
    settings = {
        "max_crops": 0,
        "crops_per_identity": 2,
        "random_identities": 8,
        "hard_negative_fraction": 0.0,
    }
    if config.keys() - settings.keys():
        raise ValueError("unknown train.identity_sampling setting")
    settings.update(config)
    if settings["max_crops"] < 0 or settings["crops_per_identity"] < 1:
        raise ValueError("invalid identity crop limits")
    if settings["random_identities"] < 0:
        raise ValueError("random_identities must be nonnegative")
    fraction = settings["hard_negative_fraction"]
    if (
        type(fraction) not in (int, float)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("hard_negative_fraction must be finite and in [0, 1]")
    return settings


@torch.no_grad()
def select_identity_observations(query_identity, target_identity, batch, settings):
    """Return indices only; callers gather the original tensors for SupCon.

    Preserve the sampler's cross-image positive quota before allocating hard
    crops. Random fallback uses the baseline sampler's remaining observations,
    so its per-identity/random-identity limits still apply. Full-negative images
    supply candidates, not negative identity labels: only unequal GT IDs compete.
    """
    limit = settings["max_crops"]
    positive = batch["extra_identity_positive_mask"].nonzero(as_tuple=True)[0][:limit]
    remaining = max(0, limit - positive.numel())
    quota = math.floor(settings["hard_negative_fraction"] * remaining)
    candidates = batch["supplemental_target_indices"]
    hard = candidates[:0]
    anchors = query_identity.detach()[batch["identity_anchor_mask"]]
    if quota and len(anchors) and len(candidates):
        with torch.autocast(query_identity.device.type, enabled=False):
            target = target_identity.detach().flatten(0, 2)[candidates].float()
            similarity = (
                F.normalize(anchors.float(), dim=-1) @ F.normalize(target, dim=-1).T
            )
        labels = batch["query_identity_labels"][batch["identity_anchor_mask"]]
        different = labels[:, None] != batch["supplemental_target_labels"][None]
        hardness = similarity.masked_fill(~different, -torch.inf).amax(0)
        # Some sampled positives are already in negative candidate images. They
        # keep their reserved slot and cannot be mined again under another role.
        reserved = batch["extra_identity_target_indices"][positive]
        hardness.masked_fill_(torch.isin(candidates, reserved), -torch.inf)
        order = hardness.argsort(descending=True, stable=True)
        order = order[torch.isfinite(hardness[order])][:quota]
        hard = candidates[order]
    random = (~batch["extra_identity_positive_mask"]).nonzero(as_tuple=True)[0]
    random = random[~torch.isin(batch["extra_identity_target_indices"][random], hard)][
        : remaining - hard.numel()
    ]
    return hard, torch.cat((positive, random))


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
                other[index] for index in order[: self.settings["random_identities"]]
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
