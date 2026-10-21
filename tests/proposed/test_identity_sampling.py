"""Gallery crop sampling extends identity supervision without new retrieval I/O."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from rcr.proposed.batch import build_batch, prepare_visual_batch
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.identity import IdentityCropSampler, identity_sampling_settings
from rcr.proposed.sampling import sample_candidates
from tests.proposed import test_batch as fixtures


def _sampler(**overrides):
    known = {
        "10": [("q1", 0), ("p1", 1), ("n1", 0), ("only10", 0)],
        "20": [("q1", 1), ("p1", 0), ("only20a", 0), ("only20b", 0)],
        "30": [("other30a", 0), ("other30b", 0)],
        "single": [("single_image", 0), ("single_image", 1)],
    }
    return IdentityCropSampler(
        SimpleNamespace(identity_crops=lambda: known),
        identity_sampling_settings(
            {
                "max_crops": 8,
                "random_identities": 1,
                **overrides,
            }
        ),
    )


@pytest.mark.parametrize("seed_only", [False, True])
def test_gallery_sampling_needs_each_id_not_cooccurrence_or_full_positive(seed_only):
    sample = dict(fixtures._samples()[0])
    if seed_only:
        sample["target_image_id"] = sample.pop("positive_image_ids")[0]
    candidates = [["p1", "n1"]]
    sampler = _sampler()
    crops = sampler.sample([sample], candidates, torch.Generator().manual_seed(5))
    assert Counter(identity for _, _, identity in crops) == {"10": 2, "20": 2, "30": 2}
    assert len({(image, person) for image, person, _ in crops}) == len(crops)
    assert not {"q1", "p1"} & {image for image, _, _ in crops}
    assert ("n1", 0, "10") in crops  # Retrieval negative, identity positive.
    assert all(
        len({i for i, _, y in crops if y == identity}) == 2
        for identity in ("10", "20", "30")
    )
    assert candidates == [["p1", "n1"]]


@pytest.mark.parametrize("limit", [0, 1, 3, 8])
def test_sampling_is_bounded_reproducible_and_preserves_retrieval_rng(limit):
    sample = {**fixtures._samples()[0], "sample_id": "query"}
    sampler = _sampler(max_crops=limit)
    crops = sampler.sample([sample], [["p1", "n1"]], torch.Generator().manual_seed(19))
    assert len(crops) <= limit
    assert crops == sampler.sample(
        [sample], [["p1", "n1"]], torch.Generator().manual_seed(19)
    )
    gallery = ["q1", "p1", "n1", "n2", "n3"]
    baseline = torch.Generator().manual_seed(7)
    augmented = torch.Generator().manual_seed(7)
    auxiliary = torch.Generator().manual_seed(991)
    for _ in range(4):
        expected = sample_candidates([sample], gallery, 3, baseline)
        actual = sample_candidates([sample], gallery, 3, augmented)
        sampler.sample([sample], actual, auxiliary)
        assert actual == expected
    assert torch.equal(baseline.get_state(), augmented.get_state())


def test_missing_detections_and_exhausted_gallery_need_no_fake_positive_pairs():
    sampler = _sampler(random_identities=0)
    sample = fixtures._samples()[0]
    sample = {
        **sample,
        "positive_image_ids": ["p1", "n1", "only10", "only20a", "only20b"],
    }
    assert (
        sampler.sample(
            [sample], [sample["positive_image_ids"]], torch.Generator().manual_seed(0)
        )
        == []
    )
    sample = {**sample, "subjects": [{"identity_ids": ["missed"]}]}
    assert sampler.sample([sample], [[]], torch.Generator().manual_seed(0)) == []


@pytest.mark.parametrize(
    "settings",
    [
        {"max_crops": -1},
        {"max_crops": True},
        {"crops_per_identity": 0},
        {"random_identities": -1},
        {"crops_per_identity": 1.5},
    ],
)
def test_invalid_identity_sampling_settings_fail_early(settings):
    with pytest.raises(ValueError, match="train.identity_sampling"):
        identity_sampling_settings(settings)


@pytest.mark.parametrize("packed", [False, True])
def test_auxiliary_batch_reads_only_pooled_rows_and_shares_labels(packed):
    class Cache(fixtures._Cache):
        def __init__(self):
            super().__init__()
            people = torch.tensor(
                [[[float(i)] * 4, [float(i + 10)] * 4] for i in range(6)]
            ).half()
            self.mask = torch.tensor([[False, True, False, True]] * 6)
            self.person_offsets = torch.arange(0, 13, 2) if packed else None
            if packed:
                self.persons = people.flatten(0, 1)
            else:
                self.persons = torch.zeros(6, 4, 4).half()
                self.persons[:, [1, 3]] = people

        def load_groups(self, *groups):
            self.groups = [g.tolist() for g in groups]
            return super().load_groups(*groups)

        def person_row(self, index):
            assert index in (3, 5), "query/positive crops should have been deduplicated"
            return GalleryCache.person_row(self, index)

        def native_targets(self, indices, max_people):
            assert indices.tolist() == [2, 4]
            return super().native_targets(indices, max_people)

    cache = Cache()
    # q1/p1 and repeated n1 observations must not be counted a second time.
    crops = [
        ("q1", 0, "10"),
        ("p1", 1, "10"),
        ("n1", 0, "10"),
        ("n1", 0, "10"),
        ("n2", 1, "new-id"),
    ]
    batch = prepare_visual_batch(
        fixtures._samples()[:1],
        [["p1", "p2"]],
        cache,
        supervision=fixtures._Supervision(),
        identity_crops=crops,
    )
    assert cache.groups == [[0], [2, 4]]  # No auxiliary image in visual/native groups.
    assert batch["extra_identity_labels"].tolist() == [0, 2]
    assert batch["query_identity_labels"][0, 0] == batch["extra_identity_labels"][0]
    assert batch["extra_identity_persons"].dtype == torch.float32
    torch.testing.assert_close(
        batch["extra_identity_persons"], torch.tensor([[3.0] * 4, [15.0] * 4])
    )
    assert batch["target_identity_labels"][0, 1].tolist() == [-1, -1, -1]
    assert batch["positive_mask"].tolist() == [[True, False]]
    synchronous = build_batch(
        fixtures._samples()[:1],
        [["p1", "p2"]],
        cache,
        fixtures._Tokenizer(),
        fixtures._TextEncoder(),
        "cpu",
        supervision=fixtures._Supervision(),
        identity_crops=crops,
    )
    for key in ("extra_identity_persons", "extra_identity_labels"):
        torch.testing.assert_close(synchronous[key], batch[key], rtol=0, atol=0)
