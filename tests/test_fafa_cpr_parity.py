"""CPU contracts shared with cpr_baseline_bench FAFA + Partial SetMatch."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from rcr.baselines.fafa_adapter import select_subject_members, subject_changes
from rcr.baselines.fafa_matching import aggregate_people, partial_setmatch


def _reference(s, b):
    if not s.shape[1]:
        return b
    rows, cols = linear_sum_assignment(-s)
    return (s[rows, cols].sum() + (len(s) - len(rows)) * b) / len(s)


@pytest.mark.parametrize(
    "k,n", [(1, 1), (1, 5), (2, 1), (2, 2), (2, 6), (3, 1), (3, 2), (3, 4), (5, 3)]
)
def test_partial_setmatch_exact(k, n):
    rng = np.random.default_rng(42 + k + n)
    matrix = rng.uniform(-1, 1, (k, n))
    assert partial_setmatch(matrix, -1.0) == pytest.approx(_reference(matrix, -1.0))


def test_mean_penalty_not_minimum():
    s = np.array([[0.8], [0.6]])
    assert partial_setmatch(s, -1) == pytest.approx(-0.1)
    matrix = np.array([[0.81, 0.23], [0.20, 0.79]])
    assert partial_setmatch(matrix, -1) == pytest.approx(0.8)


def test_batch_aggregation_against_rectangular_reference():
    rng = np.random.default_rng(3)
    offsets = np.r_[0, np.cumsum([1, 2, 4, 1, 5, 3, 2])]
    for k in range(1, 8):
        scores = rng.uniform(-1, 1, (k, offsets[-1]))
        got = aggregate_people(scores, offsets, -1)
        expected = np.array(
            [
                _reference(scores[:, a:b], -1)
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)


def test_anchor_relative_margin_does_not_use_row_max():
    # Global Hungarian reserves person 0 for Subject 2; Subject 1 anchor=person 1.
    sim = np.array([[0.92, 0.88, 0.86], [0.99, 0.1, 0.1]])
    members = select_subject_members(sim, threshold=0.2, margin=0.03, max_members=5)
    assert members == [[1, 2], [0]]


def test_cpr_text_independent_and_relational():
    sample = {
        "subjects": [{"subject_id": 1}, {"subject_id": 2}],
        "final_change": (
            "then retrieve target images where Subject 1 is wearing red "
            "and Subject 2 is holding a bag"
        ),
    }
    assert subject_changes(sample) == ["is wearing red", "is holding a bag"]
    sample["final_change"] = (
        "then retrieve target images where Subject 1 stands behind Subject 2"
    )
    assert subject_changes(sample) == [
        "Subject 1 stands behind Subject 2",
        "Subject 1 stands behind Subject 2",
    ]


def test_baseline_and_proposed_share_128_token_policy():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / "configs/fafa.yaml").read_text())
    proposed = yaml.safe_load((root / "configs/proposed.yaml").read_text())
    native = yaml.safe_load(
        (root / proposed["person_encoder"]["fafa_config"]).read_text()
    )
    assert baseline["checkpoint"]["max_txt_len"] == 128
    assert native["checkpoint"]["max_txt_len"] == 128
    assert proposed["person_encoder"]["fafa_config"] == "configs/fafa.yaml"
    assert baseline["checkpoint"]["path"] == native["checkpoint"]["path"]


def test_native_fafa_text_policy_matches_cpr_128():
    from rcr.baselines.fafa import text_policy

    assert text_policy({"checkpoint": {"max_txt_len": 128}}) == {
        "max_txt_len": 128,
        "max_words": 128,
    }


def test_shared_detector_reads_proposed_boxes_without_renormalization(tmp_path):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from rcr.baselines.fafa_detector import candidates_from_cache

    image_paths = []
    for i, size in enumerate(((160, 100), (120, 80))):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", size).save(path)
        image_paths.append(path)

    class FakeCache:
        by_id = {"first": 0, "empty": 1}
        mask = torch.tensor([[True, True], [False, False]])

        def load_geometry(self, index):
            if index == 0:
                return {
                    "boxes_pixel": torch.tensor(
                        [
                            [7.25, 3.5, 58.75, 77.5],
                            [15.0, 9.0, 100.0, 85.0],
                        ]
                    )
                }
            return {"boxes_pixel": torch.empty(0, 4)}

    data = SimpleNamespace(image_path=lambda iid: image_paths[FakeCache.by_id[iid]])
    cfg = {"scene_size": [224, 224]}
    candidates = candidates_from_cache(data, ["first", "empty"], FakeCache(), cfg)
    assert [x["box"] for x in candidates[0]] == [
        [7.25, 3.5, 58.75, 77.5],
        [15.0, 9.0, 100.0, 85.0],
    ]
    assert all(x["score"] == 1.0 and not x["fallback"] for x in candidates[0])
    assert candidates[1] == [
        {
            "box": [0, 0, 120, 80],
            "score": -1.0,
            "fallback": True,
        }
    ]


def test_shared_detector_inverts_legacy_non_square_scene_size(tmp_path):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from rcr.baselines.fafa_detector import candidates_from_cache
    from rcr.proposed.cache.dino import boxes_to_scene

    path = tmp_path / "input.png"
    Image.new("RGB", (400, 200)).save(path)
    original = torch.tensor([[40.0, 20.0, 200.0, 100.0]])

    class FakeCache:
        by_id = {"image": 0}
        mask = torch.tensor([[True]])

        def load_geometry(self, _index):
            return {"boxes_scene": boxes_to_scene(original, (400, 200), (256, 128))}

    data = SimpleNamespace(image_path=lambda _image_id: path)
    candidates = candidates_from_cache(
        data, ["image"], FakeCache(), {"scene_size": [128, 256]}
    )
    torch.testing.assert_close(torch.tensor(candidates[0][0]["box"]), original[0])


def test_proposed_dino_settings_match_fafa_baseline():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    proposed = yaml.safe_load((root / "configs/proposed.yaml").read_text())
    fafa = yaml.safe_load((root / "configs/fafa.yaml").read_text())
    detector = fafa["localization"]["detector"]
    assert detector["backend"] == "proposed_dino_tiny_cache"
    assert detector["cache_dir"] == proposed["data"]["dino_cache"]
    assert detector["model"] == proposed["detector"]["model"]
    assert detector["threshold"] == proposed["detector"]["threshold"]
    assert detector["text_threshold"] == proposed["detector"]["text_threshold"]
    assert detector["scene_size"] == proposed["image_encoder"]["scene_size"]
    assert fafa["checkpoint"]["max_txt_len"] == 128


def test_shared_dino_cache_rejects_wrong_detector_metadata(tmp_path, monkeypatch):

    import torch

    from rcr.baselines.fafa_detector import detect_gallery
    from rcr.proposed.cache import store

    class FakeCache:
        cache_id = "verified-id"
        image_ids = ["image-1"]
        by_id = {"image-1": 0}
        mask = torch.tensor([[True]])
        encoder_metadata = {
            "detector": {
                "model": "IDEA-Research/grounding-dino-tiny",
                "threshold": 0.3,
                "text_threshold": 0.25,
            },
            "image_encoder": {"scene_size": [224, 224]},
        }

        def validate_gallery(self, image_ids):
            assert image_ids == self.image_ids

        def load_geometry(self, index):
            return {"boxes_pixel": torch.tensor([[1.0, 2.0, 30.0, 40.0]])}

    from PIL import Image

    path = tmp_path / "input.png"
    Image.new("RGB", (100, 80)).save(path)
    monkeypatch.setattr(store, "GalleryCache", lambda _: FakeCache())
    from types import SimpleNamespace

    data = SimpleNamespace(gallery_ids=["image-1"], image_path=lambda _: path)
    cfg = {
        "backend": "proposed_dino_tiny_cache",
        "cache_dir": str(tmp_path),
        "model": "IDEA-Research/grounding-dino-tiny",
        "threshold": 0.3,
        "text_threshold": 0.25,
        "scene_size": [224, 224],
    }
    candidates, cache_id = detect_gallery(data, ["image-1"], cfg, torch.device("cpu"))
    assert candidates[0][0]["box"] == [1.0, 2.0, 30.0, 40.0]
    assert cache_id.endswith("#cache_id=verified-id")
    with pytest.raises(ValueError, match="provenance"):
        detect_gallery(
            data, ["image-1"], {**cfg, "threshold": 0.4}, torch.device("cpu")
        )
    FakeCache.encoder_metadata["detector"] = None
    with pytest.raises(ValueError, match="lacks detector provenance"):
        detect_gallery(data, ["image-1"], cfg, torch.device("cpu"))
    _, uncertain = detect_gallery(
        data,
        ["image-1"],
        {**cfg, "allow_legacy_cache": True},
        torch.device("cpu"),
    )
    assert uncertain.endswith("#legacy-unverified")
