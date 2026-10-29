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


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("invalid", ["truncated", "wrong_dtype"])
def test_fafa_rebuilds_invalid_features_then_reuses_the_replacement(
    tmp_path, query, invalid
):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from rcr.baselines.fafa import encode_crops

    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)
    path = tmp_path / "features.npy"
    shape = (1, 3) if query else (1, 2, 3)
    if invalid == "truncated":
        path.write_bytes(b"interrupted NPY")
    else:
        np.save(path, np.zeros(shape, dtype=np.int64))
    calls = []

    class Model:
        def extract_features(self, inputs):
            calls.append("query")
            return SimpleNamespace(
                multimodal_embeds=torch.ones(len(inputs["image"]), 3)
            )

        def extract_target_features(self, images, mode):
            calls.append("image")
            return torch.ones(len(images), 2, 3), None

    kwargs = {
        "model": Model(),
        "processors": {"eval": str},
        "preprocess": lambda _: torch.zeros(3, 4, 4),
        "items": [(str(image), (0, 0, 4, 4), "condition")],
        "runtime": {
            "query_batch_size": 1,
            "image_batch_size": 1,
            "gallery_feature_dtype": "float16",
        },
        "device": torch.device("cpu"),
        "query": query,
        "path": path,
    }
    features = encode_crops(**kwargs)
    assert features.shape == shape
    assert features.dtype == (np.float32 if query else np.float16)
    np.testing.assert_array_equal(features, np.ones(shape))
    np.testing.assert_array_equal(encode_crops(**kwargs), features)
    assert calls == ["query" if query else "image"]


def test_fafa_rebuilds_truncated_component_scores(tmp_path):
    import torch

    from rcr.baselines.fafa import score_components

    path = tmp_path / "scores.npy"
    path.write_bytes(b"interrupted NPY")
    scores = score_components(
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[[0.2, 0.8], [0.6, 0.4]]], dtype=np.float32),
        path,
        {"score_person_batch_size": 1, "score_query_batch_size": 1},
        torch.device("cpu"),
        {"fda_k": 1, "use_soft": True},
    )
    np.testing.assert_allclose(scores, [[0.6]])
    assert scores.dtype == np.float32


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
                    ),
                    "image_size": (160, 100),
                }
            return {"boxes_pixel": torch.empty(0, 4), "image_size": (120, 80)}

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


def test_pixel_boxes_follow_non_square_letterbox_coordinates():
    import torch

    from rcr.common.geometry import boxes_to_scene

    original = torch.tensor([[40.0, 20.0, 200.0, 100.0]])

    torch.testing.assert_close(
        boxes_to_scene(original, (400, 200), (256, 128)),
        torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
    )
    torch.testing.assert_close(
        boxes_to_scene(original, (400, 200), (224, 224)),
        torch.tensor([[0.1, 0.3, 0.5, 0.5]]),
    )


def test_proposed_dino_settings_match_fafa_baseline():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    proposed = yaml.safe_load((root / "configs/proposed.yaml").read_text())
    fafa = yaml.safe_load((root / "configs/fafa.yaml").read_text())
    detector = fafa["localization"]["detector"]
    assert detector["backend"] == "grounding_dino_cache"
    assert detector["cache_dir"] == proposed["data"]["detection_cache"]
    assert detector["model"] == proposed["detector"]["model"]
    assert detector["threshold"] == proposed["detector"]["threshold"]
    assert detector["text_threshold"] == proposed["detector"]["text_threshold"]
    assert "scene_size" not in detector
    assert fafa["checkpoint"]["max_txt_len"] == 128


def test_shared_dino_cache_rejects_wrong_detector_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from rcr.baselines.fafa_detector import detect_gallery
    from rcr.common import detection

    path = tmp_path / "input.png"
    Image.new("RGB", (100, 80)).save(path)
    monkeypatch.setattr(
        detection,
        "detect_persons",
        lambda *a: torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
    )
    spec = {
        "model": "IDEA-Research/grounding-dino-tiny",
        "threshold": 0.3,
        "text_threshold": 0.25,
    }
    root = tmp_path / "detections"
    detection.build_detection_cache(
        ["image-1"], [path], root, spec, model=torch.nn.Identity()
    )
    data = SimpleNamespace(gallery_ids=["image-1"], image_path=lambda _: path)
    cfg = {
        "backend": "grounding_dino_cache",
        "cache_dir": str(root),
        **spec,
    }
    candidates, cache_id = detect_gallery(data, ["image-1"], cfg, torch.device("cpu"))
    assert candidates[0][0]["box"] == [1.0, 2.0, 30.0, 40.0]
    assert cache_id == f"detections:{detection.DetectionCache(root).cache_id}"
    with pytest.raises(ValueError, match="provenance"):
        detect_gallery(
            data, ["image-1"], {**cfg, "threshold": 0.4}, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="grounding_dino_cache"):
        detect_gallery(
            data,
            ["image-1"],
            {**cfg, "backend": "proposed_dino_tiny_cache"},
            torch.device("cpu"),
        )
